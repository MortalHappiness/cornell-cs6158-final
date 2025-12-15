import contextlib
import re
import unittest
from unittest.mock import patch
import numpy as np
import tvm
from tvm import relay
from tvm.ir.module import IRModule
from tvm.relay import transform
import tvm.testing
from tvm.runtime import container
import math

# TVM equivalent for TorchInductor config.
class TVMConfig:
    def __init__(self):
        self.split_cat_fx_passes = False
        self.max_pointwise_cat_inputs = 0
        self.realize_opcount_threshold = 0
        self.pattern_matcher = False
        self.pre_grad_fusion_options = {}
        self.post_grad_fusion_options = {}
        self.triton = TVMTritonConfig()

class TVMTritonConfig:
    def __init__(self):
        self.cooperative_reductions = False

config = TVMConfig() # Global config object

# Inductor metrics are untranslatable, will be mocked or replaced.
class TVMMetrics:
    def reset(self):
        pass # No-op
    @property
    def nodes_num_elem(self):
        return {} # Empty dict
    @property
    def num_bytes_accessed(self):
        return 0 # Dummy value

metrics = TVMMetrics()

# Constants / Helpers for TVM
DEVICE = "cuda" # Default for original tests
TARGET = tvm.target.Target(DEVICE)
CTX = tvm.cuda(0) if DEVICE == "cuda" else tvm.cpu(0)

TVM_METRIC_PLACEHOLDER = "TVM_METRIC_TODO"

# Helper function to convert numeric dtypes to TVM string dtypes
def _to_tvm_dtype(dtype_like):
    if isinstance(dtype_like, str):
        return dtype_like
    return str(dtype_like)

# MockTensor: Acts as a proxy for a relay.Expr during graph construction
class MockTensor:
    _relay_expr_counter = 0

    def __init__(self, shape, dtype='float32', name=None, _relay_expr=None):
        self.shape = shape
        self.dtype = _to_tvm_dtype(dtype)
        self.name = name if name else f"__mock_tensor_{MockTensor._relay_expr_counter}__"
        MockTensor._relay_expr_counter += 1

        if _relay_expr is None:
            self._relay_expr = relay.var(self.name, shape=self.shape, dtype=self.dtype)
        else:
            self.name = name if name else f"__mock_expr_{MockTensor._relay_expr_counter-1}__"
            self._relay_expr = _relay_expr
            try:
                checked_type = _relay_expr.checked_type
                self.shape = tuple(checked_type.shape)
                self.dtype = str(checked_type.dtype)
            except tvm.error.TVMError:
                pass

    @property
    def ndim(self): return len(self.shape)
    @property
    def numel(self): return np.prod(self.shape)
    @property
    def device(self): return DEVICE # Mock device property
    @property
    def requires_grad(self): return False # Assume no grad for forward pass tracing

    def __repr__(self): return f"MockTensor(shape={self.shape}, dtype='{self.dtype}', name='{self.name}')"

    # --- Element-wise ops ---
    def cos(self): return MockTensor(self.shape, self.dtype, _relay_expr=relay.op.tensor.cos(self._relay_expr))
    def sin(self): return MockTensor(self.shape, self.dtype, _relay_expr=relay.op.tensor.sin(self._relay_expr))
    def ceil(self): return MockTensor(self.shape, self.dtype, _relay_expr=relay.op.tensor.ceil(self._relay_expr))
    def floor(self): return MockTensor(self.shape, self.dtype, _relay_expr=relay.op.tensor.floor(self._relay_expr))
    def round(self): return MockTensor(self.shape, self.dtype, _relay_expr=relay.op.tensor.round(self._relay_expr))
    def clone(self): return MockTensor(self.shape, self.dtype, _relay_expr=relay.op.tensor.copy(self._relay_expr))
    def sigmoid(self): return MockTensor(self.shape, self.dtype, _relay_expr=relay.op.tensor.sigmoid(self._relay_expr))

    # --- Reduction ops ---
    def sum(self, dim=None, keepdim=False, dtype=None):
        axis = dim if isinstance(dim, (list, tuple)) else ([dim] if dim is not None else None)
        new_shape = self._calculate_reduction_shape(axis, keepdim)
        output_dtype = _to_tvm_dtype(dtype) if dtype else self.dtype
        return MockTensor(new_shape, output_dtype, _relay_expr=relay.op.reduce.sum(self._relay_expr, axis=axis, keepdims=keepdim))
    
    def amax(self, dim=None, keepdim=False):
        axis = dim if isinstance(dim, (list, tuple)) else ([dim] if dim is not None else None)
        new_shape = self._calculate_reduction_shape(axis, keepdim)
        return MockTensor(new_shape, self.dtype, _relay_expr=relay.op.reduce.max(self._relay_expr, axis=axis, keepdims=keepdim))

    def _calculate_reduction_shape(self, axis, keepdim):
        current_shape = list(self.shape)
        if axis is None:
            return (1,) if keepdim else ()
        
        axes_to_reduce = sorted([a % len(current_shape) for a in axis])
        
        new_shape = []
        for i, dim_size in enumerate(current_shape):
            if i in axes_to_reduce:
                if keepdim:
                    new_shape.append(1)
            else:
                new_shape.append(dim_size)
        return tuple(new_shape)

    # --- Shape/Structure ops ---
    def t(self): # Transpose for 2D only
        if self.ndim != 2: raise ValueError("Transpose .t() is only defined for 2D tensors")
        return MockTensor(self.shape[::-1], self.dtype, _relay_expr=relay.op.transform.transpose(self._relay_expr, axes=[1, 0]))

    def permute(self, *dims):
        new_shape = [self.shape[d] for d in dims]
        return MockTensor(tuple(new_shape), self.dtype, _relay_expr=relay.op.transform.transpose(self._relay_expr, axes=list(dims)))

    def view(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = new_shape[0]
        return MockTensor(new_shape, self.dtype, _relay_expr=relay.op.transform.reshape(self._relay_expr, newshape=new_shape))

    # --- Inplace ops (mapped to functional + identity for tracing) ---
    def copy_(self, other):
        self._relay_expr = relay.op.tensor.copy(_get_relay_expr(other))
        self.shape = _get_mock_tensor(other).shape
        self.dtype = _get_mock_tensor(other).dtype
        return self
    def add_(self, other):
        other_expr = _get_relay_expr(other)
        self._relay_expr = relay.op.tensor.add(self._relay_expr, other_expr)
        self.shape = self._calculate_binary_op_shape(other)
        return self
    def mul_(self, other):
        other_expr = _get_relay_expr(other)
        self._relay_expr = relay.op.tensor.multiply(self._relay_expr, other_expr)
        self.shape = self._calculate_binary_op_shape(other)
        return self
    
    # --- Indexing (for tracing, assuming functional access or scatter) ---
    def __getitem__(self, key):
        if isinstance(key, MockTensor) and key.ndim == 1:
            new_shape = (key.shape[0],) + self.shape[1:]
            return MockTensor(new_shape, self.dtype, _relay_expr=relay.op.transform.take(self._relay_expr, key._relay_expr, axis=0, mode="clip"))
        
        if isinstance(key, tuple):
            index_tensor = None
            axis_to_gather = -1
            for i, k_item in enumerate(key):
                if isinstance(k_item, MockTensor):
                    index_tensor = k_item
                    axis_to_gather = i
                    break
            
            if index_tensor and axis_to_gather != -1:
                output_shape = list(self.shape)
                output_shape[axis_to_gather] = index_tensor.shape[0]
                return MockTensor(tuple(output_shape), self.dtype, _relay_expr=relay.op.transform.gather(self._relay_expr, _get_relay_expr(index_tensor), axis=axis_to_gather))

        return MockTensor((1,), self.dtype, f"{self.name}_get_item")

    def __setitem__(self, key, value):
        if isinstance(key, MockTensor) and key.ndim == 1: # a[b] = 1
            updates_expr = _get_relay_expr(value)
            if not isinstance(value, MockTensor): 
                updates_expr = relay.op.transform.full(updates_expr, shape=key.shape, dtype=self.dtype)
            
            self._relay_expr = relay.op.transform.scatter(self._relay_expr, key._relay_expr, updates_expr, axis=0)
            return self
        
        if isinstance(key, tuple):
            index_tensor = None
            scatter_axis = -1
            for i, k_item in enumerate(key):
                if isinstance(k_item, MockTensor):
                    index_tensor = k_item
                    scatter_axis = i
                    break
            
            if index_tensor is not None and scatter_axis != -1:
                updates_expr = _get_relay_expr(value)
                if not isinstance(value, MockTensor):
                    target_shape_list = list(self.shape)
                    target_shape_list[scatter_axis] = index_tensor.shape[0]
                    updates_expr = relay.op.transform.full(updates_expr, shape=tuple(target_shape_
