import functools
import itertools
import math
import numpy as np
import pytest
import unittest

import tvm
from tvm import relay
from tvm.relay.testing import run_infer_type # Used for checking types if needed
import tvm.testing
from tvm.contrib import graph_executor # To run compiled modules

# --- Mocking PyTorch-specific components and constants ---

# Mocking PyTorch Tensor attributes and methods for eager path simulation
class DummyTensor:
    def __init__(self, np_array, device_str="cpu"):
        self._array = np_array
        self.shape = np_array.shape
        self.dtype = self._numpy_dtype_to_torch_str(np_array.dtype)
        self.device = device_str

    def _numpy_dtype_to_torch_str(self, dtype):
        if dtype == np.float32: return "torch.float"
        if dtype == np.float16: return "torch.half"
        if dtype == np.bool_: return "torch.bool"
        if dtype == np.int32: return "torch.int32"
        if dtype == np.int64: return "torch.int64"
        return str(dtype)

    def __getitem__(self, item):
        return self._array[item]

    def size(self, dim=None):
        if dim is None:
            return self.shape
        # PyTorch can take negative dims, calculate absolute
        if isinstance(dim, int) and dim < 0:
            dim = len(self.shape) + dim
        return self.shape[dim]

    def is_floating_point(self):
        return self._array.dtype.kind == 'f'

    def clone(self):
        return DummyTensor(self._array.copy(), self.device)

    def requires_grad_(self, requires_grad):
        # Dummy for PyTorch specific, TVM Relay graph does not have this concept directly
        pass

    @property
    def ndim(self):
        return len(self.shape)

    # Needed for math.sqrt(key.shape[-1]) and similar attribute access
    def __len__(self):
        return len(self.shape)

    def transpose(self, dim0, dim1):
        # Simulate PyTorch's transpose. For numpy, swapaxes is equivalent.
        # Handle negative dimensions.
        actual_dim0 = dim0 if dim0 >= 0 else self.ndim + dim0
        actual_dim1 = dim1 if dim1 >= 0 else self.ndim + dim1
        new_axes = list(range(self.ndim))
        new_axes[actual_dim0], new_axes[actual_dim1] = new_axes[actual_dim1], new_axes[actual_dim0]
        return DummyTensor(self._array.transpose(new_axes), self.device)

    def permute(self, dims):
        return DummyTensor(self._array.transpose(dims), self.device)

    def reshape(self, *shape):
        # PyTorch reshape can take multiple args or a tuple
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return DummyTensor(self._array.reshape(shape), self.device)

    def view(self, *shape):
        # view is similar to reshape for numpy
        return self.reshape(*shape)

    def to(self, dtype):
        # Simulate .to(dtype)
        if isinstance(dtype, str) and dtype.startswith("torch."):
            np_dtype_map = {
                "torch.float": np.float32,
                "torch.half": np.float16,
                "torch.bool": np.bool_,
                "torch.int32": np.int32,
                "torch.int64": np.int64,
            }
            np_dtype = np_dtype_map.get(dtype)
            if np_dtype is None:
                raise ValueError(f"Unsupported torch dtype string: {dtype}")
        elif isinstance(dtype, str): # direct TVM dtype string
            np_dtype = {
                "float32": np.float32,
                "float16": np.float16,
                "bool": np.bool_,
                "int32": np.int32,
                "int64": np.int64,
            }.get(dtype)
            if np_dtype is None:
                raise ValueError(f"Unsupported TVM dtype string: {dtype}")
        else: # Assume it's a direct numpy dtype
            np_dtype = dtype
        return DummyTensor(self._array.astype(np_dtype), self.device)

    def type(self, dtype):
        return self.to(dtype)


# Helper for softmax on numpy arrays
def _softmax_numpy(x, dim):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / np.sum(e_x, axis=dim, keepdims=True)

# Helper for handling torch.finfo(dtype).min
class FInfoMock:
    def __init__(self):
        pass
    def __call__(self, dtype_str):
        if "float16" in dtype_str: return -np.float16(np.inf)
        if "float32" in dtype_str or "float" in dtype_str: return -np.float32(np.inf)
        return -np.inf # Default for other float types
finfo_mock = FInfoMock()


# Mocking aot_graph_input_parser from torch._dynamo.debug_utils
# This function inspects a PyTorch function's signature and generates dummy inputs.
# We need a NumPy-based equivalent that respects the expected tensor_shape patterns.
def aot_graph_input_parser_mock(func, device):
    import inspect
    sig = inspect.signature(func)
    dummy_inputs = {}
    default_tensor_shape = (4, 2, 16, 32) # Common shape used in tests

    for param_name, param in sig.parameters.items():
        if param_name == "training":
            dummy_inputs[param_name] = False # Assume inference mode for inputs in parsing
        elif param_name == "is_inv_factor":
            dummy_inputs[param_name] = True # Default for Model
        elif param_name == "attn_mask":
            # Determine shape for attn_mask based on context.
            # For _test_insignificant_strides, mul_2 is (1,1,1,1)
            # For _test_pattern_fails_with_unsupported_mask, it's (2,4,4,4)
            # For _test_sdpa_rewriter_19, it's (16,16)
            # We'll use a generic shape for default, but tests might override.
            if func.__name__ == "forward_eager": # For _test_insignificant_strides
                if param_name == "mul_2": # Used as attn_mask in that specific test
                     dummy_inputs[param_name] = np.random.randn(1, 1, 1, 1).astype(np.float32)
                elif param_name in ["permute_3", "permute_4", "permute_5"]:
                    dummy_inputs[param_name] = np.random.randn(1, 32, 1, 128).astype(np.float32)
                elif param_name == "permute_6":
                    dummy_inputs[param_name] = np.random.randn(1, 1, 64).astype(np.float32)
                else:
                    dummy_inputs[param_name] = np.random.randn(*default_tensor_shape).astype(np.float32)
            elif func.__name__ == "forward": # For Model in _test_pattern_fails_with_unsupported_mask
                # This needs to be a Numpy array, not a DummyTensor, to be passed to relay builder as such
                dummy_inputs[param_name] = np.random.randn(*default_tensor_shape).astype(np.int32) # Example int mask
            else: # Default for general attn_mask
                dummy_inputs[param_name] = np.random.randn(*default_tensor_shape).astype(np.float32)
        elif param_name == "causal_mask":
             # Shape (16,16) assumed in _test_sdpa_rewriter_18
             dummy_inputs[param_name] = np.tril(np.ones((16, 16), dtype=np.bool_), k=0)
        elif param_name == "scale_factor":
            dummy_inputs[param_name] = np.random.randn(4, 1, 1).astype(np.float32)
        elif param_name.startswith("permute_") or param_name.startswith("mul_"):
             # For _test_insignificant_strides, params are named specifically
            if param_name in ["permute_3", "permute_4", "permute_5"]:
                dummy_inputs[param_name] = np.random.randn(1, 32, 1, 128).astype(np.float32)
            elif param_name == "permute_6":
                dummy_inputs[param_name] = np.random.randn(1, 1, 64).astype(np.float32)
            elif param_name == "mul_2":
                dummy_inputs[param_name] = np.random.randn(1, 1, 1, 1).astype(np.float32)
            else:
                dummy_inputs[param_name] = np.random.randn(*default_tensor_shape).astype(np.float32)
        else:
            # Default tensor shape for other tensor inputs
            dummy_inputs[param_name] = np.random.randn(*default_tensor_shape).astype(np.float32)
    return dummy_inputs

# Mock for `counters`
class MockCounters:
    def __init__(self):
        self.data = {"inductor": {"fuse_attention": 0}}
    def clear(self):
        self.data = {"inductor": {"fuse_attention": 0}}
    def __getitem__(self, key):
        return self.data[key]
counters = MockCounters()

# Mapping for torch.utils.checkpoint.checkpoint
# torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True) -> tvm.relay.op.annotation.annotation.checkpoint(expr)
def checkpoint_wrapper(fn):
    # For TVM, checkpoint_wrapper just returns the original function, then during relay build,
    # we'd annotate the resulting expression. The actual annotation is applied in the `relay_builder` function.
    return fn
