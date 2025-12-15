import contextlib
import copy
import functools
import itertools
import math
import os
import platform
import sys
import unittest
from typing import Callable, Sequence, Tuple, Union
from unittest.mock import patch

import numpy as np

# TVM imports
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.op import nn
from tvm.relay.op import transform
from tvm.relay.op import reduce
from tvm.relay.op import random
from tvm.relay.op import algorithm
from tvm.topi import transform as topi_transform
from tvm.topi import utils as topi_utils
from tvm.tir import op as tir_op
from tvm.relay.frontend import common as frontend_common
from tvm.relay import qnn

# Dummy replacements for PyTorch Inductor specific utilities
class DummyConfig:
    class Cpp:
        simdlen = None
        fallback_scatter_reduce_sum = False
        dynamic_threads = False
        max_horizontal_fusion_size = 0
        enable_tiling_heuristics = True
        enable_kernel_profile = False
        descriptive_names = ""
    cpp = Cpp()

class DummyMetrics:
    generated_cpp_vec_kernel_count = 0
    generated_kernel_count = 0
    cpp_to_dtype_count = 0
    cpp_outer_loop_fused_inner_counts = []
    parallel_reduction_count = 0

config = DummyConfig()
metrics = DummyMetrics()

class DummyCpuVecIsa:
    def valid_vec_isa_list(self):
        class DummyISA:
            def bit_width(self): return 256
            def nelements(self, dtype_str=None):
                if dtype_str in ["bfloat16", "float16"]: return 32
                return 16 # Default for float32 or int32

            def __eq__(self, other):
                return self.bit_width() == other.bit_width()
            def __str__(self): return "avx2"
        return [DummyISA()] if os.getenv("TVM_TEST_VECTORIZATION", "1") == "1" else []

    @property
    def supported_vec_isa_list(self):
        return self.valid_vec_isa_list()

    def pick_vec_isa(self):
        valid = self.valid_vec_isa_list()
        return valid[0] if valid else None

cpu_vec_isa = DummyCpuVecIsa()

class DummyTestOperators:
    def realize(self, x):
        return x
test_operators = DummyTestOperators()

# Dummy for TorchDispatchMode (PyTorch-specific)
class TorchDispatchMode:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return None # Should not be called in TVM context

# Mapping for PyTorch dtypes to TVM string dtypes
PYTORCH_DTYPE_TO_TVM_STR = {
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.int8": "int8",
    "torch.uint8": "uint8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.bool": "bool",
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "torch.float8_e5m2": "float8_e5m2",
}

def convert_pt_dtype_to_tvm_str(pt_dtype):
    if isinstance(pt_dtype, str) and pt_dtype.startswith("torch."):
        return PYTORCH_DTYPE_TO_TVM_STR.get(pt_dtype, pt_dtype)
    if hasattr(pt_dtype, '__module__') and pt_dtype.__module__ == 'torch':
        return PYTORCH_DTYPE_TO_TVM_STR.get(f"torch.{pt_dtype.name}", str(pt_dtype))
    if pt_dtype is float: return "float32"
    if pt_dtype is int: return "int32"
    if pt_dtype is bool: return "bool"
    return str(pt_dtype)

# Helper to execute TVM model and compare with numpy reference
def check_model_tvm(
    pt_func_or_module,
    pt_inputs,
    relay_builder: Callable[[Sequence[relay.Var]], relay.Expr], # Callable to build relay graph
    atol=1e-5,
    rtol=1e-5,
    target="llvm",
    num_expect_outputs=None, # Explicitly tell how many outputs to expect
    test_case_instance=None, # For calling self.assertEqual/assertTrue etc.
    output_code_check_fn=None, # For checking generated code (mocked in TVM context)
    output_code_check_not_fn=None, # For checking absence of code (mocked)
    **kwargs # Catch other args not used by TVM
):
    # 1. Run original PyTorch model/function for expected output
    pt_inputs_cloned = []
    for inp in pt_inputs:
        if isinstance(inp, torch.Tensor):
            pt_inputs_cloned.append(inp.clone().detach())
        elif isinstance(inp, (tuple, list)):
            pt_inputs_cloned.append(tuple(x.clone().detach() if isinstance(x, torch.Tensor) else x for x in inp))
        else:
            pt_inputs_cloned.append(inp)

    if isinstance(pt_func_or_module, torch.nn.Module):
        pt_func_or_module.eval()
        with torch.no_grad():
            expected_outputs_pt = pt_func_or_module(*pt_inputs_cloned)
    elif callable(pt_func_or_module):
        with torch.no_grad():
            expected_outputs_pt = pt_func_or_module(*pt_inputs_cloned)
    else:
        raise TypeError("pt_func_or_module must be an nn.Module or a callable.")

    if not isinstance(expected_outputs_pt, (tuple, list)):
        expected_outputs_pt = (expected_outputs_pt,)

    # 2. Prepare TVM inputs and Relay input variables
    relay_input_vars_list = []
    tvm_input_ndarrays = []

    for i, pt_input in enumerate(pt_inputs):
        if isinstance(pt_input, torch.Tensor):
            tvm_input_ndarrays.append(tvm.nd.array(pt_input.cpu().numpy()))
            relay_input_vars_list.append(relay.var(f"p{i}", shape=pt_input.shape, dtype=convert_pt_dtype_to_tvm_str(pt_input.dtype)))
        elif isinstance(pt_input, (tuple, list)): # For LSTM (h, c)
            sub_vars = []
            for j, sub_pt_input in enumerate(pt_input):
                if isinstance(sub_pt_input, torch.Tensor):
                    tvm_input_ndarrays.append(tvm.nd.array(sub_pt_input.cpu().numpy()))
                    sub_vars.append(relay.var(f"p{i}_{j}", shape=sub_pt_input.shape, dtype=convert_pt_dtype_to_tvm_str(sub_pt_input.dtype)))
                else: # Scalar like values
                    tvm_input_ndarrays.append(sub_pt_input)
                    sub_vars.append(relay.var(f"p{i}_{j}", dtype=convert_pt_dtype_to_tvm_str(type(sub_pt_input))))
            relay_input_vars_list.append(tuple(sub_vars))
        elif isinstance(pt_input, (int, float, bool, str)): # Direct scalars
            tvm_input_ndarrays.append(pt_input)
            relay_input_vars_list.append(relay.var(f"p{i}", dtype=convert_pt_dtype_to_tvm_str(type(pt_input))))
        else:
            raise TypeError(f"Unsupported PyTorch input type for TVM conversion: {type(pt_input)}")

    flat_relay_input_vars = []
    flat_tvm_input_ndarrays = []
    for var_or_tuple, data_or_tuple in zip(relay_input_vars_list, tvm_input_ndarrays):
        if isinstance(var_or_tuple, (tuple, list)):
            flat_relay_input_vars.extend(var_or_tuple)
            if isinstance(data_or_tuple, (tuple, list)):
                flat_tvm_input_ndarrays.extend(data_or_tuple)
            else: # Should not happen if `tvm_input_ndarrays` is consistently built
                raise TypeError(f"Mismatched data structure for tuple input: {type(data_or_tuple)}")
        else:
            flat_relay_input_vars.append(var_or_tuple)
            flat_tvm_input_ndarrays.append(data_or_tuple)

    # 3. Build Relay graph
    relay_expr = relay_builder(*relay_input_vars_list)
    relay_func = relay.Function(flat_relay_input_vars, relay_expr)
    mod = tvm.IRModule.from_expr(relay_func)

    # 4. Compile and run the TVM Relay IRModule
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    dev = tvm.device(str(target), 0)
    executor = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    input_idx_counter = 0
    for data in flat_tvm_input_ndarrays:
        if isinstance(data, (tvm.nd.NDArray, np.ndarray)):
            executor.set_input(input_idx_counter, data)
            input_idx_counter += 1
        elif isinstance(data, (int, float, bool, str)):
            pass
        else:
            raise TypeError(f"Unexpected data type in flat_tvm_input_ndarrays for executor.set_input: {type(data)}")

    executor.run()
    num_outputs_tvm = executor.get_num_outputs()

    if num_expect_outputs is not None and num_outputs_tvm != num_expect_outputs:
