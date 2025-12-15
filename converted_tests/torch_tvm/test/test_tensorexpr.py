import numpy as np
import pytest
import tvm
from tvm import relay
from tvm import topi
from tvm.relay import op as relay_op
from tvm.relay.op import nn as nn_op
from tvm.relay.op import transform as transform_op
from tvm.relay.op import algorithm as algorithm_op
from tvm.relay.op import random as random_op
from tvm.tir import op as tir_op
import tvm.testing
import itertools
import scipy.special # For erf, erfc, lgamma reference computations

# Helper for dtypes
def _to_tvm_dtype(torch_dtype_str):
    dtype_map = {
        'float32': 'float32',
        'bfloat16': 'bfloat16',
        'float64': 'float64',
        'float16': 'float16',
        'int32': 'int32',
        'int16': 'int16',
        'int8': 'int8',
        'int64': 'int64',
        'bool': 'bool',
        'torch.float32': 'float32',
        'torch.bfloat16': 'bfloat16',
        'torch.double': 'float64',
        'torch.half': 'float16',
        'torch.int32': 'int32',
        'torch.int16': 'int16',
        'torch.int8': 'int8',
        'torch.int64': 'int64',
        'torch.bool': 'bool',
        'torch.float': 'float32',
    }
    return dtype_map.get(torch_dtype_str, str(torch_dtype_str))

def _to_tvm_device(torch_device_str):
    if torch_device_str == 'cpu':
        return tvm.cpu(0)
    elif torch_device_str == 'cuda':
        return tvm.cuda(0)
    else:
        raise ValueError(f"Unsupported device: {torch_device_str}")

# A simple wrapper to build and run a Relay module
def build_and_run_relay_mod(relay_func_builder, inputs_np, target, func_name="main", enable_bf16=False):
    params = {}
    
    # Sort inputs_np keys to ensure consistent order for relay.var creation
    sorted_input_keys = sorted(inputs_np.keys())
    inputs_vars = []

    for name in sorted_input_keys:
        arr = inputs_np[name]
        inputs_vars.append(relay.var(name, shape=arr.shape, dtype=str(arr.dtype)))
        params[name] = tvm.nd.array(arr, device=target)

    relay_expr = relay_func_builder(*inputs_vars)

    func = relay.Function(list(inputs_vars), relay_expr)
    mod = tvm.IRModule.from_expr(func)
    
    with tvm.transform.PassContext(opt_level=3):
        executor = relay.build(mod, target=target, params=params)

    # Prepare TVM NDArrays for inputs
    tvm_inputs = []
    for name in sorted_input_keys:
        arr = inputs_np[name]
        tvm_inputs.append(tvm.nd.array(arr, device=target))
    
    # Execute
    rt_mod = tvm.runtime.GraphModule(executor["default"](target))
    for i, name in enumerate(sorted_input_keys):
        rt_mod.set_input(name, tvm_inputs[i])
    rt_mod.run()
    
    # Get outputs
    output_count = rt_mod.get_num_outputs()
    if output_count == 1:
        return rt_mod.get_output(0)
    else:
        return [rt_mod.get_output(i) for i in range(output_count)]

def warmup_and_run_forward_tvm(relay_mod_builder, initial_inputs_np, target, func_name="main", enable_bf16=False):
    return build_and_run_relay_mod(relay_mod_builder, initial_inputs_np, target, func_name=func_name, enable_bf16=enable_bf16)


@pytest.fixture(scope="module")
def devices():
    res = ['cpu']
    if tvm.cuda().exist:
        res.append('cuda')
    return res

@pytest.fixture(scope="module")
def dtypes():
    # TVM's bfloat16 is often available with LLVM.
    return ['float32', 'bfloat16']

def assert_allclose_tvm(actual, desired, rtol=1e-5, atol=1e-8, equal_nan=False):
    actual_np = actual.asnumpy() if isinstance(actual, tvm.nd.NDArray) else actual
    desired_np = desired.asnumpy() if isinstance(desired, tvm.nd.NDArray) else desired
    tvm.testing.assert_allclose(actual_np, desired_np, rtol=rtol, atol=atol, equal_nan=equal_nan)

def assert_equal_tvm(actual, desired, atol=1e-8, rtol=1e-5):
    if isinstance(actual, tvm.nd.NDArray) or isinstance(desired, tvm.nd.NDArray) or \
       isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        assert_allclose_tvm(actual, desired, atol=atol, rtol=rtol)
    else:
        assert actual == desired

def test_easy(devices, dtypes):
    def easy_relay(x, y):
        return relay_op.add(x, y)

    a_np = np.random.rand(1024).astype('float32')
    b_np = np.random.rand(1024).astype('float32')
    inputs_np = {"x": a_np, "y": b_np}

    target = _to_tvm_device(devices[0])
    x_tvm = warmup_and_run_forward_tvm(easy_relay, inputs_np, target, func_name="easy")

    assert_allclose_tvm(a_np + b_np, x_tvm)
    # TODO: self.assertLastGraphAllFused() # PyTorch specific

def test_three_arg(devices, dtypes):
    def easy_relay(x, y, z):
        aaa = relay_op.add(x, y)
        bbb = relay_op.add(aaa, z)
        return bbb

    a_np = np.random.rand(1024).astype('float32')
    b_np = np.random.rand(1024).astype('float32')
    c_np = np.random.rand(1024).astype('float32')
    inputs_np = {"x": a_np, "y": b_np, "z": c_np}

    target = _to_tvm_device(devices[0])
    x_tvm = warmup_and_run_forward_tvm(easy_relay, inputs_np, target, func_name="three_arg")

    npr = a_np + b_np + c_np
    assert_allclose_tvm(npr, x_tvm)
    # TODO: self.assertLastGraphAllFused() # PyTorch specific

def test_four_arg(devices, dtypes):
    def run_addcmul_relay(w, x, y, z): # Sorted alphabetically for relay.var consistency
        temp_add = relay_op.add(x, y)
        temp_mul = relay_op.multiply(z, w)
        c = relay_op.add(temp_add, temp_mul)
        return c

    for dev_str in devices:
        target = _to_tvm_device(dev_str)
        rand_a_np = np.random.rand(1024).astype('float32')
        rand_b_np = np.random.rand(1024).astype('float32')
        rand_c_np = np.random.rand(1024).astype('float32')
        rand_d_np = np.random.rand(1024).astype('float32')

        inputs_for_exec = {
            "w": rand_d_np, # PyTorch 'w' corresponds to 'w' here
            "x": rand_a_np, # PyTorch 'x' corresponds to 'x' here
            "y": rand_b_np, # PyTorch 'y' corresponds to 'y' here
            "z": rand_c_np, # PyTorch 'z' corresponds to 'z' here
        }

        x_tvm = warmup_and_run_forward_tvm(run_addcmul_relay, inputs_for_exec, target, func_name="four_arg")

        y_np = (rand_a_np + rand_b_np) + (rand_c_np * rand_d_np)
        assert_allclose_tvm(x_tvm, y_np, atol=1e-6)
    # TODO: self.assertLastGraphAllFused() # PyTorch specific

def test_three_arg2(devices, dtypes):
    def test_relay(x, y, z):
        aaa = relay_op.add(x, y)
        bbb = relay_op.add(aaa, z)
        return bbb

    M = 32
    N = 32
    for dev_str in devices:
        target = _to_tvm_device(dev_str)

        a_np = np.random.rand(M, N).astype('float32')
        b_np = np.random.rand(M, N).astype('float32')
        c_np = np.random.rand(M, N).astype('float32')
        inputs_np = {"x": a_np, "y": b_np, "z": c_np}

        x_tvm = warmup_and_run_forward_tvm(test_relay, inputs_np, target, func_name="three_arg2")

        npr = a_np + b_np + c_np
        assert_allclose_tvm(npr, x_tvm)
    # TODO: self.assertLastGraphAllFused() # PyTorch specific

def test_broadcast3(devices, dtypes):
    def test_body(M, N, L, K, dev_str):
        def test_relay(x, y, z):
            v1 = relay_op.add(x, y)
            v2 = relay_
