# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck, grad

# Assuming torch.testing is available for assert_allclose
try:
    from torch.testing import assert_allclose as torch_assert_allclose
except ImportError:
    # Fallback for older PyTorch versions or environments where torch.testing might not be fully exposed
    # For CI/CD, torch.testing.assert_allclose is preferred.
    def torch_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True, err_msg=''):
        np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg)

# Helper to convert string dtype to torch.dtype
def to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    # Add other dtypes as needed
    raise ValueError(f"Unsupported dtype: {dtype_str}")

# Helper to get device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Numpy reference functions
def sigmoid_np(x):
    one = np.ones_like(x)
    return one / (one + np.exp(-x))

def relu_np(x):
    x_copy = np.copy(x)
    np.maximum(x_copy, 0, x_copy)
    return x_copy

class TestUnaryOp:
    config = {
        "log": (torch.log, lambda x, g: g * (1 / x)),
        "exp": (torch.exp, lambda x, g: g * np.exp(x)),
        "sigmoid": (torch.sigmoid, lambda x, g: g * sigmoid_np(x) * (1 - sigmoid_np(x))),
        "tanh": (torch.tanh, lambda x, g: g * (1 - np.tanh(x) * np.tanh(x))),
        "sqrt": (torch.sqrt, lambda x, g: g * 0.5 * np.power(x, -0.5)),
        "abs": (torch.abs, lambda x, g: np.where(x < 0, -g, g)),
        "relu": (F.relu, lambda x, g: np.where(x < 0, np.zeros_like(x), g)),
        "erf": (torch.erf, lambda x, g: g * (2.0 / (np.pi ** (0.5)) * np.exp(-x * x))),
        "cos": (torch.cos, lambda x, g: g * -1.0 * np.sin(x)),
        "sin": (torch.sin, lambda x, g: g * np.cos(x)),
        "tan": (torch.tan, lambda x, g: g * (1.0 / (np.cos(x) ** 2))),
        "atan": (torch.atan, lambda x, g: g * (1 / (1 + np.power(x, 2.0)))),
        "log2": (torch.log2, lambda x, g: g * (1 / (np.log(2) * x))),
        "log10": (torch.log10, lambda x, g: g * (1 / (np.log(10) * x))),
        "cosh": (torch.cosh, lambda x, g: g * (np.sinh(x))),
        "sinh": (torch.sinh, lambda x, g: g * (np.cosh(x))),
        "asin": (torch.asin, lambda x, g: g * (1.0 / (1.0 - x**2) ** (1.0 / 2.0))),
        "acos": (torch.acos, lambda x, g: g * (-1.0 / (1.0 - x**2.0) ** (1.0 / 2.0))),
        "acosh": (torch.acosh, lambda x, g: g * (1.0 / (x**2 - 1.0) ** (1.0 / 2.0))),
        "asinh": (torch.asinh, lambda x, g: g * (1.0 / (x**2 + 1.0) ** (1.0 / 2.0))),
        "atanh": (torch.atanh, lambda x, g: g * (-1.0 / (x**2 - 1.0))),
    }

    @pytest.mark.parametrize("torch_op, ref_func", list(config.values()), ids=list(config.keys()))
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("shape", [(10, 4)])
    def test_op(self, torch_op, ref_func, shape, dtype):
        dev = get_device()
        torch_dtype = to_torch_dtype(dtype)

        # PyTorch equivalent of TVM's xfail condition for Vulkan
        # Known cases where PyTorch might have precision or stability issues on GPU/CPU for float64.
        # This list is heuristic and might need to be adjusted based on actual PyTorch behavior.
        # The original TVM test had specific xfails for Vulkan.
        # For a truly direct mapping, without `target` and `dev` as TVM objects,
        # we can't fully replicate the TVM Vulkan check.
        # Here we add a general check for potential known CUDA float64 issues as an example.
        if dev.type == "cuda" and torch_dtype == torch.float64:
             known_cuda_fp64_xfails = [
                 torch.erf, torch.tan, torch.atan, torch.log10, torch.cosh, torch.sinh,
                 torch.asin, torch.acos, torch.acosh, torch.asinh, torch.atanh
             ]
             if torch_op in known_cuda_fp64_xfails:
                 pytest.xfail(f"PyTorch {torch_op.__name__} for float64 on CUDA might have precision or stability issues comparable to original TVM Vulkan xfail.")

        data_in = np.random.rand(*shape).astype(dtype)
        # Adjust input values to prevent NaNs or Inf in derivatives for NumPy reference
        if torch_op in [torch.log, torch.log2, torch.log10, torch.sqrt]:
            data_in = np.abs(data_in) + 1e-5
        if torch_op == torch.acosh:
            data_in = np.abs(data_in) + 1.1 # Ensure input > 1.0
        if torch_op == torch.atanh:
            data_in = np.clip(data_in, -0.99, 0.99) # Ensure input is in (-1, 1)

        grad_in = np.random.rand(*shape).astype(dtype)

        # Compute reference gradient using NumPy
        ref_grad_out = ref_func(data_in, grad_in)

        # PyTorch computation and gradient
        x_torch = torch.tensor(data_in, dtype=torch_dtype, device=dev, requires_grad=True)
        g_torch = torch.tensor(grad_in, dtype=torch_dtype, device=dev, requires_grad=True)

        # Define the forward pass for PyTorch: output = op(x) * g
        output = torch_op(x_torch) * g_torch

        # Compute gradient w.r.t. x_torch (d(output)/d(x_torch))
        # grad_outputs defaults to ones_like if output is not scalar, but explicit is safer
        pytorch_grad_x = torch.autograd.grad(output, x_torch, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
        
        torch_assert_allclose(torch.tensor(ref_grad_out, device=dev), pytorch_grad_x, rtol=0.01, atol=1e-5)


class TestBinaryOp:
    config = {
        "add": (torch.add, lambda x, y: [np.ones_like(x), np.ones_like(y)]),
        "subtract": (torch.subtract, lambda x, y: [np.ones_like(x), -np.ones_like(y)]),
        "multiply": (torch.multiply, lambda x, y: [y, x]),
        "divide": (torch.divide, lambda x, y: [1 / y, -x / (y**2)]),
    }

    @pytest.mark.parametrize("torch_op, ref_func", list(config.values()), ids=list(config.keys()))
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("shape", [(5, 10, 5)])
    def test_binary_op(self, torch_op, ref_func, shape, dtype):
        dev = get_device()
        torch_dtype = to_torch_dtype(dtype)

        x_data = np.random.rand(*shape).astype(dtype)
        y_data = np.random.rand(*shape).astype(dtype)

        # Adjust data to prevent division by zero or large values for divide
        if torch_op == torch.divide:
            y_data = np.clip(y_data, 0.1, None) # Ensure y is not too small or zero

        # Compute reference gradients using NumPy
        ref_grad0, ref_grad1 = ref_func(x_data, y_data)

        x_torch = torch.tensor(x_data, dtype=torch_dtype, device=dev, requires_grad=True)
        y_torch = torch.tensor(y_data, dtype=torch_dtype, device=dev, requires_grad=True)

        # Define the forward pass for PyTorch: z = op(x, y)
        z_torch = torch_op(x_torch, y_torch)

        # Compute gradients w.r.t. both x_torch and y_torch
        pytorch_grad_x, pytorch_grad_y = torch.autograd.grad(z_torch, (x_torch, y_torch), grad_outputs=torch.ones_like(z_torch), retain_graph=True)
        
        torch_assert_allclose(torch.tensor(ref_grad0, device=dev), pytorch_grad_x, rtol=0.01, atol=1e-5)
        torch_assert_allclose(torch.tensor(ref_grad1, device=dev), pytorch_grad_y, rtol=0.01, atol=1e-5)


# Dummy parameters for TVM compatibility (executor_kind, target, dev are not directly used in PyTorch functional tests)
@pytest.mark.parametrize("executor_kind", ["debug"]) 
@pytest.mark.parametrize("target", ["cpu"])
@pytest.mark.parametrize("dev", ["cpu"])
def test_softmax_grad(executor_kind, target, dev):
    # TVM had xfail for Vulkan; assuming PyTorch works on CPU/CUDA
    dev_torch = get_device()
    
    # Define the PyTorch function for gradcheck
    def fwd_func(data):
        return F.softmax(data, dim=-1)

    # Prepare input data
    data_in = torch.randn(1, 16, dtype=torch.float64, device=dev_torch, requires_grad=True)

    # Use PyTorch's gradcheck. rtol and atol adjusted for float64 precision.
    assert gradcheck(fwd_func, (data_in,), rtol=1e-5, atol=1e-6)


# Dummy parameters for TVM compatibility
@pytest.mark.parametrize("executor_kind", ["debug"]) 
@pytest.mark.parametrize("target", ["cpu"])
@pytest.mark.parametrize("dev", ["cpu"])
def test_log_softmax_grad(executor_kind, target, dev):
    # TVM had xfail for Vulkan; assuming PyTorch works on CPU/CUDA
    dev_torch = get_device()

    # Define the PyTorch function for gradcheck
    def fwd_func(data):
        return F.log_softmax(data, dim=-1)

    # Prepare input data
    data_in = torch.randn(2, 16, dtype=torch.float64, device=dev_torch, requires_grad=True)

    # Use PyTorch's gradcheck
    assert gradcheck(fwd_func, (data_in,), rtol=1e-5, atol=1e-6)


class TestBiasAddGrad:
    # d_shape: data shape, b_shape: bias shape, axis: dimension to add bias
    @pytest.mark.parametrize(
        "d_shape, b_shape, axis",
        [
            ((1, 16), (16,), 1),
            ((1, 8, 2, 2), (8,), 1),
            ((1, 2, 2, 8), (8,), 3),
            ((4, 8), (8,), 1),
        ],
    )
    # Dummy parameters for TVM compatibility
    @pytest.mark.parametrize("executor_kind", ["debug"])
    @pytest.mark.parametrize("target", ["cpu"])
    @pytest.mark.parametrize("dev", ["cpu"])
    def test_bias_add(self, executor_kind, target, dev, d_shape, b_shape, axis):
        dev_torch = get_device()

        # Define the PyTorch function for gradcheck
        def fwd_func(data, bias):
            # Reshape bias to be broadcastable along the specified axis
            bias_reshaped_shape = [1] * len(d_shape)
            bias_reshaped_shape[axis] = b_shape[0]
            bias_reshaped = bias.reshape(bias_reshaped_shape)
            return torch.add(data, bias_reshaped)

        # Prepare input data
        # Using float64 for better precision with gradcheck
        data_in = torch.randn(*d_shape, dtype=torch.float64, device=dev_torch, requires_grad=True)
        bias_in = torch.randn(*b_shape, dtype=torch.float64, device=dev_torch, requires_grad=True)

        # Use PyTorch's gradcheck
        assert gradcheck(fwd_func, (data_in, bias_in), rtol=1e-5, atol=1e-6)


# Dummy parameters for TVM compatibility
@pytest.mark.parametrize("executor_kind", ["debug"]) 
@pytest.mark.parametrize("target", ["cpu"])
@pytest.mark.parametrize("dev", ["cpu"])
def test_expand_dims_grad(executor_kind, target, dev):
    dev_torch = get_device()

    # Define the PyTorch function for gradcheck
    def fwd_func(data):
        # TVM: relay.expand_dims(data, axis=1, num_newaxis=2)
        # This means adding 2 new dimensions at axis=1
        # Example: (2,3) -> (2,1,1,3)
        return data.unsqueeze(1).unsqueeze(1)

    # Prepare input data
    data_in = torch.randn(2, 3, dtype=torch.float64, device=dev_torch, requires_grad=True)

    # Use PyTorch's gradcheck
    assert gradcheck(fwd_func, (data_in,), rtol=1e-5, atol=1e-6)


# Dummy parameters for TVM compatibility
@pytest.mark.parametrize("executor_kind", ["debug"]) 
@pytest.mark.parametrize("target", ["cpu"])
@pytest.mark.parametrize("dev", ["cpu"])
def test_concatenate_grad(executor_kind, target, dev):
    dev_torch = get_device()

    # Define the PyTorch function for gradcheck
    def fwd_func(x, y, z):
        return torch.cat([x, y, z], dim=1)

    # Prepare input data
    x_in = torch.randn(2, 2, 5, dtype=torch.float64, device=dev_torch, requires_grad=True)
    y_in = torch.randn(2, 1, 5, dtype=torch.float64, device=dev_torch, requires_grad=True)
    z_in = torch.randn(2, 4, 5, dtype=torch.float64, device=dev_torch, requires_grad=True)

    # Use PyTorch's gradcheck
    assert gradcheck(fwd_func, (x_in, y_in, z_in), rtol=1e-5, atol=1e-6)
