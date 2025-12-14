import torch
import numpy as np
import pytest
from torch import nn
from torch.nn import functional as F

# A helper for converting string dtypes to torch dtypes
_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

def to_torch_dtype(tvm_dtype_str):
    return _TORCH_DTYPE_MAP.get(tvm_dtype_str, None)

# Helper to initialize weights for PyTorch in a way that mimics Relay.testing.init.Xavier
# The original Xavier initializer works on numpy arrays which are then converted to tvm.nd.array.
# This implementation populates a numpy array directly using Xavier formula.
class XavierInitializer:
    def __call__(self, name, array):
        if len(array.shape) == 4:  # Assuming OIHW for conv2d weights
            # fan_in: C_in_per_group * K_H * K_W
            # fan_out: C_out_per_group * K_H * K_W, where C_out_per_group is typically K_H*K_W*groups (effectively O channels)
            fan_in = array.shape[1] * array.shape[2] * array.shape[3]
            fan_out = array.shape[0] * array.shape[2] * array.shape[3]
        elif len(array.shape) == 2: # Assuming for dense layers, or 1x1 conv type bias
            fan_in = array.shape[1] if array.shape[1] > 0 else 1
            fan_out = array.shape[0] if array.shape[0] > 0 else 1
        else: # For bias like (1, C, 1, 1), assume fan_in/out from C
            if array.numel() > 0:
                fan_in = array.numel()
                fan_out = array.numel()
            else: # Fallback for empty or very small arrays
                fan_in = 1
                fan_out = 1

        limit = np.sqrt(6 / (fan_in + fan_out))
        array[:] = np.random.uniform(-limit, limit, size=array.shape).astype(array.dtype)


# The original TVM test uses `tvm.testing.parameter("float32")` for dtype.
# We replace this with `pytest.mark.parametrize` for PyTorch.
# The original TVM tests also use `@tvm.testing.requires_opencl` and
# `@tvm.testing.parametrize_targets("opencl -device=adreno")`.
# These are TVM-specific infrastructure for remote device execution and are not
# directly translatable to PyTorch unit tests. The PyTorch tests will run on
# CPU/GPU (CUDA if available) without remote setup.

@pytest.mark.parametrize("dtype_str", ["float32"])
def test_depthwise_conv2d_bias_nchwc(dtype_str):
    dtype = to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 64, 112, 112) # NCHW
    filter_shape = (64, 1, 3, 3) # OIHW for groups=64 (output_channels, input_channels_per_group, kH, kW)
    bias_shape = (1, 64, 1, 1) # NCHW compatible bias for channel-wise addition

    np.random.seed(1) # For reproducible numpy array generation

    # Prepare input data
    data_np = np.random.rand(*input_shape).astype(dtype_str)
    A = torch.tensor(data_np, device=device, dtype=dtype)

    # Prepare weight data
    filter_data_np = np.zeros(filter_shape).astype(dtype_str)
    initializer = XavierInitializer()
    initializer("weight", filter_data_np)
    B = torch.tensor(filter_data_np, device=device, dtype=dtype)

    # Prepare bias data
    bias_data_np = np.zeros(bias_shape).astype(dtype_str)
    initializer("bias", bias_data_np)
    bias_t = torch.tensor(bias_data_np, device=device, dtype=dtype)

    # PyTorch computation, mimicking the Relay graph:
    # conv = relay.nn.conv2d(A, B, data_layout="NCHW", kernel_layout="OIHW",
    #                        padding=[1, 1, 1, 1], strides=[2, 2], out_dtype=dtype,
    #                        channels=64, groups=64, kernel_size=(3, 3))
    conv_output = F.conv2d(
        A,
        B,
        stride=(2, 2),
        padding=(1, 1), # TVM [1,1,1,1] maps to symmetric padding for H and W
        groups=64
    )

    # D = relay.op.add(conv, bias)
    add_output = conv_output + bias_t # PyTorch handles broadcasting (1, 64, 1, 1) to (1, 64, 56, 56)

    # D = relay.op.nn.relu(D)
    final_output = F.relu(add_output)

    # Calculate expected output shape based on conv2d formula
    # H_out = floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0]) + 1
    # W_out = floor((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1]) + 1
    # Here: padding=1, stride=2, dilation=1 (default), kernel_size=(3,3)
    # H_in=112, W_in=112
    # H_out = floor((112 + 2*1 - 1*(3-1) - 1) / 2) + 1 = floor((112 + 2 - 2 - 1) / 2) + 1 = floor(111 / 2) + 1 = 55 + 1 = 56
    expected_output_H = (input_shape[2] + 2*1 - (3-1) - 1) // 2 + 1
    expected_output_W = (input_shape[3] + 2*1 - (3-1) - 1) // 2 + 1
    expected_output_shape = (input_shape[0], filter_shape[0], expected_output_H, expected_output_W) # (1, 64, 56, 56)

    assert final_output.shape == expected_output_shape
    assert final_output.dtype == dtype
    # TODO: Add numerical assertion against a stable reference if one becomes available or is necessary.


@pytest.mark.parametrize("dtype_str", ["float32"])
def test_depthwise_conv2d_nchwc(dtype_str):
    dtype = to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 64, 112, 112)
    filter_shape = (64, 1, 3, 3) # OIHW

    np.random.seed(1)

    data_np = np.random.rand(*input_shape).astype(dtype_str)
    A = torch.tensor(data_np, device=device, dtype=dtype)

    filter_data_np = np.zeros(filter_shape).astype(dtype_str)
    initializer = XavierInitializer()
    initializer("weight", filter_data_np)
    B = torch.tensor(filter_data_np, device=device, dtype=dtype)

    # PyTorch computation (no bias, no relu)
    # conv = relay.nn.conv2d(...)
    final_output = F.conv2d(
        A,
        B,
        stride=(2, 2),
        padding=(1, 1),
        groups=64
    )

    # Calculate expected output shape (same as above)
    expected_output_H = (input_shape[2] + 2*1 - (3-1) - 1) // 2 + 1
    expected_output_W = (input_shape[3] + 2*1 - (3-1) - 1) // 2 + 1
    expected_output_shape = (input_shape[0], filter_shape[0], expected_output_H, expected_output_W)

    assert final_output.shape == expected_output_shape
    assert final_output.dtype == dtype


@pytest.mark.parametrize("dtype_str", ["float32"])
def test_depthwise_conv2d_bias_nchw(dtype_str):
    dtype = to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 64, 112, 112)
    filter_shape = (64, 1, 3, 3) # OIHW
    bias_shape = (1, 64, 1, 1)

    np.random.seed(1)

    data_np = np.random.rand(*input_shape).astype(dtype_str)
    A = torch.tensor(data_np, device=device, dtype=dtype)

    filter_data_np = np.zeros(filter_shape).astype(dtype_str)
    initializer = XavierInitializer()
    initializer("weight", filter_data_np)
    B = torch.tensor(filter_data_np, device=device, dtype=dtype)

    bias_data_np = np.zeros(bias_shape).astype(dtype_str)
    initializer("bias", bias_data_np)
    bias_t = torch.tensor(bias_data_np, device=device, dtype=dtype)

    # PyTorch computation
    # conv = relay.nn.conv2d(...)
    conv_output = F.conv2d(
        A,
        B,
        stride=(2, 2),
        padding=(1, 1),
        groups=64
    )

    # D = relay.op.add(conv, bias)
    add_output = conv_output + bias_t

    # D = relay.op.nn.relu(D)
    final_output = F.relu(add_output)

    # Calculate expected output shape
    expected_output_H = (input_shape[2] + 2*1 - (3-1) - 1) // 2 + 1
    expected_output_W = (input_shape[3] + 2*1 - (3-1) - 1) // 2 + 1
    expected_output_shape = (input_shape[0], filter_shape[0], expected_output_H, expected_output_W)

    assert final_output.shape == expected_output_shape
    assert final_output.dtype == dtype


@pytest.mark.parametrize("dtype_str", ["float32"])
def test_depthwise_conv2d_repack_bias_nchw(dtype_str):
    dtype = to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 63, 112, 112) # Channels 63, not 64
    filter_shape = (63, 1, 3, 3) # OIHW
    bias_shape = (1, 63, 1, 1)

    np.random.seed(1)

    data_np = np.random.rand(*input_shape).astype(dtype_str)
    A = torch.tensor(data_np, device=device, dtype=dtype)

    filter_data_np = np.zeros(filter_shape).astype(dtype_str)
    initializer = XavierInitializer()
    initializer("weight", filter_data_np)
    B = torch.tensor(filter_data_np, device=device, dtype=dtype)

    bias_data_np = np.zeros(bias_shape).astype(dtype_str)
    initializer("bias", bias_data_np)
    bias_t = torch.tensor(bias_data_np, device=device, dtype=dtype)

    # PyTorch computation
    # conv = relay.nn.conv2d(...)
    conv_output = F.conv2d(
        A,
        B,
        stride=(2, 2),
        padding=(1, 1),
        groups=63 # Groups = channels
    )

    # D = relay.op.add(conv, bias)
    add_output = conv_output + bias_t

    # D = relay.op.nn.relu(D)
    final_output = F.relu(add_output)

    # Calculate expected output shape (same formula, but with channels=63 for output)
    expected_output_H = (input_shape[2] + 2*1 - (3-1) - 1) // 2 + 1
    expected_output_W = (input_shape[3] + 2*1 - (3-1) - 1) // 2 + 1
    expected_output_shape = (input_shape[0], filter_shape[0], expected_output_H, expected_output_W) # (1, 63, 56, 56)

    assert final_output.shape == expected_output_shape
    assert final_output.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__])
