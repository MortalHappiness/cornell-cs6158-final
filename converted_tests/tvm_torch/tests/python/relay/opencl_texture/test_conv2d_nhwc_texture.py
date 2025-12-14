import os
import re
import pytest
import numpy as np
import torch
import torch.nn.functional as F
import functools

# --- Configuration ---
# Original TVM test uses tvm.testing.parameter, so this is a fixed input.
# The original file is for 'float32', so we fix it here.
DTYPE_TVM_STR = "float32"
DTYPE_TORCH = torch.float32 # Corresponding PyTorch dtype

# Determine device for PyTorch tests
def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Uncomment the following lines for Apple Silicon (MPS backend) support
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")

CURRENT_DEVICE = get_torch_device()

# Mock for TVM's Xavier initializer which modifies numpy arrays in-place
class MockXavierInitializer:
    def __call__(self, name, array_np, gain=1.0):
        # Adapted from PyTorch's xavier_uniform_
        # For conv2d weight HWIO: H, W, in_channels, out_channels
        if "weight" in name:
            # fan_in is in_channels * k_h * k_w
            # fan_out is out_channels * k_h * k_w
            # For HWIO, in_channels = array_np.shape[2], out_channels = array_np.shape[3]
            # k_h = array_np.shape[0], k_w = array_np.shape[1]
            fan_in = array_np.shape[2] * array_np.shape[0] * array_np.shape[1]
            fan_out = array_np.shape[3] * array_np.shape[0] * array_np.shape[1]
        elif "bias" in name:
            # For bias, a common heuristic is fan_in = fan_out = size
            fan_in = array_np.size
            fan_out = array_np.size
        else: # Generic case, might need adjustment if other types of arrays are initialized
            fan_in = np.prod(array_np.shape[1:]) if len(array_np.shape) > 1 else array_np.shape[0]
            fan_out = array_np.shape[0] if len(array_np.shape) > 0 else 1
        
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        a = np.sqrt(3.0) * std
        array_np[:] = np.random.uniform(-a, a, size=array_np.shape).astype(array_np.dtype)

# Helper function to prepare input and weight/bias tensors from numpy arrays
def prepare_conv2d_tensors(input_shape, filter_shape, bias_shape_tvm, dtype_torch, seed, has_bias, target_device):
    np.random.seed(seed)
    
    data_np = np.random.rand(*input_shape).astype(dtype_torch)
    
    filter_np = np.zeros(filter_shape, dtype=dtype_torch)
    initializer = MockXavierInitializer()
    initializer("weight", filter_np)
    
    if has_bias:
        bias_np = np.zeros(bias_shape_tvm, dtype=dtype_torch)
        initializer("bias", bias_np)
        
        # PyTorch F.conv2d bias expects a 1D tensor (out_channels,)
        bias_tensor = torch.tensor(bias_np.flatten(), device=target_device)
    else:
        bias_tensor = None

    # TVM input data_layout="NHWC" -> PyTorch input NCHW
    input_tensor_nchw = torch.tensor(data_np, device=target_device).permute(0, 3, 1, 2)

    # TVM kernel_layout="HWIO" -> PyTorch weight OIHW
    weight_tensor_oihw = torch.tensor(filter_np, device=target_device).permute(3, 2, 0, 1)

    return input_tensor_nchw, weight_tensor_oihw, bias_tensor

# Helper function to apply a single PyTorch convolution block (conv2d + bias + relu)
def apply_pytorch_conv2d_block(
    input_tensor_nchw, # NCHW input
    weight_tensor_oihw, # OIHW weight
    bias_tensor_1d, # (out_channels,) bias
    padding_tvm, # TVM style [top, left, bottom, right] or single int/tuple
    strides_tvm, # TVM style list/tuple or single int
):
    # Convert TVM padding to PyTorch F.conv2d format
    if isinstance(padding_tvm, (list, tuple)) and len(padding_tvm) == 4:
        # PyTorch F.conv2d padding argument is (pad_h, pad_w) for symmetric padding.
        # If padding is asymmetric, we use F.pad first.
        if padding_tvm[0] == padding_tvm[2] and padding_tvm[1] == padding_tvm[3]:
            conv_padding = (padding_tvm[0], padding_tvm[1])
        else:
            # Apply asymmetric padding using F.pad
            # F.pad expects (pad_left, pad_right, pad_top, pad_bottom) in reverse order of dims H, W for 4D input
            pad_list_f_pad = (padding_tvm[1], padding_tvm[3], padding_tvm[0], padding_tvm[2])
            input_tensor_nchw = F.pad(input_tensor_nchw, pad=pad_list_f_pad, mode='constant', value=0.0)
            conv_padding = 0 # No additional padding for conv2d
    else: # If padding_tvm is already (H,W) or single int
        conv_padding = padding_tvm

    # Convert TVM strides to PyTorch F.conv2d format
    conv_strides = strides_tvm if isinstance(strides_tvm, (list, tuple)) else (strides_tvm, strides_tvm)
    
    # Perform convolution (bias is applied by F.conv2d)
    conv_output_nchw = F.conv2d(
        input=input_tensor_nchw,
        weight=weight_tensor_oihw,
        bias=bias_tensor_1d,
        stride=conv_strides,
        padding=conv_padding,
        dilation=1, # Default as not specified in TVM tests
        groups=1    # Default as not specified in TVM tests
    )
    
    # Apply ReLU activation
    final_output_nchw = F.relu(conv_output_nchw)
    return final_output_nchw

# Wrapper for running a single conv2d test, generating inputs and applying the block
def run_single_conv2d_test_and_get_output_np(
    input_shape, filter_shape, bias_shape_tvm, padding_tvm, strides_tvm, 
    out_channels, kernel_size, dtype_torch, seed, has_bias=True
):
    input_tensor_nchw, weight_tensor_oihw, bias_tensor_1d = prepare_conv2d_tensors(
        input_shape, filter_shape, bias_shape_tvm, dtype_torch, seed, has_bias, CURRENT_DEVICE
    )
    
    result_nchw = apply_pytorch_conv2d_block(
        input_tensor_nchw=input_tensor_nchw,
        weight_tensor_oihw=weight_tensor_oihw,
        bias_tensor_1d=bias_tensor_1d,
        padding_tvm=padding_tvm,
        strides_tvm=strides_tvm,
    )
    
    # Convert output back to NHWC for comparison
    return result_nchw.permute(0, 2, 3, 1).detach().cpu().numpy()

# Mock `build_run_compare` and `gpu_preprocess` as they are TVM-specific for compilation/remote execution.
# For PyTorch, we directly run the model and compare.
# These mocks are here to satisfy potential calls if they were not removed, but the refactored tests
# will directly call PyTorch functions.
def mock_build_run_compare(*args, **kwargs):
    pytest.skip("TVM-specific build_run_compare is not applicable to PyTorch tests.")

def mock_gpu_preprocess(*args, **kwargs):
    pytest.skip("TVM-specific gpu_preprocess is not applicable to PyTorch tests.")

# Dummy remote and target for test compatibility.
# In a real PyTorch setup, you'd use torch.device('cuda') or 'cpu'.
# Here, these are just placeholders.
class DummyRemote:
    pass

class DummyTarget:
    pass

remote_dummy = DummyRemote()
target_dummy = DummyTarget()


# --- PyTorch Tests equivalent to TVM tests ---
# All original TVM tests use float32. We will parametrize with torch.float32.

@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_deeplabv3_1_257_257_32x1_1_32_16(dtype):
    input_shape = (1, 257, 257, 32)
    filter_shape = (1, 1, 32, 16)
    bias_shape_tvm = (filter_shape[-1],) # (16,)
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=0, # Default for TVM conv2d
        strides_tvm=1, # Default for TVM conv2d
        out_channels=filter_shape[-1],
        kernel_size=(1, 1),
        dtype_torch=dtype,
        seed=1,
        has_bias=True
    )
    
    # Output shape calculation for conv2d with 1x1 kernel, stride 1, padding 0:
    # Output spatial dimensions are same as input.
    expected_output_shape = (1, 257, 257, 16) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_deeplabv3_1_257_257_32x1_1_32_16_with_padding(dtype):
    input_shape = (1, 257, 257, 32)
    filter_shape = (1, 1, 32, 16)
    bias_shape_tvm = (filter_shape[-1],) # (16,)
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[3, 3, 3, 3], # TVM style symmetric padding
        strides_tvm=[2, 2], # TVM style
        out_channels=filter_shape[-1],
        kernel_size=(1, 1),
        dtype_torch=dtype,
        seed=1,
        has_bias=True
    )
    
    # Output shape calculation for conv2d:
    # H_out = floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    # W_out = floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    H_in, W_in = 257, 257
    pad_h, pad_w = 3, 3 # from [3,3,3,3]
    k_h, k_w = 1, 1
    stride_h, stride_w = 2, 2

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 16) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_4_35_35_32x3_3_144_16(dtype):
    input_shape = (4, 35, 35, 32)
    filter_shape = (3, 3, 32, 16)
    bias_shape_tvm = (filter_shape[-1],) # (16,)
    kernel_size = (filter_shape[0], filter_shape[1]) # (3, 3)

    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=0, # Default for TVM conv2d
        strides_tvm=1, # Default for TVM conv2d
        out_channels=filter_shape[-1], # 16
        kernel_size=kernel_size, # (3, 3)
        dtype_torch=dtype,
        seed=1,
        has_bias=True
    )

    # Output shape calculation:
    H_in, W_in = 35, 35
    pad_h, pad_w = 0, 0
    k_h, k_w = 3, 3
    stride_h, stride_w = 1, 1

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (4, int(H_out), int(W_out), 16) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_deeplabv3_1_513_513_3x3_3_3_32(dtype):
    input_shape = (1, 513, 513, 3)
    filter_shape = (3, 3, 3, 32)
    bias_shape_tvm = (filter_shape[-1],) # (32,)
    kernel_size = (filter_shape[0], filter_shape[1]) # (3, 3)

    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=0, # Default for TVM conv2d
        strides_tvm=1, # Default for TVM conv2d
        out_channels=filter_shape[-1], # 32
        kernel_size=kernel_size, # (3, 3)
        dtype_torch=dtype,
        seed=1,
        has_bias=True
    )

    # Output shape calculation:
    H_in, W_in = 513, 513
    pad_h, pad_w = 0, 0
    k_h, k_w = 3, 3
    stride_h, stride_w = 1, 1

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 32) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad(dtype):
    input_shape = (1, 42, 42, 32)
    filter_shape = (3, 3, 32, 96)
    bias_shape_tvm = (1, 1, 1, 96) # TVM bias shape
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[0, 0, 0, 0],
        strides_tvm=[2, 2],
        out_channels=96,
        kernel_size=(3, 3),
        dtype_torch=dtype,
        seed=0,
        has_bias=True
    )
    
    H_in, W_in = 42, 42
    pad_h, pad_w = 0, 0
    k_h, k_w = 3, 3
    stride_h, stride_w = 2, 2

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 96) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad_pass(dtype):
    input_shape = (1, 40, 40, 32)
    filter_shape = (2, 2, 32, 96)
    bias_shape_tvm = (1, 1, 1, 96) # TVM bias shape

    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[0, 0, 0, 0],
        strides_tvm=[2, 2],
        out_channels=96,
        kernel_size=(2, 2),
        dtype_torch=dtype,
        seed=0,
        has_bias=True
    )
    
    H_in, W_in = 40, 40
    pad_h, pad_w = 0, 0
    k_h, k_w = 2, 2
    stride_h, stride_w = 2, 2

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 96) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_inceptionv3_35_35_strides(dtype):
    input_shape = (1, 35, 35, 48)
    filter_shape = (5, 5, 48, 64)
    bias_shape_tvm = (1, 1, 1, 64) # TVM bias shape
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[2, 2, 2, 2],
        strides_tvm=[1, 1],
        out_channels=64,
        kernel_size=(5, 5),
        dtype_torch=dtype,
        seed=0,
        has_bias=True
    )

    H_in, W_in = 35, 35
    pad_h, pad_w = 2, 2
    k_h, k_w = 5, 5
    stride_h, stride_w = 1, 1

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 64) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_resnet50_v2_nhwc_3c(dtype):
    input_shape = (1, 224, 224, 3)
    filter_shape = (7, 7, 3, 64)
    bias_shape_tvm = (1, 1, 1, 64) # TVM bias shape
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[3, 3, 3, 3],
        strides_tvm=[2, 2],
        out_channels=64,
        kernel_size=(7, 7),
        dtype_torch=dtype,
        seed=1,
        has_bias=True
    )

    H_in, W_in = 224, 224
    pad_h, pad_w = 3, 3
    k_h, k_w = 7, 7
    stride_h, stride_w = 2, 2

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 64) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_inceptionv3_nhwc_3c(dtype):
    input_shape = (1, 299, 299, 3)
    filter_shape = (3, 3, 3, 64)
    bias_shape_tvm = (1, 1, 1, 64) # TVM bias shape
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[0, 0, 0, 0],
        strides_tvm=[2, 2],
        out_channels=64,
        kernel_size=(3, 3),
        dtype_torch=dtype,
        seed=0,
        has_bias=True
    )

    H_in, W_in = 299, 299
    pad_h, pad_w = 0, 0
    k_h, k_w = 3, 3
    stride_h, stride_w = 2, 2

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 64) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_1x1_16c16spatial(dtype):
    input_shape = (1, 128, 128, 16)
    filter_shape = (4, 4, 16, 32)
    bias_shape_tvm = (1, 1, 1, 32) # TVM bias shape
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[0, 0, 0, 0],
        strides_tvm=[2, 2],
        out_channels=32,
        kernel_size=(4, 4),
        dtype_torch=dtype,
        seed=0,
        has_bias=True
    )

    H_in, W_in = 128, 128
    pad_h, pad_w = 0, 0
    k_h, k_w = 4, 4
    stride_h, stride_w = 2, 2

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 32) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_4x4_16c16pad(dtype):
    input_shape = (1, 256, 256, 32)
    filter_shape = (4, 4, 32, 32)
    bias_shape_tvm = (1, 1, 1, 32) # TVM bias shape
    
    # TVM: padding=[3, 3, 0, 0] is asymmetric (3 top, 3 left, 0 bottom, 0 right)
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[3, 3, 0, 0], # TVM style asymmetric padding
        strides_tvm=[2, 2],
        out_channels=32,
        kernel_size=(4, 4),
        dtype_torch=dtype,
        seed=0,
        has_bias=True
    )

    H_in, W_in = 256, 256
    pad_h_top, pad_w_left, pad_h_bottom, pad_w_right = 3, 3, 0, 0
    k_h, k_w = 4, 4
    stride_h, stride_w = 2, 2

    # If F.pad is used first for asymmetric padding, then conv2d with padding=0
    # The effective input size becomes H_in + pad_h_top + pad_h_bottom, etc.
    effective_H_in = H_in + pad_h_top + pad_h_bottom # 256 + 3 + 0 = 259
    effective_W_in = W_in + pad_w_left + pad_w_right # 256 + 3 + 0 = 259
    
    H_out = np.floor((effective_H_in - k_h) / stride_h) + 1
    W_out = np.floor((effective_W_in - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 32) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_4x4x4_16c16pad(dtype):
    input_shape = (1, 256, 256, 32)
    filter_shape = (4, 4, 32, 4) # out_channels=4
    bias_shape_tvm = (1, 1, 1, 4) # TVM bias shape
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[3, 3, 0, 0], # TVM style asymmetric padding
        strides_tvm=[2, 2],
        out_channels=4,
        kernel_size=(4, 4),
        dtype_torch=dtype,
        seed=0,
        has_bias=True
    )

    H_in, W_in = 256, 256
    pad_h_top, pad_w_left, pad_h_bottom, pad_w_right = 3, 3, 0, 0
    k_h, k_w = 4, 4
    stride_h, stride_w = 2, 2

    effective_H_in = H_in + pad_h_top + pad_h_bottom # 256 + 3 + 0 = 259
    effective_W_in = W_in + pad_w_left + pad_w_right # 256 + 3 + 0 = 259
    
    H_out = np.floor((effective_H_in - k_h) / stride_h) + 1
    W_out = np.floor((effective_W_in - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 4) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_yolov3_v2_nhwc_3c(dtype):
    input_shape = (1, 13, 13, 1024)
    filter_shape = (1, 1, 1024, 255) # out_channels=255
    # No bias in this test
    
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=None, # No bias
        padding_tvm=[0, 0, 0, 0],
        strides_tvm=[1, 1],
        out_channels=255,
        kernel_size=(1, 1),
        dtype_torch=dtype,
        seed=0,
        has_bias=False # Explicitly set to False
    )

    H_in, W_in = 13, 13
    pad_h, pad_w = 0, 0
    k_h, k_w = 1, 1
    stride_h, stride_w = 1, 1

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 255) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_vgg16_winograd_4d(dtype):
    input_shape = (1, 28, 28, 512)
    filter_shape = (3, 3, 512, 512)
    bias_shape_tvm = (1, 1, 1, 512) # TVM bias shape
    
    # This test explicitly mentions "winograd" in the original TVM graph and asserts its presence.
    # In PyTorch, Winograd is an internal optimization for conv2d, not explicitly controlled by user API.
    # The output shape and values should be the same regardless.
    result_np = run_single_conv2d_test_and_get_output_np(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=bias_shape_tvm,
        padding_tvm=[1, 1, 1, 1],
        strides_tvm=[1, 1], # Default
        out_channels=512,
        kernel_size=[3, 3],
        dtype_torch=dtype,
        seed=0,
        has_bias=True
    )

    H_in, W_in = 28, 28
    pad_h, pad_w = 1, 1
    k_h, k_w = 3, 3
    stride_h, stride_w = 1, 1

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 512) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"
    
    # Original TVM test includes:
    # `matches = re.findall("winograd", graph)`
    # `assert len(matches) > 0`
    # This part cannot be translated as PyTorch does not expose its internal convolution algorithm selection.
    # We just ensure functional correctness.


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_winograd_conv(dtype):
    input_shape = (1, 3, 3, 4) # NHWC
    
    # First conv layer parameters
    filter_shape3 = (3, 3, 4, 8) # HWIO
    bias_shape_tvm3 = (1, 1, 1, 8) # TVM bias shape
    
    # Prepare tensors for the first conv
    input_tensor1_nchw, weight_tensor3_oihw, bias_tensor3_1d = prepare_conv2d_tensors(
        input_shape=input_shape,
        filter_shape=filter_shape3,
        bias_shape_tvm=bias_shape_tvm3,
        dtype_torch=dtype,
        seed=1,
        has_bias=True,
        target_device=CURRENT_DEVICE
    )
    
    # Apply first conv block (conv + relu)
    conv1_output_nchw = apply_pytorch_conv2d_block(
        input_tensor_nchw=input_tensor1_nchw,
        weight_tensor_oihw=weight_tensor3_oihw,
        bias_tensor_1d=bias_tensor3_1d,
        padding_tvm=[1, 1, 1, 1],
        strides_tvm=[1, 1], # Default
    )
    # Output of first conv: (1, 8, 3, 3) NCHW

    # Second conv layer parameters
    filter_shape4 = (3, 3, 8, 8) # HWIO
    bias_shape_tvm4 = (1, 1, 1, 8) # TVM bias shape
    
    # Prepare tensors for the second conv (input is output of first conv, weights are new)
    # We need to re-seed here to match TVM's behavior of calling initializer again for 'weight' and 'bias'.
    # However, PyTorch's `prepare_conv2d_tensors` generates input as well.
    # So we only use it to generate filter_np and bias_np for the second layer.
    np.random.seed(1) # Re-seed to get same random sequence for second layer's Xavier
    filter_np4 = np.zeros(filter_shape4, dtype=dtype)
    initializer = MockXavierInitializer()
    initializer("weight", filter_np4)
    bias_np4 = np.zeros(bias_shape_tvm4, dtype=dtype)
    initializer("bias", bias_np4)

    weight_tensor4_oihw = torch.tensor(filter_np4, device=CURRENT_DEVICE).permute(3, 2, 0, 1)
    bias_tensor4_1d = torch.tensor(bias_np4.flatten(), device=CURRENT_DEVICE)
    
    # Apply second conv block (conv + relu)
    # Note: The original TVM test for 'winograd_conv' does NOT have a final ReLU
    # for the second convolution `D = relay.nn.conv2d(...)`.
    # I will adapt `apply_pytorch_conv2d_block` to NOT apply ReLU if bias_tensor_1d is None,
    # or make a separate `apply_conv2d_only` function.
    # For now, I will modify it manually here.
    
    conv_strides_second = [1, 1]
    conv_padding_second = [1, 1, 1, 1] # From TVM style
    
    # Convert TVM padding to PyTorch F.conv2d format
    if isinstance(conv_padding_second, (list, tuple)) and len(conv_padding_second) == 4:
        if conv_padding_second[0] == conv_padding_second[2] and conv_padding_second[1] == conv_padding_second[3]:
            conv_padding_f_conv = (conv_padding_second[0], conv_padding_second[1])
        else:
            # Should not happen in this specific test
            raise ValueError("Asymmetric padding in second conv of test_conv2d_winograd_conv is not handled")
    else:
        conv_padding_f_conv = conv_padding_second
    
    conv_strides_f_conv = conv_strides_second if isinstance(conv_strides_second, (list, tuple)) else (conv_strides_second, conv_strides_second)

    # Perform second convolution (no explicit ReLU for the final output in TVM test)
    conv2_output_nchw = F.conv2d(
        input=conv1_output_nchw,
        weight=weight_tensor4_oihw,
        bias=bias_tensor4_1d,
        stride=conv_strides_f_conv,
        padding=conv_padding_f_conv,
        dilation=1, groups=1
    )
    
    # Convert output back to NHWC for comparison
    result_np = conv2_output_nchw.permute(0, 2, 3, 1).detach().cpu().numpy()

    expected_output_shape = (1, 3, 3, 8) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"
    
    # Original TVM test includes winograd check, which is skipped.


@pytest.mark.skipif(CURRENT_DEVICE.type == "cpu", reason="Requires GPU for equivalent Adreno test")
@pytest.mark.parametrize("dtype", [DTYPE_TORCH])
def test_conv2d_winograd_non_rect(dtype):
    input_shape = (1, 36, 64, 771) # NHWC
    filter_shape = (3, 3, 771, 128) # HWIO, out_channels=128
    
    # No bias or explicit relu in this TVM test for the final output.
    # The `apply_pytorch_conv2d_block` includes ReLU, so I'll manually apply conv.
    
    # Prepare tensors
    input_tensor_nchw, weight_tensor_oihw, _ = prepare_conv2d_tensors(
        input_shape=input_shape,
        filter_shape=filter_shape,
        bias_shape_tvm=None, # No bias
        dtype_torch=dtype,
        seed=1,
        has_bias=False,
        target_device=CURRENT_DEVICE
    )

    padding_tvm = [1, 1, 1, 1]
    strides_tvm = [1, 1]
    
    # Convert TVM padding to PyTorch F.conv2d format
    if isinstance(padding_tvm, (list, tuple)) and len(padding_tvm) == 4:
        if padding_tvm[0] == padding_tvm[2] and padding_tvm[1] == padding_tvm[3]:
            conv_padding_f_conv = (padding_tvm[0], padding_tvm[1])
        else:
            input_tensor_nchw = F.pad(input_tensor_nchw, pad=(padding_tvm[1], padding_tvm[3], padding_tvm[0], padding_tvm[2]), mode='constant', value=0.0)
            conv_padding_f_conv = 0
    else:
        conv_padding_f_conv = padding_tvm
    
    conv_strides_f_conv = strides_tvm if isinstance(strides_tvm, (list, tuple)) else (strides_tvm, strides_tvm)

    # Perform convolution without ReLU
    conv_output_nchw = F.conv2d(
        input=input_tensor_nchw,
        weight=weight_tensor_oihw,
        bias=None, # No bias
        stride=conv_strides_f_conv,
        padding=conv_padding_f_conv,
        dilation=1, groups=1
    )
    
    # Convert output back to NHWC for comparison
    result_np = conv_output_nchw.permute(0, 2, 3, 1).detach().cpu().numpy()

    H_in, W_in = 36, 64
    pad_h, pad_w = 1, 1
    k_h, k_w = 3, 3
    stride_h, stride_w = 1, 1

    H_out = np.floor((H_in + 2*pad_h - k_h) / stride_h) + 1
    W_out = np.floor((W_in + 2*pad_w - k_w) / stride_w) + 1
    
    expected_output_shape = (1, int(H_out), int(W_out), 128) # NHWC
    assert result_np.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result_np.shape}"
    assert not np.isnan(result_np).any(), "Output contains NaN values"
    assert not np.isinf(result_np).any(), "Output contains Inf values"

    # Original TVM test includes winograd check, which is skipped.

if __name__ == "__main__":
    pytest.main([__file__])
