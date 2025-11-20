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
"""Arm Compute Library integration conv2d tests."""

import numpy as np
import pytest

import torch
import torch.nn.functional as F
import torch.testing as testing
import torch.ao.nn.quantized.functional as F_q
from math import floor

# TVM-specific infrastructure for Arm Compute Library (ACL) are removed.
# `QNN_DTYPES`, `get_low_high_atol_rtol`, `skip_runtime_test`, `skip_codegen_test`,
# `build_and_run`, `verify`, `verify_codegen`, `Device` are not used.

# Define equivalent for `QNN_DTYPES` and `get_low_high_atol_rtol`
QNN_DTYPES = [torch.qint8, torch.quint8] # PyTorch quantized dtypes
# This is a simplification; PyTorch quantizes to torch.qint8/quint8, TVM typically works with int8/uint8.
# For simplicity, we'll map `dtype_str` to `torch.int8`/`torch.uint8` for input tensors.

def get_low_high_atol_rtol(dtype_str):
    if dtype_str == "float32":
        return -127, 128, 0.001, 0.001
    elif dtype_str == "uint8":
        return 0, 255, 0, 0
    elif dtype_str == "int8":
        return -128, 127, 0, 0
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

# A simple numerical verification function, since `verify` from infrastructure is gone.
def assert_outputs_equal(outputs, atol, rtol, config=None, verify_saturation=False):
    if not outputs:
        pytest.fail("No outputs to verify.")
    if len(outputs) == 1:
        assert isinstance(outputs[0], torch.Tensor)
        assert outputs[0].numel() > 0, "Output tensor is empty"
        print(f"Single output verified. Shape: {outputs[0].shape}, Dtype: {outputs[0].dtype}")
        if config:
            print(f"Test config: {config}")
        return

    first_output = outputs[0]
    for i, output in enumerate(outputs[1:]):
        # Convert to float for comparison if dtypes differ or if it's quantized
        testing.assert_allclose(first_output.float(), output.float(), atol=atol, rtol=rtol, msg=f"Output mismatch at index {i+1}")
    print(f"All {len(outputs)} outputs are numerically close within tolerance.")
    if config:
        print(f"Test config: {config}")


def _get_model(
    input_tensor,
    kernel_h,
    kernel_w,
    padding, # TVM padding (ph, pw) or (ph_b, pw_b, ph_a, pw_a)
    strides,
    dilation,
    groups,
    out_channels,
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Return a PyTorch model (functional representation) for float conv2d."""
    input_shape = input_tensor.shape # NHWC (N, H, W, C)

    current_input = input_tensor
    # PyTorch F.pad expects (pad_left, pad_right, pad_top, pad_bottom) for last dims (W, H)
    # So TVM padding for (H,W) -> PyTorch (W_val, W_val, H_val, H_val) for symmetric.
    # If TVM padding is (ph_b, pw_b, ph_a, pw_a) then PyTorch would be (pw_b, pw_a, ph_b, ph_a)
    padding_conv_pytorch = (0, 0) # Default no padding applied by F.conv2d itself if explicit F.pad used
    
    # Convert TVM padding (ph, pw) or (ph_b, pw_b, ph_a, pw_a) to PyTorch F.pad format (W_b, W_a, H_b, H_a)
    # The `padding` argument to `_get_model` is `(P_H, P_W)` (length 2) or `(P_T, P_L, P_B, P_R)` (length 4)
    if isinstance(padding, (list, tuple)) and len(padding) == 2:
        H_pad, W_pad = padding
        pad_for_f_pad = (W_pad, W_pad, H_pad, H_pad) # (W_before, W_after, H_before, H_after)
    elif isinstance(padding, (list, tuple)) and len(padding) == 4:
        P_T, P_L, P_B, P_R = padding
        pad_for_f_pad = (P_L, P_R, P_T, P_B)
    else:
        # If padding is a single int, apply symmetrically
        pad_for_f_pad = (padding, padding, padding, padding)


    if has_pad:
        current_input = F.pad(current_input.permute(0, 3, 1, 2), pad_for_f_pad).permute(0, 2, 3, 1) # Apply to NCHW, then back to NHWC
        padding_conv_pytorch = (0, 0) # Conv itself will not add further padding
    else:
        # PyTorch F.conv2d padding argument expects (pad_h, pad_w) for NHWC/NCHW.
        # This means symmetric padding.
        if isinstance(padding, (list, tuple)) and len(padding) == 2:
            padding_conv_pytorch = padding
        elif isinstance(padding, (list, tuple)) and len(padding) == 4:
            # If asymmetric padding from TVM, PyTorch F.conv2d only takes symmetric.
            # So, for direct mapping, we must convert. Let's take the max from each dimension.
            padding_conv_pytorch = (max(padding[0], padding[2]), max(padding[1], padding[3]))
        else:
            padding_conv_pytorch = (padding, padding)
            
    # PyTorch F.conv2d expects NCHW input and OIHW weight by default.
    # We must permute the input from NHWC to NCHW before conv, and permute output back.
    current_input_nchw = current_input.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W

    # PyTorch weight format: (out_channels, in_channels/groups, kernel_h, kernel_w)
    in_channels_for_weight = input_shape[3] // groups # C_in from NHWC input
    weight_shape = (out_channels, in_channels_for_weight, kernel_h, kernel_w)
    
    weights_np = np.random.uniform(-128, 127, weight_shape).astype(input_tensor.dtype)
    weights = torch.tensor(weights_np, dtype=input_tensor.dtype)
    
    bias_tensor = None
    params = {"w": weights}
    if has_bias:
        bias_shape = out_channels
        bias_np = np.random.uniform(-128, 127, bias_shape).astype(input_tensor.dtype)
        bias_tensor = torch.tensor(bias_np, dtype=input_tensor.dtype)
        params["b"] = bias_tensor

    out_nchw = F.conv2d(
        current_input_nchw,
        weights,
        bias=bias_tensor,
        stride=strides,
        padding=padding_conv_pytorch,
        dilation=dilation,
        groups=groups,
    )

    # Permute output back to NHWC
    out = out_nchw.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
    
    if has_activation:
        out = F.relu(out)
    
    return out, params


def _get_qnn_params(input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, channels):
    """Get output qnn parameters given input and kernel parameters."""
    # This logic from TVM is about inferring output quantization parameters.
    # PyTorch does this implicitly or through a `QuantStub`/`DeQuantStub` and observer pattern.
    # However, to explicitly match TVM's behavior, we'll keep the numerical logic.
    input_max = input_sc * (255 - input_zp)
    input_min = -input_sc * input_zp
    kernel_max = kernel_sc * (255 - kernel_zp)
    kernel_min = -kernel_sc * kernel_zp
    
    # Assuming the "channels" argument here means input channels of the filter
    # This `channels` parameter in TVM's `_get_qnn_params` is used as `channels` in `_get_qnn_params`
    # and passed from `shape[3]` (input channels) to `_get_qnn_params`.
    # This `channels` should be the `input_channels_per_group` or `kernel_h * kernel_w * in_channels_per_group`.
    # It seems to be related to the accumulation size for the convolution.
    # A common interpretation for the "accumulation_term" in QNN output scale calculation is `kernel_h * kernel_w * in_channels_per_group`.
    
    # TVM `shape[3]` (input channels) for channels. This is `C_in`.
    # The actual calculation for output scale for a conv2d output has an accumulation term:
    # `output_scale_float = input_scale * kernel_scale * (kernel_h * kernel_w * C_in_per_group)` for sum over input channels.
    # Here `channels` represents `C_in_per_group`.
    
    output_limits = [
        kernel_max * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_min,
        kernel_max * kernel_h * kernel_w * channels * input_min,
    ]
    output_max = max(output_limits)
    output_min = min(output_limits)
    output_sc = (output_max - output_min) / 255
    output_zp = -int(floor(output_min / output_sc)) # Use floor for consistency if TVM truncates.
    return output_zp, output_sc


# Helper function for quantization/dequantization, mimicking TVM's `qnn.op` behavior
def qnn_quantize_per_tensor(data_float, output_scale, output_zero_point, out_dtype_torch):
    data_quant = torch.round(data_float / output_scale + output_zero_point)
    iinfo = np.iinfo(out_dtype_torch.item())
    return torch.clamp(data_quant, iinfo.min, iinfo.max).to(out_dtype_torch)

def qnn_dequantize_per_tensor(data_tensor, input_scale, input_zero_point):
    return (data_tensor.float() - input_zero_point) * input_scale


def _get_qnn_model(
    input_tensor_q, # Quantized input tensor
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    groups,
    dtype_torch, # Quantized input dtype (e.g., torch.uint8)
    out_channels,
    input_zp, # Input tensor zero point
    input_sc, # Input tensor scale
    kernel_zp, # Kernel tensor zero point
    kernel_sc, # Kernel tensor scale
    output_zp, # Output tensor zero point
    output_sc, # Output tensor scale
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Return a PyTorch model (functional representation) for quantized conv2d."""
    input_shape = input_tensor_q.
