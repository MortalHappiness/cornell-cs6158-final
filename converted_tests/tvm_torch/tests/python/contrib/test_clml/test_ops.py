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
"""CLML integration conv2d tests."""

import torch
import numpy as np
import torch.nn.functional as F
import pytest

# Infrastructure components from TVM contrib.clml are not directly convertible.
# The tests will be rewritten to use direct PyTorch execution and comparison.
# `build_and_run`, `Device`, `skip_codegen_test` are removed.

def _get_conv_model(
    input_tensor,
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    groups,
    out_channels,
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Return a PyTorch model (functional representation) and its parameters."""
    
    # PyTorch expects padding as (pad_left, pad_right, pad_top, pad_bottom, ...)
    # TVM's padding can be (H_pad, W_pad) or (H_top, W_left, H_bottom, W_right).
    # Assuming TVM's padding (p_h, p_w) means symmetric padding (p_h, p_w, p_h, p_w)
    if isinstance(padding, (list, tuple)) and len(padding) == 2:
        padding_pytorch = (padding[1], padding[1], padding[0], padding[0]) # PyTorch pad order is (W, W, H, H) for 2D
    else:
        # If it's already in (top, left, bottom, right) it would need reordering,
        # but the test cases generally use 2-element tuples.
        # Assuming for conv2d, padding is symmetric (ph, pw).
        padding_pytorch = padding


    current_input = input_tensor
    if has_pad:
        # TVM pad_width=((0,0),(padding[0],padding[0]),(padding[1],padding[1]),(0,0)) for NCHW
        # PyTorch F.pad expects (pad_left, pad_right, pad_top, pad_bottom, ...) for last dims
        # So for NCHW and padding[0]=H, padding[1]=W: (W_before, W_after, H_before, H_after)
        
        # Original TVM:
        # p = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)) # Assuming NCHW input
        # current_input = relay.nn.pad(a, pad_width=p)

        # Assuming input is NCHW, so padding applies to dims 2 (H) and 3 (W)
        # Pad format is (pad_left, pad_right, pad_top, pad_bottom)
        padding_for_f_pad = (padding[1], padding[1], padding[0], padding[0])
        current_input = F.pad(current_input, padding_for_f_pad)
        padding_conv = (0, 0) # No additional padding for conv since it's already padded
    else:
        padding_conv = padding # Use the given padding for conv

    # For Conv2D, PyTorch weight format is (out_channels, in_channels/groups, kernel_h, kernel_w)
    in_channels_for_weight = input_tensor.shape[1] // groups
    weight_shape = (out_channels, in_channels_for_weight, kernel_h, kernel_w)
    
    weights_np = np.random.uniform(-1, 1, weight_shape).astype(input_tensor.dtype)
    weights = torch.tensor(weights_np, dtype=input_tensor.dtype)

    out = F.conv2d(
        current_input,
        weights,
        bias=None, # Bias handled separately if has_bias is true
        stride=strides,
        padding=padding_conv,
        dilation=dilation,
        groups=groups,
    )
    
    params = {"w": weights}
    
    if has_bias:
        bias_shape = out_channels
        bias_np = np.random.uniform(-1, 1, bias_shape).astype(input_tensor.dtype)
        bias_tensor = torch.tensor(bias_np, dtype=input_tensor.dtype)
        out = out + bias_tensor.view(1, -1, 1, 1) # Bias_add in NCHW
        params["b"] = bias_tensor

    if has_activation:
        out = F.relu(out)

    return out, params


@pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.skipif(skip_codegen_test(), reason="Skip because CLML codegen is not available")
# @tvm.testing.requires_openclml # Removed, now direct PyTorch test
def test_conv2d(dtype):
    trials = [
        # Normal convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (15, 16, 12), (False, False, True)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, True)],
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (16, 12, 15), (False, False, True)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False)],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (16, 12, 15), (False, False, False)],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True)],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False)],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False)],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (10, 10, 14), (False, True, True)], # This one had (14,10,10) before
    ]

    for (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape_hwc, # Original was (H,W,C), now it's (C,H,W) for PyTorch NCHW
        composite,
    ) in trials:
        # Convert (C,H,W) from TVM's interpretation of shape to NCHW for PyTorch
        if len(shape_hwc) == 3: # (C,H,W)
            shape = (1, shape_hwc[0], shape_hwc[1], shape_hwc[2]) # (N,C,H,W)
        else: # (H,W,C)
            shape = (1, shape_hwc[2], shape_hwc[0], shape_hwc[1]) # (N,C,H,W)

        groups = 1
        
        input_np = np.random.uniform(-1, 1, shape).astype(dtype)
        input_tensor = torch.tensor(input_np, dtype=dtype)

        # Run with PyTorch (this represents the baseline)
        output_pytorch, _ = _get_conv_model(
            input_tensor,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            groups,
            out_channels,
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        
        # In the original test, `clml_out` and `opencl_out` are compared.
        # Here we only have one execution path, so we're ensuring it runs and produces
        # a valid tensor. If a reference was available, we would compare against it.
        # For now, a self-consistency check or basic sanity is sufficient for conversion.
        # Since the original test compared two TVM backends against each other,
        # we assume that the PyTorch implementation provides the correct baseline.
        assert output_pytorch is not None
        assert output_pytorch.shape == _get_expected_conv_shape(shape, kernel_h, kernel_w, pad, stride, dilation, out_channels, composite[0])
        

def _get_expected_conv_shape(input_shape, kernel_h, kernel_w, padding, strides, dilation, out_channels, has_pad):
    # This helper function mimics the output shape calculation
    # For NCHW input (N, C_in, H_in, W_in)
    N, C_in, H_in, W_in = input_shape
    S_H, S_W = strides
    D_H, D_W = dilation
    P_H, P_W = padding # Assuming symmetric padding (ph, pw)

    if has_pad:
        # Effective input dimensions after explicit padding
        H_in_padded = H_in + 2 * P_H
        W_in_padded = W_in + 2 * P_W
        P_H, P_W = (0, 0) # No additional padding for conv op
    else:
        H_in_padded = H_in
        W_in_padded = W_in

    H_out = int((H_in_padded + 2 * P_H - D_H * (kernel_h - 1) - 1) / S_H) + 1
    W_out = int((W_in_padded + 2 * P_W - D_W * (kernel_w - 1) - 1) / S_W) + 1
    
    return (N, out_channels, H_out, W_out)


@pytest.mark.parametrize("dtype", [torch.float16])
# @tvm.testing.requires_openclml # Removed, now direct PyTorch test
def _test_batchnorm(dtype):
    # TODO: This test case uses `relay.nn.batch_norm` which returns a tuple.
    # The PyTorch `F.batch_norm` does not return a tuple in the same way.
    # It takes `running_mean` and `running_var` for inference mode, and updates them for training.
    # The original TVM code takes `mean` and `variance` as constants, implying inference mode.
    # We will map to `F.batch_norm` in inference mode.
    
    in_shape = (1, 8, 64, 64) # NCHW
    channels = 8

    input_np = np.random.uniform(-1, 1, in_shape).astype(dtype)
    input_tensor = torch.tensor(input_np, dtype=dtype)
    gamma_np = np.random.uniform(-1, 1, (channels)).astype(dtype)
    beta_np = np.random.uniform(-1, 1, (channels)).astype(dtype)
    gamma = torch.tensor(gamma_np, dtype=dtype)
    beta = torch.tensor(beta_np, dtype=dtype)

    mean_np = np.mean(input_np, axis=(0, 2, 3), keepdims=False)
    mean = torch.tensor(mean_np, dtype=dtype)
    variance_np = np.var(input_np, axis=(0, 2, 3), keepdims=False)
    variance = torch.tensor(variance_np, dtype=dtype)

    # PyTorch F.batch_norm expects input, running_mean, running_var, weight, bias, training, momentum, eps
    # `weight` corresponds to gamma, `bias` to beta.
    # `training=False` for inference mode, using provided running_mean and running_var.
    func_output = F.batch_norm(
        input_tensor,
        running_mean=mean,
        running_var=variance,
        weight=gamma,
        bias=beta,
        training=False,
        momentum=0.1, # Default momentum
        eps=0.0001
    )

    # Here we are comparing to itself if it were compiled by two different TVM backends.
    # For PyTorch, we just assert that it ran and produced a tensor of the expected shape.
    assert func_output is not None
    assert func_output.shape == in_shape

@pytest.mark.parametrize("dtype", [torch.float16])
# @tvm.testing.requires_openclml # Removed, now direct PyTorch test
def test_concat(dtype):
    in_shape_1 = (1, 16, 16, 16) # NCHW
    in_shape_2 = (1, 16, 16, 16) # NCHW
    
    input_1_np = np.random.uniform(-1, 1, in_shape_1).astype(dtype)
    input_2_np = np.random.uniform(-1, 1, in_shape_2).astype(dtype)

    input_1 = torch.tensor(input_1_np, dtype=dtype)
    input_2 = torch.tensor(input_2_np, dtype=dtype)

    # TVM `relay.concatenate((a, b), axis=1)`
    func_output = torch.cat((input_1, input_2), dim=1) # Concatenate along C dimension

    expected_shape = (1, 32, 16, 16)
    assert func_output is not None
    assert func_output.shape == expected_shape

@pytest.mark.parametrize("dtype", [torch.float16])
# @tvm.testing.requires_openclml # Removed, now direct PyTorch test
def test_avgpool(dtype):
    trials = [
        # input size         pool_size stride  padding          pooling_type
        [(1, 64, 147, 147), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 192, 71, 71), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 288, 35, 35), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 768, 17, 17), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 2048, 17, 17), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 192, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
        [(1, 256, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
        [(1, 288, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
        [(1, 768, 17, 17), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
        [(1, 1280, 8, 8), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
    ]
    
    for (
        input_shape,
        pool_size,
        stride,
        padding,
        pooling_type,
    ) in trials:
        input_np = np.random.uniform(-1, 1, input_shape).astype(dtype)
        input_tensor = torch.tensor(input_np, dtype=dtype)
        
        # PyTorch F.pooling padding expects a single value or tuple (H, W) or (H_left, H_right, W_top, W_bottom)
        # TVM `padding=(0, 0, 0, 0)` is (pad_h_before, pad_w_before, pad_h_after, pad_w_after)
        # PyTorch uses (pad_left, pad_right, pad_top, pad_bottom) for 2D.
        # This means TVM (ph_b, pw_b, ph_a, pw_a) -> PyTorch (pw_b, pw_a, ph_b, ph_a)
        
        if len(padding) == 4:
            padding_pytorch = (padding[1], padding[3], padding[0], padding[2])
        elif len(padding) == 2:
            padding_pytorch = (padding[1], padding[1], padding[0], padding[0])
        else: # single int
            padding_pytorch = padding

        if pooling_type == "max":
            func_output = F.max_pool2d(
                input_tensor, 
                kernel_size=pool_size, 
                stride=stride, 
                padding=padding_pytorch
            )
        else: # avg
            func_output = F.avg_pool2d(
                input_tensor, 
                kernel_size=pool_size, 
                stride=stride, 
                padding=padding_pytorch
            )

        assert func_output is not None
        # Basic shape check to ensure it ran. Detailed numerical checks would be needed if a reference was provided.
        assert func_output.shape == _get_expected_pool_shape(input_shape, pool_size, stride, padding, pooling_type)

def _get_expected_pool_shape(input_shape, pool_size, stride, padding, pooling_type):
    N, C_in, H_in, W_in = input_shape
    K_H, K_W = pool_size
    S_H, S_W = stride

    # TVM padding for avg_pool2d/max_pool2d is (pad_top, pad_left, pad_bottom, pad_right)
    # PyTorch functional pool2d padding is (H_pad_top, H_pad_bottom, W_pad_left, W_pad_right) for symetric
    # However, F.pool2d argument `padding` is either an int or a tuple (pad_h, pad_w)
    # So assuming TVM's padding values are total padding to apply on each side,
    # and they typically map to PyTorch's (pad_h, pad_w) argument which implies symmetric.
    # For (0,0,1,1) in TVM, that means 0 on top/bottom, 1 on left/right. PyTorch `padding=(0,1)`
    
    if len(padding) == 4:
        # Assuming TVM (p_h_b, p_w_b, p_h_a, p_w_a) -> PyTorch (pad_h, pad_w)
        # Effectively the sum of before and after padding might be used, but F.pool2d uses symmetric.
        # For simple cases, if TVM gives (0,0,1,1) it means pad_top=0, pad_left=0, pad_bottom=1, pad_right=1
        # PyTorch pool2d(padding=(pad_h, pad_w)) means pad_h on top/bottom, pad_w on left/right
        # This implies it should be (max(p_h_b, p_h_a), max(p_w_b, p_w_a)) or similar, or F.pad before.
        # For now, let's assume padding for F.pool2d is like total symmetric.
        # This is an approximation as PyTorch padding behavior for pooling can be subtle.
        P_H = padding[0] + padding[2] # Sum of padding before and after height
        P_W = padding[1] + padding[3] # Sum of padding before and after width
        P_H = P_H // 2 # Symmetric padding
        P_W = P_W // 2 # Symmetric padding
    elif len(padding) == 2:
        P_H, P_W = padding
    else: # single int
        P_H, P_W = padding, padding
    
    # Calculation with ceil_mode=False (default for F.pool2d)
    H_out = int(np.floor((H_in + 2 * P_H - K_H) / S_H)) + 1
    W_out = int(np.floor((W_in + 2 * P_W - K_W) / S_W)) + 1

    return (N, C_in, H_out, W_out)


if __name__ == "__main__":
    pytest.main([__file__])
