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
import torch
import torch.nn.functional as F
import pytest
import numpy as np

# In PyTorch, graph transformations like CombineParallelConv2D are typically handled
# by torch.compile or by rewriting the model code directly.
# This test suite aims to verify that the transformed graph produces the same numerical
# output as the original graph.
# Thus, we'll define 'before' and 'expected' as PyTorch functions/modules and compare their outputs.


def get_conv2d_defaults():
    # TVM Relay conv2d defaults (kernel_size is part of weight shape)
    # strides, padding, dilation, groups are generally 1, 0, 1, 1 if not specified.
    return dict(stride=1, padding=0, dilation=1, groups=1, bias=None)


def get_max_pool2d_defaults():
    # Assuming common defaults for max_pool2d in TVM Relay context
    # TVM max_pool2d documentation implies 2x2 with stride 2 if not given.
    return dict(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)


def get_strided_slice_equivalent_for_channels(input_tensor, begin_channel_idx, slice_size):
    # TVM relay.strided_slice with slice_mode="size"
    # Example: y, begin=[0, channels1], end=[-1, channels2], strides=[1, 1], slice_mode="size"
    # For NCHW layout, batch dim is 0, channel dim is 1.
    # begin=[-1, X] in batch dim means "start from first element"
    # end=[-1, Y] in channel dim means "take Y elements from begin_channel_idx"
    # So this translates to normal slicing along the channel dimension.
    return input_tensor[:, begin_channel_idx : begin_channel_idx + slice_size, :, :]


class BeforeModel(torch.nn.Module):
    def __init__(self, x_shape, channels1, channels2, channels3, channels4):
        super().__init__()
        in_c = x_shape[1]
        self.w1 = torch.nn.Parameter(torch.randn(channels1, in_c, 1, 1))
        self.w2 = torch.nn.Parameter(torch.randn(channels2, in_c, 1, 1))
        self.w3 = torch.nn.Parameter(torch.randn(channels3, in_c, 3, 3)) # Different kernel size (3,3)
        self.w4 = torch.nn.Parameter(torch.randn(channels4, in_c, 1, 1))

    def forward(self, x):
        y1 = F.conv2d(x, self.w1, **get_conv2d_defaults())
        y2 = F.conv2d(x, self.w2, **get_conv2d_defaults())
        # y3 cannot be combined because of different kernel size (3,3 vs 1,1)
        y3 = F.conv2d(x, self.w3, **get_conv2d_defaults())
        y4 = F.conv2d(x, self.w4, **get_conv2d_defaults())
        y5 = F.max_pool2d(x, **get_max_pool2d_defaults())
        return (y1, y2, y3, y4, y5)


class ExpectedModel(torch.nn.Module):
    def __init__(self, x_shape, channels1, channels2, channels3, channels4):
        super().__init__()
        in_c = x_shape[1]
        # Weights are initialized here, but their data will be copied from `BeforeModel`
        self.w1 = torch.nn.Parameter(torch.randn(channels1, in_c, 1, 1))
        self.w2 = torch.nn.Parameter(torch.randn(channels2, in_c, 1, 1))
        self.w3 = torch.nn.Parameter(torch.randn(channels3, in_c, 3, 3)) # Uncombinable
        self.w4 = torch.nn.Parameter(torch.randn(channels4, in_c, 1, 1))

    def forward(self, x):
        # Combined convolutions: w1, w2, w4 have same kernel size and other conv params
        # Note: the order for concatenation matters and needs to match the slicing logic.
        # TVM `CombineParallelConv2D` typically orders them consistently.
        # Here we assume w1, w2, w4 are combined in that order, as per the `expected` definition.
        w_combined = torch.cat((self.w1, self.w2, self.w4), dim=0)
        
        y_combined = F.conv2d(x, w_combined, **get_conv2d_defaults())

        y1 = get_strided_slice_equivalent_for_channels(y_combined, 0, self.w1.shape[0])
        y2 = get_strided_slice_equivalent_for_channels(y_combined, self.w1.shape[0], self.w2.shape[0])
        
        y3 = F.conv2d(x, self.w3, **get_conv2d_defaults()) # Uncombined
        
        y4_begin = self.w1.shape[0] + self.w2.shape[0]
        y4 = get_strided_slice_equivalent_for_channels(y_combined, y4_begin, self.w4.shape[0])
        
        y5 = F.max_pool2d(x, **get_max_pool2d_defaults())
        return (y1, y2, y3, y4, y5)


def check_combine_parallel_conv2d_test_case(x_shape, channels1, channels2, channels3, channels4):
    # Setup random inputs
    x = torch.randn(x_shape, dtype=torch.float32)

    # Initialize models
    before_model = BeforeModel(x_shape, channels1, channels2, channels3, channels4)
    expected_model = ExpectedModel(x_shape, channels1, channels2, channels3, channels4)

    # Copy weights from before_model to expected_model for fair comparison
    # This simulates the graph transformation preserving weights.
    expected_model.w1.data = before_model.w1.data
    expected_model.w2.data = before_model.w2.data
    expected_model.w3.data = before_model.w3.data
    expected_model.w4.data = before_model.w4.data

    # Run models and compare outputs
    output_before = before_model(x)
    output_expected = expected_model(x)

    for i in range(len(output_before)):
        # Using a higher tolerance for potential floating point differences after graph rewrite
        torch.testing.assert_allclose(output_before[i], output_expected[i], rtol=1e-5, atol=1e-5)


def test_combine_parallel_conv2d():
    """Simple testcase."""
    check_combine_parallel_conv2d_test_case((1, 4, 16, 16), 4, 4, 4, 4)
    check_combine_parallel_conv2d_test_case((1, 4, 16, 16), 4, 8, 4, 7)


# --- test_combine_parallel_conv2d_scale_relu ---

class BeforeModelScaleRelu(torch.nn.Module):
    def __init__(self, x_shape, channels1, channels2):
        super().__init__()
        in_c = x_shape[1]
        self.w1 = torch.nn.Parameter(torch.randn(channels1, in_c, 1, 1))
        self.w2 = torch.nn.Parameter(torch.randn(channels2, in_c, 1, 1))
        # Scale parameters are (C, 1, 1) to broadcast correctly over NCHW output of conv2d
        self.scale1 = torch.nn.Parameter(torch.randn(channels1, 1, 1))
        self.scale2 = torch.nn.Parameter(torch.randn(channels2, 1, 1))
        self.bias = torch.nn.Parameter(torch.randn(channels2, 1, 1)) # Bias for y2 only

    def forward(self, x):
        y1 = F.conv2d(x, self.w1, **get_conv2d_defaults())
        y1 = y1 * self.scale1
        y1 = F.relu(y1)
        
        y2 = F.conv2d(x, self.w2, **get_conv2d_defaults())
        y2 = y2 * self.scale2
        y2 = F.relu(y2)
        y2 = y2 + self.bias
        return (y1, y2)


class ExpectedModelScaleRelu(torch.nn.Module):
    def __init__(self, x_shape, channels1, channels2):
        super().__init__()
        in_c = x_shape[1]
        self.w1 = torch.nn.Parameter(torch.randn(channels1, in_c, 1, 1))
        self.w2 = torch.nn.Parameter(torch.randn(channels2, in_c, 1, 1))
        self.scale1 = torch.nn.Parameter(torch.randn(channels1, 1, 1))
        self.scale2 = torch.nn.Parameter(torch.randn(channels2, 1, 1))
        self.bias = torch.nn.Parameter(torch.randn(channels2, 1, 1))

    def forward(self, x):
        w_combined = torch.cat((self.w1, self.w2), dim=0)
        scale_combined = torch.cat((self.scale1, self.scale2), dim=0)
        
        y = F.conv2d(x, w_combined, **get_conv2d_defaults())
        y = y * scale_combined
        y = F.relu(y)
        
        y1 = get_strided_slice_equivalent_for_channels(y, 0, self.w1.shape[0])
        y2 = get_strided_slice_equivalent_for_channels(y, self.w1.shape[0], self.w2.shape[0])
        y2 = y2 + self.bias
        return (y1, y2)


def check_combine_parallel_conv2d_scale_relu_test_case(x_shape, channels1, channels2):
    x = torch.randn(x_shape, dtype=torch.float32)

    before_model = BeforeModelScaleRelu(x_shape, channels1, channels2)
    expected_model = ExpectedModelScaleRelu(x_shape, channels1, channels2)

    # Copy weights
    expected_model.w1.data = before_model.w1.data
    expected_model.w2.data = before_model.w2.data
    expected_model.scale1.data = before_model.scale1.data
    expected_model.scale2.data = before_model.scale2.data
    expected_model.bias.data = before_model.bias.data

    output_before = before_model(x)
    output_expected = expected_model(x)

    for i in range(len(output_before)):
        torch.testing.assert_allclose(output_before[i], output_expected[i], rtol=1e-5, atol=1e-5)


def test_combine_parallel_conv2d_scale_relu():
    """Testcase of combining conv2d + scale + relu"""
    check_combine_parallel_conv2d_scale_relu_test_case((1, 4, 16, 16), 4, 8)


# --- test_combine_parallel_conv2d_scale ---

class BeforeModelScale(torch.nn.Module):
    def __init__(self, x_shape, channels1, channels2):
        super().__init__()
        in_c = x_shape[1]
        self.w1 = torch.nn.Parameter(torch.randn(channels1, in_c, 1, 1))
        self.w2 = torch.nn.Parameter(torch.randn(channels2, in_c, 1, 1))
        # Note: scales here are (1,) not (channels,1,1), meaning they broadcast
        # across all channels of the conv output.
        # This makes them "uncombinable" in the sense of concatenating scales.
        self.scale1 = torch.nn.Parameter(torch.randn(1))
        self.scale2 = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        y1 = F.conv2d(x, self.w1, **get_conv2d_defaults())
        y1 = y1 * self.scale1.reshape(1, 1, 1, 1) # Reshape to (1,1,1,1) for NCHW broadcasting
        
        y2 = F.conv2d(x, self.w2, **get_conv2d_defaults())
        y2 = y2 * self.scale2.reshape(1, 1, 1, 1) # Reshape to (1,1,1,1) for NCHW broadcasting
        return (y1, y2)


class ExpectedModelScale(torch.nn.Module):
    def __init__(self, x_shape, channels1, channels2):
        super().__init__()
        in_c = x_shape[1]
        self.w1 = torch.nn.Parameter(torch.randn(channels1, in_c, 1, 1))
        self.w2 = torch.nn.Parameter(torch.randn(channels2, in_c, 1, 1))
        self.scale1 = torch.nn.Parameter(torch.randn(1))
        self.scale2 = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        w_combined = torch.cat((self.w1, self.w2), dim=0)
        y = F.conv2d(x, w_combined, **get_conv2d_defaults())
        
        # Scales are (1,), so they cannot be concatenated and applied across channels.
        # Thus, the slices of y are multiplied by their respective scales *after* slicing.
        y1 = get_strided_slice_equivalent_for_channels(y, 0, self.w1.shape[0])
        y2 = get_strided_slice_equivalent_for_channels(y, self.w1.shape[0], self.w2.shape[0])
        
        y1 = y1 * self.scale1.reshape(1, 1, 1, 1) # Reshape for explicit broadcasting
        y2 = y2 * self.scale2.reshape(1, 1, 1, 1) # Reshape for explicit broadcasting
        return (y1, y2)


def check_combine_parallel_conv2d_scale_test_case(x_shape, channels1, channels2):
    x = torch.randn(x_shape, dtype=torch.float32)

    before_model = BeforeModelScale(x_shape, channels1, channels2)
    expected_model = ExpectedModelScale(x_shape, channels1, channels2)

    # Copy weights
    expected_model.w1.data = before_model.w1.data
    expected_model.w2.data = before_model.w2.data
    expected_model.scale1.data = before_model.scale1.data
    expected_model.scale2.data = before_model.scale2.data

    output_before = before_model(x)
    output_expected = expected_model(x)

    for i in range(len(output_before)):
        torch.testing.assert_allclose(output_before[i], output_expected[i], rtol=1e-5, atol=1e-5)


def test_combine_parallel_conv2d_scale():
    """Testcase of un-combinable scale"""
    check_combine_parallel_conv2d_scale_test_case((1, 4, 16, 16), 4, 8)


# --- test_combine_parallel_conv2d_multiple_blocks ---

class BeforeModelMultipleBlocks(torch.nn.Module):
    def __init__(self, x_shape, channels, repeat):
        super().__init__()
        in_c = x_shape[1] # Initial input channels for x
        self.w = torch.nn.Parameter(torch.randn(channels, in_c, 1, 1))
        self.repeat = repeat
        self.output_channels_per_conv = channels # Output channels for each conv2d call

        # Assert consistency for this specific test's structure:
        # The output channels after concat (2 * output_channels_per_conv) must be
        # equal to the input channels 'in_c' for the next iteration's conv2d if 'w' is reused as-is.
        # This is `2 * channels == in_c` if 'channels' is the output_channels for 'w'.
        if 2 * self.output_channels_per_conv != in_c:
             # This check is important as per the TVM test interpretation.
            raise ValueError(f"Inconsistent channel sizes: 2 * output_channels_per_conv ({2 * self.output_channels_per_conv}) must equal initial in_c ({in_c})")

    def forward(self, x):
        y = x
        for i in range(self.repeat):
            # In each iteration, y.shape[1] will be `in_c` (original x_shape[1])
            # due to the self-feeding channel consistency `2 * out_c == in_c`.
            # So, `self.w` (with `in_c` as its input channel) is always compatible.
            y1 = F.conv2d(y, self.w, **get_conv2d_defaults())
            y2 = F.conv2d(y, self.w, **get_conv2d_defaults()) # uses same weight 'w' as y1
            y = torch.cat((y1, y2), dim=1) # Concatenate along channel dimension
        return y


class ExpectedModelMultipleBlocks(torch.nn.Module):
    def __init__(self, x_shape, channels, repeat):
        super().__init__()
        in_c = x_shape[1]
        self.w = torch.nn.Parameter(torch.randn(channels, in_c, 1, 1))
        self.repeat = repeat
        self.output_channels_per_conv = channels

        if 2 * self.output_channels_per_conv != in_c:
             raise ValueError(f"Inconsistent channel sizes: 2 * output_channels_per_conv ({2 * self.output_channels_per_conv}) must equal initial in_c ({in_c})")

    def forward(self, x):
        y = x
        for i in range(self.repeat):
            # Combined weights: (2*output_channels_per_conv, in_c, 1, 1)
            w_concat = torch.cat((self.w, self.w), dim=0)
            
            # This conv2d takes `y` as input, which always has `in_c` channels.
            # `w_concat` has `in_c` as its input channel. Output channels will be `2*output_channels_per_conv`.
            y_combined_conv = F.conv2d(y, w_concat, **get_conv2d_defaults())
            
            # Slice the combined output back to individual logical branches
            y1 = get_strided_slice_equivalent_for_channels(y_combined_conv, 0, self.output_channels_per_conv)
            y2 = get_strided_slice_equivalent_for_channels(y_combined_conv, self.output_channels_per_conv, self.output_channels_per_conv)
            
            y = torch.cat((y1, y2), dim=1) # Concatenate for the next iteration's input
        return y


def check_combine_parallel_conv2d_multiple_blocks_test_case(x_shape, repeat):
    x = torch.randn(x_shape, dtype=torch.float32)
    in_c = x_shape[1]
    out_c = in_c // 2 # Derived from example `out_c = in_c // 2`

    before_model = BeforeModelMultipleBlocks(x_shape, out_c, repeat)
    expected_model = ExpectedModelMultipleBlocks(x_shape, out_c, repeat)

    # Copy shared weights
    expected_model.w.data = before_model.w.data

    output_before = before_model(x)
    output_expected = expected_model(x)

    torch.testing.assert_allclose(output_before, output_expected, rtol=1e-5, atol=1e-5)


def test_combine_parallel_conv2d_multiple_blocks():
    check_combine_parallel_conv2d_multiple_blocks_test_case((1, 4, 16, 16), 4)

if __name__ == "__main__":
    pytest.main([__file__])
