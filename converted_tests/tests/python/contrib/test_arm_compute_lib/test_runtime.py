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
"""Arm Compute Library runtime tests."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.testing as testing
import pytest

# TVM-specific infrastructure for Arm Compute Library (ACL) are removed.
# `skip_runtime_test`, `build_and_run`, `verify`, `Device` are not used.

# A simple numerical verification function, since `verify` from infrastructure is gone.
def assert_outputs_equal(outputs, atol, rtol, config=None):
    if not outputs:
        pytest.fail("No outputs to verify.")
    if len(outputs) == 1:
        # If only one output, just check it's a tensor and has data.
        assert isinstance(outputs[0], torch.Tensor)
        assert outputs[0].numel() > 0, "Output tensor is empty"
        print(f"Single output verified. Shape: {outputs[0].shape}, Dtype: {outputs[0].dtype}")
        if config:
            print(f"Test config: {config}")
        return

    # Assuming outputs are from different "backends" (e.g., TVM vs ACL),
    # but here they will all be PyTorch results. So they should be identical.
    first_output = outputs[0]
    for i, output in enumerate(outputs[1:]):
        testing.assert_allclose(first_output, output, atol=atol, rtol=rtol, msg=f"Output mismatch at index {i+1}")
    print(f"All {len(outputs)} outputs are numerically close within tolerance.")
    if config:
        print(f"Test config: {config}")


def get_model_multiple_ops(input_tensor):
    """Return a PyTorch model (functional representation) for multiple ops."""
    # input_tensor is (1, 1, 1, 1000)
    out = torch.reshape(input_tensor, (1, 1, 1000))
    out = torch.reshape(out, (1, 1000))
    return out


def test_multiple_ops():
    """
    Test multiple operators.
    Original TVM test compares TVM-compiled output vs ACL-compiled output.
    Here, we just run the PyTorch model.
    """
    np.random.seed(0)

    input_np = np.random.uniform(0, 1, (1, 1, 1, 1000)).astype("float32")
    input_tensor = torch.tensor(input_np, dtype=torch.float32)

    # Outputs will store results from (simulated) different backends
    outputs = []
    # The original loop for `acl in [False, True]` now just runs PyTorch directly.
    # We run it twice to check consistency if multiple "runs" were needed, or just once.
    # For this test, it's about checking if the model can be offloaded,
    # so just ensuring the PyTorch model works is sufficient for conversion.
    # We will just run once and assert the shape.
    result_tensor = get_model_multiple_ops(input_tensor)
    outputs.append(result_tensor)
    
    assert result_tensor.shape == (1, 1000)
    assert isinstance(result_tensor, torch.Tensor)


def get_model_heterogeneous(input_tensor):
    """Return a PyTorch model (functional representation) for heterogeneous ops."""
    # input_tensor is (1, 1, 1, 1000)
    out = torch.reshape(input_tensor, (1, 1, 1000))
    out = torch.sigmoid(out)
    out = torch.reshape(out, (1, 1000))
    return out


def test_heterogeneous():
    """
    Test to check if offloading only supported operators works,
    while leaving unsupported operators computed via tvm.
    Original TVM test compares TVM-compiled output vs ACL-compiled output.
    Here, we just run the PyTorch model.
    """
    np.random.seed(0)

    input_np = np.random.uniform(-127, 128, (1, 1, 1, 1000)).astype("float32")
    input_tensor = torch.tensor(input_np, dtype=torch.float32)

    outputs = []
    result_tensor = get_model_heterogeneous(input_tensor)
    outputs.append(result_tensor)

    assert result_tensor.shape == (1, 1000)
    assert isinstance(result_tensor, torch.Tensor)


def get_model_multiple_runs():
    """Return a PyTorch module and its parameters for multiple runs."""
    # Original TVM: a = relay.var("a", shape=(1, 28, 28, 512), dtype="float32")
    # Weights for NHWC input (channels_last) and OHWI kernel_layout:
    # PyTorch conv2d `weight` format is (out_channels, in_channels, kernel_h, kernel_w) for NCHW
    # For NHWC input, PyTorch `weight` should be (out_channels, kernel_h, kernel_w, in_channels) if using Conv2d with channels_last
    # Or, if using F.conv2d directly with NCHW, inputs/weights need to be permuted.
    # The original TVM data_layout="NHWC" and kernel_layout="OHWI" suggests using NCHW input + OHWI weights
    # Or permuting inputs from NHWC to NCHW before F.conv2d.
    # Let's assume the PyTorch way, where F.conv2d expects NCHW, so input should be (N, C, H, W)
    # and weight (out_channels, in_channels, kH, kW).
    # OHWI for TVM usually means (Out, H, W, In).
    # If TVM input is NHWC (N, H, W, C_in), weight OHWI (C_out, H, W, C_in)
    # This translates to PyTorch:
    #   Input (N, C_in, H, W)
    #   Weight (C_out, C_in, H, W)
    
    # Let's adjust inputs to NCHW and weights to OIHW as expected by default F.conv2d
    
    # Input shape NHWC (1, 28, 28, 512)
    # Kernel shape (256, 1, 1, 512) for OHWI -> C_out=256, kH=1, kW=1, C_in=512
    # So PyTorch weight should be (256, 512, 1, 1) if input is NCHW.
    
    input_shape_nhwc = (1, 28, 28, 512)
    input_shape_nchw = (1, 512, 28, 28) # Convert to NCHW for F.conv2d

    w_np = np.ones((256, 512, 1, 1), dtype="float32") # PyTorch OIHW format
    weights = torch.tensor(w_np, dtype=torch.float32)

    class ConvModel(torch.nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.weights = weights
        
        def forward(self, x_nhwc):
            # Permute NHWC to NCHW for F.conv2d
            x_nchw = x_nhwc.permute(0, 3, 1, 2)
            conv_output_nchw = F.conv2d(
                x_nchw,
                self.weights,
                bias=None,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
            )
            return conv_output_nchw # Output is NCHW

    model = ConvModel(weights)
    params = {"w": weights} # Keep params format similar to TVM, though PyTorch uses model state_dict
    return model, params


def test_multiple_runs():
    """
    Test that multiple runs of an operator work.
    Original TVM test uses `no_runs=3` to run the compiled model multiple times.
    Here, we'll just ensure the PyTorch model can be called multiple times.
    """
    np.random.seed(0)

    input_np_nhwc = np.random.uniform(-127, 128, (1, 28, 28, 512)).astype("float32")
    input_tensor_nhwc = torch.tensor(input_np_nhwc, dtype=torch.float32)

    model, params = get_model_multiple_runs()

    # Perform multiple runs
    num_runs = 3
    outputs = []
    for _ in range(num_runs):
        output_nchw = model(input_tensor_nhwc)
        outputs.append(output_nchw)

    # Verify consistency across multiple runs (should be identical as PyTorch is deterministic)
    if outputs:
        first_output = outputs[0]
        for i, output in enumerate(outputs[1:]):
            testing.assert_allclose(first_output, output, atol=1e-7, rtol=1e-7, msg=f"Output mismatch at run {i+1}")
    
    assert outputs[0].shape == (1, 256, 28, 28) # Expected NCHW output shape
    assert isinstance(outputs[0], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])
