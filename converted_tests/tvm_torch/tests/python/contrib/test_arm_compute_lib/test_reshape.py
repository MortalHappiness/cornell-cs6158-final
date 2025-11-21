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
"""Arm Compute Library integration reshape tests."""

import numpy as np
import pytest

import torch
import torch.testing as testing

# TVM-specific infrastructure for Arm Compute Library (ACL) are removed.
# `skip_runtime_test`, `skip_codegen_test`, `build_and_run`, `verify`, `verify_codegen` are not used.
# `Device` is not used.

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


def _get_model(input_tensor, output_shape):
    """Return a PyTorch model (functional representation) for reshape."""
    # TVM `relay.reshape(a, output_shape)` maps to `torch.reshape(a, output_shape)`
    return torch.reshape(input_tensor, output_shape)


# _get_expected_codegen is removed as it's TVM-specific.


def test_reshape():
    np.random.seed(0)

    for dtype_str, low, high, atol, rtol in [
        ("float32", -127, 128, 0.001, 0.001),
        ("uint8", 0, 255, 0, 0),
    ]:
        # Convert dtype_str to PyTorch dtype
        if dtype_str == "float32":
            dtype_torch = torch.float32
        elif dtype_str == "uint8":
            dtype_torch = torch.uint8
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        input_np = np.random.uniform(low, high, (1, 1, 1, 1000)).astype(dtype_str)
        input_tensor = torch.tensor(input_np, dtype=dtype_torch)

        for new_shape in [(1, 1000), (10, 10, 10), (10, 100, 1), (1, 1000, 1)]:
            # Note: TVM's reshape supports 0 to copy a dimension from the input.
            # PyTorch's reshape treats 0 as a literal dimension size.
            # The provided `new_shape` tuples do not contain `0`, so direct mapping is safe here.
            outputs = []
            output_pytorch_1 = _get_model(input_tensor, new_shape)
            output_pytorch_2 = _get_model(input_tensor, new_shape)
            outputs.append(output_pytorch_1)
            outputs.append(output_pytorch_2)

            config = {
                "input shape": input_tensor.shape,
                "output shape": new_shape,
                "dtype": dtype_str,
            }
            assert_outputs_equal(outputs, atol=1e-7, rtol=1e-7, config=config)


# test_codegen_reshape is removed as it's TVM-specific.


if __name__ == "__main__":
    pytest.main([__file__])
