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

"""PyTorch / TorchInductor equivalent depth-to-space tests"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The original TVM test had a _get_model that returned a Relay expression.
# For PyTorch, we define a simple nn.Module.
class DepthToSpaceModel(nn.Module):
    def __init__(self, block_size, layout):
        super().__init__()
        self.block_size = block_size
        self.layout = layout

    def forward(self, x):
        # PyTorch's depth_to_space has an 'original_input_format' parameter
        # that directly maps to TVM's 'layout' parameter.
        return F.depth_to_space(x, self.block_size, original_input_format=self.layout)


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 16, 16, 16),
        (1, 64, 32, 16),
    ],
)
def test_depth_to_space(dtype, shape):
    """Compare Depth To Space output with PyTorch native."""
    np.random.seed(0)

    # Map numpy dtype strings to PyTorch dtypes
    if dtype == "uint8":
        torch_dtype = torch.uint8
    elif dtype == "int8":
        torch_dtype = torch.int8
    elif dtype == "int32":
        torch_dtype = torch.int32
    else: # Fallback for other dtypes if they were to appear
        torch_dtype = torch.float32

    # Prepare input tensor from numpy array
    input_np = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
    input_tensor = torch.tensor(input_np, dtype=torch_dtype, device='cpu')

    block_size = 2
    layout = "NHWC"

    # Create and run the PyTorch model
    model = DepthToSpaceModel(block_size, layout)
    expected_output_tensor = model(input_tensor)

    # Verify against a NumPy equivalent implementation for robustness.
    # NumPy does not have a direct depth_to_space, so we implement it manually.
    N, H, W, C = shape
    block = block_size
    
    # Calculate output dimensions
    C_out = C // (block * block)
    H_out = H * block
    W_out = W * block

    # NumPy implementation for NHWC layout (based on PyTorch's F.depth_to_space doc)
    # (N, H, W, C) -> (N, H, W, block, block, C_out)
    reshaped_np = input_np.reshape(N, H, W, block, block, C_out)
    # (N, H, W, block, block, C_out) -> (N, H, block, W, block, C_out)
    # Permute block dimensions to interleave with H and W
    permuted_np = reshaped_np.transpose(0, 1, 3, 2, 4, 5)
    # Reshape to final output shape (N, H*block, W*block, C_out)
    numpy_result = permuted_np.reshape(N, H_out, W_out, C_out)

    # Convert numpy result to torch tensor for comparison
    numpy_result_tensor = torch.tensor(numpy_result, dtype=torch_dtype, device='cpu')

    # Use torch.testing.assert_close for numerical comparison
    torch.testing.assert_close(expected_output_tensor, numpy_result_tensor)


# The failure tests (`test_depth_to_space_failure`) in TVM are specific to Ethos-N backend validation
# (e.g., batch size must be 1, only block size 2, specific dtypes, specific layouts only, etc.).
# PyTorch's F.depth_to_space operator is more flexible and does not impose these constraints,
# meaning these tests would generally pass instead of failing when translated to native PyTorch.
# Per instructions: "If you are NOT confident about a safe or correct rewrite, insert a clear TODO comment".
# Therefore, this test suite is skipped with a TODO.
#
# @pytest.mark.parametrize(
#     "shape,block,dtype,layout,err_msg",
#     [
#         ((2, 16, 16, 16), 2, "uint8", "NHWC", "batch size=2, batch size must = 1"),
#         (
#             (1, 16, 16, 16),
#             2,
#             "int16",
#             "NHWC",
#             "dtype='int16', dtype must be either uint8, int8 or int32;",
#         ),
#         ((1, 16, 16, 16), 4, "uint8", "NHWC", "Only block size of 2 is supported"),
#         ((1, 16, 16, 16), 2, "uint8", "NCHW", "Input layer must be NHWC or NHWCB"),
#     ],
# )
# def test_depth_to_space_failure(shape, block, dtype, layout, err_msg):
#     """Check Depth To Space error messages - NOT APPLICABLE FOR PYTORCH NATIVE OPS."""
#     # TODO: This test checks backend-specific constraints (Ethos-N).
#     # PyTorch's `F.depth_to_space` does not have these constraints and would not raise these errors.
#     # It would require simulating a backend with similar constraints, which is out of scope.
#     # For instance, PyTorch's `F.depth_to_space` supports batch_size > 1, block_size > 2, and various dtypes.
#     pytest.skip(f"Backend-specific failure test for Ethos-N is not directly translatable to PyTorch native operator behavior for error: {err_msg}")
