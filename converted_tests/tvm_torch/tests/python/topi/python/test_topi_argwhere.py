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
"""Test for argwhere operator"""
import numpy as np
import pytest

import torch
import torch.testing

# A helper to get available devices for parametrization
def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    # Add other devices if needed for CI, e.g., if torch_xla is installed:
    # if "xla" in os.environ.get("PJRT_DEVICE", "").lower():
    #     devices.append("xla")
    return devices

data_shape = pytest.mark.parametrize(
    "data_shape",
    [
        (1,),
        (100,),
        (1, 1),
        (5, 3),
        (32, 64),
        (128, 65),
        (200, 500),
        (6, 5, 3),
        (1, 1, 1),
        (1, 1, 1, 1),
        (6, 4, 5, 3),
        (1, 1, 1, 1, 1),
        (6, 4, 5, 3, 7),
    ],
)


@data_shape
@pytest.mark.parametrize("device", get_available_devices())
def test_argwhere(data_shape, device):
    # TVM test explicitly sets the input data dtype to 'int32' and
    # creates an output buffer for indices with dtype 'int32'.
    # np.argwhere typically produces int64 indices.
    # To match the TVM test's exact output dtype expectation,
    # we will cast both PyTorch's output and the NumPy reference to int32.

    input_data_dtype = "int32" # Dtype for the condition tensor values
    output_indices_dtype = np.int32 # Dtype expected for the output indices

    # Generate input data as in the TVM test (random choice of small integers)
    np_data = np.random.choice([0, 1, 2, 3], size=data_shape).astype(input_data_dtype)
    
    # Compute reference with NumPy. np.argwhere returns int64 indices by default.
    np_out_ref = np.argwhere(np_data)
    
    # Cast the NumPy reference to the expected output_indices_dtype for accurate comparison
    # with the PyTorch output which will also be cast to int32.
    np_out_ref_casted = np_out_ref.astype(output_indices_dtype)

    # Create PyTorch tensor from NumPy data, move to target device
    torch_data = torch.tensor(np_data, device=device)

    # Define a callable for torch.compile
    def argwhere_fn(data):
        return torch.argwhere(data)

    # Compile the function with TorchInductor
    compiled_argwhere_fn = torch.compile(argwhere_fn)

    # Execute the compiled function
    torch_out = compiled_argwhere_fn(torch_data)

    # Cast the PyTorch output to the expected output_indices_dtype (torch.int32)
    # to match the TVM test's explicit output buffer dtype.
    torch_out = torch_out.to(torch.int32)

    # Convert the casted NumPy reference to a PyTorch tensor on the correct device for comparison
    torch_np_out_ref_casted = torch.from_numpy(np_out_ref_casted).to(device)

    # Compare results. rtol and atol are default for assert_allclose.
    torch.testing.assert_allclose(torch_out, torch_np_out_ref_casted)
