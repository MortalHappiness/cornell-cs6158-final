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
import numpy as np
import pytest

def test_pytorch_equivalent_computation():
    """This test function demonstrates the PyTorch equivalent of a TVM topi computation.

    It replaces the original MXNet bridge test by directly executing the computation
    in PyTorch and verifying the correctness.
    """
    n = 20
    shape = (n,)

    # Determine the device for tensor operations
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # In TVM, 'x' and 'y' were te.placeholder. In PyTorch, we use actual tensors.
    xx = torch.rand(shape, device=dev, dtype=torch.float32)
    yy = torch.rand(shape, device=dev, dtype=torch.float32)

    # In TVM, 'scale' was a te.var representing a symbolic scalar.
    # The original MXNet bridge test implicitly used a scale of 10.0.
    scale_val = torch.tensor(10.0, device=dev, dtype=torch.float32)

    # Perform the core computation using PyTorch tensor operations.
    # This replaces TVM's 'topi.broadcast_add(x, y)' and 'te.compute(shape, lambda *i: z(*i) * scale)'.
    # For element-wise operations on tensors, PyTorch directly supports operator overloading.
    result_computed_pytorch = (xx + yy) * scale_val

    # Generate the expected output using NumPy for a ground truth comparison.
    # We move tensors to CPU and convert to NumPy arrays for compatibility with the original test's
    # verification approach (which used zz.numpy()).
    expected_output_np = (xx.cpu().numpy() + yy.cpu().numpy()) * scale_val.cpu().numpy()

    # Assert that the PyTorch computed result is close to the expected NumPy result.
    # torch.testing.assert_allclose provides robust floating-point comparison.
    torch.testing.assert_allclose(result_computed_pytorch.cpu().numpy(), expected_output_np)

    # The original test involved 'to_mxnet_func' which converted a TVM compiled function
    # to an MXNet symbolic function, and then ran it in the MXNet engine.
    # This bridge-specific functionality is removed as it does not have a direct
    # PyTorch equivalent when the goal is to execute the computation itself.
    # The verification now directly confirms the PyTorch implementation of the computation.

if __name__ == "__main__":
    # If this file is run directly, use pytest to discover and execute the test function.
    pytest.main([__file__])
