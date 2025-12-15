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
import pytest
import numpy as np

# Helper function to convert string dtype to torch.dtype
def _to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float332
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    # Add other dtypes as needed
    raise ValueError(f"Unsupported dtype: {dtype_str}")


# For ROCm, PyTorch uses the CUDA backend. We'll check for CUDA availability.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA (ROCm backend)")
def test_matmul():
    n = 1024
    l = 128
    m = 235
    dtype = "float32"

    # Define the eager PyTorch operation
    def matmul_op(A_tensor, B_tensor):
        return torch.matmul(A_tensor, B_tensor)

    # Compile the operation using TorchInductor
    compiled_matmul_op = torch.compile(matmul_op, backend="inductor")

    dev = torch.device("cuda:0") # Assuming cuda:0 for ROCm

    # Prepare input NumPy arrays
    np_a = np.random.uniform(size=(n, l)).astype(dtype)
    np_b = np.random.uniform(size=(l, m)).astype(dtype)

    # Convert NumPy arrays to PyTorch tensors on the device
    A_torch = torch.tensor(np_a, dtype=_to_torch_dtype(dtype), device=dev)
    B_torch = torch.tensor(np_b, dtype=_to_torch_dtype(dtype), device=dev)

    # Execute the compiled function
    C_torch = compiled_matmul_op(A_torch, B_torch)

    # Get NumPy reference result
    np_c_ref = np.dot(np_a, np_b)

    # Verify results
    torch.testing.assert_allclose(C_torch.cpu().numpy(), np_c_ref, rtol=1e-5)


# Helper for NumPy reference batch matmul
def _get_numpy_batch_matmul_ref(np_a_original, np_b_original, transa, transb):
    # np.matmul requires (..., M, K) @ (..., K, N)
    # Original A shape: (batch, k, m) if transa else (batch, m, k)
    # Original B shape: (batch, n, k) if transb else (batch, k, n)

    np_a_processed = np_a_original
    np_b_processed = np_b_original

    # If transa is True, original `np_a_original` is (batch, k, m).
    # To conform to (batch, M, K) for matmul, it needs to be (batch, m, k).
    # So, transpose the last two dimensions.
    if transa:
        np_a_processed = np.transpose(np_a_original, (0, 2, 1))

    # If transb is True, original `np_b_original` is (batch, n, k).
    # To conform to (batch, K, N) for matmul, it needs to be (batch, k, n).
    # So, transpose the last two dimensions.
    if transb:
        np_b_processed = np.transpose(np_b_original, (0, 2, 1))

    return np.matmul(np_a_processed, np_b_processed)


def verify_batch_matmul(batch, m, k, n, transa=False, transb=False, dtype="float32"):
    # Determine input shapes based on transposition flags
    ashape = (batch, k, m) if transa else (batch, m, k)
    bshape = (batch, n, k) if transb else (batch, k, n)

    # Define the eager PyTorch batch matmul operation
    def batch_matmul_op(A_tensor, B_tensor):
        # Apply transpositions to A and B tensors if specified
        # PyTorch's .transpose(-2, -1) transposes the last two dimensions.
        A_proc = A_tensor.transpose(-2, -1) if transa else A_tensor
        B_proc = B_tensor.transpose(-2, -1) if transb else B_tensor
        return torch.matmul(A_proc, B_proc)

    # Compile the operation using TorchInductor
    compiled_batch_matmul_op = torch.compile(batch_matmul_op, backend="inductor")

    # This check is duplicated with the decorator on `test_batch_matmul`,
    # but kept here to maintain structure if `verify_batch_matmul` were called independently.
    if not torch.cuda.is_available():
        pytest.skip("skip because CUDA (ROCm backend) is not enabled...")
        return

    dev = torch.device("cuda:0")

    # Prepare input NumPy arrays with their initial shapes
    np_a = np.random.uniform(size=ashape).astype(dtype)
    np_b = np.random.uniform(size=bshape).astype(dtype)

    # Convert NumPy arrays to PyTorch tensors on the device
    A_torch = torch.tensor(np_a, dtype=_to_torch_dtype(dtype), device=dev)
    B_torch = torch.tensor(np_b, dtype=_to_torch_dtype(dtype), device=dev)

    # Execute the compiled function
    C_torch = compiled_batch_matmul_op(A_torch, B_torch)

    # Get NumPy reference result, applying transpositions before actual np.matmul
    np_c_ref = _get_numpy_batch_matmul_ref(np_a, np_b, transa, transb)

    # Verify results
    torch.testing.assert_allclose(C_torch.cpu().numpy(), np_c_ref, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA (ROCm backend)")
def test_batch_matmul():
    # Calling the helper function for various batch matmul configurations
    verify_batch_matmul(128, 64, 512, 512, transa=False, transb=False)
    verify_batch_matmul(128, 64, 512, 512, transa=False, transb=True)
    verify_batch_matmul(128, 64, 512, 512, transa=True, transb=False)
    verify_batch_matmul(128, 64, 512, 512, transa=True, transb=True)
    verify_batch_matmul(128, 512, 512, 64, transa=False, transb=False)
    verify_batch_matmul(128, 512, 512, 64, transa=False, transb=True)
    verify_batch_matmul(128, 512, 512, 64, transa=True, transb=False)
    verify_batch_matmul(128, 512, 512, 64, transa=True, transb=True)
    verify_batch_matmul(128, 512, 64, 512, transa=False, transb=False)
    verify_batch_matmul(128, 512, 64, 512, transa=False, transb=True)
    verify_batch_matmul(128, 512, 64, 512, transa=True, transb=False)
    verify_batch_matmul(128, 512, 64, 512, transa=True, transb=True)
    verify_batch_matmul(128, 64, 128, 128, transa=False, transb=False)
    verify_batch_matmul(128, 64, 128, 128, transa=False, transb=True)
    verify_batch_matmul(128, 64, 128, 128, transa=True, transb=False)
    verify_batch_matmul(128, 64, 128, 128, transa=True, transb=True)
    verify_batch_matmul(128, 128, 128, 64, transa=False, transb=False)
    verify_batch_matmul(128, 128, 128, 64, transa=False, transb=True)
    verify_batch_matmul(128, 128, 128, 64, transa=True, transb=False)
    verify_batch_matmul(128, 128, 128, 64, transa=True, transb=True)


if __name__ == "__main__":
    pytest.main([__file__])
