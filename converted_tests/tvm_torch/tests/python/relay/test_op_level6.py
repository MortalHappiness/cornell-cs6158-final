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
""" Equivalent PyTorch / TorchInductor test cases for level6 operators.
"""
import pytest
import numpy as np
import torch
from torch.testing import assert_allclose

# Helper to map string dtypes to torch dtypes
_TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sort(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def verify_sort(shape, axis, is_ascend, in_dtype="float32"):
        torch_dtype = _TORCH_DTYPE_MAP[in_dtype]
        x_data_np = np.random.uniform(size=shape).astype(in_dtype)
        x_data = torch.tensor(x_data_np, dtype=torch_dtype, device=device)

        # PyTorch function for comparison
        def pytorch_sort_op(data_tensor):
            return torch.sort(data_tensor, dim=axis, descending=not is_ascend).values

        # Reference numpy result
        if is_ascend:
            ref_res = np.sort(x_data_np, axis=axis)
        else:
            # numpy sort is always ascending; negate data to effectively sort descending
            ref_res = np.sort(-x_data_np, axis=axis)
            ref_res = -ref_res # negate back to get original magnitude, sorted descending

        # Eager execution
        op_res_eager = pytorch_sort_op(x_data)
        assert_allclose(op_res_eager.cpu().numpy(), ref_res, rtol=1e-5, atol=1e-5)

        # TorchInductor execution
        compiled_pytorch_sort_op = torch.compile(pytorch_sort_op, dynamic=True)
        op_res_compiled = compiled_pytorch_sort_op(x_data)
        assert_allclose(op_res_compiled.cpu().numpy(), ref_res, rtol=1e-5, atol=1e-5)

    verify_sort((2, 3, 4), axis=0, is_ascend=False)
    verify_sort((1, 4, 6), axis=1, is_ascend=True)
    verify_sort((3, 5, 6), axis=-1, is_ascend=False)
    verify_sort((3, 2000, 6), axis=1, is_ascend=False)
    verify_sort((1, 122640), axis=1, is_ascend=False)
    # Check float16 (requires GPU for typically meaningful results and for PyTorch's sort on float16)
    if device == "cuda":
        verify_sort((1, 122640), axis=1, is_ascend=False, in_dtype="float16")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_argsort(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def verify_argsort(shape, axis, is_ascend, out_dtype, in_dtype="float32"):
        torch_in_dtype = _TORCH_DTYPE_MAP[in_dtype]
        torch_out_dtype = _TORCH_DTYPE_MAP[out_dtype]

        x_data_np = np.random.uniform(size=shape).astype(in_dtype)
        x_data = torch.tensor(x_data_np, dtype=torch_in_dtype, device=device)

        # PyTorch function for comparison
        def pytorch_argsort_op(data_tensor):
            # torch.argsort returns indices as torch.long by default
            # if out_dtype is int32, we explicitly convert
            result = torch.argsort(data_tensor, dim=axis, descending=not is_ascend)
            return result.to(torch_out_dtype) if torch_out_dtype != torch.long else result

        # Reference numpy result (kind='stable' for consistency)
        if is_ascend:
            ref_res = np.argsort(x_data_np, axis=axis, kind="stable")
        else:
            ref_res = np.argsort(-x_data_np, axis=axis, kind="stable")
        ref_res = ref_res.astype(out_dtype)

        # Eager execution
        op_res_eager = pytorch_argsort_op(x_data)
        assert_allclose(op_res_eager.cpu().numpy(), ref_res, rtol=1e-5, atol=1e-5)

        # TorchInductor execution
        compiled_pytorch_argsort_op = torch.compile(pytorch_argsort_op, dynamic=True)
        op_res_compiled = compiled_pytorch_argsort_op(x_data)
        assert_allclose(op_res_compiled.cpu().numpy(), ref_res, rtol=1e-5, atol=1e-5)

    for out_dtype in ["int32", "int64", "float32", "float64"]:
        # For float32/float64 output, it will convert the long indices to float.
        # This is unusual but supported by TVM and can be matched in PyTorch.
        verify_argsort((2, 3, 4), axis=0, is_ascend=False, out_dtype=out_dtype)
        verify_argsort((1, 4, 6), axis=1, is_ascend=True, out_dtype=out_dtype)

    # Specific dtype and shapes
    out_dtype = "int32"
    verify_argsort((3, 5, 6), axis=-1, is_ascend=False, out_dtype=out_dtype)
    verify_argsort((3, 6000, 6), axis=1, is_ascend=False, out_dtype=out_dtype)
    verify_argsort((1000, 1, 1), axis=0, is_ascend=False, out_dtype=out_dtype)
    verify_argsort((1, 122640), axis=1, is_ascend=False, out_dtype=out_dtype)
    # Check float16 input (requires GPU)
    if device == "cuda":
        verify_argsort((1, 122640), axis=1, is_ascend=False, out_dtype=out_dtype, in_dtype="float16")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("inductor_compiled", [False, True])  # Corresponds to TVM's "graph" vs "vm"
def test_topk(device, inductor_compiled):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def verify_topk(k, axis, ret_type, is_ascend, out_dtype, in_dtype="float32"):
        shape = (20, 100)
        torch_in_dtype = _TORCH_DTYPE_MAP[in_dtype]
        torch_out_dtype = _TORCH_DTYPE_MAP[out_dtype]

        np.random.seed(0) # Seed numpy for consistent results across runs
        np_data = np.random.uniform(size=shape).astype(in_dtype)
        data_tensor = torch.tensor(np_data, dtype=torch_in_dtype, device=device)

        # Reference numpy calculation
        if is_ascend:  # get smallest k elements
            np_indices_full = np.argsort(np_data, axis=axis, kind="stable")
        else:  # get largest k elements
            # Negate data for argsort to find indices of largest values (as argsort is ascending)
            np_indices_full = np.argsort(-np_data, axis=axis, kind="stable")

        actual_k = k if k >= 1 else shape[axis]

        if axis == 0:
            np_indices = np_indices_full[:actual_k, :]
        else:  # axis 1 or -1 (which is 1 for 2D for shape (20,100))
            np_indices = np_indices_full[:, :actual_k]

        np_values = np.take_along_axis(np_data, np_indices, axis=axis)
        np_indices = np_indices.astype(out_dtype)

        def pytorch_topk_op(data_tensor_input):
            return torch.topk(data_tensor_input, k=actual_k, dim=axis, largest=not is_ascend, sorted=True)

        func_to_test = pytorch_topk_op
        if inductor_compiled:
            func_to_test = torch.compile(pytorch_topk_op, dynamic=True)

        op_res_values, op_res_indices = func_to_test(data_tensor)

        # Convert indices to target dtype if not long (torch.topk returns long by default)
        op_res_indices_typed = op_res_indices.to(torch_out_dtype) if torch_out_dtype != torch.long else op_res_indices

        if ret_type == "both":
            assert_allclose(op_res_values.cpu().numpy(), np_values, rtol=1e-5, atol=1e-5)
            assert_allclose(op_res_indices_typed.cpu().numpy(), np_indices, rtol=1e-5, atol=1e-5)
        elif ret_type == "values":
            assert_allclose(op_res_values.cpu().numpy(), np_values, rtol=1e-5, atol=1e-5)
        else:  # ret_type == "indices"
            assert_allclose(op_res_indices_typed.cpu().numpy(), np_indices, rtol=1e-5, atol=1e-5)

    for k in [0, 1, 5]:
        for axis in [0, -1, 1]:
            for ret_type in ["both", "values", "indices"]:
                verify_topk(k, axis, ret_type, True, "int64")
                verify_topk(k, axis, ret_type, False, "float32")
                if device == "cuda": # float16 input requires GPU
                    verify_topk(k, axis, ret_type, False, "int64", "float16")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_searchsorted(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def verify_searchsorted(right, out_dtype):
        shape = (8, 9, 10)
        values_shape = shape[:-1] + (10,)

        sorted_sequence_np = np.sort(np.random.randn(*shape).astype("float32"), axis=-1)
        values_np = np.random.randn(*values_shape).astype("float32")

        sorted_sequence_tensor = torch.tensor(sorted_sequence_np, dtype=torch.float32, device=device)
        values_tensor = torch.tensor(values_np, dtype=torch.float32, device=device)

        # Reference numpy result
        # np.searchsorted defaults to int64-like output.
        # Cast to match TVM's out_dtype (int32 or int64).
        np_indices = np.searchsorted(sorted_sequence_np, values_np, side='right' if right else 'left')
        np_indices = np_indices.astype(out_dtype)

        torch_out_dtype = _TORCH_DTYPE_MAP[out_dtype]
        torch_out_int32 = (torch_out_dtype == torch.int32)

        def pytorch_searchsorted_op(seq, vals):
            return torch.searchsorted(seq, vals, right=right, out_int32=torch_out_int32)

        # Eager execution
        op_res_eager = pytorch_searchsorted_op(sorted_sequence_tensor, values_tensor)
        assert_allclose(op_res_eager.cpu().numpy(), np_indices, rtol=1e-5, atol=1e-5)

        # TorchInductor execution
        compiled_pytorch_searchsorted_op = torch.compile(pytorch_searchsorted_op, dynamic=True)
        op_res_compiled = compiled_pytorch_searchsorted_op(sorted_sequence_tensor, values_tensor)
        assert_allclose(op_res_compiled.cpu().numpy(), np_indices, rtol=1e-5, atol=1e-5)

    verify_searchsorted(False, "int32")
    verify_searchsorted(True, "int64")


if __name__ == "__main__":
    pytest.main([__file__])
