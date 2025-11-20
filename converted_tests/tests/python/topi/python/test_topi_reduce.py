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
"""Test code for reduce."""
import os
import sys

import numpy as np
import pytest

import torch
import torch.testing as testing
import torch.nn.functional as F

# TVM-specific imports and infrastructure are removed.
# `tvm`, `te`, `topi`, `tvm.testing`, `tvm.topi.testing` are not used.

# Define the parameterized test cases similar to TVM's `tvm.testing.parameters`
_reduce_test_cases = [
    # in_shape, axis, keepdims, reduce_type, dtype
    ((32,), 0, False, "argmax", "float32"),
    ((128, 24, 128, 24), (1, 2, 3), True, "sum", "float32"),
    ((2, 3), None, True, "all", "bool"),
    ((128, 24 * 128 * 24), (1,), False, "max", "float32"),
    ((32, 128, 24), None, True, "sum", "float32"),
    ((32, 128, 24), None, True, "all", "bool"),
    ((128, 24, 128, 24), (0, 2), False, "min", "float32"),
    ((32, 128), 1, True, "argmax", "float32"),
    ((32, 24, 32, 24), 2, False, "argmin", "float32"),
    ((31, 21, 15), None, True, "argmax", "float32"),
    ((31, 21, 15), None, False, "sum", "float32"),
    ((128, 24, 128, 24), (1, 2, 3), True, "sum", "float64"),
    ((2, 3), None, True, "any", "bool"),
    ((32, 128, 24), None, True, "any", "bool"),
    ((1, 4, 7), 1, True, "any", "bool"),
    ((128, 24, 128, 24), 2, False, "any", "bool"),
]


def _my_npy_argmax(arr, axis, keepdims):
    if not keepdims:
        return arr.argmax(axis=axis)
    else:
        if axis is None:
            out_shape = [1 for _ in arr.shape]
        else:
            out_shape = list(arr.shape)
            out_shape[axis] = 1

        return arr.argmax(axis=axis).reshape(out_shape)


def _my_npy_argmin(arr, axis, keepdims):
    if not keepdims:
        return arr.argmin(axis=axis)
    else:
        if axis is None:
            out_shape = [1 for _ in arr.shape]
        else:
            out_shape = list(arr.shape)
            out_shape[axis] = 1
        return arr.argmin(axis=axis).reshape(out_shape)


# Fixture to generate reference data, adapted from TVM's `tvm.testing.fixture`
@pytest.fixture(params=_reduce_test_cases, ids=lambda x: f"shape={x[0]}_axis={x[1]}_keepdims={x[2]}_type={x[3]}_dtype={x[4]}")
def ref_data(request):
    in_shape, axis, keepdims, reduce_type, dtype_str = request.param

    # Convert dtype_str to numpy dtype
    if dtype_str == "bool":
        npy_dtype = np.bool_
    elif dtype_str == "float32":
        npy_dtype = np.float32
    elif dtype_str == "float64":
        npy_dtype = np.float64
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype_str}")

    # Generate input data
    if npy_dtype == np.bool_:
        in_npy = np.random.choice([True, False], size=in_shape)
    else:
        in_npy = np.random.uniform(-1, 1, size=in_shape).astype(npy_dtype)
    
    in_npy_map = np.sqrt(np.exp(in_npy)).astype(npy_dtype) if npy_dtype != np.bool_ else in_npy

    # Compute reference output using NumPy
    if reduce_type == "sum":
        out_npy = in_npy_map.sum(axis=axis, keepdims=keepdims)
    elif reduce_type == "all" and npy_dtype == np.bool_:
        out_npy = in_npy_map.all(axis=axis, keepdims=keepdims)
    elif reduce_type == "any" and npy_dtype == np.bool_:
        out_npy = in_npy_map.any(axis=axis, keepdims=keepdims)
    elif reduce_type == "max":
        out_npy = in_npy_map.max(axis=axis, keepdims=keepdims)
    elif reduce_type == "min":
        out_npy = in_npy_map.min(axis=axis, keepdims=keepdims)
    elif reduce_type == "argmax":
        out_npy = _my_npy_argmax(in_npy_map, axis=axis, keepdims=keepdims)
    elif reduce_type == "argmin":
        out_npy = _my_npy_argmin(in_npy_map, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError

    return in_npy, in_npy_map, out_npy, in_shape, axis, keepdims, reduce_type, dtype_str


def test_reduce_map(ref_data):
    in_npy, in_npy_map, out_npy, in_shape, axis, keepdims, reduce_type, dtype_str = ref_data

    # Convert string dtype to torch dtype
    if dtype_str == "float32":
        dtype_torch = torch.float32
    elif dtype_str == "float64":
        dtype_torch = torch.float64
    elif dtype_str == "bool":
        dtype_torch = torch.bool
    elif dtype_str == "int32": # Argmax/argmin output dtype
        dtype_torch = torch.int32
    else:
        raise NotImplementedError(f"Unsupported dtype_str: {dtype_str}")

    data_torch = torch.tensor(in_npy, dtype=dtype_torch)
    
    # Apply operations based on original TVM logic
    if dtype_torch != torch.bool:
        # topi.sqrt(topi.exp(A))
        a1_tensor = torch.sqrt(torch.exp(data_torch))
    else:
        # For boolean inputs, these operations are not meaningful, use original data.
        a1_tensor = data_torch

    # Map TVM reduce_type to PyTorch equivalent
    if reduce_type == "sum":
        b_tensor = torch.sum(a1_tensor, dim=axis, keepdim=keepdims)
    elif reduce_type == "all":
        b_tensor = torch.all(data_torch, dim=axis, keepdim=keepdims)
    elif reduce_type == "any":
        b_tensor = torch.any(data_torch, dim=axis, keepdim=keepdims)
    elif reduce_type == "max":
        b_tensor = torch.max(a1_tensor, dim=axis, keepdim=keepdims).values
    elif reduce_type == "min":
        b_tensor = torch.min(a1_tensor, dim=axis, keepdim=keepdims).values
    elif reduce_type == "argmax":
        b_tensor = torch.argmax(a1_tensor, dim=axis, keepdim=keepdims)
        # PyTorch argmax/argmin returns `torch.int64` by default.
        # Original TVM output dtype was 'int32' so we cast for comparison.
        b_tensor = b_tensor.to(torch.int32)
    elif reduce_type == "argmin":
        b_tensor = torch.argmin(a1_tensor, dim=axis, keepdim=keepdims)
        b_tensor = b_tensor.to(torch.int32)
    else:
        raise NotImplementedError(f"Unsupported reduce_type: {reduce_type}")

    # Compare PyTorch output with NumPy reference
    if reduce_type in ["argmax", "argmin"]:
        # For argmax/argmin, `out_npy` contains the indices.
        # Compare the computed indices directly.
        testing.assert_allclose(b_tensor, torch.tensor(out_npy, dtype=torch.int32), rtol=1e-3, atol=1e-3)

        # Additional check to ensure indices correctly point to max/min values (similar to TVM's internal check)
        out_tvm_indices = b_tensor.numpy()
        if keepdims:
            out_tvm_indices_flat = np.take(out_tvm_indices, indices=0, axis=axis)
        else:
            out_tvm_indices_flat = out_tvm_indices

        if axis is None:
            out_tvm_val = in_npy_map.ravel()[out_tvm_indices_flat]
        else:
            # Construct slices for dimensions other than `axis`
            other_indices = tuple(np.indices(in_shape[0:axis] + in_shape[(axis + 1):]))
            sel_indices = other_indices[0:axis] + (out_tvm_indices_flat,) + other_indices[axis:]
            out_tvm_val = in_npy_map[sel_indices]
        
        if reduce_type == "argmax":
            testing.assert_allclose(torch.from_numpy(out_tvm_val), torch.from_numpy(in_npy_map.max(axis=axis)), 1e-3, 1e-3)
        elif reduce_type == "argmin":
            testing.assert_allclose(torch.from_numpy(out_tvm_val), torch.from_numpy(in_npy_map.min(axis=axis)), 1e-3, 1e-3)
    else:
        testing.assert_allclose(b_tensor, torch.tensor(out_npy, dtype=dtype_torch), rtol=1e-3, atol=1e-3)


def test_complex_reduce():
    in_shape = (2, 3)
    dtype_torch = torch.float32
    axis = 0
    keepdims = False

    # TVM: A = te.placeholder(shape=in_shape, name="A", dtype=dtype)
    # in_npy is the input tensor `A`
    in_npy = np.random.uniform(-1, 1, size=in_shape).astype(np.float32)
    data_torch = torch.tensor(in_npy, dtype=dtype_torch)

    # TVM: B = topi.sum(A, axis=axis, keepdims=keepdims)
    b_tensor = torch.sum(data_torch, dim=axis, keepdim=keepdims)
    
    # TVM: C = topi.add(B, B)
    c_tensor = torch.add(b_tensor, b_tensor)
    
    # TVM: D = topi.multiply(B, B)
    d_tensor = torch.mul(b_tensor, b_tensor)
    
    # TVM: E = topi.add(C, D)
    e_tensor = torch.add(c_tensor, d_tensor)

    # NumPy reference computation
    sum_npy = in_npy.sum(axis=axis, keepdims=keepdims)
    out_npy = sum_npy * 2 + sum_npy * sum_npy

    # Compare PyTorch output with NumPy reference
    testing.assert_allclose(e_tensor, torch.tensor(out_npy, dtype=dtype_torch), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
