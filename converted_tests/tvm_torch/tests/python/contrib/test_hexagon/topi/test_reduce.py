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
"""Test code for reduce"""
import numpy as np
import pytest
import torch
import torch.testing


# Helper numpy functions, used for computing reference values.
# These functions emulate NumPy's argmax/argmin behavior with keepdims.
# NumPy's `argmax` and `argmin` methods on arrays do not directly support `keepdims=True`
# when `axis=None`, so manual reshaping is needed for consistency with PyTorch.
def _my_npy_argmax(arr, axis, keepdims):
    res = arr.argmax(axis=axis)
    if keepdims:
        if axis is None:
            out_shape = [1 for _ in arr.shape]
            return np.array(res).reshape(out_shape)  # Handle scalar result from flat argmax
        else:
            out_shape = list(arr.shape)
            out_shape[axis] = 1
            return res.reshape(out_shape)  # Reshape array result from axis-specific argmax
    else:
        return res  # No reshape needed if keepdims is False


def _my_npy_argmin(arr, axis, keepdims):
    res = arr.argmin(axis=axis)
    if keepdims:
        if axis is None:
            out_shape = [1 for _ in arr.shape]
            return np.array(res).reshape(out_shape)
        else:
            out_shape = list(arr.shape)
            out_shape[axis] = 1
            return res.reshape(out_shape)
    else:
        return res


class TestReduce:
    """Test reduce class."""

    # Parameterize the test cases using pytest.mark.parametrize
    in_shape, axis, keepdims, reduce_type, dtype = pytest.mark.parametrize(
        "in_shape, axis, keepdims, reduce_type, dtype",
        [
            ((32,), 0, False, "argmax", "float32"),
            ((32, 24, 32, 24), (1, 2, 3), True, "sum", "float32"),
            ((2, 3), None, True, "all", "bool"),
            ((32, 24 * 32 * 24), (1,), False, "max", "float32"),
            ((32, 128, 24), None, True, "sum", "float32"),
            ((32, 128, 24), None, True, "all", "bool"),
            ((32, 24, 32, 24), (0, 2), False, "min", "float32"),
            ((32, 128), 1, True, "argmax", "float32"),
            ((32, 24, 32, 24), 2, False, "argmin", "float32"),
            ((31, 21, 15), None, True, "argmax", "float32"),
            ((31, 21, 15), None, False, "sum", "float32"),
            ((2, 3), None, True, "any", "bool"),
            ((32, 128, 24), None, True, "any", "bool"),
            ((1, 4, 7), 1, True, "any", "bool"),
            ((32, 24, 32, 24), 2, False, "any", "bool"),
        ],
    )

    @pytest.fixture(name="ref_data")
    def _ref_data(self, in_shape, axis, keepdims, reduce_type, dtype):
        """Generate test reference data."""
        if dtype == "bool":
            in_npy = np.random.choice([True, False], size=in_shape)
            in_npy_map = in_npy # For bool, map is identity
        else:
            in_npy = np.random.uniform(-1, 1, size=in_shape).astype(dtype)
            # This intermediate mapping (sqrt(exp(x))) is part of the computation graph
            in_npy_map = np.sqrt(np.exp(in_npy)).astype(dtype)

        if reduce_type == "sum":
            out_npy = in_npy_map.sum(axis=axis, keepdims=keepdims)
        elif reduce_type == "all" and dtype == "bool":
            out_npy = in_npy_map.all(axis=axis, keepdims=keepdims)
        elif reduce_type == "any" and dtype == "bool":
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

        return in_npy, in_npy_map, out_npy

    def test_reduce_map(
        self, ref_data, in_shape, axis, keepdims, reduce_type, dtype
    ):
        """Test reduce map."""
        in_npy, in_npy_map, out_npy = ref_data

        # Convert numpy inputs to PyTorch tensors
        torch_dtype = getattr(torch, dtype) if dtype != "bool" else torch.bool
        a_tensor_torch = torch.tensor(in_npy, dtype=torch_dtype)

        # Apply intermediate operations: `sqrt(exp(a_tensor))`
        a1_tensor_torch = torch.sqrt(torch.exp(a_tensor_torch))

        # Perform the reduction operation based on reduce_type
        if reduce_type == "sum":
            b_tensor_torch = torch.sum(a1_tensor_torch, dim=axis, keepdim=keepdims)
        elif reduce_type == "all":
            b_tensor_torch = torch.all(a_tensor_torch, dim=axis, keepdim=keepdims)
        elif reduce_type == "any":
            b_tensor_torch = torch.any(a_tensor_torch, dim=axis, keepdim=keepdims)
        elif reduce_type == "max":
            # torch.max(input, dim) returns (values, indices), need to get .values
            b_tensor_torch = torch.max(a1_tensor_torch, dim=axis, keepdim=keepdims).values
        elif reduce_type == "min":
            # torch.min(input, dim) returns (values, indices), need to get .values
            b_tensor_torch = torch.min(a1_tensor_torch, dim=axis, keepdim=keepdims).values
        elif reduce_type == "argmax":
            # TVM's argmax/argmin specifies int32 output dtype. PyTorch defaults to long.
            b_tensor_torch = torch.argmax(a1_tensor_torch, dim=axis, keepdim=keepdims).to(torch.int32)
        elif reduce_type == "argmin":
            b_tensor_torch = torch.argmin(a1_tensor_torch, dim=axis, keepdim=keepdims).to(torch.int32)
        else:
            raise NotImplementedError

        # Move PyTorch result to CPU and convert to NumPy for comparison
        out_torch_numpy = b_tensor_torch.cpu().numpy()

        if reduce_type in ["argmax", "argmin"]:
            # TVM test uses a specific method to verify argmax/argmin:
            # it takes the computed indices and uses them to fetch values from the original data,
            # then compares these fetched values to the actual max/min values.
            out_torch_indices = out_torch_numpy

            # Adjust indices shape if keepdims was True, for proper fancy indexing later.
            # PyTorch's `argmax(keepdim=True)` keeps the singleton dimension, but for NumPy
            # fancy indexing, we need to remove it if `axis` is specified.
            processed_indices = out_torch_indices
            if keepdims:
                if axis is not None:
                    # Squeeze the singleton dimension for fancy indexing
                    processed_indices = np.squeeze(out_torch_indices, axis=axis)
                else:
                    # If axis is None and keepdims=True, result is like [[[idx]]] (shape (1,1,1)).
                    # Extract the scalar index value for ravel().
                    processed_indices = out_torch_indices.item()

            if axis is None:
                # For global argmax/argmin, flatten the original map and get value by index
                out_torch_val = in_npy_map.ravel()[processed_indices]
            else:
                # For axis-specific argmax/argmin, reconstruct the full indices for fancy indexing
                other_dims = [d for d in range(in_npy_map.ndim) if d != axis]
                grid_dims_shapes = [in_shape[d] for d in other_dims]
                
                # Use np.indices to create coordinate arrays for all dimensions except 'axis'
                mesh_grids = np.indices(grid_dims_shapes)
                
                # Combine mesh_grids with the argmax/argmin result for fancy indexing
                sel_indices_list = [None] * in_npy_map.ndim
                current_mesh_idx = 0
                for d in range(in_npy_map.ndim):
                    if d == axis:
                        sel_indices_list[d] = processed_indices # The actual argmax/argmin results
                    else:
                        sel_indices_list[d] = mesh_grids[current_mesh_idx]
                        current_mesh_idx += 1
                
                sel_indices = tuple(sel_indices_list)
                out_torch_val = in_npy_map[sel_indices]

            if reduce_type == "argmax":
                torch.testing.assert_allclose(out_torch_val, in_npy_map.max(axis=axis), 1e-3, 1e-3)
            elif reduce_type == "argmin":
                torch.testing.assert_allclose(out_torch_val, in_npy_map.min(axis=axis), 1e-3, 1e-3)
        else:
            # For sum, all, any, max, min, direct comparison of output arrays is sufficient
            # `out_npy` (reference) and `out_torch_numpy` (computed) should match.
            torch.testing.assert_allclose(out_torch_numpy, out_npy, 1e-3, 1e-3)
