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
import torch.testing as testing

# TVM-specific imports and infrastructure for Hexagon are removed.
# `tvm`, `te`, `topi`, `Session`, `get_hexagon_target` are not used.

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


class TestReduce:
    """Test reduce class."""

    in_shape, axis, keepdims, reduce_type, dtype_str = pytest.mark.parametrize(
        "in_shape, axis, keepdims, reduce_type, dtype_str",
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
    )(lambda in_shape, axis, keepdims, reduce_type, dtype_str: (in_shape, axis, keepdims, reduce_type, dtype_str))

    @pytest.fixture(name="ref_data", params=[])
    def ref_data_fixture(self, in_shape, axis, keepdims, reduce_type, dtype_str):
        # Dynamically set params for this fixture
        return self._generate_ref_data(in_shape, axis, keepdims, reduce_type, dtype_str)

    def _generate_ref_data(self, in_shape, axis, keepdims, reduce_type, dtype_str):
        """Generate test reference data."""
        if dtype_str == "bool":
            in_npy = np.random.choice([True, False], size=in_shape)
        else:
            in_npy = np.random.uniform(-1, 1, size=in_shape).astype(dtype_str)
        
        in_npy_map = np.sqrt(np.exp(in_npy)).astype(dtype_str) if dtype_str != "bool" else in_npy

        if reduce_type == "sum":
            out_npy = in_npy_map.sum(axis=axis, keepdims=keepdims)
        elif reduce_type == "all" and dtype_str == "bool":
            out_npy = in_npy_map.all(axis=axis, keepdims=keepdims)
        elif reduce_type == "any" and dtype_str == "bool":
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
            raise NotImplementedError(f"Unsupported reduce_type: {reduce_type}")

        return in_npy, in_npy_map, out_npy

    @pytest.mark.parametrize(
        "in_shape, axis, keepdims, reduce_type, dtype_str",
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
    # @tvm.testing.requires_hexagon # Removed, not relevant for PyTorch
    def test_reduce_map(
        self, in_shape, axis, keepdims, reduce_type, dtype_str
    ):
        """Test reduce map."""
        in_npy, in_npy_map, out_npy = self._generate_ref_data(in_shape, axis, keepdims, reduce_type, dtype_str)

        # Convert string dtype to torch dtype
        if dtype_str == "float32":
            dtype_torch = torch.float32
        elif dtype_str == "bool":
            dtype_torch = torch.bool
        elif dtype_str == "int32":
            dtype_torch = torch.int32
        else:
            raise NotImplementedError(f"Unsupported dtype_str: {dtype_str}")

        data_torch = torch.tensor(in_npy, dtype=dtype_torch)
        
        # Applying mapping operations based on original TVM logic
        if dtype_str != "bool":
            a1_tensor = torch.sqrt(torch.exp(data_torch))
        else:
            a1_tensor = data_torch # for boolean inputs, sqrt(exp(bool)) is not meaningful

        out_dtype_actual = dtype_torch
        if reduce_type == "sum":
            b_tensor = torch.sum(a1_tensor, dim=axis, keepdim=keepdims)
        elif reduce_type == "all":
            b_tensor = torch.all(data_torch, dim=axis, keepdim=keepdims)
        elif reduce_type == "any":
            b_tensor = torch.any(data_torch, dim=axis, keepdim=keepdims)
        elif reduce_type == "max":
            # torch.max returns a tuple (values, indices)
            b_tensor = torch.max(a1_tensor, dim=axis, keepdim=keepdims).values
        elif reduce_type == "min":
            # torch.min returns a tuple (values, indices)
            b_tensor = torch.min(a1_tensor, dim=axis, keepdim=keepdims).values
        elif reduce_type == "argmax":
            b_tensor = torch.argmax(a1_tensor, dim=axis, keepdim=keepdims)
            out_dtype_actual = torch.int64 # PyTorch argmax returns long by default
        elif reduce_type == "argmin":
            b_tensor = torch.argmin(a1_tensor, dim=axis, keepdim=keepdims)
            out_dtype_actual = torch.int64 # PyTorch argmin returns long by default
        else:
            raise NotImplementedError(f"Unsupported reduce_type: {reduce_type}")
        
        out_tensor_pytorch = b_tensor

        if reduce_type in ["argmax", "argmin"]:
            # TVM output dtype was 'int32', PyTorch is 'int64'. Cast for comparison.
            out_tensor_pytorch = out_tensor_pytorch.to(torch.int32)
            out_tvm_indices = out_tensor_pytorch.numpy()
            
            # The following logic is for converting indices back to values
            # for comparison, which is a bit complex for PyTorch direct comparison.
            # We'll compare the indices themselves, and check if argmax/argmin is correct
            # for the original data.

            # Convert expected out_npy (indices) to the correct dtype for comparison
            # For argmax/argmin, `out_npy` is already the indices computed by numpy.
            # Cast `out_npy` to `int32` for direct comparison with `out_tvm_indices`.
            testing.assert_allclose(torch.from_numpy(out_tvm_indices), torch.from_numpy(out_npy).to(torch.int32), rtol=1e-3, atol=1e-3)
            
            # Additional check to ensure indices correctly point to max/min values (similar to TVM's internal check)
            # This logic comes from the TVM test and uses NumPy.
            if keepdims:
                out_tvm_indices_flat = np.take(out_tvm_indices, indices=0, axis=axis)
            else:
                out_tvm_indices_flat = out_tvm_indices

            if axis is None:
                out_tvm_val = in_npy_map.ravel()[out_tvm_indices_flat]
            else:
                # Construct slice objects for dimensions other than `axis`
                other_indices = tuple(np.indices(in_shape[0:axis] + in_shape[(axis + 1):]))
                sel_indices = other_indices[0:axis] + (out_tvm_indices_flat,) + other_indices[axis:]
                out_tvm_val = in_npy_map[sel_indices]
            
            if reduce_type == "argmax":
                testing.assert_allclose(torch.from_numpy(out_tvm_val), torch.from_numpy(in_npy_map.max(axis=axis)).to(dtype_torch), 1e-3, 1e-3)
            elif reduce_type == "argmin":
                testing.assert_allclose(torch.from_numpy(out_tvm_val), torch.from_numpy(in_npy_map.min(axis=axis)).to(dtype_torch), 1e-3, 1e-3)

        else:
            testing.assert_allclose(out_tensor_pytorch, torch.tensor(out_npy, dtype=dtype_torch), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
