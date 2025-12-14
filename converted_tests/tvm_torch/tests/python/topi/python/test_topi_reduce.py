import os
import sys

import numpy as np
import pytest
import torch
import torch.testing

# Mapping of TVM/NumPy dtypes to PyTorch dtypes
TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,  # For TVM's specific argmax/argmin output dtype
    "int64": torch.int64,  # PyTorch default for argmax/argmin
    "bool": torch.bool,
}


# `tvm.testing.parameters` maps to `pytest.mark.parametrize`
# The order of parameters is (in_shape, axis, keepdims, reduce_type, dtype)
@pytest.mark.parametrize(
    "in_shape, axis, keepdims, reduce_type, dtype",
    [
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
    ],
)
class TestReduce:
    def _my_npy_argmax(self, arr, axis, keepdims):
        if not keepdims:
            return arr.argmax(axis=axis)
        else:
            if axis is None:
                out_shape = (1,) * arr.ndim
            else:
                out_shape = list(arr.shape)
                if isinstance(axis, int):
                    out_shape[axis] = 1
                else:  # axis is a tuple
                    for ax in axis:
                        out_shape[ax] = 1
            return arr.argmax(axis=axis).reshape(out_shape)

    def _my_npy_argmin(self, arr, axis, keepdims):
        if not keepdims:
            return arr.argmin(axis=axis)
        else:
            if axis is None:
                out_shape = (1,) * arr.ndim
            else:
                out_shape = list(arr.shape)
                if isinstance(axis, int):
                    out_shape[axis] = 1
                else:  # axis is a tuple
                    for ax in axis:
                        out_shape[ax] = 1
            return arr.argmin(axis=axis).reshape(out_shape)

    @pytest.fixture(name="device")
    def device_fixture(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture(name="ref_data")
    def ref_data_fixture(self, in_shape, axis, keepdims, reduce_type, dtype):
        if dtype == "bool":
            in_npy_map = in_npy = np.random.choice([True, False], size=in_shape)
        else:
            in_npy = np.random.uniform(-1, 1, size=in_shape).astype(dtype)
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
            # For argmax/argmin, the axis is always an int or None in the parameters.
            out_npy = self._my_npy_argmax(in_npy_map, axis=axis, keepdims=keepdims)
        elif reduce_type == "argmin":
            # For argmax/argmin, the axis is always an int or None in the parameters.
            out_npy = self._my_npy_argmin(in_npy_map, axis=axis, keepdims=keepdims)
        else:
            raise NotImplementedError

        return in_npy, in_npy_map, out_npy

    def test_reduce_map(self, device, ref_data, in_shape, axis, keepdims, reduce_type, dtype):
        # The `target` and `dev` fixtures from TVM are replaced by `device`.
        # The Vulkan xfail logic is TVM-specific and removed.
        in_npy, in_npy_map, out_npy_ref = ref_data

        input_tensor_dtype = TORCH_DTYPE_MAP[dtype]
        input_tensor = torch.tensor(in_npy, device=device, dtype=input_tensor_dtype)
        in_npy_map_tensor = torch.tensor(in_npy_map, device=device, dtype=input_tensor_dtype)

        # PyTorch equivalent of A = te.placeholder(...) and A1 = topi.sqrt(topi.exp(A))
        if dtype == "bool":
            A1_tensor = input_tensor
        else:
            A1_tensor = torch.sqrt(torch.exp(input_tensor))

        actual_result_tensor = None
        # PyTorch ops for various reduction types
        if reduce_type == "sum":
            actual_result_tensor = torch.sum(A1_tensor, dim=axis, keepdim=keepdims)
        elif reduce_type == "all":
            actual_result_tensor = torch.all(input_tensor, dim=axis, keepdim=keepdims)
        elif reduce_type == "any":
            actual_result_tensor = torch.any(input_tensor, dim=axis, keepdim=keepdims)
        elif reduce_type == "max":
            actual_result_tensor = torch.max(A1_tensor, dim=axis, keepdim=keepdims).values
        elif reduce_type == "min":
            actual_result_tensor = torch.min(A1_tensor, dim=axis, keepdim=keepdims).values
        elif reduce_type == "argmax":
            actual_result_tensor = torch.argmax(A1_tensor, dim=axis, keepdim=keepdims)
            # TVM explicit out_dtype="int32" for argmax/argmin. PyTorch defaults to int64 (long).
            actual_result_tensor = actual_result_tensor.to(torch.int32)
        elif reduce_type == "argmin":
            actual_result_tensor = torch.argmin(A1_tensor, dim=axis, keepdim=keepdims)
            # TVM explicit out_dtype="int32" for argmax/argmin. PyTorch defaults to int64 (long).
            actual_result_tensor = actual_result_tensor.to(torch.int32)
        else:
            raise NotImplementedError

        # Convert PyTorch result to numpy for comparison
        actual_result_npy = actual_result_tensor.cpu().numpy()

        if reduce_type in ["argmax", "argmin"]:
            # First, verify that the computed indices match the reference NumPy indices.
            torch.testing.assert_close(actual_result_npy, out_npy_ref, rtol=1e-3, atol=1e-3)

            # Second, replicate TVM's specific check: verify that the indices point to the actual max/min values.
            # `argmax_indices_for_val_check_tensor` will have `keepdims=False` shape or be a scalar.
            argmax_indices_for_val_check_tensor = actual_result_tensor.squeeze()

            out_val_from_indices_tensor = None
            if axis is None:
                # `argmax_indices_for_val_check_tensor` is a scalar (0-dim tensor) here.
                # Use .item() to get the Python scalar index for indexing a flattened tensor.
                out_val_from_indices_tensor = in_npy_map_tensor.flatten()[
                    argmax_indices_for_val_check_tensor.item()
                ]
            else:
                # For `axis` being an integer.
                # Construct an index tuple for advanced indexing on `in_npy_map_tensor`.
                # `argmax_indices_for_val_check_tensor` has `keepdims=False` shape (e.g., (32,)).

                if any(s == 0 for s in in_shape):
                    # Handle empty tensors, indexing often results in empty tensors.
                    out_val_from_indices_tensor = torch.empty(
                        (0,), device=device, dtype=in_npy_map_tensor.dtype
                    )
                else:
                    index_tuple = []
                    for dim_idx in range(in_npy_map_tensor.ndim):
                        if dim_idx == axis:
                            # This is the dimension where we use the computed argmax/argmin indices.
                            index_tuple.append(argmax_indices_for_val_check_tensor)
                        else:
                            # For other dimensions, create a range of indices.
                            index_tuple.append(
                                torch.arange(
                                    in_npy_map_tensor.shape[dim_idx], device=device, dtype=torch.long
                                )
                            )
                    # Perform advanced indexing
                    out_val_from_indices_tensor = in_npy_map_tensor[tuple(index_tuple)]

            # Determine the expected max/min value (NumPy reference) for this check.
            # These are computed *without* keepdims in the reference for comparison.
            expected_actual_val_ref = None
            if reduce_type == "argmax":
                expected_actual_val_ref = in_npy_map.max(axis=axis, keepdims=False)
            else:  # argmin
                expected_actual_val_ref = in_npy_map.min(axis=axis, keepdims=False)

            # Compare the extracted values with the reference max/min values.
            if out_val_from_indices_tensor is not None:
                torch.testing.assert_close(
                    out_val_from_indices_tensor.cpu(),
                    torch.tensor(expected_actual_val_ref, dtype=input_tensor_dtype).cpu(),
                    rtol=1e-3,
                    atol=1e-3,
                )

        else:
            # For sum, all, any, max, min, direct comparison of result tensor with out_npy_ref
            torch.testing.assert_close(actual_result_npy, out_npy_ref, rtol=1e-3, atol=1e-3)

    def test_complex_reduce(self, device):
        in_shape = (2, 3)
        dtype = "float32"
        axis = 0
        keepdims = False

        in_npy = np.random.uniform(-1, 1, size=in_shape).astype(dtype)
        input_tensor = torch.tensor(in_npy, device=device, dtype=TORCH_DTYPE_MAP[dtype])

        # PyTorch equivalents of TVM TE operations:
        # B = topi.sum(A, axis=axis, keepdims=keepdims)
        B_tensor = torch.sum(input_tensor, dim=axis, keepdim=keepdims)

        # C = topi.add(B, B)
        C_tensor = B_tensor + B_tensor  # Using operator for simplicity

        # D = topi.multiply(B, B)
        D_tensor = B_tensor * B_tensor  # Using operator for simplicity

        # E = topi.add(C, D)
        E_tensor = C_tensor + D_tensor

        # Reference NumPy computation
        sum_npy = in_npy.sum(axis=axis, keepdims=keepdims)
        out_npy = sum_npy * 2 + sum_npy * sum_npy

        # Convert PyTorch result to numpy for comparison
        out_pytorch_npy = E_tensor.cpu().numpy()
        torch.testing.assert_close(out_pytorch_npy, out_npy, rtol=1e-3, atol=1e-3)


# Main guard for pytest execution (optional for pytest, but common practice)
if __name__ == "__main__":
    pytest.main([__file__])
