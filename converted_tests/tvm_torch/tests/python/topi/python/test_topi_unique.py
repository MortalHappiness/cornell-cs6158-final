import pytest
import torch
import numpy as np

# Helper to convert string dtypes to torch dtypes
def str_to_torch_dtype(dtype_str):
    if dtype_str == "int32":
        return torch.int32
    if dtype_str == "int64":
        return torch.int64
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    # Add other dtypes as needed
    raise ValueError(f"Unsupported dtype string: {dtype_str}")

# Pytest parametrize for various inputs
@pytest.mark.parametrize("in_dtype_str", ["int32", "int64"])
@pytest.mark.parametrize("is_sorted_param", [True, False], ids=["sorted", "unsorted"])
@pytest.mark.parametrize("with_counts", [True, False], ids=["with_counts", "no_counts"])
@pytest.mark.parametrize("arr_size, maxval", [(1, 100), (10, 10), (10000, 100)])
@pytest.mark.parametrize("device", ["cpu", "cuda"], ids=["cpu", "cuda"])
def test_unique(in_dtype_str, is_sorted_param, with_counts, arr_size, maxval, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    dev = torch.device(device)
    torch_dtype = str_to_torch_dtype(in_dtype_str)

    def calc_numpy_unique_ref(data_np, is_sorted_output=False):
        """
        Calculates NumPy unique outputs, matching PyTorch's `sorted` parameter
        (i.e., output unique values can be in sorted order or first-appearance order).
        """
        # np.unique always sorts its unique values.
        # We need return_index, return_inverse, return_counts to match PyTorch's capabilities.
        np_unique_sorted, np_indices_sorted, np_inverse_indices_sorted, np_counts_sorted = np.unique(
            data_np, return_index=True, return_inverse=True, return_counts=True
        )

        # If we need 'first-appearance' order (i.e., not sorted), we reorder.
        if not is_sorted_output:
            # `np_indices_sorted` gives the first occurrence index for each sorted unique value.
            # If we sort `np_unique_sorted` based on these first occurrence indices,
            # we get the "first-appearance" order.
            order = np.argsort(np_indices_sorted) # `order` is permutation indices for `np_unique_sorted`
            
            # Apply `order` to reorder unique values, their first indices, and their counts
            np_unique = np_unique_sorted[order].astype(data_np.dtype)
            np_indices = np_indices_sorted[order].astype(np.int32) # Indices of first occurrence in original array,
                                                                   # now corresponding to the first-appearance ordered unique values.
            np_counts = np_counts_sorted[order].astype(np.int32)

            # `np_inverse_indices_sorted` maps original elements to their position in `np_unique_sorted`.
            # We need `np_inverse` to map original elements to their position in the new `np_unique`
            # (which is in first-appearance order).
            # The `map_sorted_to_first_appearance_pos` maps a position in `np_unique_sorted`
            # to its new position in `np_unique`.
            map_sorted_to_first_appearance_pos = np.argsort(order)
            np_inverse = map_sorted_to_first_appearance_pos[np_inverse_indices_sorted].astype(np.int32)
        else:
            # If output is desired to be sorted, directly use the results from np.unique
            np_unique = np_unique_sorted.astype(data_np.dtype)
            np_indices = np_indices_sorted.astype(np.int32)
            np_inverse = np_inverse_indices_sorted.astype(np.int32)
            np_counts = np_counts_sorted.astype(np.int32)
        
        return np_unique, np_indices, np_inverse, np_counts

    # Generate input data
    data_np = np.random.randint(0, maxval, size=(arr_size,)).astype(in_dtype_str)
    input_tensor = torch.tensor(data_np, device=dev)

    # Compute NumPy reference outputs
    np_unique_ref, np_indices_ref, np_inverse_indices_ref, np_counts_ref = calc_numpy_unique_ref(
        data_np, is_sorted_output=is_sorted_param
    )

    # Call PyTorch's torch.unique.
    # When return_inverse=True, return_counts=True, return_indices=True,
    # PyTorch returns (unique, inverse_indices, counts, indices).
    # When return_counts=False, it returns (unique, inverse_indices, indices).
    pt_outputs = torch.unique(
        input_tensor, 
        sorted=is_sorted_param, 
        return_inverse=True, 
        return_counts=with_counts, 
        return_indices=True
    )
    
    # Unpack PyTorch outputs based on `with_counts` flag
    if with_counts:
        pt_unique, pt_inverse_indices, pt_counts, pt_indices = pt_outputs
    else:
        pt_unique, pt_inverse_indices, pt_indices = pt_outputs
        pt_counts = None # Explicitly set to None for consistent checking later

    # Convert PyTorch tensor outputs to NumPy arrays of consistent dtype for comparison
    pt_unique_np = pt_unique.cpu().numpy()
    pt_indices_np = pt_indices.to(torch.int32).cpu().numpy() # Ensure int32 for comparison
    pt_inverse_indices_np = pt_inverse_indices.to(torch.int32).cpu().numpy() # Ensure int32 for comparison
    if pt_counts is not None:
        pt_counts_np = pt_counts.to(torch.int32).cpu().numpy()
    else:
        pt_counts_np = None

    # Assertions using torch.testing.assert_close for numerical stability
    torch.testing.assert_close(pt_unique_np, np_unique_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pt_indices_np, np_indices_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pt_inverse_indices_np, np_inverse_indices_ref, atol=1e-5, rtol=1e-5)

    if with_counts:
        torch.testing.assert_close(pt_counts_np, np_counts_ref, atol=1e-5, rtol=1e-5)

    # The original TVM test also asserted the number of unique elements
    assert len(pt_unique_np) == len(np_unique_ref)
