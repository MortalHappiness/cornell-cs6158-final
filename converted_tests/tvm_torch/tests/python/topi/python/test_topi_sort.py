import pytest
import numpy as np
import torch
import torch.testing

# Map TVM string dtypes to torch dtypes
_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "int64": torch.int64,
}

# The original TVM tests used `tvm.testing.parameter` which maps to pytest.mark.parametrize.
# The `target` parameter in TVM dictates the backend (generic, gpu).
# In PyTorch, the device (cpu, cuda) serves a similar purpose for tensor placement and operation dispatch.
# We replace `target` and `dev` with a single `device` fixture for PyTorch.

# Fixture to provide device for PyTorch tensors
@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return torch.device(request.param)

@pytest.mark.parametrize("axis", [0, -1, 1])
@pytest.mark.parametrize("is_ascend", [True, False], ids=["is_ascend", "not_ascend"])
def test_sort(device, axis, is_ascend):
    np.random.seed(0)

    dshape = (20, 100)
    data_dtype = "float32" # Original TVM test hardcoded data_dtype here

    perm = np.arange(dshape[0] * dshape[1], dtype=data_dtype)
    np.random.shuffle(perm)
    np_data = perm.reshape(dshape)

    if is_ascend:
        np_sort = np.sort(np_data, axis=axis)
    else:
        np_sort = -np.sort(-np_data, axis=axis)
    
    # The original TVM test had redundant slicing:
    # if axis == 0: np_sort = np_sort[: dshape[axis], :]
    # else: np_sort = np_sort[:, : dshape[axis]]
    # This slicing is a no-op as dshape[axis] is the full dimension size.
    # We omit it here for simplicity as it doesn't change the NumPy reference.

    torch_data = torch.tensor(np_data, device=device, dtype=_TORCH_DTYPE_MAP[data_dtype])
    
    # torch.sort returns (values, indices). TVM's topi.sort returns only values.
    torch_out_values, _ = torch.sort(torch_data, dim=axis, descending=not is_ascend)

    torch.testing.assert_allclose(torch_out_values, torch.from_numpy(np_sort).to(device), rtol=1e0)


@pytest.mark.parametrize("axis", [0, -1, 1])
@pytest.mark.parametrize("is_ascend", [True, False], ids=["is_ascend", "not_ascend"])
def test_argsort(device, axis, is_ascend):
    np.random.seed(0)

    dshape = (20, 100)
    data_dtype = "float32" # Original TVM test hardcoded data_dtype here

    perm = np.arange(dshape[0] * dshape[1], dtype=data_dtype)
    np.random.shuffle(perm)
    np_data = perm.reshape(dshape)

    if is_ascend:
        np_indices = np.argsort(np_data, axis=axis)
    else:
        np_indices = np.argsort(-np_data, axis=axis)

    # The original TVM test had redundant slicing (similar to test_sort)
    # and then cast np_indices to data_dtype ("float32") for comparison.
    # We keep this behavior for fidelity.
    
    torch_data = torch.tensor(np_data, device=device, dtype=_TORCH_DTYPE_MAP[data_dtype])
    
    # torch.argsort returns indices as torch.long by default.
    # The original TVM test compares to float32, so we convert PyTorch output and NumPy reference.
    torch_out_indices = torch.argsort(torch_data, dim=axis, descending=not is_ascend)

    # Convert NumPy reference to the expected dtype for comparison
    np_indices_casted = np_indices.astype(data_dtype)

    torch.testing.assert_allclose(
        torch_out_indices.to(_TORCH_DTYPE_MAP[data_dtype]), # Cast PyTorch long indices to float32
        torch.from_numpy(np_indices_casted).to(device),
        rtol=1e0,
    )


@pytest.mark.parametrize("topk_val", [0, 1, 5], ids=lambda x: f"k_{x}")
@pytest.mark.parametrize("axis", [0, -1, 1])
@pytest.mark.parametrize("topk_ret_type", ["values", "indices", "both"])
@pytest.mark.parametrize("is_ascend", [True, False], ids=["is_ascend", "not_ascend"])
@pytest.mark.parametrize("out_dtype_param", ["int64", "float32"], ids=lambda x: f"dtype_{x}") # Renamed to avoid collision with builtin 'dtype'
def test_topk(device, topk_val, axis, topk_ret_type, is_ascend, out_dtype_param):
    np.random.seed(0)

    shape = (20, 100)
    data_dtype = "float32"
    np_data = np.random.uniform(size=shape).astype(data_dtype)

    # Calculate numpy reference values and indices
    if is_ascend:
        np_indices_full = np.argsort(np_data, axis=axis)
    else:
        np_indices_full = np.argsort(-np_data, axis=axis)
    
    kk = topk_val if topk_val >= 1 else shape[axis]

    if axis == 0:
        np_indices = np_indices_full[:kk, :]
        np_values = np.zeros(np_indices.shape, dtype=data_dtype)
        for i in range(shape[1]):
            np_values[:, i] = np_data[np_indices[:, i], i]
    else:
        np_indices = np_indices_full[:, :kk]
        np_values = np.zeros(np_indices.shape, dtype=data_dtype)
        for i in range(shape[0]):
            np_values[i, :] = np_data[i, np_indices[i, :]]

    # Cast NumPy indices to the specified output dtype for comparison
    np_indices = np_indices.astype(out_dtype_param)

    # PyTorch computation
    torch_data = torch.tensor(np_data, device=device, dtype=_TORCH_DTYPE_MAP[data_dtype])
    
    # k for PyTorch's topk needs to be positive
    k_pytorch = topk_val if topk_val >= 1 else shape[axis]
    
    torch_values, torch_indices = torch.topk(
        torch_data, k=k_pytorch, dim=axis, largest=not is_ascend, sorted=True
    )
    
    # Convert PyTorch indices to the specified output dtype for comparison
    torch_indices = torch_indices.to(_TORCH_DTYPE_MAP[out_dtype_param])

    # Convert NumPy references to tensors and move to device for comparison
    np_values_torch = torch.from_numpy(np_values).to(device)
    np_indices_torch = torch.from_numpy(np_indices).to(device)

    # Assertions based on ret_type
    if topk_ret_type == "both":
        torch.testing.assert_allclose(torch_values, np_values_torch, rtol=1e0)
        torch.testing.assert_allclose(torch_indices, np_indices_torch, rtol=1e0)
    elif topk_ret_type == "values":
        torch.testing.assert_allclose(torch_values, np_values_torch, rtol=1e0)
    else:  # topk_ret_type == "indices"
        torch.testing.assert_allclose(torch_indices, np_indices_torch, rtol=1e0)
