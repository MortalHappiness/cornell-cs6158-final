import pytest
import numpy as np
import torch

# Dtype mapping helper
DTYPE_MAP = {
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int8": torch.int8,
}
NUMPY_DTYPE_MAP = {
    "float32": np.float32,
    "uint8": np.uint8,
    "int8": np.int8,
}


def _get_model(axis):
    """Return a model (functional operation) for concatenation."""
    def model_fn(a, b, c):
        return torch.cat([a, b, c], dim=axis)
    return model_fn

# TODO: The original TVM test included _get_expected_codegen and test_codegen_concatenate
# which are highly TVM-specific for inspecting its generated IR/code and cannot be directly converted
# to a functional PyTorch test. These functions and their calls have been omitted.

@pytest.mark.parametrize(
    "input_shape_a, input_shape_b, input_shape_c, axis, dtype",
    [
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], 0, "float32"),
        ([1, 1, 234, 256], [1, 2, 234, 256], [1, 3, 234, 256], 1, "float32"),
        ([1, 234, 234, 1], [1, 234, 234, 2], [1, 234, 234, 3], -1, "float32"),
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], -4, "float32"),
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], 0, "uint8"),
        ([1, 1, 234, 256], [1, 2, 234, 256], [1, 3, 234, 256], 1, "uint8"),
        ([1, 234, 234, 1], [1, 234, 234, 2], [1, 234, 234, 3], -1, "uint8"),
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], -4, "uint8"),
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], 0, "int8"),
        ([1, 1, 234, 256], [1, 2, 234, 256], [1, 3, 234, 256], 1, "int8"),
        ([1, 234, 234, 1], [1, 234, 234, 2], [1, 234, 234, 3], -1, "int8"),
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], -4, "int8"),
    ],
)
def test_concatenate(input_shape_a, input_shape_b, input_shape_c, axis, dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)

    torch_dtype = DTYPE_MAP[dtype]
    numpy_dtype = NUMPY_DTYPE_MAP[dtype]

    # For integer types, use randint to avoid float conversion issues and ensure integer range.
    # For float types, use randn.
    def _generate_random_np_array(shape, np_dtype):
        if np_dtype in [np.uint8, np.int8]:
            low = np.iinfo(np_dtype).min
            high = np.iinfo(np_dtype).max + 1
            return np.random.randint(low=low, high=high, size=shape, dtype=np_dtype)
        else: # float32
            return np.random.randn(*shape).astype(np_dtype)

    a_np = _generate_random_np_array(input_shape_a, numpy_dtype)
    b_np = _generate_random_np_array(input_shape_b, numpy_dtype)
    c_np = _generate_random_np_array(input_shape_c, numpy_dtype)

    a_torch = torch.tensor(a_np, dtype=torch_dtype, device=device)
    b_torch = torch.tensor(b_np, dtype=torch_dtype, device=device)
    c_torch = torch.tensor(c_np, dtype=torch_dtype, device=device)

    # Get the PyTorch model (functional operation)
    model_fn = _get_model(axis)

    # Execute the PyTorch model
    pytorch_output = model_fn(a_torch, b_torch, c_torch)

    # Calculate expected output using NumPy
    numpy_expected_output = np.concatenate([a_np, b_np, c_np], axis=axis)

    # Determine tolerance based on dtype
    if dtype == "float32":
        atol_val = 1e-7
        rtol_val = 1e-7
    else: # uint8, int8
        atol_val = 0
        rtol_val = 0

    # Verify results
    torch.testing.assert_close(pytorch_output.cpu(), numpy_expected_output, rtol=rtol_val, atol=atol_val)
