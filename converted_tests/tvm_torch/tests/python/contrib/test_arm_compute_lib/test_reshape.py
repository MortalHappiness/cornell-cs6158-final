import numpy as np
import torch
import pytest

# Helper to map string dtypes to torch.dtype objects
def _to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "uint8":
        # Note: PyTorch does not have a native uint8 tensor type for general computation.
        # It has torch.uint8 for specific purposes like image processing or storing boolean masks.
        # For arithmetic operations, uint8 might be implicitly cast to float or raise errors.
        # For these tests, we'll convert to float32 for operations and then check.
        # If strict uint8 operations were required, more complex handling would be needed.
        # For simplicity, if input is uint8, it will be converted to float for reshape and then compared.
        # Or, if reshape preserves values directly, we can use it.
        # Since reshape is a view operation, values are preserved. We can use torch.uint8 directly for input.
        return torch.uint8
    # Add other dtypes as needed
    raise ValueError(f"Unsupported dtype: {dtype_str}")

def _get_model(input_tensor, output_shape):
    """Return a reshaped PyTorch tensor."""
    return torch.reshape(input_tensor, output_shape)

def test_reshape():
    # In PyTorch, we typically use an available device, e.g., 'cpu' or 'cuda'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(0)

    for dtype_str, low, high, atol, rtol in [
        ("float32", -127, 128, 0.001, 0.001),
        ("uint8", 0, 255, 0, 0),
    ]:
        # Create numpy input
        np_input_data = np.random.uniform(low, high, (1, 1, 1, 1000)).astype(dtype_str)
        
        # Convert to torch tensor
        torch_dtype = _to_torch_dtype(dtype_str)
        torch_input_data = torch.tensor(np_input_data, dtype=torch_dtype, device=device)

        for new_shape in [(1, 1000), (10, 10, 10), (10, 100, 1), (1, 1000, 1)]:
            # Compute expected output using numpy
            np_expected_output = np_input_data.reshape(new_shape)

            # Compute actual output using PyTorch model
            torch_actual_output = _get_model(torch_input_data, new_shape)

            config = {
                "input_shape": np_input_data.shape,
                "new_shape": new_shape,
                "dtype": dtype_str,
                "device": str(device)
            }
            
            # Compare PyTorch output (converted to numpy) with numpy expected output
            torch.testing.assert_close(
                torch_actual_output.cpu().numpy(),
                np_expected_output,
                rtol=rtol,
                atol=atol,
                msg=f"Mismatch for config: {config}"
            )

# TVM-specific codegen tests and related infrastructure functions are not convertible
# and are therefore omitted from the PyTorch version of the test.

if __name__ == "__main__":
    pytest.main([__file__])
