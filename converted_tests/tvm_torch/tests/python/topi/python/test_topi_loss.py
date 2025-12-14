import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch.testing

# Mapping for TVM string dtypes to PyTorch dtypes
_TVM_DTYPE_TO_TORCH = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "qint8": torch.qint8,  # Added for quantized types, if needed elsewhere
}

def get_torch_dtype(tvm_dtype_str):
    """Converts a TVM string dtype to its PyTorch equivalent."""
    dtype = _TVM_DTYPE_TO_TORCH.get(tvm_dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported TVM dtype string: {tvm_dtype_str}")
    return dtype

# Define a pytest fixture for the device, enabling easy switching between CPU and CUDA
@pytest.fixture
def device():
    """Returns the appropriate torch.device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.mark.parametrize(
    "prediction_shape, reduction, ignore_index, dtype",
    [
        ((10, 5), "mean", -100, "float32"),
        ((10, 5, 2, 2), "mean", -100, "float32"),
        ((10, 5), "sum", -100, "float32"),
        ((10, 5), "none", -100, "float32"),
        ((10, 5), "mean", 3, "float32"),
        ((10, 5), "mean", -100, "float64"),
    ],
)
def test_nll_loss(device, prediction_shape, reduction, ignore_index, dtype):
    """
    Tests torch.nn.functional.nll_loss against a PyTorch-based reference
    for various input shapes, reductions, and dtypes.
    """
    C = prediction_shape[1] # Number of classes
    # Target shape is batch size + any dimensions after channels, excluding channels
    target_shape = prediction_shape[:1] + prediction_shape[2:]

    # Generate numpy inputs
    # Predictions are typically log-probabilities
    predictions_npy = np.random.uniform(low=-5.0, high=0.0, size=prediction_shape).astype(dtype)
    # Targets are class indices
    targets_npy = np.random.randint(0, C, target_shape).astype("int64") # PyTorch targets usually int64
    # Weights are per-class weights
    weights_npy = np.random.uniform(size=(C,)).astype(dtype)

    # --- Reference computation using PyTorch's F.nll_loss on CPU ---
    # Convert numpy inputs to PyTorch tensors for reference (on CPU)
    predictions_ref_pt = torch.tensor(predictions_npy, dtype=get_torch_dtype(dtype), device="cpu")
    targets_ref_pt = torch.tensor(targets_npy, dtype=get_torch_dtype("int64"), device="cpu")
    weights_ref_pt = torch.tensor(weights_npy, dtype=get_torch_dtype(dtype), device="cpu")

    # Perform reference computation
    expected_result_pt = F.nll_loss(
        predictions_ref_pt,
        targets_ref_pt,
        weight=weights_ref_pt,
        reduction=reduction,
        ignore_index=ignore_index,
    )
    expected_result_npy = expected_result_pt.cpu().numpy()

    # --- Actual computation using PyTorch's F.nll_loss on specified device ---
    # Convert numpy inputs to PyTorch tensors for the actual test (on the chosen device)
    predictions_test_pt = torch.tensor(predictions_npy, dtype=get_torch_dtype(dtype), device=device)
    targets_test_pt = torch.tensor(targets_npy, dtype=get_torch_dtype("int64"), device=device)
    weights_test_pt = torch.tensor(weights_npy, dtype=get_torch_dtype(dtype), device=device)

    # Perform actual computation
    actual_result_pt = F.nll_loss(
        predictions_test_pt,
        targets_test_pt,
        weight=weights_test_pt,
        reduction=reduction,
        ignore_index=ignore_index,
    )

    # --- Compare results ---
    # Use torch.testing.assert_close for modern PyTorch, assert_allclose is deprecated.
    # The tolerance values are typical for float comparisons.
    torch.testing.assert_close(
        actual_result_pt.cpu().numpy(), expected_result_npy, rtol=1e-4, atol=1e-5
    )

# The tvm.testing.main() call is not needed for standard pytest execution.
# Pytest automatically discovers and runs tests in files named test_*.py or *_test.py.
