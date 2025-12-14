import numpy as np
import torch
import torch.nn.functional as F
import pytest

# --- Dtype conversion helper ---
DTYPE_MAP = {
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
    # Add other dtypes as needed based on TVM's usage
}

def to_torch_dtype(tvm_dtype_str):
    dtype = DTYPE_MAP.get(tvm_dtype_str)
    if dtype is None:
        # Fallback for common dtypes if not explicitly mapped
        if hasattr(torch, tvm_dtype_str):
            dtype = getattr(torch, tvm_dtype_str)
        else:
            raise ValueError(f"Unknown or unmapped dtype: {tvm_dtype_str}")
    return dtype

# --- TVM pad_before/pad_after to PyTorch pad tuple conversion ---
def _convert_tvm_pad_to_torch_pad_tuple(tvm_pad_before, tvm_pad_after):
    # TVM pad_before/after are lists/tuples like [N_b, H_b, W_b, C_b] for NHWC
    # PyTorch F.pad 'pad' argument expects a tuple of (last_dim_before, last_dim_after,
    #                                                  second_last_dim_before, second_last_dim_after, ...)
    # The order of dimensions is reversed for PyTorch's 'pad' argument.
    torch_pad = []
    for pb, pa in zip(reversed(tvm_pad_before), reversed(tvm_pad_after)):
        torch_pad.extend([pb, pa])
    return tuple(torch_pad)


def test_nn_pad():
    """Test nn pad operation."""
    # TODO: This test was originally designed for the Hexagon backend in TVM.
    # The PyTorch version will run on standard CPU/CUDA using torch.compile.
    # Specific Hexagon backend features, scheduling, or optimizations are not directly translated.

    dtype_str = "uint8"
    in_shape = (1, 56, 56, 32) # Assumed NHWC layout from NumPy reference

    data_in_np = np.ones(in_shape).astype(dtype_str)

    # Define the padding values in TVM style (per dimension: N, H, W, C)
    tvm_pad_before = [0, 1, 1, 0]
    tvm_pad_after = [0, 1, 1, 0]
    pad_value = 0

    # Convert TVM-style padding to PyTorch-style padding tuple
    torch_pad_tuple = _convert_tvm_pad_to_torch_pad_tuple(tvm_pad_before, tvm_pad_after)

    # Define the PyTorch operation as a callable
    def pad_op(input_tensor):
        return F.pad(
            input_tensor,
            pad=torch_pad_tuple,
            mode="constant",
            value=pad_value
        )

    # Determine device for PyTorch execution
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Prepare input tensor for PyTorch
    input_torch = torch.tensor(data_in_np, dtype=to_torch_dtype(dtype_str), device=device)

    # Compile the operation using TorchInductor
    compiled_pad_op = torch.compile(pad_op, backend="inductor")

    # Execute the compiled operation
    output_torch = compiled_pad_op(input_torch)

    # Reference numpy pad output
    # The numpy pad_width argument directly corresponds to TVM's pad_before/after structure
    ref_out_np = np.pad(
        data_in_np,
        pad_width=(
            (tvm_pad_before[0], tvm_pad_after[0]),  # N dimension
            (tvm_pad_before[1], tvm_pad_after[1]),  # H dimension
            (tvm_pad_before[2], tvm_pad_after[2]),  # W dimension
            (tvm_pad_before[3], tvm_pad_after[3]),  # C dimension
        ),
        mode="constant",
        constant_values=pad_value
    )

    # Assertions: compare PyTorch output with NumPy reference
    torch.testing.assert_close(output_torch.cpu().numpy(), ref_out_np)
