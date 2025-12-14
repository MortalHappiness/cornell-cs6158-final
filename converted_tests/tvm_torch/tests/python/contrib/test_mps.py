import torch
import pytest
import numpy as np
import math # For ceil in conv2d calculation

# Helper to map numpy dtypes to torch dtypes. Assuming float32 for most cases here.
def get_torch_dtype(np_dtype_str):
    if np_dtype_str == 'float32':
        return torch.float32
    if np_dtype_str == 'float64':
        return torch.float64
    if np_dtype_str == 'int32':
        return torch.int32
    if np_dtype_str == 'int64':
        return torch.int64
    # Add other dtypes as needed
    return None

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_matmul():
    n = 1024
    l = 128
    m = 256
    
    # Assuming A and B are float32 based on common TVM test patterns
    # and the use of np.random.uniform().astype(A.dtype)
    dtype_np = np.float32
    dtype_torch = get_torch_dtype(str(dtype_np))

    # Define the PyTorch model/function
    def model(A_input, B_input):
        C_output = torch.matmul(A_input, B_input)
        D_output = C_output + 1.0
        return D_output

    # Compile the model for TorchInductor.
    # Add a fallback for environments where torch.compile might not be available.
    if hasattr(torch, 'compile'):
        compiled_model = torch.compile(model, mode="reduce-overhead")
    else:
        compiled_model = model # Fallback if compile is not available

    # Prepare inputs
    dev = torch.device('mps')
    A_np = np.random.uniform(size=(n, l)).astype(dtype_np)
    B_np = np.random.uniform(size=(l, m)).astype(dtype_np)

    A_torch = torch.from_numpy(A_np).to(dev).to(dtype_torch)
    B_torch = torch.from_numpy(B_np).to(dev).to(dtype_torch)

    # Execute compiled model
    C_torch_output = compiled_model(A_torch, B_torch)

    # Calculate expected result using NumPy
    expected_np = np.dot(A_np, B_np) + 1.0

    # Assertions
    torch.testing.assert_allclose(C_torch_output.cpu().numpy(), expected_np, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_conv2d():
    n = 1
    h = 14
    w = 14
    ci = 2
    co = 4
    kh = 3
    kw = 3
    stride = 2

    # Assuming A and B are float32
    dtype_np = np.float32
    dtype_torch = get_torch_dtype(str(dtype_np))

    # Define the PyTorch model/function for conv2d
    def model(A_input, B_weight):
        # TVM A is (N, H, W, CI) -> PyTorch input needs (N, CI, H, W)
        A_input_nchw = A_input.permute(0, 3, 1, 2)
        # TVM B is (CO, KH, KW, CI) -> PyTorch weight needs (CO, CI, KH, KW)
        B_weight_oihw = B_weight.permute(0, 3, 1, 2)

        C_output_nchw = torch.nn.functional.conv2d(
            A_input_nchw,
            B_weight_oihw,
            stride=stride,
            padding='same' # PyTorch 'same' padding
        )
        # Convert output back to NHWC layout to match TVM's implied output layout (if it was NHWC).
        # Assuming the original TVM output was NHWC because input was NHWC.
        C_output_nhwc = C_output_nchw.permute(0, 2, 3, 1)
        return C_output_nhwc

    # Compile the model for TorchInductor.
    if hasattr(torch, 'compile'):
        compiled_model = torch.compile(model, mode="reduce-overhead")
    else:
        compiled_model = model # Fallback if compile is not available

    # Prepare inputs
    dev = torch.device('mps')
    A_np = np.random.uniform(size=(n, h, w, ci)).astype(dtype_np)
    B_np = np.random.uniform(size=(co, kh, kw, ci)).astype(dtype_np)

    A_torch = torch.from_numpy(A_np).to(dev).to(dtype_torch)
    B_torch = torch.from_numpy(B_np).to(dev).to(dtype_torch)

    # Execute compiled model
    C_torch_output = compiled_model(A_torch, B_torch)

    # Calculate expected reference using the uncompiled PyTorch functional conv2d
    # to ensure numerical correctness against PyTorch's eager mode.
    C_ref_output = model(A_torch, B_torch)

    # Assertions for shape and dtype
    expected_h = math.ceil(h / stride)
    expected_w = math.ceil(w / stride)
    expected_shape = (n, expected_h, expected_w, co)

    assert C_torch_output.shape == expected_shape, f"Expected shape {expected_shape}, got {C_torch_output.shape}"
    assert C_torch_output.dtype == dtype_torch, f"Expected dtype {dtype_torch}, got {C_torch_output.dtype}"

    # Numerical assertion
    torch.testing.assert_allclose(C_torch_output.cpu(), C_ref_output.cpu(), rtol=1e-5, atol=1e-5)
