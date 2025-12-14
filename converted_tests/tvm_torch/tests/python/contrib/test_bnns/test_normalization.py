import torch
import numpy as np
import pytest
import math # Original TVM code had this import, keeping it for parity although not explicitly used in current PyTorch translation logic.
import functools # Needed for any potential functools.reduce usage, not directly in this specific translation but for broader composite mappings.

# Placeholder for TVM-specific infrastructure components
# These are mocked or replaced to enable running PyTorch-native tests.

class Device:
    """Mock TVM Device. Returns a PyTorch device."""
    def __init__(self):
        # BNNS is Apple-specific. For general PyTorch testing, we use CUDA if available, else CPU.
        self.device_name = "cuda" if torch.cuda.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

    def __call__(self):
        return self.device

# Mocking TVM-specific testing utilities.
# These typically check for TVM runtime/codegen availability.
# For PyTorch, we generally want to run tests on available devices (CPU/CUDA)
# and skip TVM-specific codegen tests.
def skip_runtime_test():
    """Mock function for skipping runtime tests."""
    # This test targets BNNS, which is an Apple-specific backend for TVM.
    # In a PyTorch context, we might skip if not on macOS or without MPS backend,
    # but for a generic conversion, we assume the test should run on available PyTorch backends.
    # Returning False allows the test to run on CPU/CUDA.
    # If a specific 'BNNS' like backend is desired in PyTorch (e.g., MPS),
    # further logic would be needed.
    return False

def skip_codegen_test():
    """Mock function for skipping codegen tests."""
    # TVM's codegen tests verify the structure of generated TVM IR.
    # This has no direct equivalent in PyTorch/TorchInductor tests,
    # as TorchInductor's output is usually not directly exposed for structural verification
    # in user-facing tests. Hence, these are always skipped.
    return True

# Custom build_and_run for PyTorch
def build_and_run_pytorch(func_model, inputs, expected_output_len, params, device_obj, enable_inductor=False):
    """
    Simulates TVM's build_and_run by executing a PyTorch model
    with and without torch.compile (TorchInductor).
    """
    device = device_obj.device

    # Move input and parameters to the target device
    src_tensor_on_device = inputs["src"].to(device)
    gamma_param_on_device = params["gamma_param"].to(device)
    beta_param_on_device = params["beta_param"].to(device)

    # Wrap the functional call in a lightweight nn.Module for consistency, especially with torch.compile
    class _WrapperModule(torch.nn.Module):
        def __init__(self, fn_to_wrap, gamma, beta):
            super().__init__()
            self.fn_to_wrap = fn_to_wrap
            # Register gamma and beta as buffers if they are parameters for the functional op
            self.register_buffer("gamma", gamma)
            self.register_buffer("beta", beta)
        
        def forward(self, src):
            return self.fn_to_wrap(src, self.gamma, self.beta)
            
    model_to_run = _WrapperModule(func_model, gamma_param_on_device, beta_param_on_device)
    
    if enable_inductor:
        # Use torch.compile to simulate TVM's optimized execution path
        compiled_model = torch.compile(model_to_run)
        out = compiled_model(src_tensor_on_device)
    else:
        # Run without compilation for baseline comparison
        out = model_to_run(src_tensor_on_device)
    
    # TVM's build_and_run returns a list of outputs. Match this format.
    return [out]


def verify(outputs, atol, rtol, config):
    """
    Verifies that all outputs in the list are close to the first output.
    Equivalent to tvm.testing.utils.assert_allclose.
    """
    if len(outputs) < 2:
        raise ValueError("Need at least two outputs to verify against each other (e.g., eager vs. compiled).")
    expected = outputs[0]
    for i in range(1, len(outputs)):
        torch.testing.assert_allclose(actual=outputs[i], expected=expected, rtol=rtol, atol=atol)

# --- End of infrastructure mocks ---

# Type conversion helper for TVM string dtypes to PyTorch dtypes
def to_torch_dtype(tvm_dtype_str):
    """Converts TVM string dtype to PyTorch torch.dtype object."""
    if tvm_dtype_str == "float32":
        return torch.float32
    elif tvm_dtype_str == "int32":
        return torch.int32
    elif tvm_dtype_str == "float64":
        return torch.float64
    elif tvm_dtype_str == "int8":
        return torch.int8
    elif tvm_dtype_str == "bool":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype: {tvm_dtype_str}")


def _get_model(
    shape, b_shape, s_shape, dtype_str, var_names, axis=1, epsilon=1e-5, center=True, scale=True
):
    """
    Returns a PyTorch functional model (lambda) for instance_norm
    and a dictionary of its associated parameters (gamma, beta).
    """
    
    # Generate parameters (gamma and beta) as NumPy arrays, then convert to PyTorch tensors.
    # b_shape and s_shape are expected to be 1D tuples/lists representing the channel dimension size.
    np_beta = np.random.uniform(-128, 127, b_shape).astype(dtype_str)
    np_gamma = np.random.uniform(-128, 127, s_shape).astype(dtype_str)

    # `params` will store the actual tensors for gamma and beta
    params = {
        "gamma_param": torch.tensor(np_gamma, dtype=to_torch_dtype(dtype_str)),
        "beta_param": torch.tensor(np_beta, dtype=to_torch_dtype(dtype_str)),
    }

    # Define the PyTorch functional model.
    # This lambda will take `src_tensor`, and `gamma_param`, `beta_param` as explicit arguments.
    # It encapsulates the necessary permutation logic based on `axis` to align with PyTorch's
    # `F.instance_norm` expectations.
    def model_fn(src_tensor, gamma_param_arg, beta_param_arg):
        weight_to_use = gamma_param_arg if scale else None
        bias_to_use = beta_param_arg if center else None

        original_ndim = src_tensor.ndim
        
        src_tensor_for_norm = src_tensor
        current_channel_dim_for_norm = axis # Initially, assume TVM's axis is current channel dim
        dims_to_permute_back_for_2d = None
        is_2d_input_reshaped = False

        # Handle 2D inputs: PyTorch's F.instance_norm requires at least 3D input (N, C, L).
        if original_ndim == 2:
            is_2d_input_reshaped = True
            # TVM's `axis` specifies the channel dimension for a 2D input (H, W).
            # We transform (H,W) to (1, C, Spatial) for PyTorch's F.instance_norm.
            # If axis=0 (H is channels): (H,W) -> (1, H, W). C=H, Spatial=W.
            # If axis=1 (W is channels): (H,W) -> (1, W, H). C=W, Spatial=H.
            
            if axis == 0:
                src_tensor_for_norm = src_tensor.unsqueeze(0) # Becomes (1, H, W)
                current_channel_dim_for_norm = 1 # C is now at index 1
            elif axis == 1:
                src_tensor_for_norm = src_tensor.permute(1, 0).unsqueeze(0) # (W, H) -> (1, W, H)
                current_channel_dim_for_norm = 1 # C is now at index 1
                dims_to_permute_back_for_2d = (1, 0) # Store for inverse permute later
            else:
                raise ValueError(f"Unsupported axis {axis} for 2D input (shape {shape}) for instance norm.")
        elif not (3 <= original_ndim <= 5):
            raise ValueError(f"PyTorch F.instance_norm supports 3D, 4D, 5D inputs, got {original_ndim}D (shape {shape}).")
        else:
            # Resolve negative axis to positive index for the tensor's current dimensions.
            current_channel_dim_for_norm = axis if axis >= 0 else src_tensor.ndim + axis

        # Apply F.instance_norm, with pre/post permutation if the channel dimension is not at index 1
        if current_channel_dim_for_norm != 1:
            # Move the designated channel dimension to index 1 for F.instance_norm
            src_permuted = torch.movedim(src_tensor_for_norm, current_channel_dim_for_norm, 1)
            
            out_permuted = torch.nn.functional.instance_norm(
                src_permuted,
                running_mean=None, # F.instance_norm doesn't use running stats
                running_var=None,  # F.instance_norm doesn't use running stats
                weight=weight_to_use,
                bias=bias_to_use,
                use_input_stats=True, # Always True for instance norm
                momentum=0.1, # Default momentum, not provided in TVM API
                eps=epsilon,
            )
            
            # Move back to original layout
            output = torch.movedim(out_permuted, 1, current_channel_dim_for_norm)
        else: # Channel is already at dim 1, no permutation needed for F.instance_norm
            output = torch.nn.functional.instance_norm(
                src_tensor_for_norm,
                running_mean=None,
                running_var=None,
                weight=weight_to_use,
                bias=bias_to_use,
                use_input_stats=True,
                momentum=0.1,
                eps=epsilon,
            )
        
        # If input was originally 2D and reshaped, revert the reshape and permutations
        if is_2d_input_reshaped:
            output = output.squeeze(0) # Remove the added batch dimension
            if dims_to_permute_back_for_2d is not None:
                output = output.permute(dims_to_permute_back_for_2d)
        
        return output

    return model_fn, params


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because original test depends on BNNS runtime which is TVM specific")
def test_normalization():
    device_obj = Device()
    np.random.seed(0) # Seed numpy for reproducibility in input generation
    torch.manual_seed(0) # Seed torch for reproducibility if random ops were inside model
    dtype_str = "float32"

    # Define test configurations for shapes and corresponding channel axes
    # PyTorch F.instance_norm expects 3D, 4D, or 5D inputs (N, C, ...).
    # The `axis` here denotes the channel dimension in the *original* TVM input `shape`.
    shapes_config_with_axes = [
        ([1, 2, 3, 4], 1),   # NCHW, C at axis 1 (standard)
        ([3, 2, 3, 4], 1),   # NCHW, C at axis 1
        ([2, 2, 3], 1),      # NCW, C at axis 1
        ([16, 32, 32], 1),   # NCW, C at axis 1
        ([1, 4, 8, 16], 3),  # NHWC-like, C at axis 3, requires permute
        ([1, 16, 8, 4], 0),  # CHWN-like (if N=1), C at axis 0, requires permute
        ([5, 3], 0),         # 2D input (H, W), assume H is channel. Reshaped to (1,H,W) internally.
        ([5, 3], 1),         # 2D input (H, W), assume W is channel. Reshaped to (1,W,H) internally.
        ([2, 3, 4, 5, 6], 1), # NCDHW, C at axis 1
        ([2, 3, 4, 5, 6], 0), # DCDHW, C at axis 0, requires permute
    ]

    for shape, axis in shapes_config_with_axes:
        # Resolve negative axis to a positive index for validation and channel dimension size
        actual_axis_idx = axis if axis >= 0 else len(shape) + axis

        # Validate axis for the given shape. The `_get_model` function has internal checks,
        # but this upfront check helps skip invalid combinations early.
        if not (0 <= actual_axis_idx < len(shape)):
            continue # Skip invalid axis-shape combinations

        # Determine the size of the channel dimension for gamma/beta parameters
        actual_channel_dim_size = shape[actual_axis_idx]

        for center in [False, True]:
            for scale in [False, True]:
                outputs = []
                
                # Create input tensor for the model
                inputs = {
                    "src": torch.tensor(
                        np.random.uniform(-128, 127, shape).astype(dtype_str),
                        dtype=to_torch_dtype(dtype_str)
                    ),
                }

                # Get the PyTorch functional model and its associated parameters
                func_model, params = _get_model(
                    shape,
                    [actual_channel_dim_size], # b_shape (size of channel dim for beta)
                    [actual_channel_dim_size], # s_shape (size of channel dim for gamma)
                    dtype_str,
                    var_names=iter(inputs), # TVM-specific, not used in PyTorch _get_model
                    axis=axis,
                    center=center,
                    scale=scale,
                    epsilon=1e-5 # Pass epsilon
                )
                
                # Run the model with and without torch.compile (analogous to enable_bnns in TVM)
                for enable_inductor in [False, True]:
                    # Call the custom build_and_run_pytorch function
                    results = build_and_run_pytorch(
                        func_model,
                        inputs,
                        1, # expected_output_len (from TVM signature, not strictly used in mock)
                        params,
                        device_obj,
                        enable_inductor=enable_inductor,
                    )
                    outputs.append(results[0].cpu()) # Get the tensor result and move to CPU for comparison

                config = {
                    "dtype": dtype_str,
                }
                verify(outputs, atol=0.001, rtol=0.01, config=config)


@pytest.mark.skip(reason="TVM BNNS codegen test has no direct PyTorch equivalent")
def test_codegen_normalization():
    """
    This test is entirely TVM codegen specific and has no direct PyTorch equivalent.
    It verifies the structure of the TVM generated graph, which is not applicable
    to PyTorch/TorchInductor.
    """
    pass

if __name__ == "__main__":
    # This block allows running the tests directly using `python <filename.py>`
    # in addition to `pytest`.
    # For CI/CD, `pytest` is typically used.
    test_normalization()
    # test_codegen_normalization is intentionally skipped as it's TVM-specific.
