import torch
import torch.nn.functional as F
import numpy as np
import pytest

# Mapping of TVM dtypes to PyTorch dtypes
tvm_to_torch_dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
}

def test_simplify_batchnorm(dtype="float32"):
    # This `simple_bn` function explicitly implements the math
    # that TVM's SimplifyInference pass would fold `nn.batch_norm` into
    # at inference time.
    def simple_bn(x, gamma, beta, moving_mean, moving_var, epsilon=1e-5, axis=1, shape=None):
        # Equivalent to: (x - moving_mean) / sqrt(moving_var + eps) * gamma + beta
        # The key is to correctly broadcast gamma/beta/mean/var (which are 1D)
        # to match the input 'x' tensor's shape.
        
        # Calculate scale and shift factors
        scale = (1.0 / torch.sqrt(moving_var + epsilon)) * gamma
        shift = ((-moving_mean) * scale) + beta

        # Reshape scale and shift for broadcasting
        # The 'axis' argument determines the channel dimension in 'x'.
        # We need to reshape the 1D 'scale'/'shift' to have 1s before 'axis',
        # the original dimension at 'axis', and 1s after 'axis'.
        num_leading_ones = axis
        # shape is the full shape of x, e.g., (N, C, H, W)
        # axis is the index of C, e.g., 1
        # len(shape) is the number of dimensions, e.g., 4
        # (axis + 1) is the count of dimensions up to and including the channel.
        # So, len(shape) - (axis + 1) is the number of spatial dimensions trailing C.
        num_trailing_ones = len(shape) - (axis + 1)

        if num_leading_ones > 0 or num_trailing_ones > 0:
            current_scale_shape = scale.shape
            # Construct the new shape for broadcasting, e.g., (1, C, 1, 1) for NCHW, axis=1
            new_scale_shape = tuple([1] * num_leading_ones + list(current_scale_shape) + [1] * num_trailing_ones)
            scale = scale.reshape(new_scale_shape)
            shift = shift.reshape(new_scale_shape)
        
        return x * scale + shift

    def check(dim, axis, nstep):
        eps = 0.01
        torch_dtype = tvm_to_torch_dtype[dtype]
        
        # Use CPU for simplicity, or detect CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define shapes for input tensors
        x_shape = tuple(10 for _ in range(dim))
        param_shape = (10,) # Assuming 10 channels for this example

        # Create dummy input tensors for PyTorch
        # These correspond to rly.var in the TVM code
        x_val = torch.randn(x_shape, dtype=torch_dtype, device=device)
        beta_val = torch.randn(param_shape, dtype=torch_dtype, device=device)
        gamma_val = torch.randn(param_shape, dtype=torch_dtype, device=device) + 1.0 # Gamma usually positive
        moving_var_val = torch.abs(torch.randn(param_shape, dtype=torch_dtype, device=device)) + 0.1 # Variance usually positive
        moving_mean_val = torch.randn(param_shape, dtype=torch_dtype, device=device)

        # Clone inputs for the two computation paths
        y1_input = x_val.clone()
        y2_input = x_val.clone()

        # Iterate nstep times to simulate a sequence of operations
        for _ in range(nstep):
            # Path 1: Original computation using F.batch_norm and F.dropout (assuming inference behavior)
            # F.batch_norm uses given running_mean/running_var if training=False
            y1_output = F.batch_norm(
                y1_input + torch.tensor(1.0, dtype=torch_dtype, device=device),
                running_mean=moving_mean_val,
                running_var=moving_var_val,
                weight=gamma_val,
                bias=beta_val,
                training=False,  # Simulate inference mode
                momentum=0.1,  # Not used in inference for functional batch_norm, but required arg
                eps=eps,
            )
            # Dropout is a no-op in inference (training=False)
            y1_output = F.dropout(y1_output, p=0.5, training=False)
            y1_input = y1_output

            # Path 2: Simplified computation using the `simple_bn` function
            y2_output = simple_bn(
                y2_input + torch.tensor(1.0, dtype=torch_dtype, device=device),
                gamma_val,
                beta_val,
                moving_mean_val,
                moving_var_val,
                epsilon=eps,
                axis=axis,
                shape=x_shape,
            )
            y2_input = y2_output

        # Assert numerical equivalence of the two paths
        # TVM's structural_equal is replaced by numerical comparison here
        pytest.approx(y1_input.cpu().numpy(), rel=1e-5, abs=1e-5) == y2_input.cpu().numpy()
        torch.testing.assert_allclose(y1_input, y2_input, rtol=1e-5, atol=1e-5)


    # Run checks with various dimensions and axis configurations
    check(dim=2, axis=1, nstep=1)
    check(dim=4, axis=1, nstep=1)
    check(dim=4, axis=0, nstep=3) # More steps for robustness

if __name__ == "__main__":
    test_simplify_batchnorm(dtype="float32")
    test_simplify_batchnorm(dtype="float16")
