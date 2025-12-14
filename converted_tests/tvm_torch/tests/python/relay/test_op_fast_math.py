import numpy as np
import scipy.special
import torch
import pytest
import functools

# Using torch.testing.assert_close for modern PyTorch, aliasing to assert_allclose for consistency
try:
    from torch.testing._comparison import assert_close as assert_allclose
except ImportError:
    from torch.testing._core import assert_allclose

# Helper to map string dtypes to torch dtypes
_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

def to_torch_dtype(dtype_str):
    return _TORCH_DTYPE_MAP.get(dtype_str)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
class TestFastMath:
    def test_fastmath(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        def _test_apply_pytorch(torch_op, name, f_numpy, low, high, step, dtype="float32"):
            # Prepare numpy inputs and expected outputs
            a_np = np.arange(low, high, step).astype(dtype).reshape((1, -1))
            b_np = f_numpy(a_np)

            # Create PyTorch tensor
            x_torch = torch.tensor(a_np, device=device, dtype=to_torch_dtype(dtype))

            # Define the PyTorch model/function to be compiled
            def model(x):
                return torch_op(x)

            # Compile the model with TorchInductor
            compiled_model = torch.compile(model)

            # Execute and get output
            torch_output = compiled_model(x_torch)

            # Assertions
            # For fast math tests, the rtol/atol are crucial.
            # Softmax might have slightly higher error due to internal math optimizations
            rtol_val = 1e-5
            atol_val = 1e-5
            if name == "nn_fast_softmax":
                rtol_val = 1e-4
                atol_val = 1e-4

            assert_allclose(torch_output.cpu().numpy(), b_np, rtol=rtol_val, atol=atol_val)

        # Test relay.exp -> torch.exp
        _test_apply_pytorch(torch.exp, "fast_exp", np.exp, low=-88, high=88, step=0.01)

        # Test relay.erf -> torch.erf
        _test_apply_pytorch(torch.erf, "fast_erf", scipy.special.erf, low=-10, high=10, step=0.01)

        # Test relay.tanh -> torch.tanh
        _test_apply_pytorch(torch.tanh, "fast_tanh", np.tanh, low=-10, high=10, step=0.01)

        # Test relay.nn.fast_softmax -> torch.nn.functional.softmax (with dim=-1)
        # Using a stable numpy softmax implementation for reference
        def numpy_softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # for numerical stability
            return e_x / np.sum(e_x, axis=-1, keepdims=True)

        _test_apply_pytorch(
            functools.partial(torch.nn.functional.softmax, dim=-1),
            "nn_fast_softmax",
            numpy_softmax,
            low=-10,
            high=10,
            step=0.01,
        )
