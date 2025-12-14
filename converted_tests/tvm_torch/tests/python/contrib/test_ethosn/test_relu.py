import numpy as np
import pytest
import torch
import torch.nn.functional as F

# --- Mappings for TVM Dtypes to PyTorch Dtypes ---
# Assuming common dtypes used in TVM tests are convertible to PyTorch
DTYPE_MAP = {
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

# --- Dummy Ethos-N related decorators/infrastructure ---
def requires_ethosn(f):
    """Decorator for tests that conceptually require Ethos-N.
    In the PyTorch context, these tests will run without a specific Ethos-N backend.
    """
    return f

class MockEthosNInfrastructure:
    """A mock infrastructure to bridge TVM-specific test utilities to PyTorch."""

    def make_module(self, model_func, _params):
        """Simulates TVM's module creation by simply returning the PyTorch callable."""
        return model_func

    def build_and_run(self, model_func, inputs, _num_runs, _params, npu, additional_config_args):
        """Simulates TVM's build_and_run.
        `npu=False` maps to eager execution.
        `npu=True` maps to `torch.compile` execution.
        """
        input_tensor = inputs["a"]
        if npu:
            # Simulate "NPU" path with torch.compile
            compiled_model = torch.compile(model_func)
            result = compiled_model(input_tensor)
        else:
            # Simulate "non-NPU" path with eager execution
            result = model_func(input_tensor)
        # Convert to numpy and then back to TVM-like ndarray for compatibility with verify
        return result.cpu()

    def verify(self, outputs, _dtype, _tol):
        """Verifies outputs. Expects two outputs (eager, compiled) and asserts their closeness."""
        assert len(outputs) == 2, "Expected two outputs (eager and compiled)"
        # Convert outputs to numpy for comparison
        output_0_np = outputs[0].numpy() if isinstance(outputs[0], torch.Tensor) else outputs[0]
        output_1_np = outputs[1].numpy() if isinstance(outputs[1], torch.Tensor) else outputs[1]
        np.testing.assert_allclose(output_0_np, output_1_np, rtol=1e-5, atol=1e-5)

    def make_ethosn_partition(self, model_func):
        """Mock for Ethos-N graph partitioning; returns the original model."""
        return model_func

    def test_error(self, model_or_module, inputs, err_msg):
        """Mock for testing specific backend compilation errors.
        Since these errors are typically TVM/Ethos-N specific and not reproducible
        by standard PyTorch ops or `torch.compile` (which generally handles valid ops),
        these tests are skipped with a clear message.
        """
        pytest.skip(f"TODO: Cannot directly replicate TVM backend error: {err_msg}")

# Instantiate the mock infrastructure
tei = MockEthosNInfrastructure()

def _get_model(shape, dtype_str, a_min, a_max):
    """
    Creates a PyTorch functional model for clip (ReLU-like behavior).
    TVM's `relay.clip` maps to PyTorch's `torch.clamp`.
    """
    dtype = DTYPE_MAP[dtype_str]

    # In PyTorch, a model is a callable that takes tensors
    def model_func(input_tensor):
        # Ensure the input tensor is of the correct dtype as specified by the test
        # and has requires_grad=False for inference-like tests, similar to TVM's default
        return torch.clamp(input_tensor.to(dtype), min=a_min, max=a_max)
    return model_func


@requires_ethosn
@pytest.mark.parametrize(
    "shape,a_min,a_max,dtype",
    [
        ((1, 4, 4, 4), 65, 178, "uint8"),
        ((1, 8, 4, 2), 1, 254, "uint8"),
        ((1, 8, 4, 2), -100, 100, "int8"),
        ((1, 16), -120, -20, "int8"),
    ],
)
def test_relu(dtype, shape, a_min, a_max):
    """Compare Relu output with PyTorch (eager and compiled)."""
    np.random.seed(0)

    # Convert dtype string to NumPy dtype for random data generation
    np_dtype = np.dtype(dtype)

    inputs = {
        "a": torch.tensor(
            np.random.randint(
                low=np.iinfo(np_dtype).min,
                high=np.iinfo(np_dtype).max + 1,
                size=shape,
                dtype=np_dtype,
            ),
            dtype=DTYPE_MAP[dtype],
            device='cpu'
        ),
    }
    outputs = []
    # Simulate TVM's npu=[False, True] with eager vs. compiled PyTorch
    for is_compiled in [False, True]:
        model_func = _get_model(inputs["a"].shape, dtype, a_min, a_max)
        output = tei.build_and_run(model_func, inputs, 1, {}, npu=is_compiled, additional_config_args={})
        outputs.append(output)

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,dtype,a_min,a_max,err_msg",
    [
        ((1, 4, 4, 4, 4), "uint8", 65, 78, "dimensions=5, dimensions must be <= 4"),
        ((1, 8, 4, 2), "int16", 1, 254, "dtype='int16', dtype must be either uint8, int8 or int32"),
        ((1, 8, 4, 2), "uint8", 254, 1, "Relu has lower bound > upper bound"),
        ((2, 2, 2, 2), "uint8", 1, 63, "batch size=2, batch size must = 1; "),
    ],
)
def test_relu_failure(shape, dtype, a_min, a_max, err_msg):
    """Check Relu error messages.
    These TVM/Ethos-N specific validation errors are not directly convertible
    to PyTorch's native `torch.clamp` or `torch.compile` behavior, which
    would generally allow these operations (e.g., 5D tensors, int16 dtypes,
    or a_min > a_max which produces an output without error).
    """
    model = _get_model(shape, dtype, a_min, a_max)
    mod = tei.make_ethosn_partition(model) # This might represent a failed compilation in TVM
    tei.test_error(mod, {}, err_msg)
