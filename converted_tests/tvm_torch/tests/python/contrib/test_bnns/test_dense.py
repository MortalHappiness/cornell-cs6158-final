import torch
import numpy as np
import pytest
import math
import itertools

# Mapping TVM dtype strings to PyTorch dtypes
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int32": torch.int32,
    "int8": torch.int8,
    "int64": torch.int64,
    # Add other dtypes as needed
}

def _to_torch_dtype(tvm_dtype_str):
    if tvm_dtype_str in _DTYPE_MAP:
        return _DTYPE_MAP[tvm_dtype_str]
    raise ValueError(f"Unknown TVM dtype: {tvm_dtype_str}")

# PyTorch Module equivalent to the TVM Relay graph in _get_model
class DenseAndGELUModel(torch.nn.Module):
    def __init__(self, input_shape, weight_shape, units, dtype, has_bias=False, has_gelu=False):
        super().__init__()
        self.has_gelu = has_gelu
        self.output_torch_dtype = _to_torch_dtype(dtype)

        in_features = input_shape[-1]
        out_features = units

        # torch.nn.Linear takes in_features and out_features
        # TVM's _get_model 'weight_shape' is (units, last_dim_of_input_shape),
        # which corresponds to PyTorch's (out_features, in_features)
        self.linear = torch.nn.Linear(in_features, out_features, bias=has_bias, dtype=self.output_torch_dtype)

        # Initialize weights and bias to match TVM's random initialization logic for reproducibility
        # np.random.seed(0) is set in the test function, ensuring deterministic initialization
        weight_np = np.random.uniform(-128, 127, weight_shape).astype(dtype)
        self.linear.weight.data = torch.from_numpy(weight_np).to(self.output_torch_dtype)

        if has_bias:
            # weight_shape[0] in TVM's _get_model is 'units' (output features)
            bias_np = np.random.randint(-128, 127, weight_shape[0]).astype(dtype)
            self.linear.bias.data = torch.from_numpy(bias_np).to(self.output_torch_dtype)

    def forward(self, a):
        out = self.linear(a)

        if self.has_gelu:
            # Replicating the specific GELU approximation from the original TVM test
            x_for_gelu = out
            # Constants should be on the same device and dtype as x_for_gelu
            c1 = torch.tensor(0.044715, dtype=self.output_torch_dtype, device=x_for_gelu.device)
            c2 = torch.tensor(math.sqrt(2 / math.pi), dtype=self.output_torch_dtype, device=x_for_gelu.device)

            term1 = x_for_gelu.pow(3.0) * c1
            term2 = x_for_gelu + term1
            term3 = term2 * c2
            term4 = term3.tanh()
            term5 = term4 + 1.0
            term6 = term5 * 0.5
            out = term6 * x_for_gelu

        return out

# Helper to generate test combinations (simplified from TVM's generate_trials)
def _generate_test_combinations(param_lists):
    return list(itertools.product(*param_lists))

# Determine default device for PyTorch
_CURRENT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The original TVM test used `skip_runtime_test()` and `skip_codegen_test()`
# for BNNS backend availability. For PyTorch/TorchInductor conversion,
# we run the functional tests on available PyTorch backends (CPU/CUDA)
# and omit the TVM-specific codegen verification.

def test_dense_and_gelu():
    # Set numpy seed for reproducible results, matching TVM test's initialization logic
    np.random.seed(0)
    
    # Parameters from the original TVM test
    dtypes = ["float32"]
    shapes_configs = [ # (input_shape, weight_shape, units)
        ((1, 128), (16, 128), 16),
        ((32, 32), (32, 32), 32),
        ((1, 64), (1, 64), 1),
        ((11, 2), (2, 2), 2),
        ((2, 2), (1, 2), 1),
    ]
    
    booleans = [False, True] # For has_bias and has_gelu
    
    # Generate test combinations (TVM's `generate_trials` with `num_trials=3` was a subset,
    # here we run all combinations for thoroughness unless explicit subset is needed.)
    trials = _generate_test_combinations([dtypes, shapes_configs, booleans, booleans])

    for dtype, (input_shape, weight_shape, units), with_bias, with_gelu in trials:
        print(f"Running test for: dtype={dtype}, input_shape={input_shape}, weight_shape={weight_shape}, units={units}, with_bias={with_bias}, with_gelu={with_gelu}")

        # Ensure random states are consistent for both eager and compiled models
        # This is critical because DenseAndGELUModel uses numpy.random for weight init.
        initial_seed = np.random.get_state() 

        # --- Eager execution (reference output) ---
        np.random.set_state(initial_seed) # Reset numpy random state
        model_eager = DenseAndGELUModel(input_shape, weight_shape, units, dtype, with_bias, with_gelu)
        model_eager.to(_CURRENT_DEVICE)
        
        # Prepare input data (using a fresh random state for input for each trial)
        input_np = np.random.uniform(-128, 127, input_shape).astype(dtype)
        input_torch = torch.from_numpy(input_np).to(_CURRENT_DEVICE)

        output_eager = model_eager(input_torch)

        # --- Compiled execution (TorchInductor output) ---
        np.random.set_state(initial_seed) # Reset numpy random state again for compiled model init
        model_compiled = DenseAndGELUModel(input_shape, weight_shape, units, dtype, with_bias, with_gelu)
        model_compiled.to(_CURRENT_DEVICE)
        
        # Compile the model using TorchInductor
        compiled_fn = torch.compile(model_compiled, fullgraph=True) # fullgraph=True for compiler tests
        
        output_compiled = compiled_fn(input_torch)
        
        # Verify correctness using PyTorch's assert_allclose
        # Original TVM test used atol=0.001, rtol=0.01
        torch.testing.assert_allclose(output_eager, output_compiled, rtol=0.01, atol=0.001)

# The original `_get_expected_codegen` and `test_codegen_dense` functions
# are specific to TVM's IR checking and have no direct functional equivalent
# in PyTorch/TorchInductor. They are omitted from this conversion.
# The `test_dense_and_gelu` function implicitly verifies TorchInductor's
# functional correctness by comparing its output against eager execution.

if __name__ == "__main__":
    pytest.main([__file__])
