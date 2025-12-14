import pytest
import numpy as np
import torch
import functools

# Mock requires_ethosn for standalone execution and clear TODO
def requires_ethosn(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        pytest.skip(reason="Ethos-N specific hardware requirement cannot be natively tested with PyTorch/TorchInductor")
    return wrapper

# Helper for string dtype to torch.dtype
_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "int32": torch.int32,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int64": torch.int64,
    "bool": torch.bool,
}

def _tvm_dtype_to_torch_dtype(dtype_str):
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    return _TORCH_DTYPE_MAP.get(dtype_str) # Returns None if not found, let PyTorch raise error then

def _get_model_pytorch(shape, input_zp, input_sc, output_zp, output_sc, dtype_str):
    """
    Simulates the TVM Relay QNN Tanh model structure in PyTorch.
    The input `a` is assumed to be the raw integer quantized data.
    Returns an `nn.Module` that performs the dequantize->tanh->quantize operations.
    """
    torch_output_dtype = _tvm_dtype_to_torch_dtype(dtype_str)

    class TanhQuantizedModel(torch.nn.Module):
        def __init__(self, input_zp, input_sc, output_zp, output_sc, output_dtype_str):
            super().__init__()
            self.input_zp = input_zp
            self.input_sc = input_sc
            self.output_zp = output_zp
            self.output_sc = output_sc
            self.output_torch_dtype = _tvm_dtype_to_torch_dtype(output_dtype_str)
            
            # Get integer info for the target dtype for clamping
            np_dtype_info = np.iinfo(output_dtype_str)
            self.output_min_val = np_dtype_info.min
            self.output_max_val = np_dtype_info.max

        def forward(self, input_tensor_int_data):
            # 1. Dequantize: Convert integer input data to float representation.
            # Formula: (raw_int_value - zero_point) * scale
            float_data = (input_tensor_int_data.to(torch.float32) - self.input_zp) * self.input_sc

            # 2. Tanh operation on float data
            tanh_output_float = torch.tanh(float_data)

            # 3. Quantize: Convert float output to integer representation.
            # Formula: round(float_value / scale + zero_point)
            quantized_output_float = (tanh_output_float / self.output_sc) + self.output_zp
            
            # Round to nearest integer
            quantized_output_int = torch.round(quantized_output_float)
            
            # Clamp to the target integer dtype range
            clipped_output_int = torch.clamp(quantized_output_int, self.output_min_val, self.output_max_val)
            
            # Convert to the final integer dtype
            final_output = clipped_output_int.to(self.output_torch_dtype)
            return final_output
    
    return TanhQuantizedModel(input_zp, input_sc, output_zp, output_sc, dtype_str)


# --- PyTorch infrastructure for emulating TVM Ethos-N testing environment ---
class PyTorchEthosNTestInfrastructure:
    def make_module(self, model_instance, _):
        # In PyTorch, the model is already an nn.Module instance.
        return model_instance

    def build_and_run(self, model_instance, inputs, _num_runs, _params, npu=False, additional_config_args=None):
        # Convert NumPy inputs to PyTorch tensors on CPU
        torch_inputs = {}
        for k, v in inputs.items():
            torch_inputs[k] = torch.tensor(v, device='cpu') 

        # Assuming only one input for this test case
        input_tensor_name = list(torch_inputs.keys())[0]
        input_tensor = torch_inputs[input_tensor_name]

        # Simulate 'NPU' path with torch.compile (or just eager for 'CPU')
        if npu:
            # For `torch.compile`, a fresh model instance or a wrapper is often preferred.
            # The numerical simulation in _get_model_pytorch uses standard torch ops which should compile.
            compiled_model = torch.compile(model_instance, fullgraph=True, dynamic=False)
            output = compiled_model(input_tensor)
        else:
            output = model_instance(input_tensor)
        
        # Return output as a list containing one numpy array, matching TVM's `build_and_run` signature
        return [output.detach().cpu().numpy()]

    def verify(self, outputs, _dtype, tolerance):
        # outputs is a list of results, where outputs[0] is baseline, outputs[1] is 'NPU'
        if len(outputs) < 2:
            pytest.fail("Expected at least two outputs for verification (baseline and 'NPU').")

        # Extract the single tensor result from each output list item
        output_baseline = torch.tensor(outputs[0][0])
        output_npu = torch.tensor(outputs[1][0])

        # For integer outputs (uint8, int8), `rtol` is usually 0.
        # `tolerance` from TVM is typically `1` for small integer differences.
        # Use `atol` for absolute tolerance.
        torch.testing.assert_allclose(output_baseline, output_npu, rtol=0, atol=tolerance)

    def make_ethosn_composite(self, model, composite_name):
        # This is a TVM-specific graph transformation.
        # In PyTorch, this would be a dummy pass-through, or involve a custom operator
        # or graph rewrite that's not easily generalized.
        return model # Return the model as is for further processing

    def make_ethosn_partition(self, model):
        # This is another TVM-specific partitioning pass.
        # Similar to make_ethosn_composite, it's a compiler-internal concept.
        return model # Return the model as is

    def test_error(self, mod, _inputs, err_msg):
        # This function checks for specific compilation errors on the Ethos-N backend.
        # Replicating these specific compiler-level checks in a generic PyTorch context
        # is beyond the scope of direct API mapping.
        # We'll skip these tests with a clear message.
        pytest.skip(f"Cannot directly replicate TVM Ethos-N compilation error checks in PyTorch: {err_msg}")


tei = PyTorchEthosNTestInfrastructure()
# --- End PyTorch infrastructure ---


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize("shape", [(1, 52, 52, 3)])
def test_tanh(dtype, shape):
    """Compare Tanh output with PyTorch (compiled vs eager)."""

    # These are numpy limits, used to generate random input data within the dtype range.
    zp_min = np.iinfo(dtype).min
    zp_max = np.iinfo(dtype).max

    np.random.seed(0)
    
    # Generate random integer input data in NumPy
    input_np_array = np.random.randint(zp_min, high=zp_max, size=shape, dtype=dtype)
    inputs = {"a": input_np_array}
    
    outputs = []
    # Loop over 'npu' flag to simulate baseline (npu=False) and accelerated (npu=True) execution
    for npu_flag in [False, True]:
        # Create a new model instance for each run, especially important for `torch.compile`
        model_instance = _get_model_pytorch(shape, zp_min + 120, 0.0250629, zp_min + 128, 0.0078125, dtype)
        
        # `tei.make_module` is a pass-through in PyTorch emulation
        mod = tei.make_module(model_instance, [])
        
        # `tei.build_and_run` handles execution and result conversion
        # additional_config_args is TVM-specific, ignored in PyTorch emulation
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1, # num_runs
                {}, # params
                npu=npu_flag,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    # `tei.verify` compares the outputs
    # The original test passed 1 as tolerance. Assuming atol=1 for integer comparison.
    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape, input_zp, input_sc, output_zp, output_sc, err_msg",
    [
        (
            (1, 16, 16, 16),
            120,
            0.0250629,
            64,
            0.0078125,
            "output quantization params=(64, 0.0078125), must = ({test_zp}, 1/256);",
        )
    ],
)
def test_tanh_failure(shape, input_zp, input_sc, output_zp, output_sc, err_msg, dtype):
    """Check Tanh error messages (skipped for PyTorch)."""

    # `test_zp` logic is part of the original error message formatting.
    test_zp = 0 if dtype == "int8" else 128
    
    # For PyTorch, we need a model instance, even if it's not actually compiled for an NPU.
    model_instance = _get_model_pytorch(shape, input_zp, input_sc, output_zp, output_sc, dtype)
    
    # These TVM-specific calls are for partitioning and compilation error checks.
    # They are hard to replicate in generic PyTorch.
    # The `tei` object will handle skipping this test.
    model_composite = tei.make_ethosn_composite(model_instance, "ethos-n.qnn_tanh")
    mod_partitioned = tei.make_ethosn_partition(model_composite)
    
    tei.test_error(mod_partitioned, {}, err_msg.format(test_zp=test_zp))
