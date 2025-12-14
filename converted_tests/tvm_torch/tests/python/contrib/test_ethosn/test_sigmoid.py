import pytest
import numpy as np
import torch
import torch.nn.functional as F

# Helper to map string dtypes to torch quantized dtypes
_TORCH_QINT_DTYPE_MAP = {
    "uint8": torch.quint8,
    "int8": torch.qint8,
}

def _numpy_to_torch_qint_dtype(np_dtype_str):
    return _TORCH_QINT_DTYPE_MAP.get(np_dtype_str, None)


def _get_pytorch_quantized_sigmoid_model_fn(output_scale, output_zero_point, output_dtype_str):
    """
    Returns a callable that simulates the TVM Relay QNN Sigmoid pattern:
    (quantized input) -> dequantize -> sigmoid (float) -> quantize (to target output params).
    It expects an already quantized tensor as input.
    """
    torch_qout_dtype = _numpy_to_torch_qint_dtype(output_dtype_str)
    if torch_qout_dtype is None:
        raise ValueError(f"Unsupported quantized output dtype: {output_dtype_str}")

    def model_fn(q_input_tensor):
        # 1. Dequantize input (q_input_tensor carries its own scale and zero_point)
        dequantized_input = torch.dequantize(q_input_tensor)
        
        # 2. Apply float sigmoid
        sigmoid_output_float = torch.sigmoid(dequantized_input)
        
        # 3. Quantize the float output to the specified output parameters
        quantized_output = torch.quantize_per_tensor(
            sigmoid_output_float,
            scale=output_scale,
            zero_point=output_zero_point,
            dtype=torch_qout_dtype,
        )
        return quantized_output
    return model_fn


# Original `requires_ethosn` and other TVM-specific decorators are removed/skipped.
# This test now validates the functional behavior of quantized sigmoid in PyTorch.
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 16, 16, 16),
        (1, 8, 8),
    ],
)
def test_sigmoid(dtype, shape):
    """Compare quantized Sigmoid output with a NumPy reference in PyTorch."""
    np.random.seed(0)

    # Determine quantization parameters based on dtype
    if dtype == "uint8":
        input_zp = 0
        output_zp = 0
    else:  # dtype == "int8"
        input_zp = 127
        output_zp = -128
    
    input_sc = 0.02
    output_sc = 1.0 / 256.0
    
    # 1. Prepare NumPy input data
    # TVM uses np.random.randint for the raw quantized values.
    # PyTorch's quantize_per_tensor expects float input.
    raw_np_data = np.random.randint(
        np.iinfo(dtype).min, # Use the string dtype for np.iinfo
        np.iinfo(dtype).max + 1,
        size=shape,
        dtype=dtype, # Use the string dtype for numpy array creation
    )
    
    # Create the quantized input tensor for PyTorch
    # torch.quantize_per_tensor expects a float tensor as input for quantization
    float_input_for_quantization = torch.tensor(raw_np_data.astype(np.float32))
    
    q_input_tensor = torch.quantize_per_tensor(
        float_input_for_quantization,
        scale=input_sc,
        zero_point=input_zp,
        dtype=_numpy_to_torch_qint_dtype(dtype),
    )

    # 2. Get the PyTorch quantized sigmoid model callable
    pytorch_sigmoid_q_model_fn = _get_pytorch_quantized_sigmoid_model_fn(
        output_scale=output_sc,
        output_zero_point=output_zp,
        output_dtype_str=dtype,
    )

    # 3. Run the PyTorch quantized model.
    actual_output_q_first_run = pytorch_sigmoid_q_model_fn(q_input_tensor)
    # The original TVM test collects two outputs (NPU=False and NPU=True).
    # Since we can't run on actual Ethos-N, we simulate two runs by cloning the output.
    actual_outputs_q = [actual_output_q_first_run, actual_output_q_first_run.clone()]

    # 4. Generate reference (expected) float output using NumPy
    # Simulate the entire dequantize -> sigmoid -> quantize -> dequantize process with NumPy
    np_input_dequantized = (raw_np_data.astype(np.float32) - input_zp) * input_sc
    np_sigmoid_float = 1 / (1 + np.exp(-np_input_dequantized))
    
    np_quantized_intermediate = np.round(np_sigmoid_float / output_sc + output_zp)
    np_quantized_clamped = np.clip(
        np_quantized_intermediate, 
        np.iinfo(dtype).min, 
        np.iinfo(dtype).max
    ).astype(dtype) # Cast to the output quantized NumPy dtype
    
    np_expected_dequantized = (np_quantized_clamped.astype(np.float32) - output_zp) * output_sc

    # 5. Verify results
    # Dequantize PyTorch outputs for comparison with float reference
    dequantized_pytorch_outputs = [torch.dequantize(o) for o in actual_outputs_q]

    # Verify that the two "runs" (cloned outputs) produce identical float values
    torch.testing.assert_close(dequantized_pytorch_outputs[0], dequantized_pytorch_outputs[1], rtol=1e-5, atol=1e-5)
    
    # Verify against the NumPy reference. Relax tolerance for quantized operations.
    torch.testing.assert_close(
        dequantized_pytorch_outputs[0],
        torch.tensor(np_expected_dequantized, dtype=torch.float32),
        rtol=1e-3, 
        atol=1e-3,
    )


@pytest.mark.skip(reason="This test checks TVM Ethos-N backend compilation failure modes, which are not applicable to PyTorch.")
@pytest.mark.parametrize(
    "shape,input_zp,input_sc,output_zp,output_sc,err_msg",
    [
        ((2, 4, 4, 4), 64, 0.2, 0, 1 / 256, "batch size=2, batch size must = 1"),
        (
            (1, 4, 4, 4),
            64,
            0.2,
            3,
            1,
            "output quantization params=(3, 1), must = (0, 1/256)",
        ),
    ],
)
def test_sigmoid_failure(shape, input_zp, input_sc, output_zp, output_sc, err_msg):
    """TODO: This test checks Ethos-N specific backend compilation errors.
    Direct translation to PyTorch is not straightforward as PyTorch's functional APIs
    don't typically have such strict compile-time checks for quantization parameters
    or batch size limitations directly in the functional call.
    """
    # Placeholder to ensure the function is runnable and explicitly fails
    _ = shape, input_zp, input_sc, output_zp, output_sc, err_msg
    pytest.fail("Skipped: Ethos-N specific compilation failure test.")
