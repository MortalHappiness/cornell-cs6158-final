import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define helper to convert string dtype to torch.dtype
# TVM uses "uint8", "int8", "float32", "int32"
# PyTorch uses torch.quint8, torch.qint8, torch.float32, torch.int32 etc.
DTYPE_MAP = {
    "uint8": torch.quint8,
    "int8": torch.qint8,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}

def get_torch_qdtype(dtype_str):
    """Returns the PyTorch quantized dtype from a string."""
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype string for quantized tensor: {dtype_str}")
    return DTYPE_MAP[dtype_str]

# PyTorch Module for the QNN Leaky ReLU operation
class LeakyReluQNN(nn.Module):
    def __init__(self, output_scale, output_zero_point, dtype_str, alpha):
        super().__init__()
        # Store quantization parameters for the output
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point
        self.output_qdtype = get_torch_qdtype(dtype_str)
        self.alpha = alpha

    def forward(self, x_quantized_input_tensor):
        # x_quantized_input_tensor is assumed to be a quantized torch.Tensor (e.g., qint8, quint8).
        # PyTorch's torch.dequantize uses the scale and zero_point embedded in the input tensor.
        x_dequantized = torch.dequantize(x_quantized_input_tensor)
        
        # Apply Leaky ReLU (this is a float operation)
        x_leaky_relu = F.leaky_relu(x_dequantized, negative_slope=self.alpha)
        
        # Quantize the float result to the desired output quantization parameters
        output_quantized = torch.quantize_per_tensor(
            x_leaky_relu,
            scale=self.output_scale,
            zero_point=self.output_zero_point,
            dtype=self.output_qdtype,
        )
        return output_quantized

@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize("shape", [(1, 52, 52, 3), (1, 3, 8, 2)])
@pytest.mark.parametrize("alpha", [0.001, 0.5678])
def test_leaky_relu(dtype, shape, alpha):
    """Compare Leaky ReLU output with PyTorch/TorchInductor."""

    np.random.seed(0) # For reproducibility

    iinfo = np.iinfo(dtype)
    zp_min = iinfo.min
    zp_max = iinfo.max
    input_zp = zp_min + 120
    input_sc = 0.0068132
    output_zp = zp_min + 128
    output_sc = 0.0078125

    # 1. Prepare input NumPy data (raw quantized integer values)
    # The `high` argument for np.random.randint is exclusive, so add 1 to include zp_max.
    input_np_raw_quantized = np.random.randint(zp_min, high=zp_max + 1, size=shape, dtype=dtype)
    
    # 2. Create a quantized PyTorch input tensor
    # torch.quantize_per_tensor expects a float32 tensor as input.
    input_float_tensor_for_quantization = torch.from_numpy(input_np_raw_quantized.astype(np.float32))
    
    input_q_tensor_pytorch = torch.quantize_per_tensor(
        input_float_tensor_for_quantization,
        scale=input_sc,
        zero_point=input_zp,
        dtype=get_torch_qdtype(dtype)
    )
    
    # 3. Instantiate and compile the PyTorch model for TorchInductor
    model = LeakyReluQNN(output_sc, output_zp, dtype, alpha)
    compiled_model = torch.compile(model)

    # 4. Run the compiled PyTorch model
    output_q_tensor_pytorch = compiled_model(input_q_tensor_pytorch)
    
    # 5. Dequantize the PyTorch output for comparison in float precision
    output_float_pytorch = torch.dequantize(output_q_tensor_pytorch).cpu().numpy()

    # 6. Calculate the expected "golden" output using pure NumPy/float logic
    # This simulates the QNN operation directly in float precision, matching TVM's behavior.
    # a. Dequantize the raw input NumPy array to float
    input_float_golden = (input_np_raw_quantized.astype(np.float32) - input_zp) * input_sc
    
    # b. Apply Leaky ReLU in float
    leaky_relu_float_golden = np.where(input_float_golden > 0, input_float_golden, input_float_golden * alpha)
    
    # c. Quantize the float result using the output quantization parameters
    # This involves scaling, shifting by zero_point, rounding, and clamping to the integer range
    quantized_golden_values = np.round(leaky_relu_float_golden / output_sc + output_zp)
    
    # Clamp to the valid range of the output integer dtype (e.g., -128 to 127 for int8)
    quantized_golden_values = np.clip(quantized_golden_values, zp_min, zp_max).astype(dtype)
    
    # d. Dequantize this "golden" quantized integer array back to float for final comparison
    expected_output_float_golden = (quantized_golden_values.astype(np.float32) - output_zp) * output_sc

    # 7. Verify the results using NumPy's assert_allclose
    np.testing.assert_allclose(output_float_pytorch, expected_output_float_golden, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("dtype", ["int8"])
@pytest.mark.parametrize("shape", [(1, 14, 14, 2)])
@pytest.mark.parametrize("alpha", [-1.34, 2.32, 1, 0])
def test_leaky_relu_unsupported_alpha(dtype, shape, alpha):
    """Test unsupported values of alpha (<= 0, >= 1) in Leaky ReLU for Ethos-N.

    PyTorch's F.leaky_relu does not raise an error for negative_slope (alpha) <= 0 or >= 1.
    This test case checks for a backend-specific validation in TVM Ethos-N.
    Therefore, it is marked as xfail in a generic PyTorch/TorchInductor context,
    as the error condition itself is not present in PyTorch's `leaky_relu` operator.
    If TorchInductor were to implement a similar backend-specific validation, this test
    would need to be adapted to catch that specific error.
    """
    pytest.xfail(
        "PyTorch's F.leaky_relu does not raise an error for alpha values outside (0, 1). "
        "This test case checks for backend-specific validation in TVM Ethos-N. "
        "The operation would execute successfully in PyTorch."
    )

    # The setup for the unsupported case would look like this if PyTorch *did* raise an error:
    # iinfo = np.iinfo(dtype)
    # zp_min = iinfo.min
    #
    # input_zp = zp_min + 120
    # input_sc = 0.0068132
    # output_zp = zp_min + 128
    # output_sc = 0.0078125
    #
    # input_np_raw_quantized = np.random.randint(iinfo.min, high=iinfo.max + 1, size=shape, dtype=dtype)
    # input_float_tensor_for_quantization = torch.from_numpy(input_np_raw_quantized.astype(np.float32))
    # input_q_tensor_pytorch = torch.quantize_per_tensor(
    #     input_float_tensor_for_quantization,
    #     scale=input_sc,
    #     zero_point=input_zp,
    #     dtype=get_torch_qdtype(dtype)
    # )
    #
    # # The expected error message from TVM
    # expected_error_msg = f"leaky relu alpha must be less than 1 and greater than 0, but was {alpha}"
    #
    # # To replicate TVM's `tei.test_error`, you would wrap the problematic call in pytest.raises
    # # However, as noted, F.leaky_relu does not raise for these alpha values.
    # # For example, if a custom TorchInductor pass were to add this validation:
    # # with pytest.raises(RuntimeError, match=re.escape(expected_error_msg)): # use re.escape if message contains special regex chars
    # #    model = LeakyReluQNN(output_sc, output_zp, dtype, alpha)
    # #    compiled_model = torch.compile(model)
    # #    _ = compiled_model(input_q_tensor_pytorch)
    # pass # Removed actual execution due to xfail
