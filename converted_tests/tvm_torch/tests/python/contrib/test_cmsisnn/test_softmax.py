import itertools
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch.testing

# --- Helper functions ---
def get_range_for_dtype_str(dtype_str):
    """Returns the min and max values for a given dtype string."""
    if dtype_str in ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]:
        iinfo = np.iinfo(dtype_str)
        return iinfo.min, iinfo.max
    elif dtype_str in ["float16", "float32", "float64"]:
        finfo = np.finfo(dtype_str)
        return finfo.min, finfo.max
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")

def get_torch_q_dtype(dtype_str):
    """Maps dtype string to PyTorch quantized dtype."""
    if dtype_str == "int8":
        return torch.qint8
    elif dtype_str == "uint8":
        return torch.quint8
    else:
        raise ValueError(f"Unsupported quantized dtype for PyTorch: {dtype_str}")

# --- PyTorch Model Equivalent ---
class QnnSoftmax(torch.nn.Module):
    """
    PyTorch equivalent of TVM's Relay QNN Softmax model logic.
    Assumes quantized_input_int_repr contains the integer representation
    (e.g., int8 numpy array converted to torch.Tensor) of the quantized tensor.
    """
    def __init__(self, shape, in_dtype_str, out_dtype_str, in_zero_point, in_scale, out_zero_point, out_scale):
        super().__init__()
        self.in_scale = in_scale
        self.in_zero_point = in_zero_point
        self.out_scale = out_scale
        self.out_zero_point = out_zero_point
        
        # Determine the PyTorch quantized dtype for the output
        self.torch_out_q_dtype = get_torch_q_dtype(out_dtype_str)

    def forward(self, quantized_input_int_repr):
        # 1. Simulate TVM's qnn.dequantize operation
        # Dequantization formula: (quant_val - zero_point) * scale
        # The input `quantized_input_int_repr` is the tensor of integer values.
        float_input = (quantized_input_int_repr.to(torch.float32) - self.in_zero_point) * self.in_scale

        # 2. Apply Softmax (on float values).
        # In TVM Relay, if 'axis' is not specified, it often defaults to the last dimension.
        # This is common in many deep learning frameworks for classification outputs.
        softmax_output_float = F.softmax(float_input, dim=-1)

        # 3. Simulate TVM's qnn.quantize operation
        # PyTorch's `quantize_per_tensor` takes a float input, along with the desired
        # output scale, zero_point, and quantized dtype.
        quantized_output = torch.quantize_per_tensor(
            softmax_output_float,
            scale=self.out_scale,
            zero_point=self.out_zero_point,
            dtype=self.torch_out_q_dtype
        )
        return quantized_output

# --- Test functions ---

# Removed TVM specific decorators like @tvm.testing.requires_cmsisnn or @skip_if_no_reference_system
@pytest.mark.parametrize(["zero_point", "scale"], [[33, 0.256], [-64, 0.0128]])
@pytest.mark.parametrize(
    "compiler_cpu, cpu_flags", [("cortex-m55", "+nomve"), ("cortex-m55", ""), ("cortex-m7", "")]
)
# Note: compiler_cpu and cpu_flags are retained for parameterization to match the original
# test structure but are not used in the PyTorch test logic, as they are TVM/CMSIS-NN specific.
def test_op_int8(zero_point, scale, compiler_cpu, cpu_flags):
    """Tests int8 QNN Softmax functionality for PyTorch."""
    dtype_str = "int8"
    shape = [1, 16, 16, 3]
    # Default output quantization parameters as specified in original `make_model`
    out_zero_point = -128
    out_scale = 1.0 / 256

    # Instantiate the PyTorch QNN Softmax module
    qnn_softmax_model = QnnSoftmax(
        shape, dtype_str, dtype_str, zero_point, scale, out_zero_point, out_scale
    )

    # Generate input data as raw integer values (e.g., int8)
    in_min, in_max = get_range_for_dtype_str(dtype_str)
    np.random.seed(0)
    # np.random.randint generates values in [low, high), so +1 to include high
    input_data_np = np.random.randint(in_min, high=in_max + 1, size=shape, dtype=dtype_str)
    
    # Convert input_data_np to a PyTorch tensor, representing the integer contents
    torch_input_int_repr = torch.from_numpy(input_data_np)

    # Calculate reference (float) output by performing dequantization, softmax, then requantization and dequantization
    # 1. Dequantize the input_data_np using its scale and zero_point
    float_input_data_for_ref = (input_data_np.astype(np.float32) - zero_point) * scale
    
    # 2. Apply softmax on the float data
    ref_float_output_softmax = F.softmax(torch.from_numpy(float_input_data_for_ref), dim=-1)

    # 3. Quantize the float softmax output using the specified output parameters
    ref_quantized_output_tensor = torch.quantize_per_tensor(
        ref_float_output_softmax,
        scale=out_scale,
        zero_point=out_zero_point,
        dtype=qnn_softmax_model.torch_out_q_dtype
    )
    
    # 4. Dequantize the reference quantized output for comparison with floating point tolerance
    ref_dequantized_output = ref_quantized_output_tensor.dequantize()

    # Get actual output from the PyTorch QNN model
    actual_quantized_output = qnn_softmax_model(torch_input_int_repr)
    actual_dequantized_output = actual_quantized_output.dequantize()

    # Compare results with a relative and absolute tolerance
    torch.testing.assert_allclose(actual_dequantized_output, ref_dequantized_output, rtol=1e-4, atol=1e-4)


def parameterize_for_invalid_model(test):
    """Generates parameters for non-int8 input/output or non-default out_scale/out_zero_point
    that the original CMSIS-NN integration would not offload.
    """
    in_dtype = ["uint8", "int8"]
    out_dtype = ["uint8", "int8"]
    zero_point = [-128, 64]
    scale = [1.0 / 256, 0.2]
    out_zero_point = [-128, 33]
    out_scale = [1.0 / 256, 0.2]
    all_combinations = itertools.product(
        in_dtype, out_dtype, zero_point, scale, out_zero_point, out_scale
    )
    # Filter for combinations that are *not* the 'valid' int8 case for CMSIS-NN's direct offload
    all_combinations = filter(
        lambda params: not (
            params[0] == "int8"
            and params[1] == "int8"
            and params[4] == -128
            and params[5] == 1.0 / 256
        ),
        all_combinations,
    )
    # Convert filter object to list, as pytest.mark.parametrize expects a sequence
    return pytest.mark.parametrize(
        ["in_dtype", "out_dtype", "zero_point", "scale", "out_zero_point", "out_scale"],
        list(all_combinations),
    )(test)


@parameterize_for_invalid_model
# Removed TVM specific decorator: @tvm.testing.requires_cmsisnn
def test_invalid_parameters(in_dtype, out_dtype, zero_point, scale, out_zero_point, out_scale):
    """
    Tests for parameter combinations that would be considered 'invalid' for direct CMSIS-NN
    offload in TVM, meaning they would fall back to a generic (non-accelerated) implementation.
    In PyTorch, we verify that these combinations execute without errors using its native
    quantization capabilities.
    """
    shape = [1, 16, 16, 3]

    try:
        qnn_softmax_model = QnnSoftmax(
            shape, in_dtype, out_dtype, zero_point, scale, out_zero_point, out_scale
        )

        in_min, in_max = get_range_for_dtype_str(in_dtype)
        np.random.seed(0)
        input_data_np = np.random.randint(in_min, high=in_max + 1, size=shape, dtype=in_dtype)
        torch_input_int_repr = torch.from_numpy(input_data_np)

        actual_quantized_output = qnn_softmax_model(torch_input_int_repr)
        
        # Verify that the output tensor's dtype matches the expected quantized dtype
        expected_torch_q_dtype = get_torch_q_dtype(out_dtype)
        assert actual_quantized_output.dtype == expected_torch_q_dtype, \
            f"Expected output dtype {expected_torch_q_dtype}, but got {actual_quantized_output.dtype}"
        
        # Ensure that the quantized output can be dequantized, implying a valid quantized tensor
        _ = actual_quantized_output.dequantize()

    except Exception as e:
        pytest.fail(f"QnnSoftmax model failed with parameters (in_dtype={in_dtype}, out_dtype={out_dtype}, "
                    f"in_zp={zero_point}, in_scale={scale}, out_zp={out_zero_point}, out_scale={out_scale}): {e}")

# The original `if __name__ == "__main__": tvm.testing.main()` is omitted as it's typically
# not needed for pytest-driven execution of test files.
