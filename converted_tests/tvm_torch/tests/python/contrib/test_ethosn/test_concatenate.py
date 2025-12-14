import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Helper to define the PyTorch callable that simulates the qnn.concatenate operation.
# This callable takes raw NumPy inputs, quantizes them using provided parameters,
# performs the dequantize-concatenate-quantize sequence, and returns a quantized PyTorch tensor.
def _get_qnn_concatenate_pytorch_callable(shapes, tvm_dtype_str, axis, 
                                       input_scales_vals, input_zero_points_vals, 
                                       output_scale_val, output_zero_point_val):
    
    # Determine the PyTorch quantized dtype for the output.
    # TVM's qnn.concatenate in this context targets uint8/int8 output.
    q_output_torch_dtype = {
        "uint8": torch.quint8,
        "int8": torch.qint8,
        "int32": torch.qint32, # Though not directly used for output in this test's params, useful for completeness
    }.get(tvm_dtype_str)

    if q_output_torch_dtype is None:
        # This path is hit for unsupported dtypes like 'int16' in failure tests.
        raise ValueError(f"Unsupported quantized output dtype for PyTorch: {tvm_dtype_str}")

    def qnn_concatenate_func(raw_inputs_dict):
        q_tensors = []
        for i, _ in enumerate(shapes):
            input_name = "in" + str(i)
            np_data = raw_inputs_dict[input_name]
            
            # Convert raw integer NumPy data to a float32 PyTorch tensor.
            # PyTorch's `quantize_per_tensor` expects a float tensor as input.
            float_data = torch.from_numpy(np_data.astype(np.float32))
            
            # Determine the PyTorch quantized dtype for the input tensor.
            q_input_torch_dtype = {
                "uint8": torch.quint8,
                "int8": torch.qint8,
            }.get(tvm_dtype_str)

            if q_input_torch_dtype is None:
                # This path is hit for unsupported dtypes like 'int16' in failure tests.
                raise ValueError(f"Unsupported quantized input dtype for PyTorch: {tvm_dtype_str}")

            # Quantize the float tensor with its specific input scale and zero point.
            q_tensor_input = torch.quantize_per_tensor(
                float_data,
                scale=input_scales_vals[i],
                zero_point=input_zero_points_vals[i],
                dtype=q_input_torch_dtype
            )
            q_tensors.append(q_tensor_input)

        # Dequantize all input tensors to float32, perform concatenation, then requantize the result.
        float_tensors_deq = [torch.dequantize(t) for t in q_tensors]
        float_concat = torch.cat(float_tensors_deq, dim=axis)
        
        # Requantize the concatenated float tensor to the specified output quantized dtype.
        quantized_output = torch.quantize_per_tensor(
            float_concat,
            scale=output_scale_val,
            zero_point=output_zero_point_val,
            dtype=q_output_torch_dtype
        )
        return quantized_output
    return qnn_concatenate_func


# Helper to generate raw NumPy inputs, which represent the integer values of quantized data.
def _get_inputs(shapes, tvm_dtype_str):
    inputs = {}
    for i, shape in enumerate(shapes):
        inputs["in" + str(i)] = np.random.randint(
            np.iinfo(tvm_dtype_str).min, np.iinfo(tvm_dtype_str).max + 1, size=shape, dtype=tvm_dtype_str
        )
    return inputs


# Helper to compute the reference output using NumPy, mimicking the dequantize-float_op-quantize sequence
# of the qnn.concatenate operation.
def _get_reference_output(inputs_raw, shapes, tvm_dtype_str, axis, 
                          input_scales_vals, input_zero_points_vals, 
                          output_scale_val, output_zero_point_val):
    
    float_inputs_dequantized = []
    for i, _ in enumerate(shapes):
        input_name = "in" + str(i)
        np_data = inputs_raw[input_name]
        
        # Dequantize each input NumPy array to float32: float_val = (quantized_int_val - zero_point) * scale
        dequantized_data = (np_data.astype(np.float32) - input_zero_points_vals[i]) * input_scales_vals[i]
        float_inputs_dequantized.append(dequantized_data)

    # Perform float concatenation using NumPy.
    float_concat_ref = np.concatenate(float_inputs_dequantized, axis=axis)

    # Requantize the result back to integer type: quantized_int_val = round(float_val / output_scale) + output_zero_point
    q_min = np.iinfo(tvm_dtype_str).min
    q_max = np.iinfo(tvm_dtype_str).max
    
    quantized_ref = np.round(float_concat_ref / output_scale_val) + output_zero_point_val
    quantized_ref = np.clip(quantized_ref, q_min, q_max).astype(tvm_dtype_str)
    
    return quantized_ref


# Remove @requires_ethosn decorator as it is TVM-specific.
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shapes,axis",
    [
        ([(1, 4), (1, 6)], 1),
        ([(1, 16, 4), (1, 16, 4)], 1),
        ([(1, 25, 4, 16)] * 3, 3),
        ([(1, 25, 4, 16), (1, 25, 5, 16), (1, 25, 6, 16)], 2),
        ([(1, 4), (1, 6)], -1),
        ([(1, 16, 4), (1, 16, 4)], -2),
    ],
)
def test_concatenate(dtype, shapes, axis):
    np.random.seed(0)

    # Define constant quantization parameters as in TVM's _get_model function.
    # These values are derived from `relay.const(1, "int32")` and `relay.const(0.5, "float32")`.
    zeroi_val = 1 
    zerof_val = 0.5
    
    input_scales_vals = [zerof_val] * len(shapes)
    input_zero_points_vals = [zeroi_val] * len(shapes)
    output_scale_val = zerof_val
    output_zero_point_val = zeroi_val

    # 1. Get raw NumPy inputs (representing quantized integer values).
    inputs_raw = _get_inputs(shapes, dtype)

    # 2. Compute the reference output using NumPy, mimicking TVM's qnn.concatenate behavior.
    ref_output_np = _get_reference_output(
        inputs_raw, shapes, dtype, axis, 
        input_scales_vals, input_zero_points_vals, 
        output_scale_val, output_zero_point_val
    )
    
    # 3. Create the PyTorch callable that performs the qnn.concatenate operation.
    pytorch_qnn_concatenate_op = _get_qnn_concatenate_pytorch_callable(
        shapes, dtype, axis,
        input_scales_vals, input_zero_points_vals,
        output_scale_val, output_zero_point_val
    )

    # 4. Run the PyTorch model without compilation (baseline).
    output_q_torch_uncompiled = pytorch_qnn_concatenate_op(inputs_raw)
    # Convert the quantized PyTorch tensor to its integer representation and then to NumPy for comparison.
    output_int_torch_uncompiled = output_q_torch_uncompiled.int_repr().numpy()
    
    # Compare uncompiled PyTorch output with the NumPy reference.
    # For quantized outputs, exact equality of integer representations is generally expected.
    np.testing.assert_array_equal(output_int_torch_uncompiled, ref_output_np)
    
    # 5. If CUDA is available, run the PyTorch model compiled with TorchInductor.
    if torch.cuda.is_available():
        # Compile the callable with `fullgraph=True` to allow TorchInductor to optimize the entire function.
        compiled_pytorch_qnn_concatenate_op = torch.compile(pytorch_qnn_concatenate_op, fullgraph=True)
        output_q_torch_compiled = compiled_pytorch_qnn_concatenate_op(inputs_raw)
        output_int_torch_compiled = output_q_torch_compiled.int_repr().numpy()
        
        # Compare compiled PyTorch output with the NumPy reference.
        np.testing.assert_array_equal(output_int_torch_compiled, ref_output_np)
        
        # Additionally, compare compiled vs. uncompiled output for consistency.
        np.testing.assert_array_equal(output_int_torch_compiled, output_int_torch_uncompiled)


@pytest.mark.parametrize(
    "shapes,dtype,axis,err_msg", # `err_msg` is the TVM/Ethos-N specific message, not directly matched in PyTorch.
    [
        # Case 1: 5-dimensional input. Ethos-N has a <= 4 dim restriction.
        ([(1, 4, 4, 4, 4), (1, 4, 4, 4, 4)], "uint8", 1, "dimensions=5, dimensions must be <= 4;"),
        # Case 2: Channel count not a multiple of 16. Ethos-N restriction.
        (
            [(1, 4, 4, 4), (1, 4, 4, 4)],
            "uint8",
            3,
            "Concatenation along the channels dimension (axis 3) "
            "requires input tensors with a multiple of 16 channels;",
        ),
        # Case 3: Unsupported dtype 'int16'. PyTorch's `quantize_per_tensor` does not support `qint16`.
        (
            [(1, 4, 4, 4), (1, 4, 4, 4)],
            "int16",
            2,
            "dtype='int16', dtype must be either uint8, int8 or int32; dtype='int16', "
            "dtype must be either uint8, int8 or int32;",
        ),
        # Case 4: Batch size > 1. Ethos-N restriction.
        (
            [(2, 4, 4, 4), (2, 4, 4, 4)],
            "uint8",
            2,
            "batch size=2, batch size must = 1; batch size=2, batch size must = 1;",
        ),
        # Case 5: Concatenation along batch axis. Ethos-N restriction.
        (
            [(1, 4, 4, 4)],
            "uint8",
            0,
            "Concatenation cannot be performed along batch axis (axis 0);",
        ),
    ],
)
def test_concatenate_failure(shapes, dtype, axis, err_msg):
    # TODO: The original TVM test asserts specific failure conditions related to the Ethos-N backend,
    # such as dimension limits, channel alignment, batch size restrictions, and concatenation axis limitations.
    # PyTorch's native `torch.cat` and quantization functions are generally more permissive and
    # do not enforce these specific backend-dependent constraints.
    # Therefore, most of these failure cases will not raise an error in PyTorch for the stated reasons,
    # leading to a semantic divergence from the original TVM test's intent to assert a failure.

    np.random.seed(0)

    zeroi_val = 1
    zerof_val = 0.5
    
    input_scales_vals = [zerof_val] * len(shapes)
    input_zero_points_vals = [zeroi_val] * len(shapes)
    output_scale_val = zerof_val
    output_zero_point_val = zeroi_val

    inputs_raw = _get_inputs(shapes, dtype)

    # For `dtype='int16'`, PyTorch's `quantize_per_tensor` does not support `qint16` as a target dtype,
    # and our `_get_qnn_concatenate_pytorch_callable` will correctly raise a ValueError.
    # This specific case aligns with an expected failure, albeit for a PyTorch-native reason.
    if dtype == "int16":
        expected_exception_type = ValueError
        expected_msg_fragment = "Unsupported quantized"
    else:
        # For other cases (5D input, non-multiple-of-16 channels, batch_size > 1, concat along batch axis),
        # PyTorch's `torch.cat` is more flexible and will generally succeed where Ethos-N would fail.
        # This test cannot be directly converted to assert failure for the original backend-specific reasons.
        # We explicitly skip these tests with an explanation of the behavioral divergence.
        pytest.skip(
            f"This failure test case (original reason: '{err_msg}') is specific to "
            f"Ethos-N backend restrictions that are not enforced by PyTorch's native operations. "
            f"PyTorch's `torch.cat` is more permissive and would likely succeed here, "
            f"causing the original failure assertion to be invalid for PyTorch."
        )
        return # Skip the rest of the test function

    # Execute the test for cases where a PyTorch-native error is expected (e.g., `dtype='int16'`).
    with pytest.raises(expected_exception_type) as excinfo:
        # Attempt to create the PyTorch callable for qnn.concatenate.
        # This might fail during callable creation itself (e.g., due to unsupported dtypes)
        # or during its execution.
        pytorch_qnn_concatenate_op = _get_qnn_concatenate_pytorch_callable(
            shapes, dtype, axis,
            input_scales_vals, input_zero_points_vals,
            output_scale_val, output_zero_point_val
        )
        
        # Run the PyTorch model (uncompiled).
        pytorch_qnn_concatenate_op(inputs_raw)

        # If CUDA is available, also try the compiled version.
        if torch.cuda.is_available():
            compiled_pytorch_qnn_concatenate_op = torch.compile(pytorch_qnn_concatenate_op, fullgraph=True)
            compiled_pytorch_qnn_concatenate_op(inputs_raw)
    
    # Assert that the captured exception's message contains the expected fragment.
    assert expected_msg_fragment in str(excinfo.value)
    # The exact error message text or type might still differ slightly from TVM's,
    # but asserting a relevant fragment verifies the failure.
