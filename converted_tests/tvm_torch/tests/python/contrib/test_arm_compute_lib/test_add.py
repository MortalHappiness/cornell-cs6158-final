import numpy as np
import torch
import pytest
import functools

# Helper to map TVM dtype strings to torch dtypes
_TVM_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}

# Adjusted _qnn_params: use raw Python numbers instead of relay.const
_qnn_params = {
    "lhs_scale": 0.0156863,
    "lhs_zero_point": 127,
    "rhs_scale": 0.0117647,
    "rhs_zero_point": 85,
    "output_scale": 0.0235294,
    "output_zero_point": 128,
}

# Mock TVM relay ops for test compatibility without importing TVM
class MockRelayOp:
    def __init__(self, name, torch_op=None):
        self.name = name
        self.torch_op = torch_op
    def __call__(self, *args, **kwargs):
        # In PyTorch context, this object serves as an identifier for _get_model
        return self
    def __str__(self):
        return f"MockRelayOp('{self.name}')"
    def __repr__(self):
        return self.__str__()

class MockRelay:
    def add(self, *args, **kwargs):
        return MockRelayOp("add", torch.add)
    
    class qnn:
        @staticmethod
        def op_add(*args, **kwargs):
            return MockRelayOp("qnn.op.add") # This specific op is handled by composite logic

    # Mimic the structure `relay.qnn.op.add`
    qnn = type('qnn_mod', (object,), {'op': type('qnn_op_mod', (object,), {'add': qnn.op_add})})()


# Instantiate the mock relay object
relay = MockRelay()


def _get_model(shape, dtype, var_names_iter, op_identifier, op_params):
    # This function returns a callable that takes raw float inputs
    # and performs the operation, including any quantization/dequantization
    # if it's a QNN op.

    # Determine if it's a QNN operation based on the op_identifier's name
    is_qnn_op = "qnn.op." in op_identifier.name

    if is_qnn_op:
        # Determine the quantized dtype for inputs/outputs
        q_dtype_name = _TVM_TO_TORCH_DTYPE[dtype]
        if q_dtype_name == torch.uint8:
            actual_q_dtype = torch.quint8
        elif q_dtype_name == torch.int8:
            actual_q_dtype = torch.qint8
        else:
            raise ValueError(f"Unsupported QNN dtype for PyTorch quantization: {dtype}")

        def qnn_add_model(a_float, b_float):
            # 1. Quantize inputs from float to specified quantized dtype based on op_params
            q_a = torch.quantize_per_tensor(
                a_float,
                op_params["lhs_scale"],
                op_params["lhs_zero_point"],
                actual_q_dtype,
            )
            q_b = torch.quantize_per_tensor(
                b_float,
                op_params["rhs_scale"],
                op_params["rhs_zero_point"],
                actual_q_dtype,
            )

            # 2. Dequantize to float for computation (as per mapping table for qnn.add)
            float_lhs = torch.dequantize(q_a)
            float_rhs = torch.dequantize(q_b)

            # 3. Perform float addition
            float_output = torch.add(float_lhs, float_rhs)

            # 4. Quantize output back to specified output q_dtype based on op_params
            return torch.quantize_per_tensor(
                float_output,
                op_params["output_scale"],
                op_params["output_zero_point"],
                actual_q_dtype,
            )
        
        return qnn_add_model
    else: # Float op (e.g., relay.add -> torch.add)
        return lambda a, b: op_identifier.torch_op(a, b) # Direct call for float operations

# Mock infrastructure functions
class Device:
    def __init__(self, name="cpu"):
        self.name = name
    def load(self, config_file):
        pass
    def __call__(self):
        return self # Return self to allow .name access directly

def skip_runtime_test():
    # In PyTorch, tests are typically always runnable if device is available.
    # We can add actual device check later if needed.
    return False

def skip_codegen_test():
    # Codegen tests are TVM-specific for IR inspection. No direct PyTorch equivalent.
    return True

# This `build_and_run` simulates the TVM runtime execution for PyTorch
def build_and_run(model_callable, inputs_dict_np, _, __, device_obj, enable_acl):
    # model_callable is the function returned by _get_model, which handles PyTorch ops
    # inputs_dict_np contains numpy arrays from random generation

    # Convert numpy inputs to torch tensors and move to device
    torch_inputs = {
        k: torch.tensor(v, dtype=_TVM_TO_TORCH_DTYPE[inputs_dict_np[k].dtype.name], device=device_obj.name)
        for k, v in inputs_dict_np.items()
    }

    # Run the model callable
    result = model_callable(torch_inputs["a"], torch_inputs["b"])

    # If the result is a quantized tensor, dequantize it for comparison with NumPy
    if isinstance(result, torch.Tensor) and result.is_quantized:
        result = torch.dequantize(result)

    return [result.cpu().numpy()] # Return as a list of numpy arrays, consistent with TVM

# This `verify` function simulates tvm.testing.verify for PyTorch tests
def verify(outputs, atol, rtol, config, verify_saturation):
    # outputs is a list of numpy arrays: [pytorch_result_np]
    # We need a NumPy reference for comparison.

    np_a = config["inputs"]["a"] # This is the original numpy array
    np_b = config["inputs"]["b"]

    op_identifier = config["operation"] # Use the mocked op object to check type
    is_qnn_op = "qnn.op." in op_identifier.name

    if is_qnn_op:
        # Simulate PyTorch's dequantize-add-quantize for NumPy reference
        lhs_scale = config["op_params"]["lhs_scale"]
        lhs_zero_point = config["op_params"]["lhs_zero_point"]
        rhs_scale = config["op_params"]["rhs_scale"]
        rhs_zero_point = config["op_params"]["rhs_zero_point"]
        output_scale = config["op_params"]["output_scale"]
        output_zero_point = config["op_params"]["output_zero_point"]

        q_val_dtype = np_a.dtype # Use the input numpy dtype for quantized values
        
        # Simulate torch.quantize_per_tensor for inputs (float -> quantized int representation)
        q_a_val_float = (np_a / lhs_scale + lhs_zero_point)
        q_a_val = np.round(q_a_val_float).clip(
            np.iinfo(q_val_dtype).min, np.iinfo(q_val_dtype).max
        ).astype(q_val_dtype)
        
        q_b_val_float = (np_b / rhs_scale + rhs_zero_point)
        q_b_val = np.round(q_b_val_float).clip(
            np.iinfo(q_val_dtype).min, np.iinfo(q_val_dtype).max
        ).astype(q_val_dtype)

        # Simulate torch.dequantize for inputs (quantized int representation -> float)
        dequant_a_np = (q_a_val.astype(np.float32) - lhs_zero_point) * lhs_scale
        dequant_b_np = (q_b_val.astype(np.float32) - rhs_zero_point) * rhs_scale

        # Perform float addition
        float_add_result_np = dequant_a_np + dequant_b_np

        # Simulate torch.quantize_per_tensor for output (float -> quantized int representation)
        quant_out_val_float = (float_add_result_np / output_scale + output_zero_point)
        quant_out_val = np.round(quant_out_val_float).clip(
            np.iinfo(q_val_dtype).min, np.iinfo(q_val_dtype).max
        ).astype(q_val_dtype)

        # Simulate torch.dequantize for output to get final float reference
        expected_np_output = (quant_out_val.astype(np.float32) - output_zero_point) * output_scale

    else: # Float operation
        expected_np_output = np_a + np_b

    # Compare the PyTorch result with the NumPy reference
    # Using `check_dtype=False` because `torch.dequantize` produces float32,
    # and the input numpy data might be uint8/int8 which would normally be float64 for numpy arithmetic,
    # but the actual numerical values should be close.
    torch.testing.assert_close(outputs[0], expected_np_output, rtol=rtol, atol=atol, check_dtype=False)


# Dummy for TVM codegen check, will be skipped
# This function is not called in PyTorch context, but helps with signature mapping.
def _get_expected_codegen(shape, dtype, op_name, qnn_params):
    return []

@pytest.mark.skipif(skip_codegen_test(), reason="TVM codegen verification has no direct PyTorch equivalent.")
def verify_codegen(*args, **kwargs):
    pass # This function will never be called due to the skipif, but kept for completeness.

# Dummy for ACL specific verification, will be skipped
def verify_saturation(*args, **kwargs):
    pytest.skip("ACL saturation verification has no direct PyTorch equivalent.")


@pytest.mark.parametrize("dtype, low, high, atol, rtol, op, op_params", [
    ("float32", -127, 128, 1e-7, 1e-7, relay.add(), {}), # Note: calling `relay.add()` to get the MockRelayOp instance
    ("uint8", 0, 255, 1.0, 0.0, relay.qnn.op.add(), _qnn_params), # Note: calling `relay.qnn.op.add()`
    ("int8", -127, 128, 1.0, 0.0, relay.qnn.op.add(), _qnn_params), # Note: calling `relay.qnn.op.add()`
])
def test_runtime_add(dtype, low, high, atol, rtol, op, op_params):
    _current_device = Device() # Instantiate Device for testing context

    if skip_runtime_test():
        pytest.skip("Runtime tests are skipped.")
        return

    np.random.seed(0)

    shape = (2, 2)
    # Generate inputs as numpy arrays first
    inputs_np = {
        "a": np.random.uniform(low, high, shape).astype(dtype),
        "b": np.random.uniform(low, high, shape).astype(dtype),
    }

    # Prepare the model callable (potentially wrapping quantization logic for QNN)
    var_names = iter(["a_var", "b_var"]) # Dummy var_names for signature, not used in PyTorch _get_model
    model_callable = _get_model(shape, dtype, var_names, op, op_params)

    outputs = []
    # Run the PyTorch version of the model (ACL=False path in original TVM test)
    # `build_and_run` returns a list of outputs, we take the first element (the only one)
    outputs.append(build_and_run(model_callable, inputs_np, 1, None, _current_device, enable_acl=False)[0])

    config = {
        "shape": shape,
        "dtype": dtype,
        "inputs": inputs_np, # Pass original numpy inputs to derive reference
        "operation": op,
        "op_params": op_params,
    }

    # Pass the list containing the single output to verify
    verify([outputs[0]], atol=atol, rtol=rtol, config=config, verify_saturation=False)


@pytest.mark.skipif(skip_codegen_test(), reason="TVM codegen test has no direct PyTorch equivalent")
def test_codegen_add():
    # This test is entirely TVM-specific and skipped.
    pass


if __name__ == "__main__":
    pytest.main([__file__])
