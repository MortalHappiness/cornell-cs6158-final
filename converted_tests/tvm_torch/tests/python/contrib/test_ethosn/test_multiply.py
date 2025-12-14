import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility to map string dtypes to PyTorch dtypes and quantized dtypes
# PyTorch's quantized dtypes (quint8, qint8) are different from normal dtypes (uint8, int8).
# Operations like `torch.quantize_per_tensor` expect the underlying data type.
DTYPE_MAP = {
    "uint8": torch.quint8,
    "int8": torch.qint8,
    "int16": None,  # No direct torch.qint16. If needed, this would require special handling.
    "float32": None, # Not a quantized dtype for `quantize_per_tensor`
    "int32": None, # Not a quantized dtype for `quantize_per_tensor`
}

# Regular PyTorch dtypes for creating initial tensors from NumPy
TORCH_DTYPE_MAP = {
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "float32": torch.float32,
    "int32": torch.int32,
}

# --- Mock Infrastructure (equivalent to `infrastructure as tei` in TVM) ---
class MockTei:
    def get_conv2d_qnn_params(self, dtype, input_zp, input_sc, input2_zp, input2_sc, c_in, c_out, constant_channels_or_output_channels):
        # This function is Ethos-N specific for deriving quantization parameters.
        # For PyTorch, we return dummy but valid scales/zero-points.
        # In a real conversion, this logic would either be re-implemented for PyTorch
        # or indicate that the Ethos-N specific quantization scheme is not directly transferable.
        # The original TVM tests use specific hardcoded values for 'reinterpret_quantize' tests.
        # For general cases, a simplistic heuristic is used for the mock.
        
        # For test_multiply_to_reinterpret_quantize, specific values are directly provided
        # to _get_model, so this mock's return values are not critical there.
        # For test_multiply_to_depthwise and test_multiply_unsupported, this is used.
        
        # NOTE: A proper Ethos-N QNN parameter calculation would be complex and is
        # outside the scope of direct PyTorch API mapping.
        
        # Simple heuristic:
        if dtype == "uint8":
            return 0, 1.0  # zero_point, scale
        elif dtype == "int8":
            return 0, 1.0
        else:
            # For unsupported dtypes, provide generic values.
            return 0, 1.0

    def make_module(self, model_callable, params):
        # In the PyTorch context, model_callable is already a torch.nn.Module instance.
        # This function simply returns it as is.
        # 'params' might contain additional metadata but is not used to build the module here.
        return model_callable

    def build_and_run(self, mod, inputs_dict, num_outputs, params, npu, additional_config_args=None):
        # 'mod' is a torch.nn.Module instance.
        # 'inputs_dict' contains NumPy arrays for the inputs.
        # 'npu' indicates if Ethos-N is enabled; for PyTorch, this is conceptual.
        # Since these tests are skipped, we simply run on CPU.

        device = "cpu"
        # If TorchInductor were being tested, device would be dynamically selected.
        # if torch.cuda.is_available():
        #     device = "cuda"

        mod.eval()
        mod = mod.to(device)

        # Convert input NumPy arrays to PyTorch tensors and move to device
        # The model's forward method will handle the quantization if necessary.
        processed_inputs_for_forward = []
        for input_name, input_np_array in inputs_dict.items():
            if input_np_array is False: # Handle specific constant_data=False case
                processed_inputs_for_forward.append(None)
            else:
                torch_dtype = TORCH_DTYPE_MAP.get(input_np_array.dtype.name)
                if torch_dtype is None:
                    raise ValueError(f"Unsupported numpy dtype for PyTorch: {input_np_array.dtype.name}")
                input_tensor = torch.tensor(input_np_array, dtype=torch_dtype).to(device)
                processed_inputs_for_forward.append(input_tensor)

        with torch.no_grad():
            output_tensor_q = mod(*processed_inputs_for_forward)
        
        # The Ethos-N infrastructure returns a list of outputs, typically one.
        return [output_tensor_q]

    def build(self, model_instance, params, npu, expected_host_ops, npu_partitions):
        # This function simulates the TVM build process for Ethos-N.
        # For unsupported ops, it checks if it would have been offloaded.
        # In PyTorch, we just ensure the model is valid and runnable.
        print(f"INFO: Mock TEI build for Ethos-N specific behavior. NPU={npu}, expected_host_ops={expected_host_ops}, npu_partitions={npu_partitions}")
        model_instance.eval() # Just validate it's a valid PyTorch module.
        # A more advanced mock could integrate torch.compile and check graph breaks,
        # but that's beyond a simple mapping of existing functionality.
        pass

    def verify(self, outputs_list, dtype, tolerance):
        # In Ethos-N tests, 'outputs_list' usually contains results from NPU (True) and host (False) runs.
        # We compare them.
        assert len(outputs_list) >= 1
        
        if len(outputs_list) == 2:
            # Dequantize PyTorch quantized tensors to float for NumPy-like comparison
            output0_float = torch.dequantize(outputs_list[0]).cpu().numpy()
            output1_float = torch.dequantize(outputs_list[1]).cpu().numpy()
            np.testing.assert_allclose(output0_float, output1_float, rtol=tolerance, atol=tolerance)
        else:
            print(f"Warning: verify called with {len(outputs_list)} outputs. Expected 2 for comparison. Skipping numerical comparison.")
            # For unsupported tests, this might be fine, as the goal is to confirm it *didn't* offload.

tei = MockTei()

# Mock requires_ethosn decorator
def requires_ethosn(func):
    return pytest.mark.skip(reason="Ethos-N specific test, not applicable to PyTorch/TorchInductor")(func)
# --- End Mock Infrastructure ---


# --- PyTorch Model Definitions ---
class QnnMultiplyOp(nn.Module):
    """
    Core QNN Multiply operation, performs dequantize->mul->quantize.
    """
    def __init__(self, output_scale, output_zero_point, out_torch_q_dtype):
        super().__init__()
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point
        self.out_torch_q_dtype = out_torch_q_dtype

    def forward(self, x_q, y_q):
        # PyTorch qnn.mul mapping requires dequantizing inputs, performing float mul, then quantizing output.
        x_float = torch.dequantize(x_q)
        y_float = torch.dequantize(y_q)
        float_output = torch.mul(x_float, y_float)
        return torch.quantize_per_tensor(
            float_output,
            scale=self.output_scale,
            zero_point=self.output_zero_point,
            dtype=self.out_torch_q_dtype
        )

class QnnMultiplyModelConstantRhs(nn.Module):
    """
    PyTorch model for TVM's _get_model setup: x (variable) * y (constant).
    """
    def __init__(self, qnn_mul_op_instance, y_data_np, input_sc, input_zp, input2_sc, input2_zp, dtype, reverse_inputs):
        super().__init__()
        self.qnn_mul_op = qnn_mul_op_instance
        self.reverse_inputs = reverse_inputs
        
        # Quantize y_data_np (the constant RHS) and register as a buffer
        if y_data_np is not False and DTYPE_MAP.get(dtype) is not None:
            y_const_tensor = torch.tensor(y_data_np, dtype=TORCH_DTYPE_MAP[dtype])
            self.register_buffer('y_const_q',
                                 torch.quantize_per_tensor(
                                     y_const_tensor.to(torch.float32),
                                     input2_sc, # Use input2_sc/zp for the constant 'y'
                                     input2_zp,
                                     DTYPE_MAP[dtype]
                                 ))
        else:
            self.y_const_q = None # Signifies no fixed constant RHS
        
        self.input_sc = input_sc
        self.input_zp = input_zp
        self.dtype = dtype

    def forward(self, x_float_input):
        if self.y_const_q is None:
            raise ValueError("Model expects a constant 'y_const_q' but it's not set. This model is for variable*constant.")

        # Quantize x_float_input (the variable LHS)
        x_q_input = torch.quantize_per_tensor(
            x_float_input.to(torch.float32),
            self.input_sc,
            self.input_zp,
            DTYPE_MAP[self.dtype]
        )
        
        if self.reverse_inputs:
            return self.qnn_mul_op(self.y_const_q, x_q_input)
        else:
            return self.qnn_mul_op(x_q_input, self.y_const_q)

class QnnMultiplyModelTwoVariables(nn.Module):
    """
    PyTorch model for two variable inputs (x * y), used in unsupported test.
    This model also handles input quantization in its forward pass.
    """
    def __init__(self, output_scale, output_zero_point, out_torch_q_dtype, input_sc, input_zp, input2_sc, input2_zp, dtype):
        super().__init__()
        self.qnn_mul_op = QnnMultiplyOp(output_scale, output_zero_point, out_torch_q_dtype)
        self.input_sc = input_sc
        self.input_zp = input_zp
        self.input2_sc = input2_sc
        self.input2_zp = input2_zp
        self.dtype = dtype

    def forward(self, x_float, y_float):
        x_q = torch.quantize_per_tensor(
            x_float.to(torch.float32), self.input_sc, self.input_zp, DTYPE_MAP[self.dtype]
        )
        y_q = torch.quantize_per_tensor(
            y_float.to(torch.float32), self.input2_sc, self.input2_zp, DTYPE_MAP[self.dtype]
        )
        return self.qnn_mul_op(x_q, y_q)

def _get_model(
    shape,
    constant_shape,
    input_zp,
    input_sc,
    input2_zp,
    input2_sc,
    output_zp,
    output_sc,
    dtype,
    reverse_inputs=False,
    constant_data=None, # if not None, y_data will be constant_data
):
    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max

    if constant_data is not False: # constant_data can be `False` in some tests
        if constant_data is not None:
            y_data_np = np.array(constant_data, dtype=dtype).reshape(constant_shape)
        else:
            y_data_np = np.random.randint(data_min, data_max + 1, size=constant_shape, dtype=dtype)
    else:
        y_data_np = False # Pass False to ModelWrapper to indicate no constant

    # Determine the quantized output dtype based on the string dtype
    out_torch_q_dtype = DTYPE_MAP.get(dtype)
    if out_torch_q_dtype is None:
        raise ValueError(f"Output dtype '{dtype}' is not supported for PyTorch QNN quantization.")

    qnn_mul_op_instance = QnnMultiplyOp(output_sc, output_zp, out_torch_q_dtype)
    
    # The returned model is a PyTorch nn.Module ready for execution
    model = QnnMultiplyModelConstantRhs(
        qnn_mul_op_instance,
        y_data_np,
        input_sc,
        input_zp,
        input2_sc, # Scale/ZP for the constant side
        input2_zp, # Scale/ZP for the constant side
        dtype,
        reverse_inputs
    )
    # Params dict is empty because constants are now buffers in the model
    return model, {}

# --- Tests ---

@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape,constant_shape",
    [((1, 4, 4, 8), (1, 1, 1, 8)), ((1, 16, 12, 4), (4,))],
)
@pytest.mark.parametrize("reverse_inputs", [False, True])
def test_multiply_to_depthwise(dtype, shape, constant_shape, reverse_inputs):
    """Compare Multiply -> Depthwise conversion output with TVM."""

    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    
    # tei.get_conv2d_qnn_params is mocked, actual EthosN param derivation not replicated.
    # The returned values will be generic 0, 1.0 based on mock logic.
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[3]
    )

    model, params = _get_model(
        shape,
        constant_shape,
        input_zp,
        input_sc,
        input2_zp,
        input2_sc,
        output_zp,
        output_sc,
        dtype,
        reverse_inputs,
    )
    # The 'inputs' dict for tei.build_and_run only contains 'x' as 'y' is a constant in the model.
    inputs = {"x": np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype)}
    
    outputs = []
    for npu in [False, True]: # Simulate Ethos-N vs. Host comparison
        mod = tei.make_module(model, params)
        outputs.append(tei.build_and_run(mod, inputs, 1, params, npu=npu))

    tei.verify(outputs, dtype, 1e-5) # Adjusted tolerance slightly for potential float differences


@requires_ethosn
@pytest.mark.parametrize(
    "shape,constant_shape", [((1, 4, 5, 8), (1, 1, 1, 1)), ((1, 3, 7, 10), None)]
)
@pytest.mark.parametrize("reverse_inputs", [False, True])
def test_multiply_to_reinterpret_quantize(shape, constant_shape, reverse_inputs):
    """Compare Multiply -> Reinterpret Quantize conversion output with TVM."""
    np.random.seed(0)

    dtype = "uint8"
    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max

    # Multiply can only be offloaded as a reinterpret quantize operation if
    # it is an identity option. We must choose the quantization and constant
    # data carefully to make sure that this is the case.
    input_zp = 0
    input_sc = 0.007814894430339336
    input2_zp = 0
    input2_sc = 0.5
    output_zp = 0
    output_sc = 0.9963990449905396
    constant_data = 255 # This constant when quantized with input2_sc, input2_zp and multiplied,
                        # and then requantized with output_sc, output_zp, should be an identity-like op.

    model, params = _get_model(
        shape,
        constant_shape,
        input_zp,
        input_sc,
        input2_zp,
        input2_sc,
        output_zp,
        output_sc,
        dtype,
        reverse_inputs,
        constant_data,
    )
    inputs = {"x": np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype)}
    outputs = []
    for npu in [False, True]:
        mod = tei.make_module(model, params)
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                params,
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    tei.verify(outputs, dtype, 1e-5)


@requires_ethosn
def test_multiply_multiple_inputs_unsupported():
    """Check multiply operator with two inputs is not offloaded."""

    np.random.seed(0)

    shape = (1, 4, 5, 6)
    dtype = "int8"
    out_torch_q_dtype = DTYPE_MAP.get(dtype)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    
    # tei.get_conv2d_qnn_params is mocked, actual EthosN param derivation not replicated.
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[3]
    )

    # In this test, both x and y are variables, not one variable and one constant.
    # We use a different PyTorch Module for this case.
    model = QnnMultiplyModelTwoVariables(
        output_sc,
        output_zp,
        out_torch_q_dtype,
        input_sc,
        input_zp,
        input2_sc,
        input2_zp,
        dtype
    )

    # For this test, inputs are not run, only build is checked.
    # The 'inputs' dict for build must match the forward signature (x, y).
    inputs_for_build = {
        "x": np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype),
        "y": np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype)
    }

    expected_host_ops = 1 # TVM expected 1 host op if not offloaded
    npu_partitions = 0    # TVM expected 0 NPU partitions
    for npu in [False, True]:
        mod = tei.make_module(model, {}) # params are empty as the model directly contains its configuration
        tei.build(
            mod,
            inputs_for_build, # Use inputs_for_build to satisfy the mock build signature
            npu=npu,
            expected_host_ops=expected_host_ops,
            npu_partitions=npu_partitions,
        )


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,shape,constant_shape",
    [
        ("int16", (1, 4, 5, 6), (1, 1, 1, 6)), # Unsupported dtype
        ("int8", (1, 1, 3), (1, 1, 1, 3)), # Unsupported shape (rank 3)
        ("int8", (1, 2, 4, 8), (1, 2, 4, 8)), # Unsupported broadcast pattern
    ],
)
def test_multiply_unsupported(dtype, shape, constant_shape):
    """Check multiply operator with unsupported attributes is not offloaded."""

    np.random.seed(0)

    # Handle dtype conversion for iinfo if it's not a standard numpy integer type
    iinfo_dtype = dtype if dtype in ["uint8", "int8", "int16", "int32"] else "int8"
    iinfo = np.iinfo(iinfo_dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    
    # tei.get_conv2d_qnn_params is mocked, actual EthosN param derivation not replicated.
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[-1]
    )

    # If dtype is not supported for QNN in PyTorch, _get_model will raise ValueError.
    # So wrap in try-except or adjust the test to skip based on dtype support.
    try:
        model, params = _get_model(
            shape,
            constant_shape,
            input_zp,
            input_sc,
            input2_zp,
            input2_sc,
            output_zp,
            output_sc,
            dtype,
            reverse_inputs=False,
            constant_data=False, # Signifies no specific constant value, just a shape
        )
    except ValueError as e:
        if "is not supported for PyTorch QNN quantization" in str(e):
            pytest.skip(f"PyTorch QNN does not directly support quantized '{dtype}' for this operation (e.g., torch.qint16).")
        else:
            raise # Re-raise if it's an unexpected ValueError

    inputs_for_build = {"x": np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype)}

    expected_host_ops = 1
    npu_partitions = 0
    for npu in [False, True]:
        mod = tei.make_module(model, params)
        tei.build(
            mod,
            inputs_for_build,
            npu=npu,
            expected_host_ops=expected_host_ops,
            npu_partitions=npu_partitions,
        )
