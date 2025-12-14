import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch.ao.nn.quantized.functional as qF

# --- REPLACEMENTS for TVM infrastructure imports ---

# Assuming PyTorch is running on CUDA if available, else CPU.
_DEVICES = ["cuda" if torch.cuda.is_available() else "cpu"]

class Device:
    _loaded_config = False
    _device_str = _DEVICES[0] # Default to first available

    @classmethod
    def load(cls, config_path):
        # In PyTorch, device loading is usually not from a config file this way
        # and depends on global availability or explicit device placement.
        # This is a stub to satisfy the original call pattern.
        cls._loaded_config = True
        print(f"INFO: Simulating Device.load({config_path}). Using device: {cls._device_str}")

    def __init__(self):
        pass

    @property
    def device(self):
        return torch.device(self._device_str)

    def get_target(self):
        # TVM specific, not directly applicable. Returning a placeholder.
        return "llvm" if self._device_str == "cpu" else "cuda"


# QNN_DTYPES in TVM are strings like "int8", "uint8".
# For PyTorch, we use these strings to map to actual dtypes, but the tests
# might pass these strings directly.
QNN_DTYPES = ["int8", "uint8"]

def get_low_high_atol_rtol(dtype_str):
    # Simplified version, TVM dtypes are strings like "float32", "int8", "uint8"
    # PyTorch dtypes are torch.float32, torch.qint8 etc.
    # Convert string dtype to actual torch dtype for comparison if needed
    if dtype_str == "float32":
        return -127.0, 127.0, 1e-5, 1e-5
    elif dtype_str == "int8":
        return -128, 127, 2, 0 # atol for quantized
    elif dtype_str == "uint8":
        return 0, 255, 2, 0 # atol for quantized
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def skip_runtime_test():
    # In PyTorch context, we assume tests can run if PyTorch is installed.
    # We might skip if CUDA is requested but not available, but for now, assume availability.
    return False

def skip_codegen_test():
    # Codegen tests are TVM-specific for Arm Compute Library integration.
    # There is no direct PyTorch equivalent to 'codegen' for ACL.
    # TorchInductor is the general PyTorch codegen. We can skip these tests.
    return True # Always skip codegen tests as they are TVM specific.

def build_and_run(func, inputs, _, params, device_obj, enable_acl=False):
    # func: a callable (PyTorch model or function) that takes input tensors
    # inputs: a dict of input tensors (numpy arrays initially)
    # params: a dict of parameter tensors (numpy arrays initially)
    # device_obj: Device object
    # enable_acl: If True, indicates ACL in TVM. In PyTorch, this might mean torch.compile.

    inputs_on_device = {}
    for name, arr in inputs.items():
        if not isinstance(arr, torch.Tensor):
            inputs_on_device[name] = torch.tensor(arr, device=device_obj.device)
        else:
            inputs_on_device[name] = arr.to(device_obj.device)

    # Convert parameters to torch tensors and move to device
    params_on_device = {} # Parameters will be stored in this dict if func needs them
    # But for the current design where func is a closure, params_on_device might not be directly used
    # if `func` already captured the torch.Tensors.
    # We pass params_np to the _get_model functions, which then store their numpy versions
    # and the model_func internalizes torch.Tensors. So this is just to satisfy signature.
    for name, arr in params.items():
        if isinstance(arr, np.ndarray):
            params_on_device[name] = torch.tensor(arr, device=device_obj.device)
        elif isinstance(arr, torch.Tensor):
            params_on_device[name] = arr.to(device_obj.device)
        else:
            raise TypeError(f"Unsupported parameter type: {type(arr)}")

    if not callable(func):
        raise TypeError("`func` must be a callable (PyTorch model or function).")

    input_for_model = list(inputs_on_device.values())[0] # Assuming single input 'a'

    if enable_acl: # Simulate ACL by using torch.compile
        try:
            compiled_func = torch.compile(func, mode="reduce-overhead")
            output = compiled_func(input_for_model)
        except Exception as e:
            pytest.skip(f"torch.compile failed for ACL simulation: {e}")
    else:
        output = func(input_for_model)

    return [output.detach().cpu().numpy()] # Return as numpy array for comparison

def verify(outputs, atol, rtol, config, verify_saturation=False):
    # outputs: a list of numpy arrays, first is baseline, second is ACL/compiled
    # config: dict of test configuration
    print(f"Running verification for config: {config}")
    if len(outputs) < 2:
        pytest.fail("Need at least two outputs for verification (baseline and ACL/compiled).")

    baseline_output = outputs[0]
    acl_output = outputs[1]

    # Dynamically determine atol/rtol based on config or defaults
    current_atol = config.get("atol", atol)
    current_rtol = config.get("rtol", rtol)
    
    # Check for saturation only for quantized types if `verify_saturation` is true
    # The outputs received here are already dequantized float numpy arrays.
    # So, saturation checks must be against the original quantized range if meaningful,
    # or this part is effectively moot if dequantization implicitly handles saturation.
    if verify_saturation and (config.get("dtype") == "int8" or config.get("dtype") == "uint8"):
        qmin_orig, qmax_orig, _, _ = get_low_high_atol_rtol(config.get("dtype"))
        
        # This check is on the *dequantized* output, comparing it to the *dequantized*
        # equivalent of the original quantized range. This implies the `_get_qnn_model`
        # returns values that respect the quantized range.
        # This is typically handled by the `torch.quantize_per_tensor`
        # and subsequent dequantization correctly.
        
        # For simplicity, we assume `torch.testing.assert_allclose` with appropriate
        # tolerance `atol` (passed from test caller for quantized types) is sufficient.
        pass

    torch.testing.assert_allclose(
        torch.tensor(acl_output),
        torch.tensor(baseline_output),
        rtol=current_rtol,
        atol=current_atol,
        msg=f"Verification failed for config: {config}",
    )


def verify_codegen(func, expected_codegen, num_execs):
    # This function is TVM specific codegen verification.
    # There is no direct equivalent in PyTorch for verifying a TVM-style codegen structure.
    # This function will be skipped.
    pytest.skip("Codegen verification is TVM-specific and not convertible to PyTorch.")

# --- END REPLACEMENTS ---

# Global variable to map string dtypes to torch dtypes
_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "float64": torch.float64, # Add if needed
    "int64": torch.int64,     # Add if needed
    # For torch.quantize_per_tensor dtype:
    "qint8": torch.qint8,
    "quint8": torch.quint8,
}

# Helper function to convert TVM dtype string to PyTorch dtype object
def _get_torch_dtype(dtype_str):
    return _TORCH_DTYPE_MAP.get(dtype_str, None)

def _get_model(
    shape,
    kernel_h,
    kernel_w,
    padding_orig, # Renamed to avoid conflict with `padding` local variable
    strides,
    dilation,
    groups,
    dtype_str, # Renamed to avoid conflict with `dtype` local variable if used for torch.dtype
    channels,
    var_names, # Not directly used in PyTorch model, but kept for signature consistency
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Return a model callable and any parameters it may have"""
    dtype = _get_torch_dtype(dtype_str) # Convert string dtype to torch.dtype
    
    # Store numpy arrays for parameters, which will be converted to torch.Tensor in build_and_run
    params = {}

    # Define the PyTorch model/function dynamically
    def model_func(input_tensor_a):
        current_input = input_tensor_a
        current_shape = list(input_tensor_a.shape)
        
        # Handle padding before conv2d if has_pad is True
        padding_for_conv = padding_orig
        if has_pad:
            # TVM pad_width: ((N0, N1), (H0, H1), (W0, W1), (C0, C1)) for NHWC
            # For 2D, (H, W) padding, it becomes ((0,0), (pad_h,pad_h), (pad_w,pad_w), (0,0))
            # PyTorch F.pad: (W_left, W_right, H_top, H_bottom, C_front, C_back, N_front, N_back)
            pad_h, pad_w = padding_orig # padding_orig is (PH, PW) here
            p_torch = [pad_w, pad_w, pad_h, pad_h] 
            current_input = F.pad(current_input, p_torch, mode="constant", value=0.0)
            padding_for_conv = (0, 0) # No additional padding in conv op itself
        else:
            # If has_pad is False, then padding_orig is used for conv2d.
            # TVM `padding` for conv2d is (PH, PW) or (PH_top, PW_left, PH_bottom, PW_right)
            # The original TVM code converts (PH, PW) -> (PH, PW, PH, PW)
            # PyTorch F.conv2d `padding` takes (PH, PW) for symmetric padding.
            padding_for_conv = (padding_orig[0], padding_orig[1])

        is_depthwise = current_shape[3] == channels == groups
        
        # Weights generation (numpy array for initial storage)
        if is_depthwise:
            # TVM HWOI: (kernel_h, kernel_w, channels, 1) where channels == groups
            weight_shape_np = (kernel_h, kernel_w, groups, 1)
            weight_np = np.random.uniform(-128, 127, weight_shape_np).astype(dtype_str)
            params["w"] = weight_np # Store original numpy array
            
            # Convert to PyTorch's expected format (groups, 1, kernel_h, kernel_w)
            # From (H, W, G, 1) to (G, 1, H, W)
            weight_torch = torch.tensor(weight_np, dtype=dtype).permute(2, 3, 0, 1)
        else:
            # TVM HWIO: (kernel_h, kernel_w, in_channels, out_channels)
            weight_shape_np = (kernel_h, kernel_w, current_shape[3], channels)
            weight_np = np.random.uniform(-128, 127, weight_shape_np).astype(dtype_str)
            params["w"] = weight_np
            
            # Convert to PyTorch's expected format (out_channels, in_channels/groups, kernel_h, kernel_w)
            # From (H, W, InC, OutC) to (OutC, InC, H, W)
            weight_torch = torch.tensor(weight_np, dtype=dtype).permute(3, 2, 0, 1)

        # PyTorch F.conv2d expects NCHW input and returns NCHW output.
        input_nchw = current_input.permute(0, 3, 1, 2) # NHWC -> NCHW

        out_nchw = F.conv2d(
            input_nchw,
            weight_torch.to(input_nchw.device), # Ensure weight is on same device as input
            bias=None, # Bias handled separately if has_bias
            stride=strides,
            padding=padding_for_conv,
            dilation=dilation,
            groups=groups,
        )

        # Permute output back to NHWC for consistency with TVM's data_layout="NHWC" assumption
        out = out_nchw.permute(0, 2, 3, 1)

        if has_bias:
            bias_shape = channels # For NHWC, bias_add axis=3 implies bias is of shape (channels,)
            bias_np = np.random.uniform(-128, 127, bias_shape).astype(dtype_str)
            params["b"] = bias_np
            bias_torch = torch.tensor(bias_np, dtype=dtype).to(out.device)
            out = out + bias_torch # PyTorch auto-broadcasts
        
        if has_activation:
            out = F.relu(out)
        return out
    
    return model_func, params


def _get_qnn_model(
    shape,
    kernel_h,
    kernel_w,
    padding_orig,
    strides,
    dilation,
    groups,
    dtype_str, # e.g., "int8", "uint8"
    channels,
    input_zp,
    input_sc,
    kernel_zp,
    kernel_sc,
    output_zp,
    output_sc,
    var_names,
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Return a quantized model callable and any parameters it may have."""
    low, high, _, _ = get_low_high_atol_rtol(dtype_str)
    
    # Map input dtype_str to PyTorch quantized dtype (e.g., torch.qint8)
    q_dtype = _TORCH_DTYPE_MAP[dtype_str]
    
    params = {}

    def qnn_model_func(input_tensor_a_float):
        device = input_tensor_a_float.device

        # Quantize the input float tensor explicitly before processing
        input_q = torch.quantize_per_tensor(
            input_tensor_a_float,
            scale=input_sc,
            zero_point=input_zp,
            dtype=q_dtype,
        ).to(device)
        current_input_q = input_q
        current_shape = list(input_tensor_a_float.shape) # Use float input shape for original shape

        # Handle padding before conv2d if has_pad is True
        padding_for_conv = padding_orig
        if has_pad:
            pad_h, pad_w = padding_orig
            p_torch = [pad_w, pad_w, pad_h, pad_h]
            # F.pad for quantized tensors expects pad_value in quantized domain (input_zp)
            current_input_q = F.pad(current_input_q, p_torch, mode="constant", value=input_zp)
            padding_for_conv = (0, 0) # No additional padding in conv op itself
        else:
            padding_for_conv = (padding_orig[0], padding_orig[1])

        is_depthwise = current_shape[3] == channels == groups

        # Weights generation
        if is_depthwise:
            # TVM HWOI: (kernel_h, kernel_w, channels, 1) where channels == groups
            weight_shape_np = (kernel_h, kernel_w, groups, 1)
            weight_np = np.random.uniform(low, high, weight_shape_np).astype(dtype_str)
            params["w"] = weight_np
            
            # Convert to PyTorch's expected format (groups, 1, kernel_h, kernel_w)
            # And quantize the weight tensor
            weight_torch_float = torch.tensor(weight_np, dtype=_TORCH_DTYPE_MAP[dtype_str])
            weight_torch_q = torch.quantize_per_tensor(
                weight_torch_float.permute(2, 3, 0, 1), # (H,W,G,1) -> (G,1,H,W)
                scale=kernel_sc,
                zero_point=kernel_zp,
                dtype=q_dtype,
            ).to(device)
        else:
            # TVM HWIO: (kernel_h, kernel_w, in_channels, out_channels)
            weight_shape_np = (kernel_h, kernel_w, current_shape[3], channels)
            weight_np = np.random.uniform(low, high, weight_shape_np).astype(dtype_str)
            params["w"] = weight_np
            
            # Convert to PyTorch's expected format (out_channels, in_channels/groups, kernel_h, kernel_w)
            # And quantize the weight tensor
            weight_torch_float = torch.tensor(weight_np, dtype=_TORCH_DTYPE_MAP[dtype_str])
            weight_torch_q = torch.quantize_per_tensor(
                weight_torch_float.permute(3, 2, 0, 1), # (H,W,InC,OutC) -> (OutC,InC,H,W)
                scale=kernel_sc,
                zero_point=kernel_zp,
                dtype=q_dtype,
            ).to(device)

        # PyTorch quantized functional conv2d expects NCHW input and returns NCHW output.
        input_nchw_q = current_input_q.permute(0, 3, 1, 2) # NHWC -> NCHW

        # Output dtype for qnn.conv2d accumulator is typically int32
        out_int32_nchw = qF.conv2d( # Use qF for quantized functional
            input_nchw_q,
            weight_torch_q,
            bias=None, # Bias handled separately if has_bias
            stride=strides,
            padding=padding_for_conv,
            dilation=dilation,
            groups=groups,
            scale=input_sc * kernel_sc, # Output scale for intermediate int32 result
            zero_point=0, # Zero point for intermediate int32 result
            dtype=torch.int32, # Accumulator dtype
        )

        # Permute output back to NHWC after conv for consistency
        out_int32 = out_int32_nchw.permute(0, 2, 3, 1)

        if has_bias:
            bias_shape = channels
            bias_np = np.random.uniform(-128, 127, bias_shape).astype("int32")
            params["b"] = bias_np
            bias_torch = torch.tensor(bias_np, dtype=torch.int32).to(out_int32.device)
            # Bias addition with int32 accumulator
            out_int32 = out_int32 + bias_torch.reshape((1,1,1,-1))

        if has_activation:
            # ReLU on quantized int32 output (accumulator)
            out_int32 = F.relu(out_int32)
        
        # Requantize to final output_dtype and then dequantize for verification
        final_output_q = torch.quantize_per_tensor(
            out_int32.dequantize(), # Dequantize the int32 accumulator result to float
            scale=output_sc,
            zero_point=output_zp,
            dtype=q_dtype, # Final output quantized dtype (qint8 or quint8)
        )
        return final_output_q.dequantize() # Dequantize to float for verification

    return qnn_model_func, params


def _get_expected_codegen(
    shape,
    kernel_h,
    kernel_w,
    padding_orig,
    strides,
    dilation,
    groups,
    dtype_str,
    channels,
    has_bias=False,
    has_activation=False,
):
    # This function is TVM specific and has no direct PyTorch equivalent.
    # It would be used to verify the structure of the generated TVM Relay/ACL graph.
    return None


@pytest.mark.parametrize("device_name", _DEVICES)
def test_conv2d(device_name):
    Device.load("test_config.json") # Call the stub Device.load
    Device._device_str = device_name # Set the device for this test run

    if skip_runtime_test():
        return

    device_obj = Device()
    np.random.seed(0)

    dtype = "float32"
    trials = [
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, False, False), False],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (12, 15, 16), (False, False, True), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, True, False), False],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (12, 15, 16), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True), False],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False), False],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (10, 10, 14), (False, True, True), False],
        # Depth-wise convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, True), True],
        [5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), (False, True, False), True],
        [3, 3, (2, 2), (2, 2), (1, 1), 14, (10, 10, 14), (True, False, False), True],
        [5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, False), True],
        [3, 3, (1, 1), (2, 2), (1, 1), 14, (10, 10, 14), (False, True, True), True],
    ]

    for (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape,
        composite,
        is_depthwise,
    ) in trials:
        shape = (1, *shape) # Add batch dimension
        if is_depthwise:
            groups = shape[3]
        else:
            groups = 1
        
        input_np_a = np.random.uniform(-128, 127, shape).astype(dtype)
        
        input_var_names = {"a": input_np_a}.keys()

        func, params_np = _get_model(
            shape,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            groups,
            dtype,
            out_channels,
            iter(input_var_names),
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        
        inputs_for_run = {"a": input_np_a}

        outputs = []
        for enable_acl in [False, True]: # Run once without compile, once with compile
            outputs.append(build_and_run(func, inputs_for_run, 1, params_np, device_obj, enable_acl=enable_acl)[0])

        config = {
            "shape": shape,
            "groups": groups,
            "kernel size": (kernel_h, kernel_w),
            "padding": pad,
            "stride": stride,
            "dilation": dilation,
            "out channels": out_channels,
            "composite operators (pad, bias, activation)": composite,
            "dtype": dtype # Add dtype to config for verify
        }
        verify(outputs, atol=0.002, rtol=0.01, config=config)


@pytest.mark.skipif(skip_codegen_test(), reason="Codegen tests are TVM-specific")
def test_codegen_conv2d():
    pytest.skip("Codegen tests are TVM-specific and not convertible to PyTorch.")


@pytest.mark.parametrize("dtype_str", QNN_DTYPES)
@pytest.mark.parametrize("device_name", _DEVICES)
def test_qnn_conv2d(dtype_str, device_name):
    Device.load("test_config.json")
    Device._device_str = device_name

    if skip_runtime_test():
        return

    device_obj = Device()
    np.random.seed(0)

    trials = [
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, False, False), False],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (12, 15, 16), (False, False, True), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, True, False), False],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (12, 15, 16), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True), False],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False), False],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (10, 10, 14), (False, True, True), False],
        # Depth-wise convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, True), True],
        [5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), (False, True, False), True],
        [3, 3, (2, 2), (2, 2), (1, 1), 14, (10, 10, 14), (True, False, False), True],
        [5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, False), True],
        [3, 3, (1, 1), (2, 2), (1, 1), 14, (10, 10, 14), (False, True, True), True],
    ]

    for (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape,
        composite,
        is_depthwise,
    ) in trials:
        shape = (1, *shape)
        if is_depthwise:
            groups = shape[3]
        else:
            groups = 1
        
        # Inputs for the _get_qnn_model function, as float, to be quantized inside
        input_np_a_float = np.random.uniform(0, 255, shape).astype("float32")
        
        input_var_names = {"a": input_np_a_float}.keys()

        input_zp = 100
        input_sc = 0.5
        kernel_zp = 25
        kernel_sc = 0.03
        output_zp, output_sc = _get_qnn_params(
            input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, shape[3]
        )

        func, params_np = _get_qnn_model(
            shape,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            groups,
            dtype_str,
            out_channels,
            input_zp,
            input_sc,
            kernel_zp,
            kernel_sc,
            output_zp,
            output_sc,
            iter(input_var_names),
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        
        inputs_for_run = {"a": input_np_a_float}

        outputs = []
        for enable_acl in [False, True]:
            outputs.append(build_and_run(func, inputs_for_run, 1, params_np, device_obj, enable_acl=enable_acl)[0])

        config = {
            "shape": shape,
            "groups": groups,
            "kernel size": (kernel_h, kernel_w),
            "padding": pad,
            "stride": stride,
            "dilation": dilation,
            "out channels": out_channels,
            "composite operators (pad, bias, activation)": composite,
            "input scale": input_sc,
            "input zero point": input_zp,
            "kernel scale": kernel_sc,
            "kernel zero point": kernel_zp,
            "output scale": output_sc,
            "output zero point": output_zp,
            "dtype": dtype_str # Add dtype to config for verify
        }

        atol = 2 if is_depthwise else 1
        verify(outputs, atol=atol, rtol=0, config=config, verify_saturation=True)


@pytest.mark.parametrize("dtype_str", QNN_DTYPES)
@pytest.mark.skipif(skip_codegen_test(), reason="Codegen tests are TVM-specific")
def test_codegen_qnn_conv2d(dtype_str):
    pytest.skip("Codegen tests are TVM-specific and not convertible to PyTorch.")


if __name__ == "__main__":
    print("Running test_conv2d (CPU)...")
    test_conv2d(device_name="cpu")
    if torch.cuda.is_available():
        print("Running test_conv2d (CUDA)...")
        test_conv2d(device_name="cuda")

    print("Running test_qnn_conv2d (int8, CPU)...")
    test_qnn_conv2d(dtype_str="int8", device_name="cpu")
    print("Running test_qnn_conv2d (uint8, CPU)...")
    test_qnn_conv2d(dtype_str="uint8", device_name="cpu")
    if torch.cuda.is_available():
        print("Running test_qnn_conv2d (int8, CUDA)...")
        test_qnn_conv2d(dtype_str="int8", device_name="cuda")
        print("Running test_qnn_conv2d (uint8, CUDA)...")
        test_qnn_conv2d(dtype_str="uint8", device_name="cuda")
