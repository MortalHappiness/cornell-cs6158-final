import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
from itertools import product
import random

# Global device setup
_device = "cuda" if torch.cuda.is_available() else "cpu"
_torch_dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int8": torch.int8,
    "qint8": torch.qint8,
}

def _convert_tvm_dtype_to_torch_dtype(tvm_dtype_str):
    return _torch_dtype_map.get(tvm_dtype_str, torch.float32)

def skip_runtime_test():
    """Checks if TorchInductor is available for compilation."""
    # In PyTorch context, BNNS specific codegen check is replaced by TorchInductor availability.
    # If a specific Apple BNNS backend for TorchInductor were implemented, this would change.
    return not torch.backends.inductor.is_available()

def generate_trials(param_lists, num_trials):
    """Generates a list of random trials from given parameter lists."""
    all_combinations = list(product(*param_lists))
    if num_trials >= len(all_combinations):
        return all_combinations
    return random.sample(all_combinations, num_trials)

class Conv2dModel(nn.Module):
    """A PyTorch model mimicking the TVM Relay conv2d structure."""
    def __init__(self, input_shape, kernel, padding, strides, dilation, groups, dtype_str, out_channels, bias_type, activation_type, np_params):
        super().__init__()
        torch_dtype = _convert_tvm_dtype_to_torch_dtype(dtype_str)

        self.conv = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel,
            stride=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False # Bias is added explicitly later to match TVM Relay structure
        )
        self.conv.weight.data = torch.tensor(np_params["w"], dtype=torch_dtype)

        self.bias_tensor = None
        if bias_type in ["bias_add", "add_3d", "add_4d"]:
            self.bias_tensor = nn.Parameter(torch.tensor(np_params["b"], dtype=torch_dtype))

        self.activation = None
        if activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        if self.bias_tensor is not None:
            # PyTorch's '+' operator handles broadcasting for [C], [C,1,1] or [1,C,1,1] biases
            out = out + self.bias_tensor
        if self.activation:
            out = self.activation(out)
        return out

def _get_model(
    input_shape,
    kernel=(3, 3),
    padding=(1, 1),
    strides=(1, 1),
    dilation=(1, 1),
    groups=1,
    dtype="float32",
    channels=-1, # -1 means same as input channels
    bias_type="none",
    activation_type="none",
):
    """
    Returns a PyTorch model and the expected input configuration.
    `input_shape` is assumed to be (N, C, H, W).
    """
    if channels == -1:
        channels = input_shape[1]

    np_params = {}
    weight_shape = (channels, input_shape[1] // groups, *kernel)
    np_params["w"] = np.random.uniform(-128, 127, weight_shape).astype(dtype)

    if bias_type in ["bias_add", "add_3d", "add_4d"]:
        if bias_type == "bias_add":
            bias_np_shape = (channels,) # 1D bias
        elif bias_type == "add_3d":
            bias_np_shape = (channels, 1, 1) # 3D bias like [C, 1, 1]
        elif bias_type == "add_4d":
            bias_np_shape = (1, channels, 1, 1) # 4D bias like [1, C, 1, 1]
        np_params["b"] = np.random.uniform(-10, 10, bias_np_shape).astype(dtype)

    model = Conv2dModel(
        input_shape=input_shape,
        kernel=kernel,
        padding=padding,
        strides=strides,
        dilation=dilation,
        groups=groups,
        dtype_str=dtype,
        out_channels=channels,
        bias_type=bias_type,
        activation_type=activation_type,
        np_params=np_params
    )
    return model, input_shape, dtype

def compare_inference_with_ref(model, expected_input_shape, expected_dtype_str):
    """
    Compares the output of an eager PyTorch model with its TorchInductor compiled version.
    """
    model.to(_device)
    model.eval()

    torch_dtype = _convert_tvm_dtype_to_torch_dtype(expected_dtype_str)

    # Generate random input
    np_data = np.random.uniform(-1, 1, expected_input_shape).astype(expected_dtype_str)
    torch_data = torch.tensor(np_data, dtype=torch_dtype, device=_device)

    # Eager execution
    with torch.no_grad():
        ref_out = model(torch_data)

    # TorchInductor compilation and execution
    if skip_runtime_test():
        pytest.skip("TorchInductor is not available or disabled on this system.")

    compiled_model = torch.compile(model, fullgraph=True, dynamic=False)
    with torch.no_grad():
        compiled_out = compiled_model(torch_data)

    # Compare results
    torch.testing.assert_allclose(ref_out, compiled_out, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because TorchInductor is not available or disabled")
def test_conv2d():
    np.random.seed(0)
    random.seed(0)

    kernel_hs = [1, 2, 3, 5]
    kernel_ws = [1, 2, 3, 5]
    pads = [(1, 1), (2, 2), (2, 1)] # Renamed to avoid clash with 'padding' parameter
    strides_list = [(1, 1), (2, 2)] # Renamed
    dilations = [(1, 1)] # Renamed
    out_channels_list = [1, 4, 8, 16]
    input_spatial_shapes = [(10, 10), (12, 15), (20, 20)] # (H, W)
    input_channels_list = [1, 3, 5] # C
    batches = [1, 2]
    groups_list = [1, 2] # Renamed
    bias_kinds = ["none", "add_3d", "add_4d", "bias_add"] # Renamed "bias.add" to "bias_add" for consistency
    activation_kinds = ["none", "relu", "sigmoid"] # Renamed
    dtypes = ["float32"]

    # Generate trials combining all parameters
    trials_raw = generate_trials(
        [
            kernel_hs, kernel_ws,
            pads,
            strides_list,
            dilations,
            out_channels_list,
            input_spatial_shapes,
            input_channels_list,
            groups_list,
            batches,
            bias_kinds,
            activation_kinds,
            dtypes,
        ],
        num_trials=5 # Reduced for quicker testing, original was 3 samples on a different set. Adjusted to 5 for broader coverage
    )

    for (
        kernel_h, kernel_w,
        padding_val, # Use padding_val to avoid collision with function arg
        stride_val,
        dilation_val,
        out_channels_val,
        input_spatial_shape,
        input_c,
        group_val,
        batch_val,
        bias_type_val,
        activation_type_val,
        dtype_val
    ) in trials_raw:
        # Skip invalid configurations where output channels are not divisible by groups
        if out_channels_val % group_val != 0:
            continue

        current_input_shape = (batch_val, input_c, *input_spatial_shape) # NCHW format

        # Create the PyTorch model
        model, expected_input_shape, expected_dtype = _get_model(
            input_shape=current_input_shape,
            kernel=(kernel_h, kernel_w),
            padding=padding_val,
            strides=stride_val,
            dilation=dilation_val,
            groups=group_val,
            channels=out_channels_val,
            bias_type=bias_type_val,
            activation_type=activation_type_val,
            dtype=dtype_val,
        )
        compare_inference_with_ref(model, expected_input_shape, expected_dtype)


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because TorchInductor is not available or disabled")
def test_conv2d_dw():
    np.random.seed(0)
    random.seed(0)

    input_c = 5
    input_spatial_shape = (5, 5)

    for batch_val in [1, 2]:
        current_input_shape = (batch_val, input_c, *input_spatial_shape)
        # For depthwise, groups should be equal to input channels
        model, expected_input_shape, expected_dtype = _get_model(
            input_shape=current_input_shape,
            groups=input_c,
            channels=input_c, # Output channels same as input for depthwise
            dtype="float32",
        )
        compare_inference_with_ref(model, expected_input_shape, expected_dtype)


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because TorchInductor is not available or disabled")
def test_conv2d_with_oc1():
    np.random.seed(0)
    random.seed(0)

    input_c = 3
    input_spatial_shape = (5, 5)

    for batch_val in [1, 2]:
        for bias_type_val in ["none", "add_4d"]:
            current_input_shape = (batch_val, input_c, *input_spatial_shape)
            model, expected_input_shape, expected_dtype = _get_model(
                input_shape=current_input_shape,
                channels=1, # output channels = 1
                bias_type=bias_type_val,
                dtype="float32",
            )
            compare_inference_with_ref(model, expected_input_shape, expected_dtype)
