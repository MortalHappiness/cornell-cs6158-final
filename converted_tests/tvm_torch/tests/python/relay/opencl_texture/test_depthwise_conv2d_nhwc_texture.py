import pytest
import torch
import numpy as np
import math
import os # For pytest.mark.skipif (for 'adreno')

# Mocking tvm.testing.init.Xavier
class XavierInitializer:
    def __call__(self, name, data):
        if name == "weight":
            # Assuming depthwise convolution, TVM filter_shape (kH, kW, G, 1)
            kH, kW, G, O = data.shape
            # For depthwise, each group has 1 input channel and 1 output channel.
            # Fan_in and fan_out are per-group.
            # fan_in = kH * kW * 1
            # fan_out = kH * kW * 1
            # For Xavier, std = sqrt(2 / (fan_in + fan_out)) for normal, or limit = sqrt(6 / (fan_in + fan_out)) for uniform
            limit = math.sqrt(6.0 / (kH * kW * 1 + kH * kW * 1)) # Simplified for depthwise
            data[:] = np.random.uniform(low=-limit, high=limit, size=data.shape).astype(data.dtype)
        elif name == "bias":
            # Original TVM test sets bias_data to zeros then calls initializer.
            # Initializer sets bias to zeros. So keep as zeros.
            data[:] = np.zeros(data.shape).astype(data.dtype)

# This PyTorch Module encapsulates the Relay graph logic
class DepthwiseConv2dModel(torch.nn.Module):
    def __init__(self, filter_shape, conv_params, add_bias_relu_flag, dtype_torch):
        super().__init__()
        self.conv_params = conv_params
        self.add_bias_relu_flag = add_bias_relu_flag
        self.dtype_torch = dtype_torch

        self.data_layout = conv_params.get("data_layout", "NHWC")
        self.kernel_layout = conv_params.get("kernel_layout", "HWOI")
        self.groups = conv_params.get("groups", filter_shape[2])
        # TVM's 'channels' arg, usually C_in. In PyTorch F.conv2d, it's inferred from input.
        self.channels = conv_params.get("channels", filter_shape[2]) 
        self.kernel_size = conv_params.get("kernel_size", (filter_shape[0], filter_shape[1]))
        self.strides = conv_params.get("strides", 1)
        self.padding = conv_params.get("padding", 0)
        self.dilation = conv_params.get("dilation", 1)
        self.out_dtype = conv_params.get("out_dtype", "float32")

        # Normalize strides, padding, dilation to tuples
        self._strides = self.strides if isinstance(self.strides, (list, tuple)) else (self.strides, self.strides)
        self._dilation = self.dilation if isinstance(self.dilation, (list, tuple)) else (self.dilation, self.dilation)
        
        # Determine PyTorch-compatible padding for F.conv2d
        if isinstance(self.padding, (list, tuple)) and len(self.padding) == 4:
            # TVM padding: [p_h_top, p_h_bottom, p_w_left, p_w_right]
            # PyTorch F.conv2d padding: (padding_height, padding_width).
            # If asymmetric padding is specified (e.g. padding[0] != padding[1]),
            # an explicit F.pad call before F.conv2d would be needed.
            # For these specific tests, the 4-element padding is always symmetric [X,X,Y,Y].
            self._conv_padding = (self.padding[0], self.padding[2])
            if self.padding[0] != self.padding[1] or self.padding[2] != self.padding[3]:
                raise NotImplementedError("Asymmetric padding for F.conv2d not fully handled in this conversion script.")
        else: # single int or (H_pad, W_pad)
            self._conv_padding = self.padding if isinstance(self.padding, (list, tuple)) else (self.padding, self.padding)

    def forward(self, data, weight, bias):
        # Data layout: NHWC -> NCHW for PyTorch F.conv2d
        if self.data_layout == "NHWC":
            data = data.permute(0, 3, 1, 2) # NCHW

        # Weight layout: HWOI -> OIHW for depthwise (G, 1, kH, kW)
        if self.kernel_layout == "HWOI":
            weight = weight.permute(2, 3, 0, 1) # (kH, kW, G, 1) -> (G, 1, kH, kW)

        conv_output = torch.nn.functional.conv2d(
            input=data,
            weight=weight,
            bias=None, # Bias handled after conv in TVM Relay graph
            stride=self._strides,
            padding=self._conv_padding,
            dilation=self._dilation,
            groups=self.groups,
        )

        output = conv_output
        if self.add_bias_relu_flag:
            output = torch.add(output, bias.reshape(1, -1, 1, 1)) # Bias broadcastable to NCHW
            output = torch.relu(output)
        
        # Output layout: NCHW -> NHWC
        if self.data_layout == "NHWC":
            output = output.permute(0, 2, 3, 1) # NHWC
        
        return output

def build_run_compare(input_shapes, filter_shape, bias_shape, dtype_str, target_device, add_bias_relu_flag, conv_specific_params):
    dtype_torch = getattr(torch, dtype_str)
    device = "cpu"
    # Placeholder for 'remote' target. Actual PyTorch/Inductor tests run on CUDA if available.
    if "cuda" in target_device.lower() and torch.cuda.is_available():
        device = "cuda"

    # 1. Initialize parameters (weights/biases) using original TVM's numpy approach
    np.random.seed(1) # Seed for weight/bias initialization
    _filter_data_np = np.zeros(filter_shape).astype(dtype_str)
    _bias_data_np = np.zeros(bias_shape).astype(dtype_str)
    initializer = XavierInitializer()
    initializer("weight", _filter_data_np)
    initializer("bias", _bias_data_np)
    
    # 2. Instantiate and compile the PyTorch model
    model = DepthwiseConv2dModel(
        filter_shape=filter_shape,
        conv_params=conv_specific_params,
        add_bias_relu_flag=add_bias_relu_flag,
        dtype_torch=dtype_torch
    )
    model.to(device)
    model.eval() 
    compiled_model = torch.compile(model, dynamic=False)

    # 3. Generate input data for the compiled model run
    np.random.seed(1) # Re-seed for input data generation (consistent with TVM's `np.random.seed(1)` line)
    input_data_np = np.random.rand(*input_shapes["data"]).astype(dtype_str)
    
    torch_input_data = torch.from_numpy(input_data_np).to(device)
    torch_weight_data = torch.from_numpy(_filter_data_np).to(device)
    torch_bias_data = torch.from_numpy(_bias_data_np).to(device)

    # 4. Run compiled PyTorch model
    actual_output_torch = compiled_model(
        torch_input_data,
        torch_weight_data,
        torch_bias_data
    )
    actual_output_np = actual_output_torch.cpu().numpy()

    # 5. Generate reference output using a non-compiled PyTorch model (on CPU)
    np.random.seed(1) # Re-seed for reference input data generation
    ref_input_data_np = np.random.rand(*input_shapes["data"]).astype(dtype_str)

    ref_model = DepthwiseConv2dModel(
        filter_shape=filter_shape,
        conv_params=conv_specific_params,
        add_bias_relu_flag=add_bias_relu_flag,
        dtype_torch=dtype_torch
    )
    ref_model.eval()

    ref_input_data_torch = torch.from_numpy(ref_input_data_np) # On CPU
    ref_weight_data_torch = torch.from_numpy(_filter_data_np) # On CPU
    ref_bias_data_torch = torch.from_numpy(_bias_data_np) # On CPU

    ref_output_torch = ref_model(
        ref_input_data_torch,
        ref_weight_data_torch,
        ref_bias_data_torch
    )
    ref_output_np = ref_output_torch.cpu().numpy()

    # 6. Compare results
    # TVM tests sometimes have looser tolerances for GPU targets.
    torch.testing.assert_allclose(actual_output_np, ref_output_np, rtol=1e-4, atol=1e-4)

# Placeholder for `tvm.testing.parametrize_targets`.
# We use pytest.mark.parametrize here. For 'opencl -device=adreno', we'll just use 'cpu' or 'cuda'
# as a placeholder, possibly with a skip for unsupported device.
def parametrize_targets(target_str):
    if "opencl -device=adreno" in target_str:
        targets = ["cpu"]
        if torch.cuda.is_available():
            targets.append("cuda")
        return pytest.mark.parametrize("target", targets)
    return pytest.mark.parametrize("target", [target_str]) # Fallback for other targets


# Placeholder for tvm.testing.requires_opencl.
# We skip the test if CUDA is required but not available (if 'opencl' implies GPU here).
def requires_opencl(test_func):
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Requires a GPU-like device (OpenCL in TVM), running on CUDA if available."
    )(test_func)

# tvm.testing.parameter replacement
dtype_param = pytest.mark.parametrize("dtype", ["float32"])


@requires_opencl
@parametrize_targets("opencl -device=adreno")
@dtype_param
def test_depthwise_conv2d_deeplabv3_1_129_129_144x3_3_144_1(remote, target, dtype):
    input_shape = (1, 129, 129, 144)
    filter_shape = (3, 3, 144, 1)
    kernel_size = (filter_shape[0], filter_shape[1])
    bias_shape = (filter_shape[2],) # = (144,)

    # The original TVM test defines `conv`, then `D = add(conv, bias)`, `D = relu(D)`,
    # then `mod = relay.Function([A, B, bias], D)` which means conv+add+relu.
    # HOWEVER, it then IMMEDIATELY OVERRIDES `mod = relay.Function([A, B, bias], conv)`.
    # This means the test actually only runs the `conv` op without bias and relu.
    add_bias_relu_flag = False 

    conv_params = {
        "data_layout": "NHWC",
        "kernel_layout": "HWOI",
        "out_dtype": dtype,
        "groups": filter_shape[2], # 144
        "channels": filter_shape[2], # 144
        "kernel_size": kernel_size, # (3,3)
        "strides": (1,1), # Default
        "padding": (0,0), # Default
        "dilation": (1,1), # Default
    }

    build_run_compare(
        {"data": input_shape},
        filter_shape,
        bias_shape,
        dtype,
        target,
        add_bias_relu_flag,
        conv_params
    )


@requires_opencl
@parametrize_targets("opencl -device=adreno")
@dtype_param
def test_depthwise_conv2d_deeplabv3_4_35_35_576x3_3_576_1(remote, target, dtype):
    input_shape = (4, 35, 35, 576)
    filter_shape = (3, 3, 576, 1)
    kernel_size = (filter_shape[0], filter_shape[1])
    bias_shape = (filter_shape[2],) # = (576,)

    # The original TVM test has the same override pattern as above:
    # `mod = relay.Function([A, B, bias], D)` then `mod = relay.Function([A, B, bias], conv)`
    add_bias_relu_flag = False 

    conv_params = {
        "data_layout": "NHWC",
        "kernel_layout": "HWOI",
        "out_dtype": dtype,
        "groups": filter_shape[2], # 576
        "channels": filter_shape[2], # 576
        "kernel_size": kernel_size, # (3,3)
        "strides": (1,1), # Default
        "padding": (0,0), # Default
        "dilation": (1,1), # Default
    }

    build_run_compare(
        {"data": input_shape},
        filter_shape,
        bias_shape,
        dtype,
        target,
        add_bias_relu_flag,
        conv_params
    )


@requires_opencl
@parametrize_targets("opencl -device=adreno")
@dtype_param
def test_depthwise_conv2d_deeplabv3_1_129_129_144x3_3_144_1_with_padding(remote, target, dtype):
    input_shape = (1, 129, 129, 144)
    filter_shape = (3, 3, 144, 1)
    kernel_size = (filter_shape[0], filter_shape[1])
    bias_shape = (filter_shape[2],) # = (144,)

    # In this test, `mod = relay.Function([A, B, bias], D)` is the final assignment,
    # and `D` includes `add` and `relu`.
    add_bias_relu_flag = True 

    conv_params = {
        "data_layout": "NHWC",
        "kernel_layout": "HWOI",
        "padding": [3, 3, 3, 3], # (top, bottom, left, right)
        "strides": [2, 2],
        "out_dtype": dtype,
        "groups": filter_shape[2], # 144
        "channels": filter_shape[2], # 144
        "kernel_size": kernel_size, # (3,3)
        "dilation": (1,1), # Default
    }

    build_run_compare(
        {"data": input_shape},
        filter_shape,
        bias_shape,
        dtype,
        target,
        add_bias_relu_flag,
        conv_params
    )


@requires_opencl
@parametrize_targets("opencl -device=adreno")
@dtype_param
def test_depthwise_conv2d_1_513_513_7x3_3_7_1(remote, target, dtype):
    input_shape = (1, 513, 513, 7)
    filter_shape = (3, 3, 7, 1)
    bias_shape = (filter_shape[2],) # = (7,)
    kernel_size = (filter_shape[0], filter_shape[1])

    # In this test, `mod = relay.Function([A, B, bias], D)` is the final assignment,
    # and `D` includes `add` and `relu`.
    add_bias_relu_flag = True 

    conv_params = {
        "data_layout": "NHWC",
        "kernel_layout": "HWOI",
        "out_dtype": dtype,
        "channels": filter_shape[2], # 7
        "groups": filter_shape[2], # 7
        "kernel_size": kernel_size, # (3,3)
        "strides": (1,1), # Default
        "padding": (0,0), # Default
        "dilation": (1,1), # Default
    }

    build_run_compare(
        {"data": input_shape},
        filter_shape,
        bias_shape,
        dtype,
        target,
        add_bias_relu_flag,
        conv_params
    )


@requires_opencl
@parametrize_targets("opencl -device=adreno")
@dtype_param
def test_depthwise_conv2d_1_513_513_3x3_3_3_1(remote, target, dtype):
    input_shape = (1, 513, 513, 3)
    filter_shape = (3, 3, 3, 1)
    bias_shape = (filter_shape[2],) # = (3,)
    kernel_size = (filter_shape[0], filter_shape[1])

    # In this test, `mod = relay.Function([A, B, bias], D)` is the final assignment,
    # and `D` includes `add` and `relu`.
    add_bias_relu_flag = True 

    conv_params = {
        "data_layout": "NHWC",
        "kernel_layout": "HWOI",
        "out_dtype": dtype,
        "channels": filter_shape[2], # 3
        "groups": filter_shape[2], # 3
        "kernel_size": kernel_size, # (3,3)
        "strides": (1,1), # Default
        "padding": (0,0), # Default
        "dilation": (1,1), # Default
    }

    build_run_compare(
        {"data": input_shape},
        filter_shape,
        bias_shape,
        dtype,
        target,
        add_bias_relu_flag,
        conv_params
    )

# tvm.testing.main() is replaced by pytest's entry point if this file is run directly
if __name__ == "__main__":
    pytest.main([__file__])
