import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import os # For checking CUDA availability

# Mock tvm.nd.array for easy conversion to numpy arrays, if tvm.nd.array objects are passed.
class tvm_nd_array_mock:
    def __init__(self, np_array):
        self._array = np_array
    def asnumpy(self):
        return self._array
    def __str__(self):
        return f"tvm_nd_array_mock({self._array})"

# Mock for tvm.testing.requires_openclml
def requires_openclml(f):
    # This decorator implies a specific TVM backend requirement.
    # For PyTorch conversion, we'll run on CPU/CUDA if available.
    # If the test is truly for a niche hardware accelerator,
    # it might need `pytest.mark.skip` based on PyTorch backend availability.
    # For now, it's a pass-through.
    return f

# Mock for skip_codegen_test (can be just a pass-through decorator or ignore)
def skip_codegen_test(f):
    return f

# Mock for Device (just a string for PyTorch device)
# In PyTorch, device is typically a string like 'cpu' or 'cuda'.
# We dynamically set it based on CUDA availability for broader test coverage.
device = "cuda" if torch.cuda.is_available() else "cpu"

def to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

# Wrapper to handle PyTorch model execution and comparison
def run_pytorch_model(model_instance, inputs_dict, output_idx, device_str, enable_compiler):
    current_device = torch.device(device_str)
    model_instance = model_instance.to(current_device)

    # Convert inputs to torch.Tensor and move to device
    torch_inputs = {}
    for k, v in inputs_dict.items():
        if isinstance(v, np.ndarray):
            torch_inputs[k] = torch.tensor(v, device=current_device)
        elif isinstance(v, torch.Tensor):
            torch_inputs[k] = v.to(current_device)
        elif isinstance(v, tvm_nd_array_mock): # Handle mocked tvm.nd.array
            torch_inputs[k] = torch.tensor(v.asnumpy(), device=current_device)
        else: # Fallback for other potential types in inputs
            torch_inputs[k] = torch.tensor(v, device=current_device)

    if enable_compiler:
        # torch.compile requires the model to be a callable (nn.Module or function)
        compiled_model = torch.compile(model_instance, dynamic=False)
        output = compiled_model(**torch_inputs)
    else:
        output = model_instance(**torch_inputs)

    # Assuming the output is always a tensor, or a tuple/list of tensors if more than one.
    # The TVM `build_and_run` returns a list of outputs, typically one.
    if isinstance(output, (tuple, list)):
        return [output[output_idx].cpu().detach().numpy()]
    else:
        return [output.cpu().detach().numpy()]


def _get_conv_model(
    input_shape, # Expected as (N, C, H, W)
    kernel_h,
    kernel_w,
    padding_tuple_hw, # TVM style (h_pad, w_pad) for `relay.nn.pad` or `conv2d`
    strides_tuple_hw,
    dilation_tuple_hw,
    groups,
    dtype_str,
    out_channels,
    # The 'var' parameter from TVM (input placeholder name) is not directly used in PyTorch Module.
    var_for_input_name_dummy, 
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Returns a PyTorch ConvModule instance and a dictionary of its initial NumPy parameters."""
    torch_dtype = to_torch_dtype(dtype_str)

    class ConvModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.has_pad = has_pad
            self.has_activation = has_activation
            
            # Determine effective input channels for nn.Conv2d
            current_input_ch = input_shape[1]
            
            # Determine padding configuration for F.pad and nn.Conv2d
            if self.has_pad:
                # TVM's `relay.nn.pad` takes `pad_width` as `((0,0), (ph,ph), (pw,pw), (0,0))` for NCHW.
                # PyTorch `F.pad` takes a flat tuple `(left, right, top, bottom, ...)`.
                self.torch_pad_tuple = (padding_tuple_hw[1], padding_tuple_hw[1], # width padding (left, right)
                                        padding_tuple_hw[0], padding_tuple_hw[0]) # height padding (top, bottom)
                conv_padding_arg = (0, 0) # After explicit F.pad, conv itself needs no further padding.
            else:
                self.torch_pad_tuple = None # No explicit F.pad needed.
                conv_padding_arg = padding_tuple_hw # nn.Conv2d handles this padding directly.

            # Initialize nn.Conv2d layer
            self.conv_layer = nn.Conv2d(
                in_channels=current_input_ch,
                out_channels=out_channels,
                kernel_size=(kernel_h, kernel_w),
                stride=strides_tuple_hw,
                padding=conv_padding_arg,
                dilation=dilation_tuple_hw,
                groups=groups,
                bias=has_bias,
                dtype=torch_dtype # Specify dtype here
            )

            # Manually initialize weights and bias using np.random.uniform, matching TVM test.
            # This overwrites the default initialization of nn.Conv2d.
            weight_shape = (out_channels, current_input_ch // groups, kernel_h, kernel_w)
            self.conv_layer.weight.data = torch.tensor(
                np.random.uniform(-1, 1, weight_shape).astype(dtype_str),
                dtype=torch_dtype
            )
            if has_bias:
                bias_shape = out_channels
                self.conv_layer.bias.data = torch.tensor(
                    np.random.uniform(-1, 1, bias_shape).astype(dtype_str),
                    dtype=torch_dtype
                )

        def forward(self, a):
            x = a 
            if self.has_pad:
                x = F.pad(x, self.torch_pad_tuple, mode='constant', value=0.0)
            
            out = self.conv_layer(x)

            if self.has_activation:
                out = F.relu(out)
            
            return out

    model_instance = ConvModule()

    # Create a dictionary of NumPy parameters, matching the structure TVM's `params` dict would have.
    # This is mainly for alignment with the original TVM test structure, though not directly used by `run_pytorch_model` anymore.
    params_np_dict = {
        "w": model_instance.conv_layer.weight.data.cpu().numpy()
    }
    if has_bias:
        params_np_dict["b"] = model_instance.conv_layer.bias.data.cpu().numpy()

    return model_instance, params_np_dict


@pytest.mark.parametrize("dtype", ["float32"])
@requires_openclml
def test_conv2d(dtype): # Removed 'device' from signature as it's global
    trials = [
        # (kernel_h, kernel_w, pad_hw, stride_hw, dilation_hw, out_channels, input_chw, composite_flags)
        # input_chw is (C, H, W)
        [3, 3, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (15, 16, 12), (False, False, True)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, True)],
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (16, 12, 15), (False, False, True)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False)],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (16, 12, 15), (False, False, False)],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True)],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False)],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False)],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (14, 10, 10), (False, True, True)],
    ]

    for (
        kernel_h,
        kernel_w,
        pad, # (h_pad, w_pad)
        stride,
        dilation,
        out_channels,
        shape_chw, # (C, H, W)
        composite, # (has_pad, has_bias, has_activation)
    ) in trials:
        input_full_shape = (1, *shape_chw) # (N, C, H, W)
        groups = 1
        
        # Prepare random input for the 'a' variable
        input_np = np.random.uniform(-1, 1, input_full_shape).astype(dtype)
        inputs_dict = {"a": tvm_nd_array_mock(input_np)} # Wrap numpy array in mock tvm.nd.array

        model_instance, _ = _get_conv_model( # `_` for TVM-style params, not directly used by run_pytorch_model
            input_full_shape,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            groups,
            dtype,
            out_channels,
            list(inputs_dict.keys())[0], # The input name "a" (dummy arg now)
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        
        # Run with normal PyTorch (reference)
        reference_out = run_pytorch_model(model_instance, inputs_dict, 0, device, enable_compiler=False)[0]
        # Run with TorchInductor (compiled)
        compiled_out = run_pytorch_model(model_instance, inputs_dict, 0, device, enable_compiler=True)[0]

        torch.testing.assert_allclose(
            compiled_out, reference_out, rtol=1e-5, atol=1e-5
        )


def _get_batchnorm_model(in_shape, channels, dtype_str, input_np_data, epsilon=0.0001):
    """Returns a PyTorch BatchNormModule instance."""
    torch_dtype = to_torch_dtype(dtype_str)

    # Generate random numpy data for parameters that are not derived from input
    gamma_np = np.random.uniform(-1, 1, (channels)).astype(dtype_str)
    beta_np = np.random.uniform(-1, 1, (channels)).astype(dtype)

    # Compute mean and variance from the actual input data, matching TVM test setup
    mean_np = np.mean(input_np_data, axis=(0, 2, 3), keepdims=False).astype(dtype_str)
    variance_np = np.var(input_np_data, axis=(0, 2, 3), keepdims=False).astype(dtype_str)

    class BatchNormModule(nn.Module):
        def __init__(self):
            super().__init__()
            # For inference mode (training=False), running_mean/var are read-only.
            # We initialize them with the pre-computed values.
            self.register_buffer('running_mean', torch.tensor(mean_np, dtype=torch_dtype))
            self.register_buffer('running_var', torch.tensor(variance_np, dtype=torch_dtype))
            self.weight = nn.Parameter(torch.tensor(gamma_np, dtype=torch_dtype))
            self.bias = nn.Parameter(torch.tensor(beta_np, dtype=torch_dtype))
            self.eps = epsilon

        def forward(self, a):
            # F.batch_norm in inference mode uses these fixed values
            return F.batch_norm(
                a,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False, # Important: use pre-computed mean/var
                momentum=0.1, # PyTorch default, not exposed in TVM signature for this op
                eps=self.eps
            )
    
    model_instance = BatchNormModule()
    # The original TVM test's `params` dict for `build_and_run` was empty for batch_norm.
    # The mean/variance, gamma/beta are directly passed to `relay.nn.batch_norm` as constants/variables.
    params_np_dict_dummy = {} # Return an empty dict to match `_get_conv_model`'s signature
    return model_instance, params_np_dict_dummy


@pytest.mark.parametrize("dtype", ["float16"])
@requires_openclml
def _test_batchnorm(dtype): # Removed 'device' from signature
    in_shape = (1, 8, 64, 64)
    channels = 8

    input_np = np.random.uniform(-1, 1, in_shape).astype(dtype)
    inputs_dict = {"a": tvm_nd_array_mock(input_np)} # Mock tvm.nd.array

    model_instance, _ = _get_batchnorm_model(
        in_shape, channels, dtype, input_np, epsilon=0.0001
    )

    reference_out = run_pytorch_model(model_instance, inputs_dict, 0, device, enable_compiler=False)[0]
    compiled_out = run_pytorch_model(model_instance, inputs_dict, 0, device, enable_compiler=True)[0]

    torch.testing.assert_allclose(
        compiled_out, reference_out, rtol=1e-5, atol=1e-5
    )


def _get_concat_model(dtype_str, axis):
    """Returns a PyTorch ConcatModule instance."""
    torch_dtype = to_torch_dtype(dtype_str)

    class ConcatModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.axis = axis # Store dim for cat

        def forward(self, input_1, input_2): # Inputs correspond to relay.var names
            return torch.cat((input_1, input_2), dim=self.axis)
    
    model_instance = ConcatModule()
    params_np_dict_dummy = {} # No learnable parameters
    return model_instance, params_np_dict_dummy


@pytest.mark.parametrize("dtype", ["float16"])
@requires_openclml
def test_concat(dtype): # Removed 'device' from signature
    in_shape_1 = (1, 16, 16, 16)
    in_shape_2 = (1, 16, 16, 16)
    
    input_1_np = np.random.uniform(-1, 1, in_shape_1).astype(dtype)
    input_2_np = np.random.uniform(-1, 1, in_shape_2).astype(dtype)

    inputs_dict = {
        "input_1": tvm_nd_array_mock(input_1_np),
        "input_2": tvm_nd_array_mock(input_2_np),
    }

    model_instance, _ = _get_concat_model(dtype, axis=1)

    reference_out = run_pytorch_model(model_instance, inputs_dict, 0, device, enable_compiler=False)[0]
    compiled_out = run_pytorch_model(model_instance, inputs_dict, 0, device, enable_compiler=True)[0]

    torch.testing.assert_allclose(
        compiled_out, reference_out, rtol=1e-3, atol=1e-3
    )


def _get_pooling_model(input_shape, pool_size, stride, padding_4tuple_tvm, pooling_type, dtype_str):
    """Returns a PyTorch PoolingModule instance."""
    torch_dtype = to_torch_dtype(dtype_str)

    # PyTorch functional pooling ops take `padding` as `(pad_height, pad_width)` or a single int.
    # TVM's `padding` for pooling ops is (P_top, P_bottom, P_left, P_right).
    # We map this to (P_top, P_left) for PyTorch's `padding` argument.
    pytorch_padding_for_pooling = (padding_4tuple_tvm[0], padding_4tuple_tvm[2])

    class PoolingModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.pooling_type = pooling_type
            self.pool_size = pool_size
            self.stride = stride
            self.padding = pytorch_padding_for_pooling
            self.ceil_mode = False # TVM's default in these tests, matching PyTorch default
            self.count_include_pad = True # PyTorch `avg_pool2d` default. TVM may vary by implementation.

        def forward(self, input_1):
            if self.pooling_type == "max":
                return F.max_pool2d(
                    input_1,
                    kernel_size=self.pool_size,
                    stride=self.stride,
                    padding=self.padding,
                    ceil_mode=self.ceil_mode,
                )
            else: # "avg"
                return F.avg_pool2d(
                    input_1,
                    kernel_size=self.pool_size,
                    stride=self.stride,
                    padding=self.padding,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
    
    model_instance = PoolingModule()
    params_np_dict_dummy = {}
    return model_instance, params_np_dict_dummy


@pytest.mark.parametrize("dtype", ["float16"])
@requires_openclml
def test_avgpool(dtype): # Removed 'device' from signature
    trials = [
        # input_shape         pool_size stride  padding (TVM 4-tuple)    pooling_type
        [(1, 64, 147, 147), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 192, 71, 71), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 288, 35, 35), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 768, 17, 17), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 2048, 17, 17), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 192, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"], # PyTorch padding (0,1)
        [(1, 256, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"], # PyTorch padding (0,1)
        [(1, 288, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"], # PyTorch padding (0,1)
        [(1, 768, 17, 17), (3, 3), (1, 1), (0, 0, 1, 1), "avg"], # PyTorch padding (0,1)
        [(1, 1280, 8, 8), (3, 3), (1, 1), (0, 0, 1, 1), "avg"], # PyTorch padding (0,1)
    ]
    
    for (
        input_shape,
        pool_size,
        stride,
        padding_4tuple_tvm,
        pooling_type,
    ) in trials:
        input_np = np.random.uniform(-1, 1, input_shape).astype(dtype)
        inputs_dict = {"input_1": tvm_nd_array_mock(input_np)}

        model_instance, _ = _get_pooling_model(
            input_shape, pool_size, stride, padding_4tuple_tvm, pooling_type, dtype
        )

        reference_out = run_pytorch_model(model_instance, inputs_dict, 0, device, enable_compiler=False)[0]
        compiled_out = run_pytorch_model(model_instance, inputs_dict, 0, device, enable_compiler=True)[0]

        # Use 1e-3 for float16, as in test_concat
        torch.testing.assert_allclose(
            compiled_out, reference_out, rtol=1e-3, atol=1e-3
        )
