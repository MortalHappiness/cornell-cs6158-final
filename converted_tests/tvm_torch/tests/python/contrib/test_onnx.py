# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""PyTorch to ONNX serialization test cases"""
import pytest
import numpy as np
import os
import tempfile

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as rt

# --- Dtype mapping for TVM string dtypes to PyTorch dtypes ---
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    # PyTorch doesn't have uint16, map to int16. For actual uint values,
    # conversion to a larger signed int type or float might be necessary
    # if values exceed int16 max. For these tests, int16 should be fine for range.
    "uint16": torch.int16,
    "bool": torch.bool,
}

# Default device to use for PyTorch execution
DEFAULT_DEVICE = "cpu"  # For broader compatibility, use CPU. Could be 'cuda' if available.


def get_pytorch_dtype(dtype_str):
    if dtype_str is None:
        return None
    return DTYPE_MAP.get(dtype_str, None) or getattr(torch, dtype_str, None)


def TO_DEVICE(tensor_or_array, device=DEFAULT_DEVICE, dtype_str=None):
    if isinstance(tensor_or_array, np.ndarray):
        tensor = torch.from_numpy(tensor_or_array)
    elif isinstance(tensor_or_array, (list, tuple)):
        # Assume it's a list of tensors/arrays
        return [TO_DEVICE(t, device, dtype_str) for t in tensor_or_array]
    else:  # Assume it's a torch.Tensor
        tensor = tensor_or_array

    if dtype_str is not None:
        dtype = get_pytorch_dtype(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")
        tensor = tensor.to(dtype)
    return tensor.to(device)


# --- Re-implement func_to_onnx for PyTorch models ---
def export_torch_to_onnx(model: torch.nn.Module, input_data, name: str, is_dyn: bool = False):
    model.eval()
    dummy_input_tensors = TO_DEVICE(input_data)

    # Prepare input and output names for ONNX export
    input_names = [f"input_{i}" for i in range(len(dummy_input_tensors))]
    
    # Get example output to determine output names and dynamic axes for outputs
    with torch.no_grad():
        example_outputs = model(*dummy_input_tensors) if isinstance(dummy_input_tensors, (list, tuple)) else model(dummy_input_tensors)
    
    if isinstance(example_outputs, torch.Tensor):
        output_names = ["output_0"]
        example_outputs = [example_outputs]
    else: # Assume it's a tuple/list of tensors
        output_names = [f"output_{i}" for i in range(len(example_outputs))]

    dynamic_axes = None
    if is_dyn:
        dynamic_axes = {}
        # Make all dimensions dynamic for all inputs
        for i, input_t in enumerate(dummy_input_tensors):
            dynamic_axes[input_names[i]] = {j: f"input_{i}_dim{j}" for j in range(len(input_t.shape))}
        
        # Make all dimensions dynamic for all outputs
        for i, output_t in enumerate(example_outputs):
            dynamic_axes[output_names[i]] = {j: f"output_{i}_dim{j}" for j in range(len(output_t.shape))}

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        export_inputs = dummy_input_tensors if isinstance(dummy_input_tensors, (list, tuple)) else (dummy_input_tensors,)

        torch.onnx.export(
            model,
            export_inputs,
            tmp_path,
            export_params=True,
            opset_version=11,  # Common opset version
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        with open(tmp_path, "rb") as f:
            onnx_model_content = f.read()
    finally:
        os.remove(tmp_path)

    return onnx_model_content


# --- Re-implement run_relay for PyTorch models ---
def run_pytorch_model(model: torch.nn.Module, input_data):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        inputs = TO_DEVICE(input_data, device=DEFAULT_DEVICE)

        # Adjust input handling for models that expect multiple args vs single tensor
        if isinstance(inputs, list):
            pytorch_res = model(*inputs)
        else:
            pytorch_res = model(inputs)

    # Ensure output is a list of numpy arrays
    result = []
    if isinstance(pytorch_res, (list, tuple)):
        for res_tensor in pytorch_res:
            result.append(res_tensor.cpu().numpy())
    elif isinstance(pytorch_res, torch.Tensor):
        result.append(pytorch_res.cpu().numpy())
    else:
        # Handle cases where output might be a scalar (e.g., from sum)
        result.append(np.array(pytorch_res))

    return result


# --- Keep run_onnx as is, but ensure input_data matches expected format ---
def run_onnx(onnx_model_content, input_data):
    sess = rt.InferenceSession(onnx_model_content)
    input_names_map = {}

    # Ensure input_data is a list of numpy arrays for ONNX Runtime
    input_numpy_arrays = []
    for data_item in input_data:
        if isinstance(data_item, torch.Tensor):
            input_numpy_arrays.append(data_item.cpu().numpy())
        else:
            input_numpy_arrays.append(data_item)

    # ONNX Runtime input names are derived from torch.onnx.export's input_names
    for i, input_meta in enumerate(sess.get_inputs()):
        input_names_map[input_meta.name] = input_numpy_arrays[i]

    output_names = [out.name for out in sess.get_outputs()]
    res = sess.run(output_names, input_names_map)
    return res


# --- Modified verify_results to use PyTorch model ---
def verify_results(
    pytorch_model: torch.nn.Module, indata, test_name, rtol=1e-7, atol=0, is_dyn=False
):
    pytorch_results = run_pytorch_model(pytorch_model, indata)
    onnx_results = run_onnx(
        export_torch_to_onnx(pytorch_model, indata, test_name, is_dyn), indata
    )

    for pytorch_res, onnx_res in zip(pytorch_results, onnx_results):
        np.testing.assert_allclose(pytorch_res, onnx_res, rtol=rtol, atol=atol)


def test_add():
    class TestAdd(nn.Module):
        def forward(self, x, y):
            return x + y

    dtype_str = "float32"
    x_data = np.random.rand(5, 10, 5).astype(dtype_str)
    y_data = np.random.rand(5, 10, 5).astype(dtype_str)

    verify_results(TestAdd(), [x_data, y_data], "test_add")


def test_bias_add():
    class TestBiasAdd(nn.Module):
        def forward(self, x, bias):
            # Assuming NCHW format for x, bias added to channel dimension
            # bias_add in TVM (without axis explicitly specified) defaults to axis 1 for NCHW.
            # If x is 4D, bias is 1D (C,), then reshape bias to (1, C, 1, 1) for broadcasting
            return x + bias.reshape(1, -1, 1, 1)

    for dtype_str in ["float16", "float32"]:
        xshape = (10, 2, 3, 4)
        bshape = (2,)  # Channels are 2
        rtol = 1e-2 if dtype_str == "float16" else 1e-5
        
        # Ensure bias matches the channel dimension
        x_data = np.random.uniform(size=xshape).astype(dtype_str)
        y_data = np.random.uniform(size=bshape).astype(dtype_str)

        verify_results(TestBiasAdd(), [x_data, y_data], "test_bias_add", rtol=rtol)


def test_conv2d():
    class TestConv2d(nn.Module):
        def __init__(self, padding, groups, dilation, channels, kernel_size):
            super().__init__()
            self.padding = padding
            self.groups = groups
            self.dilation = dilation
            # channels and kernel_size are often inferred in PyTorch from weight shape
            # but stored here for completeness if needed for module init or debugging.

        def forward(self, x, w):
            # PyTorch F.conv2d expects weight in (out_channels, in_channels/groups, kH, kW)
            # TVM relay.nn.conv2d's kernel_layout is typically OIHW (out_channels, in_channels, kH, kW)
            # This matches PyTorch's expectation.
            return F.conv2d(
                x,
                w,
                bias=None,
                stride=1,  # TVM default stride is 1 if not specified
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

    def verify_conv2d(
        dtype_str, scale, dshape, kshape, padding=(1, 1), groups=1, dilation=(1, 1), **attrs
    ):
        model = TestConv2d(padding, groups, dilation, attrs["channels"], attrs["kernel_size"])
        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype_str)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype_str)
        verify_results(
            model, [data, kernel], "test_conv2d", rtol=1e-5, atol=1e-5, is_dyn=True
        )

    # All these calls specify padding, dilation, groups, channels, kernel_size explicitly.
    # The default stride for TVM conv2d is (1,1).
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_conv2d(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=32, groups=32, kernel_size=(3, 3)
    )

    dshape = (1, 32, 18, 18)
    kshape = (32, 4, 3, 3)
    verify_conv2d(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=32, groups=8, kernel_size=(3, 3)
    )

    # also group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (64, 1, 3, 3) # This is depthwise: out_channels=groups * multiplier, in_channels_per_group=1
    verify_conv2d(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=64, groups=32, kernel_size=(3, 3)
    )

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    verify_conv2d("float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(3, 3))

    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    verify_conv2d("float32", 1, dshape, kshape, padding=(2, 2), channels=10, kernel_size=(3, 3))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    verify_conv2d(
        "float32",
        1,
        dshape,
        kshape,
        padding=(1, 1),
        channels=10,
        kernel_size=(3, 3),
        dilation=(3, 3),
    )

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 2, 2)
    verify_conv2d(
        "float32",
        1,
        dshape,
        kshape,
        padding=(2, 2),
        channels=10,
        kernel_size=(2, 2),
        dilation=(1, 1),
    )

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 4, 4)
    verify_conv2d("float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 4, 4)
    verify_conv2d("float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4))


def test_conv2d_transpose():
    """Conv2d_Transpose unit tests."""

    class TestConv2dTranspose(nn.Module):
        def __init__(self, padding, groups, dilation, channels, kernel_size):
            super().__init__()
            self.padding = padding
            self.groups = groups
            self.dilation = dilation
            # channels and kernel_size inferred.

        def forward(self, x, w):
            # PyTorch F.conv_transpose2d expects weight in (in_channels, out_channels/groups, kH, kW)
            # TVM relay.nn.conv2d_transpose's kernel_layout is typically IOHW (in_channels, out_channels, kH, kW)
            # This matches PyTorch's expectation.
            return F.conv_transpose2d(
                x,
                w,
                bias=None,
                stride=1,  # TVM default stride is 1 if not specified
                padding=self.padding,
                output_padding=0, # TVM conv2d_transpose doesn't specify output_padding. Default to 0.
                dilation=self.dilation,
                groups=self.groups,
            )

    def verify_conv2d_transpose(
        dtype_str, scale, dshape, kshape, padding=(1, 1), groups=1, dilation=(1, 1), **attrs
    ):
        model = TestConv2dTranspose(
            padding, groups, dilation, attrs["channels"], attrs["kernel_size"]
        )
        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype_str)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype_str)
        verify_results(model, [data, kernel], "test_conv2d_transpose", rtol=1e-5, atol=1e-5)

    dshape = (1, 3, 224, 224)
    kshape = (3, 10, 3, 3)
    verify_conv2d_transpose(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(3, 3)
    )

    dshape = (1, 3, 224, 224)
    kshape = (3, 10, 3, 3)
    verify_conv2d_transpose(
        "float32", 1, dshape, kshape, padding=(2, 2), channels=10, kernel_size=(3, 3)
    )

    dshape = (1, 3, 18, 18)
    kshape = (3, 10, 2, 2)
    verify_conv2d_transpose(
        "float32",
        1,
        dshape,
        kshape,
        padding=(2, 2),
        channels=10,
        kernel_size=(2, 2),
        dilation=(1, 1),
    )

    dshape = (1, 3, 18, 18)
    kshape = (3, 10, 4, 4)
    verify_conv2d_transpose(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4)
    )

    dshape = (1, 3, 18, 18)
    kshape = (3, 10, 4, 4)
    verify_conv2d_transpose(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4)
    )


def test_reshape():
    class TestReshape(nn.Module):
        def __init__(self, newshape_tvm):
            super().__init__()
            self.newshape_tvm = newshape_tvm

        def forward(self, x):
            # Convert TVM's newshape (where 0 means copy input dimension) to PyTorch compatible shape
            current_shape = list(x.shape)
            actual_newshape = []
            input_dim_idx = 0
            
            for dim_val in self.newshape_tvm:
                if dim_val == 0:
                    if input_dim_idx >= len(current_shape):
                        raise ValueError(f"Cannot copy dimension: input shape {current_shape} exhausted at index {input_dim_idx}")
                    actual_newshape.append(current_shape[input_dim_idx])
                    input_dim_idx += 1
                elif dim_val == -1:
                    # PyTorch handles -1 for inference, ensure only one -1
                    actual_newshape.append(-1)
                    input_dim_idx += 1 # advance input_dim_idx for the original shape (it's conceptually 'consumed')
                else:
                    actual_newshape.append(dim_val)
                    input_dim_idx += 1
            
            # PyTorch's reshape automatically handles the -1 inference if there's exactly one.
            return x.reshape(tuple(actual_newshape))

    def verify_reshape(shape, newshape_tvm):
        model = TestReshape(newshape_tvm)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(model, [x_data], "test_reshape", rtol=1e-5, atol=1e-5)

    verify_reshape((2, 3, 4), tuple(np.array([4, 2, 3], dtype=np.int64)))
    verify_reshape((2, 3, 4), tuple(np.array([2, 0, 0], dtype=np.int64))) # 0 means copy dim
    verify_reshape((2, 3, 4), tuple(np.array([0, -1], dtype=np.int64))) # 0 means copy dim, -1 infers
    verify_reshape((2, 3, 4), tuple(np.array([-1, 0], dtype=np.int64))) # -1 infers, 0 means copy dim


def test_transpose():
    class TestTranspose(nn.Module):
        def __init__(self, axes):
            super().__init__()
            self.axes = axes

        def forward(self, x):
            # If axes is None in TVM, it means reverse the dimensions.
            # PyTorch's permute needs an explicit tuple for dims.
            if self.axes is None:
                return x.permute(tuple(reversed(range(x.ndim))))
            return x.permute(self.axes)

    def verify_transpose(shape, axes):
        model = TestTranspose(axes)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(model, [x_data], "test_transpose", rtol=1e-5, atol=1e-5)

    verify_transpose((1, 2, 3, 4), (0, 2, 3, 1))
    verify_transpose((1, 2, 3, 4), (0, 3, 2, 1))
    # Test with axes=None implicitly reversing dimensions
    verify_transpose((1, 2, 3, 4), None)


def test_dense():
    class TestDense(nn.Module):
        def forward(self, data, weight):
            # TVM relay.nn.dense often expects weight in (out_features, in_features) format.
            # PyTorch's F.linear expects (N, *, in_features) and (out_features, in_features).
            # The weights are typically stored in the (out_features, in_features) format.
            return F.linear(data, weight)

    def verify_dense(d_shape, w_shape):
        model = TestDense()
        x_data = np.random.uniform(size=d_shape).astype("float32")
        w_data = np.random.uniform(size=w_shape).astype("float32")
        verify_results(model, [x_data, w_data], "test_dense", rtol=1e-5, atol=1e-5)

    verify_dense((1, 8), (16, 8)) # (batch, in_features) @ (out_features, in_features)
    verify_dense((1, 4), (3, 4))


def test_max_pool():
    class TestMaxPool2d(nn.Module):
        def __init__(self, pool_size, strides, padding, ceil_mode):
            super().__init__()
            self.kernel_size = pool_size
            self.stride = strides
            self.padding = padding
            self.ceil_mode = ceil_mode

        def forward(self, x):
            return F.max_pool2d(
                x, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode
            )

    def verify_max_pool(x_shape, pool_size, strides, padding, ceil_mode):
        model = TestMaxPool2d(pool_size, strides, padding, ceil_mode)
        x_data = np.random.uniform(size=x_shape).astype("float32")
        verify_results(model, [x_data], "test_max_pool", rtol=1e-5, atol=1e-5)

    verify_max_pool(
        (1, 4, 16, 16), pool_size=(2, 2), strides=(2, 2), padding=(0, 0), ceil_mode=False
    )


def test_batch_flatten():
    class TestBatchFlatten(nn.Module):
        def forward(self, data):
            # Flatten all dimensions except the batch dimension (dim 0)
            return data.flatten(1)

    def verify_test_batch_flatten(d_shape):
        model = TestBatchFlatten()
        x_data = np.random.uniform(size=d_shape).astype("float32")
        verify_results(model, [x_data], "test_batch_flatten", rtol=1e-5, atol=1e-5)

    verify_test_batch_flatten((1, 2, 3, 4))
    verify_test_batch_flatten((1, 8))


def test_batch_norm():
    class TestBatchNorm(nn.Module):
        def __init__(self, axis=1):
            super().__init__()
            self.axis = axis

        def forward(self, data, gamma, beta, moving_mean, moving_var):
            # PyTorch F.batch_norm expects NCHW (channel-first) layout.
            # TVM's axis can be 1 (NCHW) or 3 (NHWC).
            # If data is NHWC and axis=3, convert to NCHW before batch_norm, then back.
            needs_permute = False
            if self.axis == 3:  # Assume input is NHWC, convert to NCHW
                data = data.permute(0, 3, 1, 2)
                # Gamma/beta/mean/var should match the channel dimension size after permute
                # They are already 1D arrays matching channel count, so no explicit reshape needed here.
                needs_permute = True
            elif self.axis != 1:
                raise ValueError(
                    f"Unsupported axis for batch_norm in PyTorch conversion: {self.axis}"
                )

            # PyTorch F.batch_norm parameters: input, running_mean, running_var, weight, bias, training, momentum, eps
            out = F.batch_norm(
                data,
                moving_mean,
                moving_var,
                gamma,
                beta,
                training=False,  # Assuming inference mode as per typical Relay graph compilation
                momentum=0.1,  # Default momentum
                eps=1e-5,  # Default epsilon, or pass from attrs if available
            )

            if needs_permute:  # Convert back to NHWC
                out = out.permute(0, 2, 3, 1)

            return out

    def verify_batch_norm(axis=1):
        for dtype_str in ["float16", "float32"]:
            # TVM test uses (2, 4, 4, 1) data and axis=1 or axis=3.
            # If axis=1, it means (N, C, H, W)
            # If axis=3, it means (N, H, W, C)
            if axis == 1: # For data_shape (2,4,4,1) and axis=1, it means C=4 which is impossible.
                          # The data.type_annotation.shape[axis].value would be 4 for axis=1.
                          # So, if axis=1, it means (N,C,H,W) with C=4. data_shape should be (2,4,4,1) or something else.
                          # Given the gamma_shape = (data.type_annotation.shape[axis].value,)
                          # For (2,4,4,1) and axis=1, gamma_shape would be (4,).
                          # For (2,4,4,1) and axis=3, gamma_shape would be (1,).
                          # Let's adjust data_shape for axis=1 to match NCHW interpretation where C=4
                data_shape = (2, 4, 4, 4) # (N,C,H,W) C=4
                gamma_shape = (4,)
            elif axis == 3: # NHWC
                data_shape = (2, 4, 4, 1) # N=2, H=4, W=4, C=1
                gamma_shape = (1,)
            else:
                raise ValueError(f"Unsupported axis for test_batch_norm: {axis}")

            model = TestBatchNorm(axis=axis)

            x_data = np.random.uniform(size=data_shape).astype(dtype_str)
            beta = np.random.uniform(size=gamma_shape).astype(dtype_str)
            gamma = np.random.uniform(size=gamma_shape).astype(dtype_str)
            moving_mean = np.random.uniform(size=gamma_shape).astype(dtype_str)
            moving_var = np.random.uniform(size=gamma_shape).astype(dtype_str)
            verify_results(
                model,
                [x_data, gamma, beta, moving_mean, moving_var],
                "test_batch_norm",
                rtol=1e-1, # Increased rtol for float16
                atol=1e-1,
            )

    verify_batch_norm(axis=1)
    verify_batch_norm(axis=3)


def test_pad():
    """Pad unit test."""

    class TestPad(nn.Module):
        def __init__(self, pad_width_tvm):
            super().__init__()
            self.pad_width_tvm = pad_width_tvm

        def forward(self, x):
            # Convert TVM pad_width to PyTorch pad format
            # TVM: ((b0,a0), (b1,a1), ..., (bN,aN)) for dim 0..N
            # PyTorch: (aN,bN, ..., a1,b1, a0,b0) for dim N..0
            pad_list = []
            for b, a in reversed(self.pad_width_tvm):
                pad_list.extend([a, b])
            return F.pad(x, pad=tuple(pad_list), mode="constant", value=0.0)

    def verify_pad():
        dshape = (4, 10, 7, 7)
        pad_width = ((1, 1), (2, 2), (3, 3), (4, 4))
        model = TestPad(pad_width)
        x_data = np.random.randint(low=-255, high=255, size=dshape).astype(np.int32)
        verify_results(model, [x_data], "test_pad", rtol=1e-5, atol=1e-5)

    verify_pad()


def test_sofmax():
    class TestSoftmax(nn.Module):
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, x):
            return F.softmax(x, dim=self.axis)

    def verify_sofmax():
        for dtype_str in ["float32"]:
            shape = (10, 4)
            model = TestSoftmax(axis=1)
            x_data = np.random.uniform(size=shape).astype(dtype_str)
            verify_results(model, [x_data], "test_softmax", rtol=1e-5, atol=1e-5)

    verify_sofmax()


def test_squeeze():
    class TestSqueezeSingleAxis(nn.Module):
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, x):
            return torch.squeeze(x, dim=self.axis) if self.axis is not None else torch.squeeze(x)
            
    class TestMultiAxisSqueeze(nn.Module):
        def __init__(self, axes):
            super().__init__()
            # PyTorch's squeeze takes a single dim. If multiple, apply sequentially.
            # Squeezing higher dimensions first prevents index shifts.
            self.axes = sorted(list(axes), reverse=True) 

        def forward(self, x):
            out = x
            for ax in self.axes:
                out = torch.squeeze(out, dim=ax)
            return out

    def verify_squeeze(shape, dtype_str, axis):
        if isinstance(axis, (list, tuple)):
            model = TestMultiAxisSqueeze(axis)
        else: # single int or None
            model = TestSqueezeSingleAxis(axis)

        x_data = np.random.random_sample(shape).astype(dtype_str)
        verify_results(model, [x_data], "test_squeeze", rtol=1e-5, atol=1e-5)

    verify_squeeze((1, 3, 2, 5), "float32", None)
    verify_squeeze(
        (1, 3, 1),
        "float32",
        [2],
    )
    verify_squeeze((1, 2, 1, 2, 1), "float32", [0, 2])


def test_mean():
    class TestMean(nn.Module):
        def __init__(self, axis, keepdims, exclude):
            super().__init__()
            self.axis = axis
            self.keepdims = keepdims
            self.exclude = exclude

        def forward(self, x):
            # PyTorch torch.mean does not have 'exclude'.
            # If exclude is True, it means mean over all axes *except* those in self.axis.
            if self.exclude:
                if self.axis is None:
                    # Exclude None implies reducing nothing.
                    # This case seems to be an identity operation in TVM Relay.
                    # This is tricky as PyTorch mean without dim reduces all.
                    # If axis is None and exclude is True, it implies mean over no axes.
                    # This is just an identity function, returning the input `x`
                    return x
                
                # Calculate non-excluded dimensions
                all_dims = set(range(x.ndim))
                if isinstance(self.axis, (list, tuple)):
                    exclude_dims = set(self.axis)
                else: # single int axis
                    exclude_dims = {self.axis}
                
                reduce_dims = sorted(list(all_dims - exclude_dims))
                # If reduce_dims is empty and keepdims is True, the result shape is same as input.
                # If reduce_dims is empty and keepdims is False, the result shape is same as input.
                # If reduce_dims is empty, torch.mean will raise error "cannot reduce dim with no dims"
                # So if no dims to reduce, return identity
                if not reduce_dims:
                    return x # No dimensions to reduce, act as identity
                
                return torch.mean(x, dim=reduce_dims, keepdim=self.keepdims)
            else:
                return torch.mean(x, dim=self.axis, keepdim=self.keepdims)

    def verify_mean(data_shape, axis, exclude, keepdims):
        dtype_str = "float32"
        model = TestMean(axis, keepdims, exclude)
        x_data = np.random.uniform(size=data_shape).astype(dtype_str)
        verify_results(model, [x_data], "test_mean", rtol=1e-5, atol=1e-5)

    verify_mean((1, 2), 0, False, False)
    verify_mean((1, 2), 0, True, False) 
    verify_mean((1, 2), 0, True, True)  
    verify_mean((1, 2), 1, True, True)  
    verify_mean((3, 2, 1), 1, False, True)


def test_split():
    class TestSplit(nn.Module):
        def __init__(self, indices_or_sections, axis):
            super().__init__()
            self.indices_or_sections = indices_or_sections
            self.axis = axis

        def forward(self, x):
            # torch.split returns a tuple of tensors. This matches relay.split().astuple()
            return torch.split(x, self.indices_or_sections, dim=self.axis)

    def verify_split(dshape, indices_or_sections, axis=None):
        dtype_str = "float32"
        model = TestSplit(indices_or_sections, axis)
        x_data = np.random.uniform(size=dshape).astype(dtype_str)
        verify_results(model, [x_data], "test_split", rtol=1e-5, atol=1e-5)

    verify_split((5, 5, 2, 2), 5, axis=1) # 5 sections along axis 1
    verify_split((5, 5, 2, 2), 5, axis=0) # 5 sections along axis 0
    verify_split((5, 5, 2, 2), [1, 3, 4], axis=0) # split by sizes [1, 3, 4] along axis 0
    verify_split((5, 5, 2, 2), [1, 3, 4], axis=1) # split by sizes [1, 3, 4] along axis 1


def test_concatenate():
    class TestConcatenate(nn.Module):
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, *in_vars):
            return torch.cat(in_vars, dim=self.axis)

    def verify_concatenate(shapes, axis, dtype_str="float32"):
        # Create a dummy model instance; actual inputs are passed as *args to forward
        model = TestConcatenate(axis)
        in_data = [np.random.uniform(size=shape).astype(dtype_str) for shape in shapes]
        verify_results(model, in_data, "test_concatenate", rtol=1e-5, atol=1e-5)

    verify_concatenate([(2,), (2,), (2,)], -1)
    verify_concatenate([(2, 3, 4), (2, 2, 4), (2, 5, 4)], 1)
    verify_concatenate([(1, 2, 4), (1, 2, 3), (1, 2, 7), (1, 2, 8), (1, 2, 1)], -1)
    verify_concatenate([(5, 6, 7, 3), (16, 6, 7, 3), (12, 6, 7, 3), (8, 6, 7, 3), (2, 6, 7, 3)], 0)
    verify_concatenate([(1, 14400), (1, 2400), (1, 640), (1, 240)], 1)


def test_strided_slice():
    # Helper for strided_slice:
    def _convert_slice_args(data_shape, begin_tvm, end_tvm, strides_tvm, slice_mode):
        slices = []

        # Pad begin, end, strides to match data_shape length, using defaults
        begin_padded = list(begin_tvm) + [0] * (len(data_shape) - len(begin_tvm))
        end_padded = list(end_tvm) + [data_shape[i] for i in range(len(end_tvm), len(data_shape))]
        strides_padded = list(strides_tvm) + [1] * (len(data_shape) - len(strides_tvm)) if strides_tvm is not None else [1] * len(data_shape)

        for i in range(len(data_shape)):
            b = begin_padded[i]
            e = end_padded[i]
            s = strides_padded[i]

            # Handle negative indices for begin/end (NumPy style)
            if b < 0:
                b += data_shape[i]
            if e < 0:
                e += data_shape[i]

            # Clip bounds to actual dimension size before further processing
            b = max(0, min(b, data_shape[i]))

            effective_end = e
            if slice_mode == "size":
                # In size mode, `e` represents the size of the slice, not the end index.
                # `0` in size mode for ONNX Slice usually means "take up to the end of the dimension"
                if e == 0:
                    effective_end = data_shape[i] # Interpret 0 size as "rest of the dimension"
                else:
                    effective_end = b + e
            
            effective_end = max(0, min(effective_end, data_shape[i]))

            if s == 0:
                raise ValueError("Stride cannot be zero")

            slices.append(slice(b, effective_end, s))

        return tuple(slices)


    class TestStridedSlice(nn.Module):
        def __init__(self, begin, end, strides, slice_mode):
            super().__init__()
            self.begin = begin
            self.end = end
            self.strides = strides
            self.slice_mode = slice_mode

        def forward(self, x):
            slices = _convert_slice_args(x.shape, self.begin, self.end, self.strides, self.slice_mode)
            return x[slices]

    def verify_strided_slice(dshape, begin, end, strides, mode):
        model = TestStridedSlice(begin, end, strides, mode)
        x_data = np.random.uniform(size=dshape).astype("float32")
        verify_results(model, [x_data], "test_strided_slice", rtol=1e-5, atol=1e-5)

    for mode in ["end", "size"]:
        verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 2, 3], None, mode) # strides=None means [1,1,1]
        verify_strided_slice((3, 4, 3), [1, -1, 0], [4, -1, 3], [1, 2], mode) # negative begin/end
        verify_strided_slice(
            (3, 4, 3),
            [
                1,
            ],
            [4, -3],
            None,
            mode,
        )
        # Note: the original test uses [4, -5, 4] for end with dshape=(3,4,3), this means the second end index (for dim 1, size 4)
        # is -5, which after conversion to positive index is -1. This means the slice for this dimension will be empty.
        # This will work as long as PyTorch's slicing and ONNX's slice operator handle empty dimensions consistently.
        verify_strided_slice((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], mode)
        verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, -3], [2, 1, 1], mode)
        verify_strided_slice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], mode)
        verify_strided_slice((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2], mode)
        verify_strided_slice((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], mode)

        verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 1000, 3], None, mode) # end out of bounds
        verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4], None, mode) # end shorter than dims
        verify_strided_slice((3, 4, 3), [1, 1], [4, 4, 3], None, mode) # begin shorter than dims
        verify_strided_slice((3, 4, 3), [1, 1], [4, 4, 3], [1, 1, 2], mode)


def test_cmp_type():
    class TestCmpType(nn.Module):
        def __init__(self, op_name):
            super().__init__()
            self.op_name = op_name

        def forward(self, x, y):
            if self.op_name == "greater":
                return x > y
            elif self.op_name == "less":
                return x < y
            elif self.op_name == "equal":
                return x == y
            else:
                raise ValueError(f"Unknown comparison op: {self.op_name}")

    for op_name, ref_np_op in (
        ("greater", np.greater),
        ("less", np.less),
        ("equal", np.equal),
    ):
        x_shape = (10, 4)
        y_shape = (5, 10, 1)
        dtype_str = "float32" # Original TVM `TensorType` for `relay.greater` doesn't specify dtype, defaults to float32
        model = TestCmpType(op_name)
        x_data = np.random.rand(*x_shape).astype(dtype_str)
        y_data = np.random.rand(*y_shape).astype(dtype_str)
        verify_results(model, [x_data, y_data], "test_cmp_type", rtol=1e-5, atol=1e-5)


def test_unary_identity():
    class TestUnaryIdentity(nn.Module):
        def __init__(self, op_name, dtype_str):
            super().__init__()
            self.op_name = op_name
            self.dtype = get_pytorch_dtype(dtype_str)

        def forward(self, x):
            if self.op_name == "zeros_like":
                return torch.zeros_like(x, dtype=self.dtype)
            elif self.op_name == "ones_like":
                return torch.ones_like(x, dtype=self.dtype)
            else:
                raise ValueError(f"Unknown unary op: {self.op_name}")

    for dtype_str in ["int16", "float32", "float64"]:
        for op_name, ref_np_op in [("zeros_like", np.zeros_like), ("ones_like", np.ones_like)]:
            shape = (8, 9, 4)
            model = TestUnaryIdentity(op_name, dtype_str)
            x_data = np.random.rand(*shape).astype(dtype_str)
            verify_results(model, [x_data], "test_unary_identity", rtol=1e-5, atol=1e-5)


def test_binary_op():
    class TestBinaryOp(nn.Module):
        def __init__(self, op_name):
            super().__init__()
            self.op_name = op_name

        def forward(self, x, y):
            if self.op_name == "add":
                return x + y
            elif self.op_name == "subtract":
                return x - y
            elif self.op_name == "multiply":
                return x * y
            elif self.op_name == "divide":
                return x / y
            else:
                raise ValueError(f"Unknown binary op: {self.op_name}")

    for op_name, ref_np_op in [
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
        ("divide", np.divide),
    ]:
        for dtype_str in ["float32"]:
            model = TestBinaryOp(op_name)
            x_data = np.random.rand(5, 10, 5).astype(dtype_str)
            y_data = np.random.rand(5, 10, 5).astype(dtype_str)
            verify_results(model, [x_data, y_data], "test_binary_op", rtol=1e-5, atol=1e-5)


def test_tuple_types():
    # Helper for various tuple operations, split and concatenate
    class TestTupleOps(nn.Module):
        def __init__(self, op_type, indices_or_sections=None, axis=None):
            super().__init__()
            self.op_type = op_type
            self.indices_or_sections = indices_or_sections
            self.axis = axis

        def forward(self, x):
            if self.op_type == "split_concat":
                # y = relay.split(x, indices_or_sections, axis=axis)
                # z = relay.concatenate(y, axis=axis)
                y_split = torch.split(x, self.indices_or_sections, dim=self.axis)
                z_concat = torch.cat(y_split, dim=self.axis)
                return z_concat
            elif self.op_type == "split_astuple":
                # split_z = relay.split(z, indices_or_sections, axis=axis)
                # func = relay.Function([x], split_z.astuple())
                # This assumes 'z' is the input here, but the test passes 'x_data' to the function
                # So we simulate one split and return its tuple output
                return torch.split(x, self.indices_or_sections, dim=self.axis)
            elif self.op_type == "tuple_add_sub":
                # out = relay.Tuple([y[0] + y[1], y[0] - y[1]])
                # Here, `x` is the original input, y[0] and y[1] are results of a split
                # Simulate the split first
                y_split = torch.split(x, self.indices_or_sections, dim=self.axis)
                return y_split[0] + y_split[1], y_split[0] - y_split[1]
            elif self.op_type == "tuple_concat":
                # z = relay.concatenate(out, axis=axis)
                # This assumes 'out' is a tuple of tensors (from previous step), but `x` is the input.
                # Simulate the split and the tuple operation (add/sub) then concatenate.
                y_split = torch.split(x, self.indices_or_sections, dim=self.axis)
                out_tuple = (y_split[0] + y_split[1], y_split[0] - y_split[1])
                z_concat = torch.cat(out_tuple, dim=self.axis)
                return z_concat
            else:
                raise ValueError(f"Unknown tuple operation type: {self.op_type}")

    def verify_tuple_types(dshape, indices_or_sections, axis=None, dtype_str="float32"):
        x_data = np.random.uniform(size=dshape).astype(dtype_str)

        # 1. split then concatenate
        model_split_concat = TestTupleOps("split_concat", indices_or_sections, axis)
        verify_results(model_split_concat, [x_data], "test_tuple_types_split_concat", rtol=1e-5, atol=1e-5)

        # 2. split and return as tuple
        model_split_astuple = TestTupleOps("split_astuple", indices_or_sections, axis)
        verify_results(model_split_astuple, [x_data], "test_tuple_types_split_astuple", rtol=1e-5, atol=1e-5)

        # 3. tuple arithmetic
        model_tuple_add_sub = TestTupleOps("tuple_add_sub", indices_or_sections, axis)
        verify_results(model_tuple_add_sub, [x_data], "test_tuple_types_tuple_add_sub", rtol=1e-5, atol=1e-5)
        
        # 4. tuple arithmetic then concatenate
        model_tuple_concat = TestTupleOps("tuple_concat", indices_or_sections, axis)
        verify_results(model_tuple_concat, [x_data], "test_tuple_types_tuple_concat", rtol=1e-5, atol=1e-5)


    verify_tuple_types((5, 5, 2, 2), 5, axis=1)
    verify_tuple_types((5, 5, 2, 2), 5, axis=0)
    verify_tuple_types((5, 5, 2, 2), [1, 3, 4], axis=0)
    verify_tuple_types((5, 5, 2, 2), [1, 3, 4], axis=1)


def test_layout_transform():
    class TestLayoutTransform(nn.Module):
        def __init__(self, src_layout, dst_layout):
            super().__init__()
            self.src_layout = src_layout
            self.dst_layout = dst_layout

        def forward(self, x):
            if self.src_layout == "NCHW" and self.dst_layout == "NHWC":
                # (N, C, H, W) -> (N, H, W, C)
                return x.permute(0, 2, 3, 1)
            elif self.src_layout == "NHWC" and self.dst_layout == "NCHW":
                # (N, H, W, C) -> (N, C, H, W)
                return x.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Unsupported layout transform: {self.src_layout} to {self.dst_layout}"
                )

    def verify_layout_transform(dshape, src_layout, dst_layout, dtype_str="float32"):
        model = TestLayoutTransform(src_layout, dst_layout)
        x_data = np.random.uniform(size=dshape).astype(dtype_str)
        verify_results(model, [x_data], "test_layout_transform", rtol=1e-5, atol=1e-5)

    verify_layout_transform((1, 3, 8, 8), "NCHW", "NHWC")
    verify_layout_transform((1, 8, 8, 3), "NHWC", "NCHW")


def test_clip():
    class TestClip(nn.Module):
        def __init__(self, a_min, a_max):
            super().__init__()
            self.a_min = a_min
            self.a_max = a_max

        def forward(self, x):
            return torch.clamp(x, min=self.a_min, max=self.a_max)

    def verify_clip(dshape, a_min, a_max, dtype_str="float32"):
        model = TestClip(a_min, a_max)
        x_data = np.random.uniform(size=dshape).astype(dtype_str)
        verify_results(model, [x_data], "test_clip", rtol=1e-5, atol=1e-5)

    verify_clip((5, 5, 2, 5), 0, 0.2)
    verify_clip((5, 5, 2, 5), 0.2, 0.5)


def test_expand_dims():
    class TestExpandDims(nn.Module):
        def __init__(self, axis, num_newaxis):
            super().__init__()
            self.axis = axis
            self.num_newaxis = num_newaxis

        def forward(self, x):
            out = x
            # Apply unsqueeze repeatedly for multiple new axes
            for _ in range(self.num_newaxis):
                out = torch.unsqueeze(out, dim=self.axis)
            return out

    def verify_expand_dims(dshape, axis, num_newaxis, dtype_str="float32"):
        model = TestExpandDims(axis, num_newaxis)
        x_data = np.random.uniform(size=dshape).astype(dtype_str)
        verify_results(model, [x_data], "test_expand_dims", rtol=1e-5, atol=1e-5)

    verify_expand_dims((1, 1001), 0, 2)
    verify_expand_dims((1, 1, 1001), 2, 2)


def test_lrn():
    """LRN unit test."""

    class TestLRN(nn.Module):
        def __init__(self, size, alpha, beta, bias):
            super().__init__()
            self.size = size
            self.alpha = float(alpha) # Ensure float
            self.beta = float(beta)   # Ensure float
            self.k = float(bias)      # PyTorch uses 'k' for bias in LRN

        def forward(self, x):
            # PyTorch F.local_response_norm assumes NCHW, axis 1 is channels.
            # TVM's lrn uses axis=1 by default.
            return F.local_response_norm(
                x, size=self.size, alpha=self.alpha, beta=self.beta, k=self.k
            )

    def verify_lrn(xshape, size, dtype_str="float32"):
        # TVM default alpha=1.0, beta=1.0, bias=1.0
        model = TestLRN(size=size, alpha=1.0, beta=1.0, bias=1.0)
        x_data = np.random.uniform(size=xshape).astype(dtype_str)
        verify_results(model, [x_data], "test_lrn", rtol=1e-5, atol=1e-5)

    isize = [(1, 1, 480, 640), (1, 3, 224, 224)]
    sizes = [1, 3]
    for i in isize:
        for s in sizes:
            verify_lrn(i, s)


def test_sigmoid():
    """Sigmoid unit test."""

    class TestSigmoid(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    def verify_sigmoid(dshape, dtype_str="float32"):
        model = TestSigmoid()
        x_data = np.random.uniform(size=dshape).astype(dtype_str)
        verify_results(model, [x_data], "test_sigmoid", rtol=1e-4, atol=1e-4)

    isize = [(1, 3, 480, 640), (1, 3, 224, 224)]

    for i in isize:
        verify_sigmoid(i)


def test_copy():
    """Copy unit test."""

    class TestCopy(nn.Module):
        def forward(self, x):
            return x.clone()

    def verify_copy(dshape, dtype_str="float32"):
        model = TestCopy()
        x_data = np.random.uniform(size=dshape).astype(dtype_str)
        verify_results(model, [x_data], "test_copy", rtol=1e-4, atol=1e-4)

    isize = [(1, 3, 480, 640), (1, 3, 224, 224)]

    for i in isize:
        verify_copy(i)


def test_round():
    """Round unit test."""

    class TestRound(nn.Module):
        def forward(self, x):
            return torch.round(x)

    def verify_round(dshape, dtype_str="float32"):
        model = TestRound()
        x_data = np.random.uniform(size=dshape).astype(dtype_str)
        verify_results(model, [x_data], "test_round", rtol=1e-4, atol=1e-4)

    isize = [(1, 3, 480, 640), (1, 3, 224, 224)]

    for i in isize:
        verify_round(i)


def test_cast():
    """Cast unit test."""

    class TestCast(nn.Module):
        def __init__(self, dtype_str):
            super().__init__()
            self.dtype = get_pytorch_dtype(dtype_str)

        def forward(self, x):
            return x.to(self.dtype)

    def verify_cast(dshape, dtype_str):
        model = TestCast(dtype_str)
        x_data = np.random.uniform(size=dshape).astype("float32")
        verify_results(model, [x_data], "test_cast", rtol=1e-4, atol=1e-4)

    isize = [(1, 3, 480, 640), (1, 3, 224, 224)]
    out_dtypes = ["int8", "int16", "uint8", "uint16"]

    for i in isize:
        for o_dtype in out_dtypes:
            verify_cast(i, o_dtype)


@pytest.mark.xfail(reason="F.interpolate's coordinate_transformation_mode and rounding_method are not directly equivalent to TVM Relay's parameters.")
def test_resize():
    """Resize unit test."""

    # Map TVM method to PyTorch mode
    RESIZE_METHOD_MAP = {
        "nearest_neighbor": "nearest",
        "linear": "bilinear",
        "cubic": "bicubic",
    }

    class TestResize(nn.Module):
        def __init__(self, outsize, method, coord_trans, rounding_method):
            super().__init__()
            self.outsize = outsize
            self.method = method
            self.coord_trans = coord_trans
            self.rounding_method = rounding_method

        def forward(self, x):
            mode = RESIZE_METHOD_MAP.get(self.method)
            if mode is None:
                raise ValueError(f"Unsupported resize method: {self.method}")

            align_corners = None
            # align_corners is only applicable for 'bilinear' and 'bicubic'
            if mode in ["bilinear", "bicubic"]:
                if self.coord_trans == "half_pixel":
                    align_corners = False
                elif self.coord_trans == "align_corners":
                    align_corners = True
                elif self.coord_trans == "asymmetric":
                    align_corners = False  # Asymmetric implies non-align, might need recompute_scale_factor too
                else:
                    align_corners = False # PyTorch default for these modes is False

            # PyTorch F.interpolate does not have a direct 'rounding_method' parameter.
            # This is a known divergence, especially for 'nearest' mode.
            # E.g., for "round" or "ceil" in nearest, needs custom implementation.
            # Rely on assert_allclose to catch differences or adjust rtol/atol.

            return F.interpolate(x, size=tuple(self.outsize), mode=mode, align_corners=align_corners)

    def verify_resize(dshape, outsize, method, coord_trans, rounding_method, dtype_str="float32"):
        model = TestResize(outsize, method, coord_trans, rounding_method)
        x_data = np.random.uniform(size=dshape).astype(dtype_str)
        verify_results(model, [x_data], "test_resize", rtol=1e-4, atol=1e-4)

    method_types = ["nearest_neighbor", "linear", "cubic"]
    coord_trans_types = ["half_pixel", "align_corners", "asymmetric"]
    rounding_method_types = ["round", "floor", "ceil"] # PyTorch only 'floor' implicitly for nearest

    isize = (1, 3, 480, 640)

    # Downsample
    osize = (240, 320)
    for method_type in method_types:
        for coord_trans_type in coord_trans_types:
            for rounding_method_type in rounding_method_types:
                # Skip known problematic combinations (as per original TVM comments) or those that diverge heavily
                if (method_type == "nearest_neighbor" and coord_trans_type == "align_corners") or \
                   (method_type == "cubic" and coord_trans_type in ["half_pixel", "align_corners"]):
                    continue
                verify_resize(isize, osize, method=method_type, coord_trans=coord_trans_type, rounding_method=rounding_method_type)

    # Upsample
    osize = (960, 1280)
    for method_type in method_types:
        for coord_trans_type in coord_trans_types:
            for rounding_method_type in rounding_method_types:
                if (method_type == "nearest_neighbor" and coord_trans_type == "align_corners") or \
                   (method_type == "cubic"): # Cubic is often problematic across frameworks for upsampling
                    continue
                verify_resize(isize, osize, method=method_type, coord_trans=coord_trans_type, rounding_method=rounding_method_type)


def test_dyn():
    """Dynamic unit test."""

    class TestDynBcast(nn.Module):
        def forward(self, x, y):
            return x + y

    def verify_dyn_bcast(lhs_shape, rhs_shape, dtype_str):
        model = TestDynBcast()
        lhs_data = np.random.uniform(size=lhs_shape).astype(dtype_str)
        rhs_data = np.random.uniform(size=rhs_shape).astype(dtype_str)
        verify_results(
            model, [lhs_data, rhs_data], "test_dyn_bcast", rtol=1e-5, atol=1e-5, is_dyn=True
        )

    verify_dyn_bcast((1, 3, 32, 1), (1, 3, 1, 3), "float32")
    verify_dyn_bcast((1, 13), (4, 3, 5, 1), "float32")


if __name__ == "__main__":
    pytest.main([__file__]) # Use pytest.main to run tests
