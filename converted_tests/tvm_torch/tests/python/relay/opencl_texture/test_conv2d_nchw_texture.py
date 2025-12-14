import pytest
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import re

# Helper to map TVM string dtypes to torch dtypes
_TVM_TO_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

def tvm_dtype_to_torch_dtype(tvm_dtype_str):
    return _TVM_TO_TORCH_DTYPE_MAP.get(tvm_dtype_str, None)

# Mock tvm.testing.init.Xavier for converting numpy data
class MockXavierInitializer:
    def __call__(self, name, data_np):
        if "weight" in name:
            # For conv2d weights (OIHW), fan_in = C_in * k_H * k_W, fan_out = C_out * k_H * k_W
            if data_np.ndim == 4: # conv2d weights
                c_in, k_h, k_w = data_np.shape[1], data_np.shape[2], data_np.shape[3]
                c_out = data_np.shape[0]
                fan_in = c_in * k_h * k_w
                fan_out = c_out * k_h * k_w
            else: # general case, fallback to fan_in/out from just 2 dimensions
                fan_in = data_np.shape[1] if data_np.ndim > 1 else data_np.size
                fan_out = data_np.shape[0] if data_np.ndim > 0 else 1
            
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            data_np[:] = np.random.uniform(-bound, bound, size=data_np.shape).astype(data_np.dtype)
        elif "bias" in name:
            data_np[:] = np.zeros(data_np.shape).astype(data_np.dtype)
        else:
            data_np[:] = np.random.uniform(-1, 1, size=data_np.shape).astype(data_np.dtype)

# Mock TVM modules for parameterization to allow the original decorators to parse
class MockRelayTestingInit:
    Xavier = MockXavierInitializer

class MockRelay:
    pass

class MockTVM:
    testing = type("testing", (), {
        "parameter": pytest.mark.parametrize,
        "requires_opencl": pytest.mark.skipif(True, reason="OpenCL-specific test, skipping in PyTorch context"),
        "parametrize_targets": lambda *args, **kwargs: pytest.mark.parametrize("target,remote", [('cpu', 'local')] if not torch.cuda.is_available() else [('cuda', 'local')])
    })
    relay = MockRelay()

# Replace original tvm imports with mocks
tvm = MockTVM()
relay = tvm.relay

# Dummy gpu_preprocess from utils.adreno_utils, not used but kept for completeness
def gpu_preprocess(*args, **kwargs):
    # This function is TVM/Adreno specific and has no direct PyTorch equivalent in this context.
    # It usually performs post-processing on TVM output (e.g., layout transform).
    # For PyTorch, the output will already be in the desired format.
    return args[0] # Just return the input

# `dtype` parameter from TVM tests is often a string. Convert it here.
# The `@tvm.testing.parametrize_targets` is replaced by Pytest's `parametrize`
# with a dummy 'cpu' or 'cuda' target, and `remote` is ignored.
# We also add a `dtype_str` parameter to convert to `torch.dtype` inside each test.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 42, 42)
    filter_shape = (96, 32, 3, 3)
    bias_shape = (96,) # PyTorch bias for conv2d is 1D (output channels)

    # Define PyTorch model (equivalent to Relay Function)
    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device)) # Squeeze bias to 1D

        def forward(self, data):
            conv = F.conv2d(
                data,
                self.weight,
                stride=(2, 2),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1) # Reshape bias for broadcasting
            D = F.relu(D)
            return D

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str) # Initialize as 1D
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)

    # Generate input data
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    # Eager execution
    eager_output = model_eager(input_data)

    # Compile and execute
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)

    # Compare results
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad_pass(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 40, 40)
    filter_shape = (96, 32, 2, 2)
    bias_shape = (96,)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device))

        def forward(self, data):
            conv = F.conv2d(
                data,
                self.weight,
                stride=(2, 2),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1)
            D = F.relu(D)
            return D

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str)
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_inceptionv3_35_35_strides(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 48, 35, 35)
    filter_shape = (64, 48, 5, 5)
    bias_shape = (64,)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device))

        def forward(self, data):
            conv = F.conv2d(
                data,
                self.weight,
                stride=(1, 1),
                padding=(2, 2), # TVM [2, 2, 2, 2] implies symmetric padding of 2 in H and W
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1)
            D = F.relu(D)
            return D

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str)
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_resnet50_v2_nchw_3c(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 3, 224, 224)
    filter_shape = (64, 3, 7, 7)
    bias_shape = (64,)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device))

        def forward(self, data):
            conv = F.conv2d(
                data,
                self.weight,
                stride=(2, 2),
                padding=(3, 3), # TVM [3, 3, 3, 3] implies symmetric padding of 3 in H and W
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1)
            D = F.relu(D)
            return D

    np.random.seed(1)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str)
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_inceptionv3_nchw_3c(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 3, 299, 299)
    filter_shape = (64, 3, 3, 3)
    bias_shape = (64,)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device))

        def forward(self, data):
            conv = F.conv2d(
                data,
                self.weight,
                stride=(2, 2),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1)
            D = F.relu(D)
            return D

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str)
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_1x1_16c16spatial(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 16, 256, 256)
    filter_shape = (32, 16, 4, 4)
    bias_shape = (32,)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device))

        def forward(self, data):
            conv = F.conv2d(
                data,
                self.weight,
                stride=(2, 2),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1)
            D = F.relu(D)
            return D

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str)
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_4x4_16c16pad(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 256, 256)
    filter_shape = (32, 32, 4, 4)
    bias_shape = (32,)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device))

        def forward(self, data):
            # TVM `padding=[3, 3, 0, 0]` means (top=3, bottom=3, left=0, right=0)
            # For PyTorch F.conv2d, `padding=(pad_H, pad_W)` implies symmetric padding.
            # So, `padding=(3, 0)` is the correct interpretation for F.conv2d.
            conv = F.conv2d(
                data,
                self.weight,
                stride=(2, 2),
                padding=(3, 0),
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1)
            D = F.relu(D)
            return D

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str)
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_4x4x4_16c16pad(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 256, 256)
    filter_shape = (4, 32, 4, 4)
    bias_shape = (4,)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device))

        def forward(self, data):
            # TVM `padding=[3, 3, 0, 0]` -> `padding=(3, 0)` for F.conv2d
            conv = F.conv2d(
                data,
                self.weight,
                stride=(2, 2),
                padding=(3, 0),
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1)
            D = F.relu(D)
            return D

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str)
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_yolov3_v2_nchw_3c(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 1024, 13, 13)
    filter_shape = (255, 1024, 1, 1)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))

        def forward(self, data):
            conv = F.conv2d(
                data,
                self.weight,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            )
            return conv

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    initializer("weight", filter_data)

    model_eager = TestModel(filter_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_vgg16_winograd_4d(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 512, 28, 28)
    filter_shape = (512, 512, 3, 3)
    bias_shape = (512,)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np, bias_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))
            self.bias = nn.Parameter(torch.from_numpy(bias_data_np).squeeze().to(device))

        def forward(self, data):
            conv = F.conv2d(
                data,
                self.weight,
                padding=(1, 1), # TVM [1, 1, 1, 1] implies symmetric padding of 1 in H and W
                stride=(1, 1),
                dilation=(1, 1),
                groups=1,
            )
            D = conv + self.bias.view(1, -1, 1, 1)
            D = F.relu(D)
            return D

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    bias_data = np.zeros(bias_shape).astype(dtype_str)
    initializer("weight", filter_data)
    initializer("bias", bias_data)

    model_eager = TestModel(filter_data, bias_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `temp`, `stat_file`, `re.findall` and graph analysis are TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_winograd_conv(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 4, 3, 3)
    filter_shape3 = (8, 4, 3, 3)
    filter_shape4 = (8, 8, 3, 3)

    class TestModel(nn.Module):
        def __init__(self, filter_data3_np, filter_data4_np):
            super().__init__()
            self.weight3 = nn.Parameter(torch.from_numpy(filter_data3_np).to(device))
            self.weight4 = nn.Parameter(torch.from_numpy(filter_data4_np).to(device))

        def forward(self, data):
            D = F.conv2d(
                data,
                self.weight3,
                padding=(1, 1), # TVM [1, 1, 1, 1] implies symmetric padding of 1 in H and W
                stride=(1, 1),
                dilation=(1, 1),
                groups=1,
            )
            D = F.conv2d(
                D,
                self.weight4,
                padding=(1, 1), # TVM [1, 1, 1, 1] implies symmetric padding of 1 in H and W
                stride=(1, 1),
                dilation=(1, 1),
                groups=1,
            )
            return D

    np.random.seed(1)
    initializer = MockRelayTestingInit.Xavier()
    filter_data3 = np.zeros(filter_shape3).astype(dtype_str)
    filter_data4 = np.zeros(filter_shape4).astype(dtype_str)

    # Manually apply Xavier init to ensure distinct initialization
    c_in3, k_h3, k_w3 = filter_shape3[1], filter_shape3[2], filter_shape3[3]
    c_out3 = filter_shape3[0]
    fan_in3 = c_in3 * k_h3 * k_w3
    fan_out3 = c_out3 * k_h3 * k_w3
    bound3 = np.sqrt(6.0 / (fan_in3 + fan_out3))
    filter_data3[:] = np.random.uniform(-bound3, bound3, size=filter_shape3).astype(dtype_str)

    c_in4, k_h4, k_w4 = filter_shape4[1], filter_shape4[2], filter_shape4[3]
    c_out4 = filter_shape4[0]
    fan_in4 = c_in4 * k_h4 * k_w4
    fan_out4 = c_out4 * k_h4 * k_w4
    bound4 = np.sqrt(6.0 / (fan_in4 + fan_out4))
    filter_data4[:] = np.random.uniform(-bound4, bound4, size=filter_shape4).astype(dtype_str)

    model_eager = TestModel(filter_data3, filter_data4).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `temp`, `stat_file`, `re.findall` and graph analysis are TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_residual_block(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 40, 40)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (32,) # PyTorch bias for conv2d is 1D

    class TestModel(nn.Module):
        def __init__(self, W1_data_np, B1_data_np, W2_data_np, W3_data_np):
            super().__init__()
            self.W1 = nn.Parameter(torch.from_numpy(W1_data_np).to(device))
            self.B1 = nn.Parameter(torch.from_numpy(B1_data_np).squeeze().to(device))
            self.W2 = nn.Parameter(torch.from_numpy(W2_data_np).to(device))
            self.W3 = nn.Parameter(torch.from_numpy(W3_data_np).to(device))

        def forward(self, A):
            conv1 = F.conv2d(A, self.W1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1)
            D_pre_relu = conv1 + self.B1.view(1, -1, 1, 1) # Residual path uses this
            D_post_relu_branch = F.relu(D_pre_relu)

            conv2_out = F.conv2d(D_post_relu_branch, self.W2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
            D_residual_sum = conv2_out + D_pre_relu # Residual connection
            D_mult = D_residual_sum * 0.15 # scalar constant
            D_relu_final = F.relu(D_mult)

            conv3_out = F.conv2d(D_relu_final, self.W3, stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1)
            final_output = F.relu(conv3_out)
            return final_output

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()

    filter_data1 = np.zeros(filter_shape1).astype(dtype_str)
    bias_data1 = np.zeros(bias_shape1).astype(dtype_str)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)

    filter_data2 = np.zeros(filter_shape2).astype(dtype_str)
    # Manually initialize since TVM uses same "weight" name repeatedly
    c_in2, k_h2, k_w2 = filter_shape2[1], filter_shape2[2], filter_shape2[3]
    c_out2 = filter_shape2[0]
    fan_in2 = c_in2 * k_h2 * k_w2
    fan_out2 = c_out2 * k_h2 * k_w2
    bound2 = np.sqrt(6.0 / (fan_in2 + fan_out2))
    filter_data2[:] = np.random.uniform(-bound2, bound2, size=filter_shape2).astype(dtype_str)

    filter_data3 = np.zeros(filter_shape3).astype(dtype_str)
    # Manually initialize
    c_in3, k_h3, k_w3 = filter_shape3[1], filter_shape3[2], filter_shape3[3]
    c_out3 = filter_shape3[0]
    fan_in3 = c_in3 * k_h3 * k_w3
    fan_out3 = c_out3 * k_h3 * k_w3
    bound3 = np.sqrt(6.0 / (fan_in3 + fan_out3))
    filter_data3[:] = np.random.uniform(-bound3, bound3, size=filter_shape3).astype(dtype_str)

    model_eager = TestModel(filter_data1, bias_data1, filter_data2, filter_data3).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `static_memory_scope` is TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_concat(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 40, 40)
    filter_shape1 = (96, 32, 2, 2)
    filter_shape2 = (32, 96, 2, 2)
    filter_shape3 = (5, 96, 2, 2)
    bias_shape1 = (96,)
    bias_shape2 = (32,)

    class TestModel(nn.Module):
        def __init__(self, W1_data_np, B1_data_np, W2_data_np, B2_data_np, W3_data_np):
            super().__init__()
            self.W1 = nn.Parameter(torch.from_numpy(W1_data_np).to(device))
            self.B1 = nn.Parameter(torch.from_numpy(B1_data_np).squeeze().to(device))
            self.W2 = nn.Parameter(torch.from_numpy(W2_data_np).to(device))
            self.B2 = nn.Parameter(torch.from_numpy(B2_data_np).squeeze().to(device))
            self.W3 = nn.Parameter(torch.from_numpy(W3_data_np).to(device))

        def forward(self, A):
            conv1 = F.conv2d(A, self.W1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1)
            D_conv1_bias = conv1 + self.B1.view(1, -1, 1, 1)
            D_conv1_relu = F.relu(D_conv1_bias)

            conv2 = F.conv2d(D_conv1_relu, self.W2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1)
            conv2_bias = conv2 + self.B2.view(1, -1, 1, 1)
            conv2_relu = F.relu(conv2_bias)

            conv3 = F.conv2d(D_conv1_relu, self.W3, stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1)

            c = torch.cat([conv2_relu, conv3], dim=1) # axis=1 in TVM relay.op.concatenate

            return c

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()

    filter_data1 = np.zeros(filter_shape1).astype(dtype_str)
    bias_data1 = np.zeros(bias_shape1).astype(dtype_str)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)

    filter_data2 = np.zeros(filter_shape2).astype(dtype_str)
    bias_data2 = np.zeros(bias_shape2).astype(dtype_str)
    # Manually initialize
    c_in2, k_h2, k_w2 = filter_shape2[1], filter_shape2[2], filter_shape2[3]
    c_out2 = filter_shape2[0]
    fan_in2 = c_in2 * k_h2 * k_w2
    fan_out2 = c_out2 * k_h2 * k_w2
    bound2 = np.sqrt(6.0 / (fan_in2 + fan_out2))
    filter_data2[:] = np.random.uniform(-bound2, bound2, size=filter_shape2).astype(dtype_str)
    bias_data2[:] = np.zeros(bias_shape2).astype(dtype_str)

    filter_data3 = np.zeros(filter_shape3).astype(dtype_str)
    # Manually initialize
    c_in3, k_h3, k_w3 = filter_shape3[1], filter_shape3[2], filter_shape3[3]
    c_out3 = filter_shape3[0]
    fan_in3 = c_in3 * k_h3 * k_w3
    fan_out3 = c_out3 * k_h3 * k_w3
    bound3 = np.sqrt(6.0 / (fan_in3 + fan_out3))
    filter_data3[:] = np.random.uniform(-bound3, bound3, size=filter_shape3).astype(dtype_str)

    model_eager = TestModel(filter_data1, bias_data1, filter_data2, bias_data2, filter_data3).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `static_memory_scope` is TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_pooling_branching_texture_params(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 40, 40)
    filter_shape0 = (32, 32, 1, 1)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (32,)

    class TestModel(nn.Module):
        def __init__(self, W0_data_np, W1_data_np, B1_data_np, W2_data_np, W3_data_np):
            super().__init__()
            self.W0 = nn.Parameter(torch.from_numpy(W0_data_np).to(device))
            self.W1 = nn.Parameter(torch.from_numpy(W1_data_np).to(device))
            self.B1 = nn.Parameter(torch.from_numpy(B1_data_np).squeeze().to(device))
            self.W2 = nn.Parameter(torch.from_numpy(W2_data_np).to(device))
            self.W3 = nn.Parameter(torch.from_numpy(W3_data_np).to(device))

        def forward(self, A):
            conv0 = F.conv2d(A, self.W0, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
            pool = F.avg_pool2d(conv0, kernel_size=(2, 2), stride=(2, 2))

            # TVM conv1 padding=[0,0,1,1] -> PyTorch F.conv2d padding=(pad_H, pad_W) => (0, 1)
            conv1 = F.conv2d(pool, self.W1, stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1)
            conv1_bias = conv1 + self.B1.view(1, -1, 1, 1)
            conv1_relu = F.relu(conv1_bias)

            conv2 = F.conv2d(pool, self.W2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)

            # TVM conv3 padding=[0,1,1,0] means (top=0, bottom=1, left=1, right=0)
            # This requires F.pad: (pad_left, pad_right, pad_top, pad_bottom)
            pad_before_conv3 = (1, 0, 0, 1)
            padded_pool_for_conv3 = F.pad(pool, pad_before_conv3, mode='constant', value=0)
            conv3 = F.conv2d(padded_pool_for_conv3, self.W3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
            conv3_relu = F.relu(conv3)

            res = conv1_relu + conv2
            res = res + conv3_relu

            return res

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()

    filter_data0 = np.zeros(filter_shape0).astype(dtype_str)
    c_in0, k_h0, k_w0 = filter_shape0[1], filter_shape0[2], filter_shape0[3]
    c_out0 = filter_shape0[0]
    fan_in0 = c_in0 * k_h0 * k_w0
    fan_out0 = c_out0 * k_h0 * k_w0
    bound0 = np.sqrt(6.0 / (fan_in0 + fan_out0))
    filter_data0[:] = np.random.uniform(-bound0, bound0, size=filter_shape0).astype(dtype_str)

    filter_data1 = np.zeros(filter_shape1).astype(dtype_str)
    bias_data1 = np.zeros(bias_shape1).astype(dtype_str)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)

    filter_data2 = np.zeros(filter_shape2).astype(dtype_str)
    c_in2, k_h2, k_w2 = filter_shape2[1], filter_shape2[2], filter_shape2[3]
    c_out2 = filter_shape2[0]
    fan_in2 = c_in2 * k_h2 * k_w2
    fan_out2 = c_out2 * k_h2 * k_w2
    bound2 = np.sqrt(6.0 / (fan_in2 + fan_out2))
    filter_data2[:] = np.random.uniform(-bound2, bound2, size=filter_shape2).astype(dtype_str)

    filter_data3 = np.zeros(filter_shape3).astype(dtype_str)
    c_in3, k_h3, k_w3 = filter_shape3[1], filter_shape3[2], filter_shape3[3]
    c_out3 = filter_shape3[0]
    fan_in3 = c_in3 * k_h3 * k_w3
    fan_out3 = c_out3 * k_h3 * k_w3
    bound3 = np.sqrt(6.0 / (fan_in3 + fan_out3))
    filter_data3[:] = np.random.uniform(-bound3, bound3, size=filter_shape3).astype(dtype_str)

    model_eager = TestModel(filter_data0, filter_data1, bias_data1, filter_data2, filter_data3).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `static_memory_scope` is TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_branching_texture_params(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 40, 40)
    filter_shape0 = (32, 32, 1, 1)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (32,)

    class TestModel(nn.Module):
        def __init__(self, W0_data_np, W1_data_np, B1_data_np, W2_data_np, W3_data_np):
            super().__init__()
            self.W0 = nn.Parameter(torch.from_numpy(W0_data_np).to(device))
            self.W1 = nn.Parameter(torch.from_numpy(W1_data_np).to(device))
            self.B1 = nn.Parameter(torch.from_numpy(B1_data_np).squeeze().to(device))
            self.W2 = nn.Parameter(torch.from_numpy(W2_data_np).to(device))
            self.W3 = nn.Parameter(torch.from_numpy(W3_data_np).to(device))

        def forward(self, A):
            conv0 = F.conv2d(A, self.W0, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)

            # TVM conv1 padding=[0,0,1,1] -> PyTorch F.conv2d padding=(pad_H, pad_W) => (0, 1)
            conv1 = F.conv2d(conv0, self.W1, stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1)
            conv1_bias = conv1 + self.B1.view(1, -1, 1, 1)
            conv1_relu = F.relu(conv1_bias)

            conv2 = F.conv2d(conv0, self.W2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)

            # TVM conv3 padding=[0,1,1,0] means (top=0, bottom=1, left=1, right=0)
            # This requires F.pad: (pad_left, pad_right, pad_top, pad_bottom)
            pad_before_conv3 = (1, 0, 0, 1)
            padded_conv0_for_conv3 = F.pad(conv0, pad_before_conv3, mode='constant', value=0)
            conv3 = F.conv2d(padded_conv0_for_conv3, self.W3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
            conv3_relu = F.relu(conv3)

            res = conv1_relu + conv2
            res = res + conv3_relu

            return res

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()

    filter_data0 = np.zeros(filter_shape0).astype(dtype_str)
    c_in0, k_h0, k_w0 = filter_shape0[1], filter_shape0[2], filter_shape0[3]
    c_out0 = filter_shape0[0]
    fan_in0 = c_in0 * k_h0 * k_w0
    fan_out0 = c_out0 * k_h0 * k_w0
    bound0 = np.sqrt(6.0 / (fan_in0 + fan_out0))
    filter_data0[:] = np.random.uniform(-bound0, bound0, size=filter_shape0).astype(dtype_str)

    filter_data1 = np.zeros(filter_shape1).astype(dtype_str)
    bias_data1 = np.zeros(bias_shape1).astype(dtype_str)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)

    filter_data2 = np.zeros(filter_shape2).astype(dtype_str)
    c_in2, k_h2, k_w2 = filter_shape2[1], filter_shape2[2], filter_shape2[3]
    c_out2 = filter_shape2[0]
    fan_in2 = c_in2 * k_h2 * k_w2
    fan_out2 = c_out2 * k_h2 * k_w2
    bound2 = np.sqrt(6.0 / (fan_in2 + fan_out2))
    filter_data2[:] = np.random.uniform(-bound2, bound2, size=filter_shape2).astype(dtype_str)

    filter_data3 = np.zeros(filter_shape3).astype(dtype_str)
    c_in3, k_h3, k_w3 = filter_shape3[1], filter_shape3[2], filter_shape3[3]
    c_out3 = filter_shape3[0]
    fan_in3 = c_in3 * k_h3 * k_w3
    fan_out3 = c_out3 * k_h3 * k_w3
    bound3 = np.sqrt(6.0 / (fan_in3 + fan_out3))
    filter_data3[:] = np.random.uniform(-bound3, bound3, size=filter_shape3).astype(dtype_str)

    model_eager = TestModel(filter_data0, filter_data1, bias_data1, filter_data2, filter_data3).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `static_memory_scope` is TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_different_lowering_same_op(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 32, 40, 40)
    filter_shape1 = (32, 32, 1, 1)

    class TestModel(nn.Module):
        def __init__(self, W1_data_np):
            super().__init__()
            self.W1 = nn.Parameter(torch.from_numpy(W1_data_np).to(device))

        def forward(self, A):
            conv1 = F.conv2d(A, self.W1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
            conv2 = F.conv2d(conv1, self.W1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
            conv3 = F.conv2d(conv2, self.W1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
            return conv3

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype_str)
    initializer("weight", filter_data1)

    model_eager = TestModel(filter_data1).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `static_memory_scope` is TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_conv2d_winograd_non_rect(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 771, 36, 64)
    filter_shape = (128, 771, 3, 3)

    class TestModel(nn.Module):
        def __init__(self, filter_data_np):
            super().__init__()
            self.weight = nn.Parameter(torch.from_numpy(filter_data_np).to(device))

        def forward(self, A):
            D = F.conv2d(
                A, self.weight, padding=(1, 1), stride=(1, 1), dilation=(1, 1), groups=1
            )
            return D

    np.random.seed(1)
    initializer = MockRelayTestingInit.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype_str)
    initializer("weight", filter_data)

    model_eager = TestModel(filter_data).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `temp`, `stat_file`, `re.findall` and graph analysis are TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_injective_nwo_inputs1(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 4, 40, 40)
    filter_shape1 = (4, 4, 3, 3)
    filter_shape2 = (4, 4, 3, 3)

    class TestModel(nn.Module):
        def __init__(self, W1_data_np, W2_data_np):
            super().__init__()
            self.W1 = nn.Parameter(torch.from_numpy(W1_data_np).to(device))
            self.W2 = nn.Parameter(torch.from_numpy(W2_data_np).to(device))

        def forward(self, A):
            mean = torch.mean(A, dim=1, keepdim=True)

            conv1 = F.conv2d(A, self.W1, padding=(1, 1), stride=(1, 1), dilation=(1, 1), groups=1)
            conv2 = F.conv2d(conv1, self.W2, padding=(1, 1), stride=(1, 1), dilation=(1, 1), groups=1)

            ad3 = conv1 + conv2
            ad1 = mean + conv1
            ad2 = ad1 * conv2
            ad4 = ad3 + ad2
            return ad4

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype_str)
    filter_data2 = np.zeros(filter_shape2).astype(dtype_str)
    initializer("weight", filter_data1)
    # Manually initialize the second weight
    c_in2, k_h2, k_w2 = filter_shape2[1], filter_shape2[2], filter_shape2[3]
    c_out2 = filter_shape2[0]
    fan_in2 = c_in2 * k_h2 * k_w2
    fan_out2 = c_out2 * k_h2 * k_w2
    bound2 = np.sqrt(6.0 / (fan_in2 + fan_out2))
    filter_data2[:] = np.random.uniform(-bound2, bound2, size=filter_shape2).astype(dtype_str)


    model_eager = TestModel(filter_data1, filter_data2).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `static_memory_scope` is TVM-specific and removed.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@tvm.testing.parametrize_targets("cuda")
@pytest.mark.parametrize("dtype_str", ["float32"])
def test_injective_nwo_inputs2(target, remote, dtype_str):
    dtype = tvm_dtype_to_torch_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_shape = (1, 4, 40, 40)
    filter_shape1 = (4, 4, 3, 3)
    filter_shape2 = (4, 4, 3, 3)

    class TestModel(nn.Module):
        def __init__(self, W1_data_np, W2_data_np):
            super().__init__()
            self.W1 = nn.Parameter(torch.from_numpy(W1_data_np).to(device))
            self.W2 = nn.Parameter(torch.from_numpy(W2_data_np).to(device))

        def forward(self, A):
            mean = torch.mean(A, dim=1, keepdim=True)

            conv1 = F.conv2d(A, self.W1, padding=(1, 1), stride=(1, 1), dilation=(1, 1), groups=1)
            conv2 = F.conv2d(conv1, self.W2, padding=(1, 1), stride=(1, 1), dilation=(1, 1), groups=1)

            ad3 = conv1 + conv2
            ad1 = mean + conv1
            ad2 = ad1 * conv2
            # The order of add is swapped compared to test_injective_nwo_inputs1, but mathematically equivalent
            ad4 = ad2 + ad3
            return ad4

    np.random.seed(0)
    initializer = MockRelayTestingInit.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype_str)
    filter_data2 = np.zeros(filter_shape2).astype(dtype_str)
    initializer("weight", filter_data1)
    # Manually initialize the second weight
    c_in2, k_h2, k_w2 = filter_shape2[1], filter_shape2[2], filter_shape2[3]
    c_out2 = filter_shape2[0]
    fan_in2 = c_in2 * k_h2 * k_w2
    fan_out2 = c_out2 * k_h2 * k_w2
    bound2 = np.sqrt(6.0 / (fan_in2 + fan_out2))
    filter_data2[:] = np.random.uniform(-bound2, bound2, size=filter_shape2).astype(dtype_str)

    model_eager = TestModel(filter_data1, filter_data2).to(device)
    input_data = torch.randn(input_shape, dtype=dtype, device=device)

    eager_output = model_eager(input_data)
    compiled_model = torch.compile(model_eager, mode="inductor")
    compiled_output = compiled_model(input_data)
    torch.testing.assert_allclose(eager_output, compiled_output, rtol=1e-4, atol=1e-4)

    # `static_memory_scope` is TVM-specific and removed.
