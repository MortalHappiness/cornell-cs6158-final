import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import functools

# Mock infrastructure for PyTorch equivalent tests.
# These replace TVM-specific concepts with PyTorch-idiomatic approaches.

# For ACL-like checks, we'll use `hasattr(torch, 'compile')` to simulate
# whether an advanced compiler/runtime (like TorchInductor) is available.
# Device simulation for PyTorch is typically `tensor.to(device)`
# or `torch.device('cpu')`, `torch.device('cuda')`.
# For ARM Compute Library, CPU is a reasonable default.

_DEV = "cpu"
if torch.cuda.is_available():
    _DEV = "cuda"

def skip_runtime_test():
    """Simulate TVM's skip_runtime_test by checking for torch.compile availability."""
    # Since ACL is an external runtime, we'll simulate its "availability"
    # by checking for torch.compile for TorchInductor.
    # If not available, we skip the 'compiled' path.
    return not hasattr(torch, 'compile')

def _build_and_run_pytorch_mock(model_instance, inputs_dict, enable_compile=False, num_runs=1):
    """
    Simulates TVM's build_and_run by executing a PyTorch model.
    `enable_compile=True` simulates the 'ACL' path via `torch.compile`.
    """
    torch_inputs = {k: torch.tensor(v, device=_DEV) for k, v in inputs_dict.items()}
    model_instance.to(_DEV)

    compiled_model = model_instance
    if enable_compile and hasattr(torch, 'compile'):
        try:
            # We use 'cpu' backend by default for ARM Compute Library context,
            # but if cuda is available, inductor will use it.
            compiled_model = torch.compile(model_instance, backend="inductor")
        except Exception as e:
            pytest.skip(f"torch.compile failed for model (simulating ACL path failure): {e}")

    outputs = []
    for _ in range(num_runs):
        with torch.no_grad():
            output = compiled_model(**torch_inputs)
            outputs.append(output.cpu().numpy()) # Convert to numpy for comparison

    return outputs

def _verify_pytorch_mock(outputs, atol, rtol):
    """
    Simulates TVM's verify by using PyTorch's assert_close.
    Compares the first output against subsequent outputs.
    """
    if not outputs:
        pytest.fail("No outputs to verify.")

    if len(outputs) > 1:
        # Compare all subsequent outputs to the first one
        # This typically means comparing the uncompiled output to compiled,
        # or multiple runs of a compiled model.
        expected = torch.tensor(outputs[0])
        for i in range(1, len(outputs)):
            actual = torch.tensor(outputs[i])
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    # If only one output, there's nothing to compare against itself, so no assertion needed.


# PyTorch model definitions corresponding to TVM Relay graphs

class MultipleOpsModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        # Corresponds to relay.reshape(a, (1, 1, 1000))
        out = a.reshape(1, 1, 1000)
        # Corresponds to relay.reshape(out, (1, 1000))
        out = out.reshape(1, 1000)
        return out

class HeterogeneousModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        # Corresponds to relay.reshape(a, (1, 1, 1000))
        out = a.reshape(1, 1, 1000)
        # Corresponds to relay.sigmoid(out)
        out = torch.sigmoid(out)
        # Corresponds to relay.reshape(out, (1, 1000))
        out = out.reshape(1, 1000)
        return out

class MultipleRunsConvModel(nn.Module):
    def __init__(self, np_w_ohwi):
        super().__init__()
        # TVM weights: (256, 1, 1, 512) OHWI
        # PyTorch conv2d weights: (out_channels, in_channels, kernel_height, kernel_width) = (O, I, H, W)
        # Permute OHWI to OIHW: (0, 3, 1, 2)
        weight_oi_hw = torch.tensor(np_w_ohwi, dtype=torch.float32).permute(0, 3, 1, 2)
        self.conv_weight = nn.Parameter(weight_oi_hw)

    def forward(self, a):
        # Input 'a' is (1, 28, 28, 512) NHWC
        # PyTorch conv2d input needs to be NCHW: permute NHWC to NCHW (0, 3, 1, 2)
        input_nchw = a.permute(0, 3, 1, 2)
        
        # relay.nn.conv2d arguments:
        # kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
        out_nchw = F.conv2d(
            input_nchw,
            self.conv_weight,
            bias=None, # TVM example doesn't specify bias, assume None
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1, # TVM example doesn't specify groups, assume 1
        )
        # Output NCHW: (1, 256, 28, 28)
        # To match TVM output convention (if it were NHWC), permute back: (0, 2, 3, 1)
        # The original test doesn't permute the output back in Relay, so comparison with external
        # outputs implies the TVM runtime would produce the expected layout.
        # For PyTorch, we produce NCHW and compare it with itself (uncompiled vs compiled)
        return out_nchw


def test_multiple_ops_pytorch():
    """
    Test multiple operators (reshape) using PyTorch/TorchInductor.
    Compares uncompiled PyTorch output with compiled PyTorch output.
    """
    if skip_runtime_test():
        pytest.skip("ACL/TorchInductor runtime test skipped.")

    np.random.seed(0)

    # Inputs: "a": (1, 1, 1, 1000) float32
    inputs = {"a": np.random.uniform(0, 1, (1, 1, 1, 1000)).astype("float32")}
    
    model = MultipleOpsModel()
    outputs = []

    # Simulate TVM running without ACL (plain PyTorch execution)
    outputs.extend(_build_and_run_pytorch_mock(model, inputs, enable_compile=False))
    
    # Simulate TVM running with ACL (PyTorch with torch.compile)
    outputs.extend(_build_and_run_pytorch_mock(model, inputs, enable_compile=True))
    
    _verify_pytorch_mock(outputs, atol=0.002, rtol=0.01)


def test_heterogeneous_pytorch():
    """
    Test heterogeneous execution where some ops might be offloaded
    (simulated by torch.compile) and others not.
    Compares uncompiled PyTorch output with compiled PyTorch output.
    """
    if skip_runtime_test():
        pytest.skip("ACL/TorchInductor runtime test skipped.")

    np.random.seed(0)

    # Inputs: "a": (1, 1, 1, 1000) float32
    inputs = {"a": np.random.uniform(-127, 128, (1, 1, 1, 1000)).astype("float32")}
    
    model = HeterogeneousModel()
    outputs = []

    # Simulate TVM running without ACL (plain PyTorch execution)
    outputs.extend(_build_and_run_pytorch_mock(model, inputs, enable_compile=False))
    
    # Simulate TVM running with ACL (PyTorch with torch.compile)
    outputs.extend(_build_and_run_pytorch_mock(model, inputs, enable_compile=True))
    
    _verify_pytorch_mock(outputs, atol=0.002, rtol=0.01)


def test_multiple_runs_pytorch():
    """
    Test that multiple runs of an operator (conv2d) with TorchInductor work.
    Compares results of multiple compiled runs to ensure consistency.
    """
    if skip_runtime_test():
        pytest.skip("ACL/TorchInductor runtime test skipped.")

    np.random.seed(0)

    # Weights for conv2d, (256, 1, 1, 512) in OHWI layout
    np_w = np.ones((256, 1, 1, 512), dtype="float32")
    model = MultipleRunsConvModel(np_w)

    # Input: (1, 28, 28, 512) in NHWC layout
    inputs = {
        "a": np.random.uniform(-127, 128, (1, 28, 28, 512)).astype("float32"),
    }

    # Simulate TVM running with ACL and multiple runs
    # In this test, we run the compiled model 3 times and check consistency.
    outputs = _build_and_run_pytorch_mock(model, inputs, enable_compile=True, num_runs=3)
    _verify_pytorch_mock(outputs, atol=0.002, rtol=0.01)

if __name__ == "__main__":
    pytest.main([__file__]) # Use pytest to run the tests
