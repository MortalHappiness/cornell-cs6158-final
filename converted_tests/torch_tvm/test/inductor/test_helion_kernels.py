import tvm
from tvm import relay
import numpy as np
import pytest

# For reference softmax calculation
from scipy.special import softmax as scipy_softmax
from tvm.testing import assert_allclose

# Dummy device and target, assuming GPU is the primary target as implied by original test
# If CUDA is not available, these lines would need to be adjusted (e.g., tvm.cpu())
try:
    tvm_device = tvm.cuda()
    tvm_target = "cuda"
    _has_gpu = True
except Exception:
    tvm_device = tvm.cpu()
    tvm_target = "llvm"
    _has_gpu = False

# Decorator to mimic requires_helion, ensuring test only runs if a GPU target is available
def requires_gpu():
    return pytest.mark.skipif(not _has_gpu, reason="Requires GPU")


class TestHelionKernels:
    @requires_gpu()
    def test_add_kernel(self):
        # Define the Relay function for addition
        x_shape = (4, 8)
        y_shape = (4, 8)
        dtype = "float16"

        x_tvm_var = relay.var("x", relay.TensorType(x_shape, dtype))
        y_tvm_var = relay.var("y", relay.TensorType(y_shape, dtype))
        
        # The core logic of the original 'add' Helion kernel is element-wise addition.
        # In Relay, `relay.add` handles broadcasting and type promotion implicitly.
        output_expr = relay.add(x_tvm_var, y_tvm_var)
        relay_func = relay.Function([x_tvm_var, y_tvm_var], output_expr)
        mod = tvm.IRModule.from_expr(relay_func)

        # Prepare numpy inputs
        x_np = np.random.randn(*x_shape).astype(dtype)
        y_np = np.random.randn(*y_shape).astype(dtype)
        
        # Reference calculation (equivalent to PyTorch's x + y)
        reference_out_np = x_np + y_np

        # Convert numpy inputs to TVM NDArrays
        x_tvm = tvm.nd.array(x_np, device=tvm_device)
        y_tvm = tvm.nd.array(y_np, device=tvm_device)

        # Compile and execute the Relay module
        with tvm.transform.PassContext(opt_level=3):
            # Mapping: torch.compile -> tvm.relay.backend.vm.compile
            # For this simple function, no explicit parameters are needed.
            vm_executor = relay.vm.compile(mod, target=tvm_target)
            compiled_out_tvm = vm_executor.invoke("main", x_tvm, y_tvm)
            compiled_out_np = compiled_out_tvm.numpy()

        # Assertions (using tvm.testing.assert_allclose for numerical comparison)
        assert_allclose(compiled_out_np, reference_out_np, rtol=1e-2, atol=1e-2)

    @requires_gpu()
    def test_softmax_view_reshape(self):
        # Define the Relay function for softmax
        x_shape = (1024, 1024)
        dtype = "float16"

        x_tvm_var = relay.var("x", relay.TensorType(x_shape, dtype))
        
        # The original Helion kernel implements softmax by combining max, exp, sum, div.
        # For semantic equivalence and idiomatic TVM, we use the direct `relay.op.nn.softmax` operator.
        # `dim=1` in PyTorch maps to `axis=1` in TVM.
        output_expr = relay.op.nn.softmax(x_tvm_var, axis=1)
        relay_func = relay.Function([x_tvm_var], output_expr)
        mod = tvm.IRModule.from_expr(relay_func)

        # Prepare numpy inputs
        x_np = np.random.randn(*x_shape).astype(dtype)
        
        # Reference calculation using scipy's softmax, which is functionally equivalent
        # to torch.nn.functional.softmax.
        reference_out_np = scipy_softmax(x_np, axis=1)

        # Convert numpy inputs to TVM NDArrays
        x_tvm = tvm.nd.array(x_np, device=tvm_device)

        # Compile and execute the Relay module
        with tvm.transform.PassContext(opt_level=3):
            # No explicit parameters for this function
            vm_executor = relay.vm.compile(mod, target=tvm_target)
            compiled_out_tvm = vm_executor.invoke("main", x_tvm)
            compiled_out_np = compiled_out_tvm.numpy()

        # Assertions
        # Original PyTorch test uses rtol=1e-2, atol=1e-1, which is a loose tolerance.
        assert_allclose(compiled_out_np, reference_out_np, rtol=1e-2, atol=1e-1)
