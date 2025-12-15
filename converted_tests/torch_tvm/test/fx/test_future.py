from __future__ import annotations

import tvm
from tvm import relay
import numpy as np
import pytest

# All PyTorch modules M1, M2, M3, M4 (and class A) in the original test
# effectively perform the same core computation: `input_tensor + input_tensor`.
# The differences in PyTorch's Python-level type annotations (forward references, list[Tensor])
# are handled by `torch.fx.symbolic_trace` to produce a unified underlying graph.
# In TVM, we directly define this resulting computation graph.

def create_add_relay_function():
    # The input to the core operation is a single tensor.
    # Its shape and dtype are derived from the example `torch.rand(2, 3)`.
    x = relay.var("x", relay.TensorType((2, 3), "float32"))
    out = relay.add(x, x)
    return relay.Function([x], out)

def test_fx_future_conversion():
    # Input data for the PyTorch models, now in NumPy format for TVM
    x_np = np.random.rand(2, 3).astype("float32")
    x_tvm = tvm.nd.array(x_np)

    # Reference calculation using NumPy to mimic PyTorch's eager execution result
    ref_np = x_np + x_np

    # Build the Relay module representing the core computation
    relay_func = create_add_relay_function()
    mod = tvm.IRModule.from_expr(relay_func)

    # Compile the Relay module
    target = "llvm" # Can be changed to "cuda" if a GPU is available
    ctx = tvm.cpu(0) # Context for execution
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    
    # Create a TVM runtime.Module for execution
    executor = tvm.runtime.GraphModule(lib["default"](ctx))

    # The original PyTorch test runs `symbolic_trace` on different `M` classes
    # and then executes the traced module. Since all these classes effectively
    # perform the same `add(x, x)` computation on a single tensor input,
    # we can reuse the same compiled TVM module for all checks.
    # This mirrors the fact that PyTorch's `symbolic_trace` would produce
    # functionally identical graphs for these modules in this simple case.

    # --- Simulate checks for M1, M2, M3, M4 ---

    # For M1 (no forward references, single Tensor input)
    executor.set_input("x", x_tvm)
    executor.run()
    res1_tvm_np = executor.get_output(0).numpy()
    tvm.testing.assert_allclose(ref_np, res1_tvm_np, rtol=1e-5, atol=1e-5)

    # For M2 (forward references, single Tensor input)
    # Computation is identical to M1's traced graph.
    executor.set_input("x", x_tvm)
    executor.run()
    res2_tvm_np = executor.get_output(0).numpy()
    tvm.testing.assert_allclose(ref_np, res2_tvm_np, rtol=1e-5, atol=1e-5)

    # For M3 (non-torch annotation list[torch.Tensor], accesses x[0])
    # The effective input to `a` is still a single tensor.
    executor.set_input("x", x_tvm) # Pass the single tensor `x_tvm` as if it were x[0]
    executor.run()
    res3_tvm_np = executor.get_output(0).numpy()
    tvm.testing.assert_allclose(ref_np, res3_tvm_np, rtol=1e-5, atol=1e-5)

    # For M4 (non-torch annotation list[torch.Tensor], accesses x[0], with forward references)
    # Computation is identical to M3's traced graph.
    executor.set_input("x", x_tvm)
    executor.run()
    res4_tvm_np = executor.get_output(0).numpy()
    tvm.testing.assert_allclose(ref_np, res4_tvm_np, rtol=1e-5, atol=1e-5)
