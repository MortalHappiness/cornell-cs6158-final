import pytest
import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor
import os


# Helper function to get device and target based on environment or availability
def get_test_target_dev():
    if tvm.cuda().exist:
        return "cuda", tvm.cuda(0)
    else:
        return "llvm", tvm.cpu(0)


class TestFuserCommonTVM:
    """
    Equivalent TVM tests for PyTorch's TestFuserCommon.

    Note: The `requires_grad` concept as a runtime attribute on tensors
    is PyTorch-specific and does not have a direct, identical mapping in TVM.
    Instead, TVM's autograd is graph-based, where a gradient graph is explicitly
    constructed and executed. The tests below verify the forward pass correctness
    and the ability to compute gradients when conceptually `requires_grad` would be True.
    The "autodiff fallback" aspect of the original PyTorch test refers to PyTorch's
    internal JIT/compiler behavior; in TVM, the graph is always explicitly constructed
    and compiled.
    """

    def _define_fn_relay(self, x_var_name="x", x_shape=(5,), x_dtype="float32"):
        """Defines the Relay function equivalent to `fn(x) = torch.max(x**2.0, x**3.0)`."""
        x_var = relay.var(x_var_name, shape=x_shape, dtype=x_dtype)
        # Ensure constants have the same dtype as x for proper element-wise operations
        const_2_0 = relay.const(2.0, dtype=x_dtype)
        const_3_0 = relay.const(3.0, dtype=x_dtype)

        x_sq = relay.op.power(x_var, const_2_0)
        x_cube = relay.op.power(x_var, const_3_0)
        result = relay.op.tensor.maximum(x_sq, x_cube)
        return relay.Function([x_var], result), x_var, result

    @pytest.mark.parametrize("rq", [False, True])
    def test_autodiff_forward_and_gradient_status(self, rq):
        x_shape = (5,)
        x_dtype = "float32"  # PyTorch default for float tensors

        # Define the forward function
        forward_func, x_var, output_expr = self._define_fn_relay(x_shape=x_shape, x_dtype=x_dtype)
        mod_fwd = tvm.IRModule.from_expr(forward_func)

        target, dev = get_test_target_dev()

        # Generate random input data for x
        # Use a range that makes x**2 and x**3 behave differently and derivatives non-zero.
        # e.g., x = 0 or x = 1 are points where x**2 == x**3.
        # For x > 1, x**3 > x**2. For 0 < x < 1, x**2 > x**3. For x < 0, behavior is complex.
        # Let's use inputs that are likely to produce non-trivial gradients.
        rng = np.random.default_rng(42) # Fixed seed for reproducibility
        x_np = rng.uniform(low=-2.0, high=2.0, size=x_shape).astype(x_dtype)
        # Avoid exact tie-points which can make derivatives ambiguous depending on implementation
        x_np[np.isclose(x_np, 0.0)] += 0.01
        x_np[np.isclose(x_np, 1.0)] += 0.01


        # --- Forward pass verification (always correct regardless of `rq`) ---
        with tvm.transform.PassContext(opt_level=3):
            lib_fwd = relay.build(mod_fwd, target=target)
        module_fwd = graph_executor.GraphModule(lib_fwd["default"](dev))
        module_fwd.set_input("x", tvm.nd.array(x_np, dev))
        module_fwd.run()
        y_tvm = module_fwd.get_output(0).numpy()

        # Calculate expected output using NumPy
        y_np_expected = np.maximum(x_np**2.0, x_np**3.0)
        tvm.testing.assert_allclose(y_tvm, y_np_expected, rtol=1e-5, atol=1e-5)

        # --- Gradient computability check (simulating `y.requires_grad == rq`) ---
        if rq:
            # When rq is True, the original PyTorch test implies that gradients should be computable.
            # In TVM, we explicitly construct and execute a backward graph to verify this.

            # Define an adjoint for the output. For a simple gradient check, typically `dL/dy = 1`.
            output_grad_var = relay.var("y_grad", shape=output_expr.shape, dtype=output_expr.dtype)

            # Build the gradient function (computes dy/dx)
            # `relay.transform.gradient` creates a function that takes original inputs and
            # adjoints for original outputs, returning original outputs and gradients w.r.t. inputs.
            backward_func = relay.transform.gradient(forward_func, [x_var], output_grad_var)
            mod_grad = tvm.IRModule.from_expr(backward_func)

            with tvm.transform.PassContext(opt_level=3):
                lib_grad = relay.build(mod_grad, target=target)
            module_grad = graph_executor.GraphModule(lib_grad["default"](dev))

            # Provide inputs for forward pass and adjoints for output gradient
            output_grad_np = np.ones_like(x_np, dtype=x_dtype)  # Adjoint dL/dy = 1.0
            module_grad.set_input("x", tvm.nd.array(x_np, dev))
            module_grad.set_input("y_grad", tvm.nd.array(output_grad_np, dev))
            module_grad.run()

            # First output is the forward result, second is the gradient w.r.t. x
            y_fwd_tvm_from_grad_mod = module_grad.get_output(0).numpy()
            dx_tvm = module_grad.get_output(1).numpy()

            # Assert forward result consistency (sanity check)
            tvm.testing.assert_allclose(y_fwd_tvm_from_grad_mod, y_np_expected, rtol=1e-5, atol=1e-5)

            # Compute expected gradient with NumPy
            # The derivative of max(f(x), g(x)) is f'(x) if f(x) > g(x) and g'(x) if g(x) > f(x).
            # At ties (f(x) == g(x)), PyTorch's `max` often defaults to the derivative of the first arg.
            # d/dx (x**2) = 2x
            # d/dx (x**3) = 3x**2
            # So, dx = 2x if x**2.0 >= x**3.0, else 3x**2.0
            dx_np_expected = np.where(x_np**2.0 >= x_np**3.0, 2 * x_np, 3 * x_np**2.0)

            tvm.testing.assert_allclose(dx_tvm, dx_np_expected, rtol=1e-5, atol=1e-5)

            # Assert that the gradient is non-trivial for the chosen inputs
            # This confirms that 'requires_grad' (conceptually) was enabled.
            assert not np.all(dx_tvm == 0), "Expected non-zero gradient when 'requires_grad' is conceptually True"
        else:
            # When rq is False, the original PyTorch test implies that `y.requires_grad` is False.
            # In TVM, the output of a forward-only compilation (like the `lib_fwd` above) does not
            # carry a `requires_grad` attribute. The absence of explicitly building a backward graph
            # for this case is the TVM equivalent. We've already verified the forward pass.
            pass


# The original `raise_on_run_directly` is a PyTorch-specific safeguard for test invocation.
# In TVM, tests are typically run via `pytest`, which handles this.
# `if __name__ == "__main__":` block is removed.
