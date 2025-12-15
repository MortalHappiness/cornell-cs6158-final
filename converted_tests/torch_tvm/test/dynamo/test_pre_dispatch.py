import tvm
import tvm.relay as relay
from tvm import te
import numpy as np
import pytest

# --- Helper functions and TestCase for TVM environment ---

# Mocking TestCase and its methods for standalone execution of generated tests
class TestCase:
    def assertEqual(self, actual, desired, rtol=1e-5, atol=1e-8):
        def _to_numpy(val):
            if isinstance(val, tvm.runtime.ndarray.NDArray):
                return val.numpy()
            return np.asarray(val)

        actual_np = _to_numpy(actual)
        desired_np = _to_numpy(desired)

        # Handle tuples of results (e.g., from gradient function)
        if isinstance(actual_np, (list, tuple)) and isinstance(desired_np, (list, tuple)):
            assert len(actual_np) == len(desired_np), "Tuple lengths mismatch"
            for a, d in zip(actual_np, desired_np):
                tvm.testing.assert_allclose(a, d, rtol=rtol, atol=atol)
        else:
            tvm.testing.assert_allclose(actual_np, desired_np, rtol=rtol, atol=atol)

    # Mimic torch._dynamo.test_case.TestCase's skipIf
    def skipIf(self, condition, reason):
        if condition:
            pytest.skip(reason)

# Helper for compiling and running a Relay function
def compile_and_run_relay(relay_func_expr, input_map, target="llvm"):
    mod = tvm.IRModule.from_expr(relay_func_expr)
    
    # Extract input names and their dtypes from the Relay function's parameters
    param_names = [p.name_hint for p in relay_func_expr.params]
    param_dtypes = [p.checked_type.dtype for p in relay_func_expr.params]

    # Create tvm.nd.array inputs
    tvm_inputs = {}
    for name, dtype in zip(param_names, param_dtypes):
        if name in input_map:
            val_np = input_map[name]
            # Ensure scalar inputs are treated as 0-dim arrays for tvm.nd.array
            if np.isscalar(val_np):
                tvm_inputs[name] = tvm.nd.array(np.array(val_np, dtype=dtype), tvm.device(target, 0))
            else:
                tvm_inputs[name] = tvm.nd.array(val_np, tvm.device(target, 0))
        else:
            raise ValueError(f"Missing input for Relay parameter: {name}")

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    
    dev = tvm.device(target, 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    for name, tvm_arr in tvm_inputs.items():
        module.set_input(name, tvm_arr)

    module.run()
    
    num_outputs = 1
    if isinstance(relay_func_expr.ret_type, tvm.ir.relay.type.TupleType):
        num_outputs = len(relay_func_expr.ret_type.fields)
    
    if num_outputs > 1:
        return tuple(module.get_output(i) for i in range(num_outputs))
    else:
        return module.get_output(0)

# Main test class (equivalent to PyTorch's TestCase)
class PreDispatchTests(TestCase):
    def test_no_grad_simple(self):
        # Reference (NumPy-based, manually derived forward and backward)
        a_ref_np = np.random.randn(4).astype(np.float32)

        def f_numpy_forward_ref(a_np):
            b_np = np.sin(a_np)
            c_np = np.cos(b_np)
            return b_np * np.sin(c_np)

        def f_numpy_backward_ref(a_np):
            # d(sum(out)) / da, where out = b * sin(c)
            # b = sin(a)
            # c = cos(b)
            # Since 'c' is in a no_grad block, dc/da is effectively 0 for autograd.
            # So, dL/da = (dL/db) * (db/da)
            # dL/db = sin(c_val) (where c_val is the computed value of c)
            # db/da = cos(a)
            b_np = np.sin(a_np)
            c_np = np.cos(b_np) # Value of c when no_grad applied
            dL_db = np.sin(c_np)
            db_da = np.cos(a_np)
            # The gradient w.r.t. `a` when sum() is called on the output
            return dL_db * db_da
        
        out_ref_np = f_numpy_forward_ref(a_ref_np)
        grad_a_ref_np = f_numpy_backward_ref(a_ref_np)


        # TVM Test (with relay.op.annotation.stop_gradient for no_grad)
        a_test_np = a_ref_np.copy() # Use a copy for the TVM input

        a_var = relay.var("a", shape=a_test_np.shape, dtype=str(a_test_np.dtype))
        
        b_expr = relay.op.sin(a_var)
        c_raw_expr = relay.op.cos(b_expr)
        c_expr = relay.op.annotation.stop_gradient(c_raw_expr) # Simulates torch.no_grad()
        output_expr = relay.op.multiply(b_expr, relay.op.sin(c_expr))
        
        f_test_relay_forward = relay.Function([a_var], output_expr)

        # Compile and run forward pass
        input_map_forward = {"a": a_test_np}
        out_test_tvm = compile_and_run_relay(f_test_relay_forward, input_map_forward)
        
        self.assertEqual(out_ref_np, out_test_tvm)

        # For backward pass: create gradient function
        # The gradient function takes original inputs and a gradient w.r.t. the output
        # Here, sum().backward() implies gradient w.r.t. scalar sum.
        # So we pass ones_like(output) as the gradient w.r.t. the output.
        grad_f_test_relay = relay.transform.gradient(f_test_relay_forward, [a_var])
        
        # The arguments of the gradient function are [original_input_vars..., grad_of_output_var]
        # So the last parameter is the gradient of the output
        grad_output_param_name = grad_f_test_relay.params[-1].name_hint
        
        output_grad_np = np.ones_like(out_test_tvm.numpy()).astype(np.float32)
        input_map_backward = {"a": a_test_np, grad_output_param_name: output_grad_np}
        
        # Compile and run backward pass
        # The result of grad_f_test_relay is (output, grad_wrt_a)
        _, grad_a_test_tvm = compile_and_run_relay(grad_f_test_relay, input_map_backward)
        
        self.assertEqual(grad_a_ref_np, grad_a_test_tvm)


    def test_enable_grad_and_no_grad(self):
        # Reference (NumPy-based, manually derived forward and backward)
        a_ref_np = np.random.randn(4).astype(np.float32)

        def f_enable_grad_numpy_forward_ref(a_np):
            b_np = a_np * 2.0
            # Within no_grad, c is just a value, its ops don't track gradients to b
            c_np = b_np * 3.0
            # Within enable_grad (nested in no_grad), d is computed using c.
            # If c doesn't track gradients (due to outer no_grad), d won't re-enable for path to 'a'.
            d_np = c_np * 4.0
            # Back to no_grad, e is computed using d.
            e_np = d_np * 5.0
            return b_np + c_np + d_np + e_np

        def f_enable_grad_numpy_backward_ref(a_np):
            # Based on PyTorch's actual behavior for no_grad/enable_grad nesting,
            # only 'b' truly contributes to the gradient path to 'a'.
            # L = sum(b + c_val + d_val + e_val)
            # dL/da = d(sum(b))/da = d(sum(a * 2))/da = 2 * ones_like(a)
            return np.full(a_np.shape, 2.0, dtype=a_np.dtype)
        
        out_ref_np = f_enable_grad_numpy_forward_ref(a_ref_np)
        grad_a_ref_np = f_enable_grad_numpy_backward_ref(a_ref_np)


        # TVM Test (with relay.op.annotation.stop_gradient)
        a_test_np = a_ref_np.copy()

        a_var = relay.var("a", shape=a_test_np.shape, dtype=str(a_test_np.dtype))
        
        b_expr = relay.op.multiply(a_var, relay.const(2.0, "float32"))
        
        # Simulate no_grad blocks:
        # Outer no_grad affects computation of c, and this effect propagates.
        c_raw_expr = relay.op.multiply(b_expr, relay.const(3.0, "float32"))
        c_expr = relay.op.annotation.stop_gradient(c_raw_expr)
        
        # Inner enable_grad is overridden by outer no_grad for path to 'a' because 'c' itself broke the chain.
        d_raw_expr = relay.op.multiply(c_expr, relay.const(4.0, "float32"))
        d_expr = relay.op.annotation.stop_gradient(d_raw_expr)
        
        # Back to outer no_grad
        e_raw_expr = relay.op.multiply(d_expr, relay.const(5.0, "float32"))
        e_expr = relay.op.annotation.stop_gradient(e_raw_expr)
        
        output_expr = relay.op.add(b_expr, relay.op.add(c_expr, relay.op.add(d_expr, e_expr)))
        
        f_test_relay_forward = relay.Function([a_var], output_expr)

        # Compile and run forward pass
        input_map_forward = {"a": a_test_np}
        out_test_tvm = compile_and_run_relay(f_test_relay_forward, input_map_forward)
        
        self.assertEqual(out_ref_np, out_test_tvm)

        # For backward pass
        grad_f_test_relay = relay.transform.gradient(f_test_relay_forward, [a_var])
        
        grad_output_param_name = grad_f_test_relay.params[-1].name_hint
        
        output_grad_np = np.ones_like(out_test_tvm.numpy()).astype(np.float32)
        input_map_backward = {"a": a_test_np, grad_output_param_name: output_grad_np}
        
        _, grad_a_test_tvm = compile_and_run_relay(grad_f_test_relay, input_map_backward)
        
        self.assertEqual(grad_a_ref_np, grad_a_test_tvm)


    def test_autocast_simple(self):
        # Reference (NumPy-based, manually derived forward and backward)
        a_ref_np = np.random.randn(4).astype(np.float32)

        def f_autocast_numpy_forward_ref(a_np):
            b_np = a_np * 2.0
            # matmul for 1D arrays is dot product (scalar output)
            c_np = np.dot(b_np, b_np)
            return b_np + c_np # scalar c_np broadcasts to 1D b_np

        def f_autocast_numpy_backward_ref(a_np):
            # L = sum(out) where out = b + c, b = a*2, c = dot(b,b) = sum(b_i**2)
            # L = sum(a*2) + len(a) * sum((a*2)_i**2)
            # dL/da_j = 2 + len(a) * d( (a_j*2)**2 )/da_j
            #         = 2 + len(a) * d( 4 * a_j**2 )/da_j
            #         = 2 + len(a) * (8 * a_j)
            # For a_np.shape=(4,), len(a)=4
            return 2.0 + 4.0 * (8.0 * a_np) # = 2.0 + 32.0 * a_np

        out_ref_np = f_autocast_numpy_forward_ref(a_ref_np)
        grad_a_ref_np = f_autocast_numpy_backward_ref(a_ref_np)

        # TVM Test (assuming autocast to float32 on CPU is transparent for float32 inputs)
        a_test_np = a_ref_np.copy()

        a_var = relay.var("a", shape=a_test_np.shape, dtype=str(a_test_np.dtype))
        
        b_expr = relay.op.multiply(a_var, relay.const(2.0, "float32"))
        
        # TODO: torch.amp.autocast in PyTorch can change dtypes (e.g., to bfloat16 on CPU).
        # This requires explicit relay.cast operations or a specific mixed-precision graph pass.
        # For simplicity and to pass current float32-only numerical checks, keeping as float32.
        # If numerical differences occur due to bfloat16, explicit casting would be needed.
        c_expr = relay.op.nn.matmul(b_expr, b_expr) # Matmul for 1D inputs is dot product (scalar output)
        output_expr = relay.op.add(b_expr, c_expr)
        
        f_test_relay_forward = relay.Function([a_var], output_expr)

        # Compile and run forward pass
        input_map_forward = {"a": a_test_np}
        out_test_tvm = compile_and_run_relay(f_test_relay_forward, input_map_forward)
        
        self.assertEqual(out_ref_np, out_test_tvm)

        # For backward pass
        grad_f_test_relay = relay.transform.gradient(f_test_relay_forward, [a_var])
        
        grad_output_param_name = grad_f_test_relay.params[-1].name_hint
        
        output_grad_np = np.ones_like(out_test_tvm.numpy()).astype(np.float32)
        input_map_backward = {"a": a_test_np, grad_output_param_name: output_grad_np}
        
        _, grad_a_test_tvm = compile_and_run_relay(grad_f_test_relay, input_map_backward)
        
        self.assertEqual(grad_a_ref_np, grad_a_test_tvm)


# This ensures the file is executable via pytest when called directly
if __name__ == "__main__":
    pytest.main([__file__])
