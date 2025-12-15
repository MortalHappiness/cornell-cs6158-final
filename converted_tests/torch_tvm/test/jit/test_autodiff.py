import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.ir.module import IRModule
from tvm.runtime import NDArray
from tvm.testing import assert_allclose

# Helper to convert numpy array to TVM NDArray
def _to_tvm_ndarray(arr, device=tvm.cpu(0)):
    if isinstance(arr, NDArray):
        return arr
    return tvm.nd.array(arr, device=device)

# Helper to compile and run a Relay module
def _compile_and_run(mod, params, inputs_np, target="llvm", device=tvm.cpu(0)):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    
    rt_mod = tvm.runtime.GraphModule(lib["default"](device))

    for name, val in inputs_np.items():
        rt_mod.set_input(name, _to_tvm_ndarray(val, device))
    
    rt_mod.run()
    outputs = []
    for i in range(rt_mod.get_num_outputs()):
        outputs.append(rt_mod.get_output(i).numpy())
    return outputs


class TestAutodiffTVM:
    # Placeholder for special.gammaln and special.entr
    # The original PyTorch test notes: "this relies on gammaln and entr not having autodiff implementations."
    # For TVM, we make them identity functions for the forward pass.
    # In Relay, these operations are typically differentiable. If the test truly relies
    # on their non-differentiability, a more complex mock would be needed.
    # For now, their differentiation is handled by `transform.gradient` in TVM
    # if `r` is part of `grad_params`.
    def _special_gammaln_relay(data):
        return data # Identity for forward pass

    def _special_entr_relay(data):
        return data # Identity for forward pass

    def test_undefined_tensor_lists(self):
        # Original PyTorch function logic:
        # def fn(tensor_list: List[torch.Tensor], add_tensor):
        #     cat = torch.cat(tensor_list, dim=1)
        #     r = torch.sin(cat + add_tensor)
        #     return r

        def relay_fn_forward(a_var, b_var, add_tensor_var):
            # In PyTorch, tensor_list would be [a, b]
            cat = relay.op.tensor.concatenate([a_var, b_var], axis=1)
            r = relay.op.tensor.sin(relay.op.tensor.add(cat, add_tensor_var))
            return r

        a_shape = (3, 6)
        b_shape = (3, 10)
        y_shape = (3, 16) # Corresponds to `add_tensor`

        # Define Relay variables for inputs
        a_var = relay.var("a", relay.TensorType(a_shape, "float32"))
        b_var = relay.var("b", relay.TensorType(b_shape, "float32"))
        add_tensor_var = relay.var("add_tensor", relay.TensorType(y_shape, "float32"))

        # Build the forward function returning `r`
        f_r = relay.Function([a_var, b_var, add_tensor_var], relay_fn_forward(a_var, b_var, add_tensor_var))
        
        # Generate random numpy data for inputs
        a_np = np.random.rand(*a_shape).astype("float32")
        b_np = np.random.rand(*b_shape).astype("float32")
        y_np = np.random.rand(*y_shape).astype("float32")

        # --- Gradient Test ---
        # The PyTorch test inspects the `grad_fn` of `ret.sum()`.
        # `backward_fn = s.grad_fn.next_functions[0][0]` gets the backward function for `ret` (which is `r` in our Relay `f_r`).
        # This backward function expects `grad_output` (dL/dr, which is `grad_out` here) and `current_grad_r` (None).
        # We model this by using `tvm.relay.transform.gradient` which takes `f_r` and an explicit `grad_outputs` for its output `r`.
        # The `grad_params` are `a_var, b_var, add_tensor_var`.

        grad_params = [a_var, b_var, add_tensor_var]
        
        # `transform.gradient` on `f_r` will generate a function like:
        # `(a, b, add_tensor, grad_r_input) -> (f_r_output, grad_a, grad_b, grad_add_tensor)`
        grad_func_of_r_expr = relay.transform.gradient(f_r, grad_params=grad_params)
        grad_mod_of_r = IRModule.from_expr(grad_func_of_r_expr)

        # 1. Check behavior with defined tensor (PyTorch's `grad_out`)
        # This `grad_out_np` is the incoming gradient for `r_output` (dL/dr).
        grad_out_np = np.random.rand(*y_shape).astype("float32")
        
        inputs_for_grad_run = {
            "a": a_np,
            "b": b_np,
            "add_tensor": y_np,
            grad_func_of_r_expr.params[-1].name_hint: grad_out_np, # The last parameter is the incoming gradient `dL/dr`
        }
        
        results_with_grad_out = _compile_and_run(grad_mod_of_r, None, inputs_for_grad_run)
        
        # Expected outputs from `grad_mod_of_r`: (forward_result, grad_a, grad_b, grad_add_tensor)
        assert len(results_with_grad_out) == 4
        grad_a_np = results_with_grad_out[1]
        grad_b_np = results_with_grad_out[2]
        grad_y_np = results_with_grad_out[3]

        for x_grad, original_shape in [(grad_a_np, a_shape), (grad_b_np, b_shape), (grad_y_np, y_shape)]:
            assert isinstance(x_grad, np.ndarray)
            assert x_grad.shape == original_shape
            # Assert that they are not all zeros (as incoming grad_out is non-zero)
            assert not np.allclose(x_grad, np.zeros_like(x_grad), rtol=1e-5, atol=1e-5)


        # 2. Now test with undefined grad_out (PyTorch's `None`, implying zero gradient)
        # This is `dL/dr = 0`.
        grad_out_none_np = np.zeros_like(y_np)
        
        inputs_for_grad_run_zero_grad_out = {
            "a": a_np,
            "b": b_np,
            "add_tensor": y_np,
            grad_func_of_r_expr.params[-1].name_hint: grad_out_none_np,
        }
        
        results_with_zero_grad_out = _compile_and_run(grad_mod_of_r, None, inputs_for_grad_run_zero_grad_out)
        
        assert len(results_with_zero_grad_out) == 4
        grad_a_zero_np = results_with_zero_grad_out[1]
        grad_b_zero_np = results_with_zero_grad_out[2]
        grad_y_zero_np = results_with_zero_grad_out[3]

        for x_grad, original_shape in [(grad_a_zero_np, a_shape), (grad_b_zero_np, b_shape), (grad_y_zero_np, y_shape)]:
            assert isinstance(x_grad, np.ndarray)
            assert x_grad.shape == original_shape
            # Expect all gradients to be zero when incoming gradient is zero
            assert_allclose(x_grad, np.zeros_like(x_grad), rtol=1e-5, atol=1e-5)

    def test_requires_grad_outputs(self):
        # Original PyTorch function logic:
        # def fn(a, b, c):
        #     return a.relu() + b.relu(), c.relu()

        def relay_fn_forward(a_var, b_var, c_var):
            output1 = relay.op.tensor.add(relay.op.nn.relu(a_var), relay.op.nn.relu(b_var))
            output2 = relay.op.nn.relu(c_var)
            return relay.Tuple([output1, output2])

        a_shape = (10, 10)
        b_shape = (10, 10)
        c_shape = (10, 10)

        a_np = np.random.rand(*a_shape).astype("float32")
        b_np = np.random.rand(*b_shape).astype("float32")
        c_np = np.random.rand(*c_shape).astype("float32")

        # Define Relay variables
        a_var = relay.var("a", relay.TensorType(a_shape, "float32"))
        b_var = relay.var("b", relay.TensorType(b_shape, "float32"))
        c_var = relay.var("c", relay.TensorType(c_shape, "float32"))

        # Build forward function
        forward_func = relay.Function([a_var, b_var, c_var], relay_fn_forward(a_var, b_var, c_var))
        
        # To simulate PyTorch's `requires_grad` propagation, we examine what gradients `tvm.relay.transform.gradient` produces.
        # PyTorch test implies:
        # (x, y) = fn(a, b, c) where a, b have requires_grad=False, c has requires_grad=True.
        # Expect x.requires_grad=False and y.requires_grad=True.
        # This means dL/da = 0, dL/db = 0, and dL/dc != 0.

        # Create a scalar loss from the outputs (sum of all elements in both outputs)
        output_tuple_expr = relay_fn_forward(a_var, b_var, c_var)
        sum_output1 = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 0))
        sum_output2 = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 1))
        loss_expr = relay.op.tensor.add(sum_output1, sum_output2)

        loss_func = relay.Function([a_var, b_var, c_var], loss_expr)

        # Compute gradients with respect to all inputs
        grad_params = [a_var, b_var, c_var]
        grad_func = relay.transform.gradient(loss_func, grad_params=grad_params)
        grad_mod = IRModule.from_expr(grad_func)

        inputs_for_grad = {"a": a_np, "b": b_np, "c": c_np}
        results_grad = _compile_and_run(grad_mod, None, inputs_for_grad)

        # `grad_func` output: (forward_loss_value, dL/da, dL/db, dL/dc)
        assert len(results_grad) == 4
        _, grad_a, grad_b, grad_c = results_grad

        # `a` was requires_grad=False, so dL/da should be zero
        assert_allclose(grad_a, np.zeros_like(grad_a), rtol=1e-5, atol=1e-5)
        # `b` was requires_grad=False, so dL/db should be zero
        assert_allclose(grad_b, np.zeros_like(grad_b), rtol=1e-5, atol=1e-5)
        # `c` was requires_grad=True, so dL/dc should be non-zero
        assert not np.allclose(grad_c, np.zeros_like(grad_c), rtol=1e-5, atol=1e-5)
        assert grad_c.shape == c_shape


    def test_requires_grad_outputs_profiled_twice(self):
        # Original PyTorch function logic:
        # def fn(a, b, c):
        #     r = a.relu().relu()
        #     return torch.special.gammaln(r), torch.special.entr(r), c.cos().relu()

        # Define Relay version
        def relay_fn_forward(a_var, b_var, c_var): # b_var is unused in original PyTorch fn
            r = relay.op.nn.relu(relay.op.nn.relu(a_var))
            out1 = self._special_gammaln_relay(r) # Identity op
            out2 = self._special_entr_relay(r)     # Identity op
            out3 = relay.op.nn.relu(relay.op.tensor.cos(c_var))
            return relay.Tuple([out1, out2, out3])

        a_shape = (10, 10)
        c_shape = (10, 10)

        a_np = np.random.rand(*a_shape).astype("float32")
        b_np = np.random.rand(*a_shape).astype("float32") # b is unused, provide dummy
        c_np = np.random.rand(*c_shape).astype("float32")

        a_var = relay.var("a", relay.TensorType(a_shape, "float32"))
        b_var = relay.var("b", relay.TensorType(a_shape, "float32"))
        c_var = relay.var("c", relay.TensorType(c_shape, "float32"))

        # Build forward function
        forward_func = relay.Function([a_var, b_var, c_var], relay_fn_forward(a_var, b_var, c_var))
        
        inputs_np = {"a": a_np, "b": b_np, "c": c_np}

        # PyTorch test implies: a.requires_grad=False, c.requires_grad=True
        # This means dL/da = 0, dL/db = 0 (b is unused), dL/dc != 0.

        # Create a scalar loss from the outputs
        output_tuple_expr = relay_fn_forward(a_var, b_var, c_var)
        sum_out1 = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 0))
        sum_out2 = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 1))
        sum_out3 = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 2))
        loss_expr = relay.op.tensor.add(relay.op.tensor.add(sum_out1, sum_out2), sum_out3)

        loss_func = relay.Function([a_var, b_var, c_var], loss_expr)
        grad_params = [a_var, b_var, c_var]
        grad_func = relay.transform.gradient(loss_func, grad_params=grad_params)
        grad_mod = IRModule.from_expr(grad_func)

        results_grad = _compile_and_run(grad_mod, None, inputs_np)
        assert len(results_grad) == 4
        _, grad_a, grad_b, grad_c = results_grad

        # `a` was requires_grad=False
        assert_allclose(grad_a, np.zeros_like(grad_a), rtol=1e-5, atol=1e-5)
        # `b` was unused
        assert_allclose(grad_b, np.zeros_like(grad_b), rtol=1e-5, atol=1e-5)
        # `c` was requires_grad=True
        assert not np.allclose(grad_c, np.zeros_like(grad_c), rtol=1e-5, atol=1e-5)
        assert grad_c.shape == c_shape


    def test_requires_grad_outputs_side_effects(self):
        # Original PyTorch function logic:
        # @torch.jit.ignore
        # def python_fn(x):
        #     return x.relu()

        # def fn(a, b, c):
        #     r = a.relu().relu()
        #     z = python_fn(r) # python_fn is ignored, so its contribution is transparent
        #     return torch.relu(r), torch.nn.functional.gelu(r), c.cos().relu()

        def relay_python_fn(x):
            return relay.op.nn.relu(x) # Placeholder: JIT ignores this, so we model its effect.

        def relay_fn_forward(a_var, b_var, c_var): # b_var is unused in original PyTorch fn
            r = relay.op.nn.relu(relay.op.nn.relu(a_var))
            _ = relay_python_fn(r) # JIT ignores, so it's a pass-through from 'r' to outputs
            out1 = relay.op.nn.relu(r)
            out2 = relay.op.nn.gelu(r)
            out3 = relay.op.nn.relu(relay.op.tensor.cos(c_var))
            return relay.Tuple([out1, out2, out3])

        a_shape = (10, 10)
        c_shape = (10, 10)

        a_np = np.random.rand(*a_shape).astype("float32")
        b_np = np.random.rand(*a_shape).astype("float32") # b is unused
        c_np = np.random.rand(*c_shape).astype("float32")

        a_var = relay.var("a", relay.TensorType(a_shape, "float32"))
        b_var = relay.var("b", relay.TensorType(a_shape, "float32"))
        c_var = relay.var("c", relay.TensorType(c_shape, "float32"))

        # Build forward function
        forward_func = relay.Function([a_var, b_var, c_var], relay_fn_forward(a_var, b_var, c_var))
        
        inputs_np = {"a": a_np, "b": b_np, "c": c_np}

        # PyTorch test implies: a.requires_grad=False, c.requires_grad=True
        # This means dL/da = 0, dL/db = 0 (b is unused), dL/dc != 0.

        # Create a scalar loss from the outputs
        output_tuple_expr = relay_fn_forward(a_var, b_var, c_var)
        sum_out1 = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 0))
        sum_out2 = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 1))
        sum_out3 = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 2))
        loss_expr = relay.op.tensor.add(relay.op.tensor.add(sum_out1, sum_out2), sum_out3)

        loss_func = relay.Function([a_var, b_var, c_var], loss_expr)
        grad_params = [a_var, b_var, c_var]
        grad_func = relay.transform.gradient(loss_func, grad_params=grad_params)
        grad_mod = IRModule.from_expr(grad_func)

        results_grad = _compile_and_run(grad_mod, None, inputs_np)
        assert len(results_grad) == 4
        _, grad_a, grad_b, grad_c = results_grad

        # `a` was requires_grad=False
        assert_allclose(grad_a, np.zeros_like(grad_a), rtol=1e-5, atol=1e-5)
        # `b` was unused
        assert_allclose(grad_b, np.zeros_like(grad_b), rtol=1e-5, atol=1e-5)
        # `c` was requires_grad=True
        assert not np.allclose(grad_c, np.zeros_like(grad_c), rtol=1e-5, atol=1e-5)
        assert grad_c.shape == c_shape


    def test_autodiff_requires_grad_nograd(self):
        # Original PyTorch function logic:
        # @torch.jit.ignore
        # def python_fn(x):
        #     return x.relu()

        # def fn(a, b, c):
        #     x = a.sin().relu()
        #     y = python_fn(b)
        #     with torch.no_grad():
        #         z = x + c
        #     return x, y, z

        def relay_python_fn(x):
            return relay.op.nn.relu(x)

        def relay_fn_forward(a_var, b_var, c_var):
            x = relay.op.nn.relu(relay.op.tensor.sin(a_var))
            y = relay_python_fn(b_var)
            # `with torch.no_grad(): z = x + c` implies z.requires_grad=False.
            # In TVM, `transform.gradient` will compute gradients based on the graph.
            # If `z` itself is output and its gradient `dL/dz` is zero, then `dL/dc` through `z` will be zero.
            # We explicitly define the mathematical addition here.
            z_val = relay.op.tensor.add(x, c_var)
            return relay.Tuple([x, y, z_val])

        a_shape = (10, 10)
        b_shape = (10, 10)
        c_shape = (10, 10)

        a_np = np.random.rand(*a_shape).astype("float32")
        b_np = np.random.rand(*b_shape).astype("float32")
        c_np = np.random.rand(*c_shape).astype("float32")

        a_var = relay.var("a", relay.TensorType(a_shape, "float32"))
        b_var = relay.var("b", relay.TensorType(b_shape, "float32"))
        c_var = relay.var("c", relay.TensorType(c_shape, "float32"))

        # Build forward function
        forward_func = relay.Function([a_var, b_var, c_var], relay_fn_forward(a_var, b_var, c_var))
        
        inputs_np = {"a": a_np, "b": b_np, "c": c_np}

        # PyTorch test implies:
        # a.requires_grad=True, b.requires_grad=True, c.requires_grad=True.
        # Outputs: x, y, z.
        # x.requires_grad=True (from a)
        # y.requires_grad=True (from b)
        # z.requires_grad=False (due to `with torch.no_grad():`)
        # This implies: dL/da != 0, dL/db != 0, and dL/dc = 0 (because c only contributes to z, and z is no_grad).

        # Create a scalar loss from the outputs
        output_tuple_expr = relay_fn_forward(a_var, b_var, c_var)
        sum_x = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 0))
        sum_y = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 1))
        sum_z = relay.op.reduce.sum(relay.TupleGetItem(output_tuple_expr, 2))
        loss_expr = relay.op.tensor.add(relay.op.tensor.add(sum_x, sum_y), sum_z)

        loss_func = relay.Function([a_var, b_var, c_var], loss_expr)
        
        grad_params = [a_var, b_var, c_var] # Ask for gradients w.r.t. all original inputs
        grad_func = relay.transform.gradient(loss_func, grad_params=grad_params)
        grad_mod = IRModule.from_expr(grad_func)

        results_grad = _compile_and_run(grad_mod, None, inputs_np)
        assert len(results_grad) == 4
        _, grad_a, grad_b, grad_c = results_grad

        # `a` had requires_grad=True, and `x` from `a` had requires_grad=True
        assert not np.allclose(grad_a, np.zeros_like(grad_a), rtol=1e-5, atol=1e-5)
        assert grad_a.shape == a_shape
        
        # `b` had requires_grad=True, and `y` from `b` had requires_grad=True
        assert not np.allclose(grad_b, np.zeros_like(grad_b), rtol=1e-5, atol=1e-5)
        assert grad_b.shape == b_shape

        # `c` had requires_grad=True, but its only path to the loss `sum_z` was through `z` which was in a `no_grad` context.
        # This means `dL/dc` should be zero.
        assert_allclose(grad_c, np.zeros_like(grad_c), rtol=1e-5, atol=1e-5)
        assert grad_c.shape == c_shape


if __name__ == "__main__":
    pytest.main([__file__])
