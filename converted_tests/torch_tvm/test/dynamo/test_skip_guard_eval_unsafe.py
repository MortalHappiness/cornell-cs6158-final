import unittest
import numpy as np
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay.op.tensor import add, multiply
from tvm.relay.op.transform import arange, reshape, squeeze, split, transpose, tile, full, full_like, zeros, ones, zeros_like, ones_like, broadcast_to, where
from tvm.relay.op.nn import adaptive_avg_pool2d, adaptive_max_pool2d, avg_pool2d, avg_pool3d, batch_norm, conv1d, conv2d, conv3d, dropout, group_norm, instance_norm, layer_norm, leaky_relu, log_softmax, nll_loss, prelu, relu, softmax
from tvm.relay.op.tensor import abs, acos, acosh, asin, asinh, atan, atanh, bitwise_and, bitwise_not, bitwise_or, bitwise_xor, ceil, cos, cosh, erf, exp, floor, isfinite, isinf, isnan, log, log10, log2, logical_and, logical_not, logical_or, logical_xor, maximum, minimum, sign, sin, sinh, sqrt, tan, tanh, trunc, divide, floor_divide, trunc_divide
from tvm.relay.op.reduce import argmax, argmin, all, any, max, mean, min, prod, sum, logsumexp, std
from tvm.relay.op.random.kernel import normal, uniform, threefry_key
from tvm.relay.op.algorithm import argsort, searchsorted, sort, topk
from tvm.runtime.ndarray import array, empty, from_dlpack
import tvm.testing
from tvm.topi.transform import flip, tensordot
from tvm.tir.op import atan2, ceildiv, copysign, exp2, fmod, hypot, ldexp, log1p
from tvm.relay.frontend.common import unbind
from tvm.relay.op.image import affine_grid, grid_sample
from tvm.relay.qnn.op import qnn # Import qnn module for qnn ops
import pytest # Although not directly used for assertRaisesRegex, useful for general testing framework

# Helper to map numpy dtypes to TVM string dtypes
def _to_tvm_dtype(np_dtype):
    if np_dtype == np.float32:
        return "float32"
    if np_dtype == np.float64:
        return "float64"
    if np_dtype == np.int64:
        return "int664"
    if np_dtype == np.int32:
        return "int32"
    if np_dtype == np.bool_:
        return "bool"
    # Fallback for other dtypes
    return str(np_dtype)


class RunDiffGuardTests(unittest.TestCase):
    def test_bool_recompile(self):
        def fn(x, y, c):
            if c:
                return x * y
            else:
                return x + y

        # Manual Relay graph construction for `fn`
        # This function builds a Relay graph specific to the Python boolean `c_val`.
        def build_bool_fn_relay(py_fn_ref, x_val_np, y_val_np, c_val):
            x_ph = relay.var("x", shape=x_val_np.shape, dtype=_to_tvm_dtype(x_val_np.dtype))
            y_ph = relay.var("y", shape=y_val_np.shape, dtype=_to_tvm_dtype(y_val_np.dtype))
            
            # The conditional 'c_val' is a Python boolean, dictating which Relay graph is built.
            # This simulates what TorchDynamo would do based on dynamic control flow,
            # resulting in different compiled graphs for True/False paths.
            if c_val:
                relay_expr = multiply(x_ph, y_ph)
            else:
                relay_expr = add(x_ph, y_ph)
            
            func = relay.Function([x_ph, y_ph], relay_expr)
            mod = tvm.IRModule.from_expr(func)
            return mod, {}, [x_ph, y_ph], {} # Return module, empty params, input placeholders, empty constant map

        # Helper to create a specific TVM runner that matches how `fn` is called with a fixed `c`
        class SpecializedTVMRunner:
            def __init__(self, c_const_value, py_fn_ref):
                self.c_const = c_const_value
                self._py_fn_ref = py_fn_ref
                self._compiled_func = None
                self._input_phs_names = []
                self._cached_input_shapes = None
                self._cached_input_dtypes = None
                
            def __call__(self, x_nd, y_nd):
                x_np = x_nd.numpy()
                y_np = y_nd.numpy()

                current_input_shapes = (x_np.shape, y_np.shape)
                current_input_dtypes = (_to_tvm_dtype(x_np.dtype), _to_tvm_dtype(y_np.dtype))

                if self._compiled_func is None or \
                   self._cached_input_shapes != current_input_shapes or \
                   self._cached_input_dtypes != current_input_dtypes:
                    
                    print(f"TVM: Building/Re-building graph for c={self.c_const}, shapes={current_input_shapes}, dtypes={current_input_dtypes}")
                    mod, _, input_phs, _ = build_bool_fn_relay(
                        self._py_fn_ref, x_np, y_np, self.c_const
                    )
                    target = tvm.target.Target("llvm", host="llvm")
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build(mod, target=target)
                    self._compiled_func = tvm.runtime.GraphModule(lib["default"](tvm.cpu(0)))
                    self._input_phs_names = [ph.name_hint for ph in input_phs]
                    self._cached_input_shapes = current_input_shapes
                    self._cached_input_dtypes = current_input_dtypes
                        
                # Execute the cached graph
                inputs = {self._input_phs_names[0]: x_nd, self._input_phs_names[1]: y_nd}
                self._compiled_func.set_input(**inputs)
                self._compiled_func.run()
                return self._compiled_func.get_output(0)
            
        # Simulate torch.compile creating different graphs for different 'c' values
        opt_fn_true = SpecializedTVMRunner(True, fn)
        opt_fn_false = SpecializedTVMRunner(False, fn)

        x_np = 2 * np.ones(4, dtype=np.float32)
        y_np = 3 * np.ones(4, dtype=np.float32)
        x_tvm = tvm.nd.array(x_np)
        y_tvm = tvm.nd.array(y_np)

        ref1_np = fn(x_np, y_np, True)
        ref2_np = fn(x_np, y_np, False)

        res1_tvm = opt_fn_true(x_tvm, y_tvm)
        res2_tvm = opt_fn_false(x_tvm, y_tvm)

        tvm.testing.assert_allclose(ref1_np, res1_tvm.numpy(), rtol=1e-5, atol=1e-5)
        tvm.testing.assert_allclose(ref2_np, res2_tvm.numpy(), rtol=1e-5, atol=1e-5)

        # The 'torch.compiler.set_stance(skip_guard_eval_unsafe=True)' context
        # is for PyTorch's internal guard evaluation.
        # TVM does not have this concept directly as its graph is static once built for a given type/shape.
        # Here, we verify that the pre-compiled graphs still produce correct results.
        # The recompilation behavior itself is not replicated or asserted here.
        # The calls below will use the already cached compiled functions.
        res2_tvm_rerun = opt_fn_false(x_tvm, y_tvm)
        res1_tvm_rerun = opt_fn_true(x_tvm, y_tvm)

        tvm.testing.assert_allclose(ref1_np, res1_tvm_rerun.numpy(), rtol=1e-5, atol=1e-5)
        tvm.testing.assert_allclose(ref2_np, res2_tvm_rerun.numpy(), rtol=1e-5, atol=1e-5)


    def test_tensor_recompile(self):
        def fn(x, y):
            return x * y

        def build_tensor_fn_relay(py_fn_ref, x_val_np, y_val_np):
            x_ph = relay.var("x", shape=x_val_np.shape, dtype=_to_tvm_dtype(x_val_np.dtype))
            y_ph = relay.var("y", shape=y_val_np.shape, dtype=_to_tvm_dtype(y_val_np.dtype))
            relay_expr = multiply(x_ph, y_ph)
            func = relay.Function([x_ph, y_ph], relay_expr)
            mod = tvm.IRModule.from_expr(func)
            return mod, {}, [x_ph, y_ph], {}

        class SimpleTVMRunner:
            def __init__(self, python_fn_ref):
                self._python_fn = python_fn_ref
                self._compiled_func_cache = {} # Cache based on (shape, dtype) of inputs
            
            def __call__(self, x_nd, y_nd):
                x_np = x_nd.numpy()
                y_np = y_nd.numpy()

                current_input_dtypes = (_to_tvm_dtype(x_np.dtype), _to_tvm_dtype(y_np.dtype))
                current_input_shapes = (x_np.shape, y_np.shape)
                cache_key = (current_input_shapes, current_input_dtypes)

                if cache_key not in self._compiled_func_cache:
                    print(f"TVM: Building/Re-building graph for shapes={current_input_shapes}, dtypes={current_input_dtypes}")
                    mod, _, input_phs, _ = build_tensor_fn_relay(self._python_fn, x_np, y_np)
                    target = tvm.target.Target("llvm", host="llvm")
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build(mod, target=target)
                    runtime_module = tvm.runtime.GraphModule(lib["default"](tvm.cpu(0)))
                    self._compiled_func_cache[cache_key] = (runtime_module, [ph.name_hint for ph in input_phs])
                
                runtime_module, input_phs_names = self._compiled_func_cache[cache_key]

                inputs = {input_phs_names[0]: x_nd, input_phs_names[1]: y_nd}
                runtime_module.set_input(**inputs)
                runtime_module.run()
                return runtime_module.get_output(0)

        opt_fn = SimpleTVMRunner(fn)

        x_f32_np = np.random.randn(4).astype(np.float32)
        y_f32_np = np.random.randn(4).astype(np.float32)
        x_f32_tvm = tvm.nd.array(x_f32_np)
        y_f32_tvm = tvm.nd.array(y_f32_np)

        ref1_np = fn(x_f32_np, y_f32_np)
        res1_tvm = opt_fn(x_f32_tvm, y_f32_tvm)
        tvm.testing.assert_allclose(ref1_np, res1_tvm.numpy(), rtol=1e-5, atol=1e-5)

        x_f64_np = np.random.randn(4).astype(np.float64)
        y_f64_np = np.random.randn(4).astype(np.float64)
        x_f64_tvm = tvm.nd.array(x_f64_np)
        y_f64_tvm = tvm.nd.array(y_f64_np)
        
        # This will trigger a "recompilation" in our simplified TVM model because dtypes changed
        ref2_np = fn(x_f64_np, y_f64_np)
        res2_tvm = opt_fn(x_f64_tvm, y_f64_tvm)
        tvm.testing.assert_allclose(ref2_np, res2_tvm.numpy(), rtol=1e-5, atol=1e-5)

        # The 'torch.compiler.set_stance' context is for PyTorch's internal guard evaluation.
        # For TVM, re-running with original inputs just uses the cached or rebuilt graph.
        res1_tvm_rerun = opt_fn(x_f32_tvm, y_f32_tvm)
        res2_tvm_rerun = opt_fn(x_f64_tvm, y_f64_tvm)

        tvm.testing.assert_allclose(ref1_np, res1_tvm_rerun.numpy(), rtol=1e-5, atol=1e-5)
        tvm.testing.assert_allclose(ref2_np, res2_tvm_rerun.numpy(), rtol=1e-5, atol=1e-5)


    def test_post_recompile(self):
        class Foo:
            def __init__(self):
                self.a = 4
                self.b = 5

        foo = Foo()

        # This `fn` closes over `foo.a` and `foo.b`
        # In TorchDynamo, changes to `foo.a` would trigger recompilation
        def fn(x_np_val): # Takes numpy array to match expected ref computation
            return x_np_val + foo.a + foo.b

        # Helper to build Relay graph for this specific fn
        def build_post_recompile_relay(py_fn_ref, x_val_np, current_foo_a, current_foo_b):
            x_ph = relay.var("x", shape=x_val_np.shape, dtype=_to_tvm_dtype(x_val_np.dtype))
            
            # These are captured Python values, so they become Relay constants
            a_const = relay.const(float(current_foo_a), dtype=_to_tvm_dtype(x_val_np.dtype))
            b_const = relay.const(float(current_foo_b), dtype=_to_tvm_dtype(x_val_np.dtype))
            
            relay_expr = add(x_ph, a_const)
            relay_expr = add(relay_expr, b_const)
            
            func = relay.Function([x_ph], relay_expr)
            mod = tvm.IRModule.from_expr(func)
            return mod, {}, [x_ph], {}

        # Custom TVM runner that rebuilds graph if captured python values change
        class PostRecompileTVMRunner:
            def __init__(self, python_fn_ref, initial_x_np):
                self._python_fn = python_fn_ref
                self._compiled_func = None
                self._input_phs_names = []
                self._cached_a = None
                self._cached_b = None
                self.frame_count = 0 # Simulate compile counter

            def __call__(self, x_nd):
                x_np = x_nd.numpy()
                
                # Check if the captured `foo.a` or `foo.b` values have changed
                # This simulates TorchDynamo's guard mechanism for global/closure variables
                if self._compiled_func is None or \
                   self._cached_a != foo.a or self._cached_b != foo.b:
                    
                    print(f"TVM: Re-building graph due to change in foo.a ({self._cached_a} -> {foo.a}) or foo.b ({self._cached_b} -> {foo.b})")
                    mod, _, input_phs, _ = build_post_recompile_relay(
                        self._python_fn, x_np, foo.a, foo.b
                    )
                    target = tvm.target.Target("llvm", host="llvm")
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build(mod, target=target)
                    self._compiled_func = tvm.runtime.GraphModule(lib["default"](tvm.cpu(0)))
                    self._input_phs_names = [ph.name_hint for ph in input_phs]
                    self._cached_a = foo.a
                    self._cached_b = foo.b
                    self.frame_count += 1 # Increment frame_count on "recompilation"

                inputs = {self._input_phs_names[0]: x_nd}
                self._compiled_func.set_input(**inputs)
                self._compiled_func.run()
                return self._compiled_func.get_output(0)

        x_np = np.random.randn(4).astype(np.float32)
        x_tvm = tvm.nd.array(x_np)

        opt_fn = PostRecompileTVMRunner(fn, x_np)

        ref = fn(x_np)
        res = opt_fn(x_tvm)
        tvm.testing.assert_allclose(ref, res.numpy(), rtol=1e-5, atol=1e-5)
        self.assertEqual(opt_fn.frame_count, 1)

        foo.a = 11
        ref = fn(x_np)
        res = opt_fn(x_tvm)
        tvm.testing.assert_allclose(ref, res.numpy(), rtol=1e-5, atol=1e-5)
        self.assertEqual(opt_fn.frame_count, 2)

        # --- Begin `torch.compiler.set_stance(skip_guard_eval_unsafe=True)` block ---
        # This context manager affects PyTorch's internal guard logic.
        # In TVM, we don't have equivalent guards to "skip", so our runner
        # will continue to rebuild the graph if the captured Python variables (foo.a, foo.b) change.
        # This means the `frame_count` will diverge from PyTorch's original test expectations
        # because PyTorch's `skip_guard_eval_unsafe=True` would prevent recompilation
        # even if `foo.a` changes, assuming its guards are truly bypassed.
        # We focus on numerical correctness.
        print("TODO: test_post_recompile: 'skip_guard_eval_unsafe' context is specific to TorchDynamo's guard management.")
        print("      TVM does not have this concept. Numerical results are checked, but frame_count assertions")
        print("      within this conceptual block are not directly translatable and are therefore omitted.")

        # Set it back to original value (foo.a = 4)
        foo.a = 4
        # Our TVM runner WILL rebuild the graph here because foo.a changed from 11 back to 4.
        ref = fn(x_np)
        res = opt_fn(x_tvm)
        tvm.testing.assert_allclose(ref, res.numpy(), rtol=1e-5, atol=1e-5)

        foo.a = 11
        ref = fn(x_np)
        res = opt_fn(x_tvm)
        tvm.testing.assert_allclose(ref, res.numpy(), rtol=1e-5, atol=1e-5)

        # --- End `torch.compiler.set_stance` block ---

        # Check that we are back to original behavior
        # (This implies Dynamo's guards would be re-enabled after the context manager exits)
        # Our TVM runner continues to rebuild if params change.
        foo.b = 8
        ref = fn(x_np)
        res = opt_fn(x_tvm)
        tvm.testing.assert_allclose(ref, res.numpy(), rtol=1e-5, atol=1e-5)
        # The original test expected frame_count to be 3 here, but in our TVM simulation
        # it would be higher due to rebuilding graphs inside the 'skipped guards' block.
        # Thus, we omit this frame_count assertion here.


    def test_fail_on_tensor_shape_change(self):
        # This test relies on a specific TorchDynamo runtime error message
        # when `skip_guard_eval_unsafe` is enabled and a tensor shape changes.
        # This behavior is deeply tied to TorchDynamo's guard management
        # and has no direct equivalent in TVM's static graph compilation.
        # TVM would simply fail to execute a pre-compiled graph with mismatched shapes,
        # or require a new graph to be built for the new shape.
        # The specific error message "Recompilation triggered with skip_guard_eval_unsafe stance"
        # cannot be replicated.

        print("TODO: test_fail_on_tensor_shape_change is highly specific to TorchDynamo's guarding logic and error messages.")
        print("      Direct replication in TVM is not possible. A simplified simulation is provided below.")

        # The original `fn` from PyTorch test:
        def fn(dt):
            return dt["x"] + 1

        # Define a mock TVM runner that captures the input shape
        class MockTVMRunnerForShapeGuard:
            def __init__(self, initial_x_shape, initial_x_dtype):
                self._cached_shape = initial_x_shape
                self._cached_dtype = initial_x_dtype
                self._compiled_func = None

                # Build the initial graph for the expected shape/dtype
                x_ph = relay.var("x", shape=initial_x_shape, dtype=initial_x_dtype)
                relay_expr = add(x_ph, relay.const(1, dtype=initial_x_dtype))
                func = relay.Function([x_ph], relay_expr)
                mod = tvm.IRModule.from_expr(func)
                target = tvm.target.Target("llvm", host="llvm")
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(mod, target=target)
                self._compiled_func = tvm.runtime.GraphModule(lib["default"](tvm.cpu(0)))
                self._input_ph_name = x_ph.name_hint
            
            def __call__(self, dt):
                current_x_nd = dt["x"]
                if current_x_nd.shape != self._cached_shape or str(current_x_nd.dtype) != self._cached_dtype:
                    # Simulate guard failure
                    raise RuntimeError("Shape or dtype change detected! (Simulated TorchDynamo guard failure)")
                
                # If shapes/dtypes match, execute the compiled graph
                inputs = {self._input_ph_name: current_x_nd}
                self._compiled_func.set_input(**inputs)
                self._compiled_func.run()
                return self._compiled_func.get_output(0)


        x_np_initial = np.random.randn(4).astype(np.float32)
        dt_initial = {"x": tvm.nd.array(x_np_initial)}
        
        # Simulate torch.compile capturing initial shape/dtype
        opt_fn = MockTVMRunnerForShapeGuard(x_np_initial.shape, _to_tvm_dtype(x_np_initial.dtype))
        
        # First call works
        ref_initial = x_np_initial + 1 # Compute reference with numpy
        res_initial = opt_fn(dt_initial)
        tvm.testing.assert_allclose(ref_initial, res_initial.numpy(), rtol=1e-5, atol=1e-5)

        # Now, change shape, simulating the failure condition that the original test checks
        with self.assertRaisesRegex(
            RuntimeError, "Shape or dtype change detected! (Simulated TorchDynamo guard failure)"
        ):
            # `torch.compiler.set_stance(skip_guard_eval_unsafe=True)` would try to bypass guards,
            # but then still recompile and raise a specific error in PyTorch.
            # Our mock simply raises a general runtime error on shape mismatch.
            x_np_new = np.random.randn(4, 4).astype(np.float32)
            dt_new = {"x": tvm.nd.array(x_np_new)}
            opt_fn(dt_new)

    def test_cache_line_pickup(self):
        def fn(x_np_val, a=None, b=None): # Takes numpy array to match expected ref computation
            x_np_val = x_np_val * 3
            if a: # Python boolean check
                x_np_val = x_np_val * 5
            if b: # Python boolean check
                x_np_val = x_np_val * 7
            return x_np_val

        # Helper to build Relay graph for this specific fn
        def build_conditional_fn_relay(py_fn_ref, x_val_np, a_val, b_val):
            x_ph = relay.var("x", shape=x_val_np.shape, dtype=_to_tvm_dtype(x_val_np.dtype))
            
            result_expr = multiply(x_ph, relay.const(3.0, dtype=_to_tvm_dtype(x_val_np.dtype)))
            if a_val:
                result_expr = multiply(result_expr, relay.const(5.0, dtype=_to_tvm_dtype(x_val_np.dtype)))
            if b_val:
                result_expr = multiply(result_expr, relay.const(7.0, dtype=_to_tvm_dtype(x_val_np.dtype)))
            
            func = relay.Function([x_ph], result_expr)
            mod = tvm.IRModule.from_expr(func)
            return mod, {}, [x_ph], {}

        # Custom TVM runner for this test that compiles a specific graph based on (a, b)
        class ConditionalTVMRunner:
            def __init__(self, python_fn_ref):
                self._python_fn = python_fn_ref
                self._compiled_cache = {} # Cache compiled graphs by (a, b, dtype, shape) tuple
                
            def __call__(self, x_nd, a=None, b=None):
                x_np = x_nd.numpy()
                cache_key = (bool(a), bool(b), _to_tvm_dtype(x_np.dtype), x_np.shape)
                
                if cache_key not in self._compiled_cache:
                    print(f"TVM: Building and compiling graph for cache key: {cache_key}")
                    mod, _, input_phs, _ = build_conditional_fn_relay(
                        self._python_fn, x_np, bool(a), bool(b)
                    )
                    target = tvm.target.Target("llvm", host="llvm")
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build(mod, target=target)
                    runtime_module = tvm.runtime.GraphModule(lib["default"](tvm.cpu(0)))
                    self._compiled_cache[cache_key] = (runtime_module, [ph.name_hint for ph in input_phs])
                
                runtime_module, input_phs_names = self._compiled_cache[cache_key]
                inputs = {input_phs_names[0]: x_nd}
                runtime_module.set_input(**inputs)
                runtime_module.run()
                return runtime_module.get_output(0)

        x_np_init = np.ones(4, dtype=np.float32)
        x_tvm_init = tvm.nd.array(x_np_init)
        opt_fn = ConditionalTVMRunner(fn)

        # Test cases for different (a, b) combinations
        ref1_np = fn(x_np_init, a=None, b=None)
        res1_tvm = opt_fn(x_tvm_init, a=None, b=None)
        tvm.testing.assert_allclose(ref1_np, res1_tvm.numpy(), rtol=1e-5, atol=1e-5)

        ref2_np = fn(x_np_init, a=1, b=None)
        res2_tvm = opt_fn(x_tvm_init, a=1, b=None)
        tvm.testing.assert_allclose(ref2_np, res2_tvm.numpy(), rtol=1e-5, atol=1e-5)

        ref3_np = fn(x_np_init, a=1, b=1)
        res3_tvm = opt_fn(x_tvm_init, a=1, b=1)
        tvm.testing.assert_allclose(ref3_np, res3_tvm.numpy(), rtol=1e-5, atol=1e-5)

        # The `torch.compiler.set_stance(skip_guard_eval_unsafe=True)` context
        # influences PyTorch's internal guard evaluation logic.
        # For TVM, the compiled functions are already determined by the (a,b) values at graph build time.
        # Re-running with `skip_guard_eval_unsafe` would simply re-execute the *same* cached graphs.
        print("TODO: test_cache_line_pickup: 'skip_guard_eval_unsafe' context is specific to TorchDynamo's guard management.")
        print("      TVM does not have this concept. Numerical results are checked, but the guard-skipping behavior is not replicated.")

        res1_tvm_rerun = opt_fn(x_tvm_init, a=None, b=None)
        res2_tvm_rerun = opt_fn(x_tvm_init, a=1, b=None)
        res3_tvm_rerun = opt_fn(x_tvm_init, a=1, b=1)

        tvm.testing.assert_allclose(ref1_np, res1_tvm_rerun.numpy(), rtol=1e-5, atol=1e-5)
        tvm.testing.assert_allclose(ref2_np, res2_tvm_rerun.numpy(), rtol=1e-5, atol=1e-5)
        tvm.testing.assert_allclose(ref3_np, res3_tvm_rerun.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    # Original: from torch._dynamo.test_case import run_tests; run_tests()
    # Replaced with standard unittest.main()
    unittest.main()
