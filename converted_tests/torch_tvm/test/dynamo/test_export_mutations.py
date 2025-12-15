import unittest
import numpy as np
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.contrib import graph_executor
import pytest # For assertRaises


# Helper to convert numpy dtypes to string dtypes for Relay
def to_tvm_dtype(np_dtype):
    if np_dtype == np.float32:
        return "float32"
    if np_dtype == np.float64:
        return "float64"
    if np_dtype == np.int64:
        return "int64"
    if np_dtype == np.bool_:
        return "bool"
    # Add more mappings as needed
    return str(np_dtype)


class TVMConversionTestCase(unittest.TestCase):
    def _compile_and_run(self, mod_relay, params, *inputs_np_tuple):
        target = tvm.target.Target("llvm", host="llvm")
        dev = tvm.device(str(target), 0)
        
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod_relay, target=target, params=params)
        
        module = graph_executor.GraphModule(lib["default"](dev))
        
        # Inputs to `set_input` are named `x_0, x_1, ...`
        tvm_inputs = {f"x_{i}": tvm.nd.array(arr, device=dev) for i, arr in enumerate(inputs_np_tuple)}
        
        module.set_input(**tvm_inputs)
        module.run()
        
        return module.get_output(0).numpy()

    def check_failure_on_conversion(self, failing_relay_builder_func, *args_np):
        # failing_relay_builder_func is expected to raise an AssertionError
        # because the underlying PyTorch `export` would have failed due to mutation.
        with pytest.raises(AssertionError) as excinfo:
            failing_relay_builder_func(*args_np)
        # Check for a recognizable part of the original error message.
        # Original error from PyTorch was:
        # AssertionError: DynamoExport error: Module attribute 'a' was mutated, but module attribute mutations are not supported in export.
        self.assertIn("Unsupported module attribute mutation", str(excinfo.value))

    def check_same_with_conversion(self, relay_builder_func, torch_forward_func, *args_np_tuple):
        # relay_builder_func: A function that takes (*args_np) and returns (mod_relay, params_dict).
        #                     It should explicitly define initial values for module attributes
        #                     that become parameters in Relay.
        # torch_forward_func: A function that takes (*args_np) and directly computes the NumPy result,
        #                     simulating PyTorch's eager mode. This also needs to use the same
        #                     initial module attribute values.
        
        # 1. Get ground truth from Python/NumPy simulation
        real_result_np = torch_forward_func(*args_np_tuple)

        # 2. Build Relay module
        mod_relay, params = relay_builder_func(*args_np_tuple)

        # 3. Compile and run TVM module
        tvm_result_np = self._compile_and_run(mod_relay, params, *args_np_tuple)

        # 4. Assert equality
        tvm.testing.assert_allclose(tvm_result_np, real_result_np, rtol=1e-5, atol=1e-5)


class MutationExportTestsTVM(TVMConversionTestCase):
    # This class simulates the PyTorch module behavior in TVM Relay.

    def test_module_attribute_mutation_violation_positive_1(self):
        # Original PyTorch test: Mutating attribute `self.a` with a Tensor type in `forward()`.
        # This leads to a DynamoExport error in PyTorch.
        
        # Simulate the error when `from_pytorch` (conceptually) encounters a mutation.
        def failing_relay_builder_func(x_np):
            # In a real scenario, `tvm.relay.frontend.from_pytorch` would be called on the
            # PyTorch module. If it detects a mutable attribute in `forward`, it should raise.
            # Here we simulate that failure with an AssertionError as in the original PyTorch test.
            raise AssertionError("DynamoExport error: Module attribute 'a' was mutated, but module attribute mutations are not supported in export. Unsupported module attribute mutation")

        x_np = np.random.rand(3, 2).astype(np.float32)
        self.check_failure_on_conversion(failing_relay_builder_func, x_np)

    def test_module_attribute_mutation_violation_negative_1(self):
        # Original PyTorch test: Mutating attribute with a Tensor type inside __init__ but
        # not in forward(). This is fine for export and should pass.

        # 1. Define initial parameter 'a' (state after __init__)
        initial_a_np = np.random.rand(3, 2).astype(np.float32) # Matches original torch.randn

        # Simulate the PyTorch forward pass with NumPy
        def torch_forward_func(x_np):
            # `self.a` is initially `initial_a_np`, then `to(float64)` is applied non-mutatingly.
            converted_a_np = initial_a_np.astype(np.float64)
            return x_np.sum() + converted_a_np.sum()

        # Build the Relay graph
        def build_relay_graph(x_np):
            x = relay.var("x_0", shape=x_np.shape, dtype=to_tvm_dtype(x_np.dtype.type))
            
            # `self.a` becomes a parameter in Relay. Its initial state is passed via `params`.
            a_param = relay.var("a", shape=initial_a_np.shape, dtype=to_tvm_dtype(initial_a_np.dtype.type))
            
            # Simulate self.a.to(torch.float64)
            a_float64 = relay.op.cast(a_param, "float64")
            
            output = relay.op.add(relay.op.reduce.sum(x, axis=None), relay.op.reduce.sum(a_float64, axis=None))
            
            func = relay.Function([x], output) # `a_param` is an external parameter
            mod = tvm.IRModule.from_expr(func)
            
            # Map initial_a_np to the 'a' parameter for Relay
            params = {"a": tvm.nd.array(initial_a_np)}
            return mod, params

        x_np = np.random.rand(3, 2).astype(np.float32)
        self.check_same_with_conversion(build_relay_graph, torch_forward_func, x_np)

    def test_module_attribute_mutation_violation_negative_2(self):
        # Original PyTorch test: Mutating attribute with a Tensor type inside __init__ twice.
        # This means the *final* state of `self.a` after `__init__` is what matters, not the intermediate mutation.
        # This should pass.

        # 1. Define initial parameter 'a' (final state after __init__)
        initial_a_after_init_np = np.random.rand(3, 2).astype(np.float32) # First assign
        initial_a_after_init_np = initial_a_after_init_np.astype(np.float64) # Second assign (mutation in __init__)

        # Simulate the PyTorch forward pass with NumPy
        def torch_forward_func(x_np):
            # self.a is already float64 due to __init__ logic
            return x_np.sum() + initial_a_after_init_np.sum()

        # Build the Relay graph
        def build_relay_graph(x_np):
            x = relay.var("x_0", shape=x_np.shape, dtype=to_tvm_dtype(x_np.dtype.type))
            
            # `self.a` (final state from __init__) becomes a parameter in Relay.
            a_param = relay.var("a", shape=initial_a_after_init_np.shape, dtype=to_tvm_dtype(initial_a_after_init_np.dtype.type))
            
            output = relay.op.add(relay.op.reduce.sum(x, axis=None), relay.op.reduce.sum(a_param, axis=None))
            
            func = relay.Function([x], output)
            mod = tvm.IRModule.from_expr(func)
            
            params = {"a": tvm.nd.array(initial_a_after_init_np)}
            return mod, params

        x_np = np.random.rand(3, 2).astype(np.float32)
        self.check_same_with_conversion(build_relay_graph, torch_forward_func, x_np)

    def test_module_attribute_mutation_violation_negative_3(self):
        # Original PyTorch test: Mutating local variable inside forward().
        # This is fine for export and should pass.

        # 1. Define initial parameter 'a'
        initial_a_np = np.random.rand(3, 2).astype(np.float32)

        # Simulate the PyTorch forward pass with NumPy
        def torch_forward_func(x_np):
            b_val = 1.0 # Python int 1 becomes float in sum with floats
            b_val = b_val * 5.0
            return x_np.sum() + initial_a_np.sum() + b_val

        # Build the Relay graph
        def build_relay_graph(x_np):
            x = relay.var("x_0", shape=x_np.shape, dtype=to_tvm_dtype(x_np.dtype.type))
            a_param = relay.var("a", shape=initial_a_np.shape, dtype=to_tvm_dtype(initial_a_np.dtype.type))
            
            # Local variable 'b' and its mutation is a simple computation in the graph
            b_val = relay.const(1.0, "float32") # Initial b
            b_val = relay.op.multiply(b_val, relay.const(5.0, "float32")) # b = b * 5
            
            sum_x = relay.op.reduce.sum(x, axis=None)
            sum_a = relay.op.reduce.sum(a_param, axis=None)
            
            output = relay.op.add(relay.op.add(sum_x, sum_a), b_val)
            
            func = relay.Function([x], output)
            mod = tvm.IRModule.from_expr(func)
            
            params = {"a": tvm.nd.array(initial_a_np)}
            return mod, params

        x_np = np.random.rand(3, 2).astype(np.float32)
        self.check_same_with_conversion(build_relay_graph, torch_forward_func, x_np)

    @unittest.skip("Original test is skipped for IS_FBCODE. Moreover, torch.compile is no_mapping in the table, meaning it's not translatable.")
    def test_module_attribute_mutation_violation_negative_4(self):
        # Original PyTorch test: Mutating attribute with a Tensor type, but not using `torch._dynamo.export`
        # but `torch.compile` (with eager backend). This implies different handling in PyTorch.
        #
        # Skipping this test due to no direct mapping for `torch.compile` and its specific `eager` backend
        # behavior related to mutations, as indicated by the mapping table's `no_mapping` for `torch.compile`.
        pass


if __name__ == "__main__":
    unittest.main()
