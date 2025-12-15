import tvm
from tvm import relay
import numpy as np
import pytest
from scipy.special import erf as scipy_erf # For numerical reference for erf

# Helper class to mock common_utils.TestCase and provide assert methods.
class TVMExportTest:
    def assert_allclose(self, actual, desired, rtol=1e-5, atol=1e-8):
        tvm.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)

    def assertEqual(self, actual, expected):
        assert actual == expected

    def assertFalse(self, condition):
        assert not condition

    def assertTrue(self, condition):
        assert condition

    def skipTest(self, reason):
        pytest.skip(reason)

# Helper function to compile and run a Relay module
def compile_and_run_relay(relay_expr, input_dict, output_count=1, target="llvm"):
    """
    Builds and runs a Relay module.
    :param relay_expr: The Relay expression representing the model.
    :param input_dict: Dictionary of input names to numpy arrays.
    :param output_count: Expected number of outputs from the Relay graph.
    :param target: TVM target string.
    :return: List of numpy arrays representing outputs.
    """
    params = {}
    main_func_inputs = []
    for name, data_np in input_dict.items():
        var = relay.var(name, shape=data_np.shape, dtype=str(data_np.dtype))
        main_func_inputs.append(var)
        params[name] = tvm.nd.array(data_np)

    mod = tvm.IRModule.from_expr(relay.Function(main_func_inputs, relay_expr))

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(target, 0)
    module = tvm.runtime.GraphModule(lib["default"](dev))

    for name, data_nd in params.items():
        module.set_input(name, data_nd)

    module.run()
    outputs = []
    for i in range(output_count):
        outputs.append(module.get_output(i).numpy())
    return outputs

class TestAutogradFuns(TVMExportTest):
    # ONNX-specific attributes like opset_version, keep_initializers_as_inputs, onnx_shape_inference
    # are removed as they are not relevant to TVM Relay graph construction/testing.

    def test_single_output(self):
        input_np = np.ones(1, dtype=np.float32)

        # Equivalent to PyTorch's Caller.forward -> SingleOut.apply(result) + 3
        # result = input + 5
        # return result.exp().log() + 3
        input_var = relay.var("input", shape=input_np.shape, dtype=str(input_np.dtype))
        intermediate_add = relay.op.add(input_var, relay.const(5.0, dtype="float32"))
        exp_result = relay.op.exp(intermediate_add)
        log_result = relay.op.log(exp_result)
        output_relay = relay.op.add(log_result, relay.const(3.0, dtype="float32"))

        # Compute expected result using NumPy
        expected_np = np.log(np.exp(input_np + 5)) + 3

        # Compile and run with TVM
        outputs_tvm = compile_and_run_relay(output_relay, {"input": input_np})
        self.assert_allclose(outputs_tvm[0], expected_np)

    def test_multi_output(self):
        input_np = np.ones((1, 5), dtype=np.float32)

        # Equivalent to PyTorch's Caller.forward -> MultiOut.apply(input)
        # result_exp = input.exp()
        # result_log = result_exp.log()
        # return result_exp, result_log
        input_var = relay.var("input", shape=input_np.shape, dtype=str(input_np.dtype))
        result_exp = relay.op.exp(input_var)
        result_log = relay.op.log(result_exp)
        output_relay = relay.Tuple([result_exp, result_log])

        # Compute expected result using NumPy
        expected_exp_np = np.exp(input_np)
        expected_log_np = np.log(expected_exp_np)
        expected_np = [expected_exp_np, expected_log_np]

        # Compile and run with TVM
        outputs_tvm = compile_and_run_relay(output_relay, {"input": input_np}, output_count=2)
        self.assert_allclose(outputs_tvm[0], expected_np[0])
        self.assert_allclose(outputs_tvm[1], expected_np[1])

    def test_partial_output(self):
        # Creating non-uniform input for topk to be meaningful
        input_np = np.array([[1.0, 5.0, 2.0, 8.0, 3.0]], dtype=np.float32)
        k_val = 3

        # Equivalent to PyTorch's Caller.forward -> PartialOut.apply(input)
        # values, _ = torch.topk(input, 3)
        # return values
        input_var = relay.var("input", shape=input_np.shape, dtype=str(input_np.dtype))
        # TVM's topk returns a tuple (values, indices). We need to select the values.
        topk_result_tuple = relay.op.algorithm.topk(
            input_var,
            k=k_val,
            axis=-1,
            ret_type="both", # Request both values and indices
            is_ascend=False # PyTorch's topk is descending by default (largest=True)
        )
        output_relay = topk_result_tuple[0] # Extract the values

        # Compute expected result using NumPy (simulating PyTorch's topk behavior)
        # For simplicity, sorting and then taking the top-k for `values`
        sorted_indices = np.argsort(input_np, axis=-1)[:, ::-1] # Descending order
        expected_np = np.take_along_axis(input_np, sorted_indices[:, :k_val], axis=-1)

        # Compile and run with TVM
        outputs_tvm = compile_and_run_relay(output_relay, {"input": input_np})
        self.assert_allclose(outputs_tvm[0], expected_np)

    def test_nested_autograd(self):
        # Input needs to be positive after exp for the log operation
        input_np = np.ones((1, 5), dtype=np.float32) + 0.1

        # Equivalent to PyTorch's Caller.forward -> Parent.apply(input)
        # Parent.forward:
        #   result_exp = i.exp()
        #   result_log = Child.apply(result_exp)
        #   return result_exp, result_log
        # Child.forward:
        #   result = i.log()
        #   result_log = result.log()
        #   return result_log

        input_var = relay.var("input", shape=input_np.shape, dtype=str(input_np.dtype))
        
        # Parent's first output (result_exp)
        parent_result_exp = relay.op.exp(input_var)
        
        # Child.apply(parent_result_exp) logic
        child_intermediate_log = relay.op.log(parent_result_exp)
        child_output = relay.op.log(child_intermediate_log)
        
        output_relay = relay.Tuple([parent_result_exp, child_output])

        # Compute expected result using NumPy
        expected_parent_result_exp_np = np.exp(input_np)
        expected_child_output_np = np.log(np.log(expected_parent_result_exp_np))
        expected_np = [expected_parent_result_exp_np, expected_child_output_np]

        # Compile and run with TVM
        outputs_tvm = compile_and_run_relay(output_relay, {"input": input_np}, output_count=2)
        self.assert_allclose(outputs_tvm[0], expected_np[0])
        self.assert_allclose(outputs_tvm[1], expected_np[1])

    def test_aten_unsupported(self):
        # This test originally checked PyTorch's ONNX exporter behavior for unsupported ops (erf, erfinv).
        # Specifically, it checked if fallback nodes (prim::PythonOp, aten::ATen) were inserted.
        # This introspection of PyTorch's ONNX graph structure is not directly translatable to TVM Relay graph analysis.
        # However, the *forward computation* of `torch.special.erf` *is* translatable to `tvm.relay.op.tensor.erf`.
        # We will translate the computation and acknowledge that the original test's assertions about
        # ONNX exporter internals cannot be mapped.

        input_np = np.array([[0.5, 1.0, -0.5, -1.0, 0.0]], dtype=np.float32)

        # Equivalent to PyTorch's Caller.forward -> Erf.apply(input)
        # erf_out = torch.special.erf(x)
        input_var = relay.var("input", shape=input_np.shape, dtype=str(input_np.dtype))
        output_relay = relay.op.tensor.erf(input_var)

        # Compute expected result using SciPy's erf for numerical reference
        # (SciPy is an external dependency but commonly used for such functions in numerical tests)
        expected_np = scipy_erf(input_np)

        # Compile and run with TVM
        outputs_tvm = compile_and_run_relay(output_relay, {"input": input_np})
        self.assert_allclose(outputs_tvm[0], expected_np)

        # TODO: The original test's assertions about `graph.nodes().kind()` (`prim::PythonOp`, `aten::ATen`)
        # and `OperatorExportTypes` are specific to PyTorch's ONNX exporter internals and cannot be directly
        # mapped or reproduced in a TVM Relay graph test. They are removed.

    def test_inline_and_symbolic(self):
        input_np = np.ones(1, dtype=np.float32) + 0.1 # Ensure positive for log

        # Equivalent to PyTorch's Caller.forward
        # exp_result = Exp.apply(input) which uses Exp.symbolic -> g.op("Exp", input)
        # return LogLog.apply(exp_result) which uses LogLog.forward -> i.log().log()
        input_var = relay.var("input", shape=input_np.shape, dtype=str(input_np.dtype))
        exp_result = relay.op.exp(input_var)
        output_relay = relay.op.log(relay.op.log(exp_result))

        # Compute expected result using NumPy
        expected_np = np.log(np.log(np.exp(input_np)))

        # Compile and run with TVM
        outputs_tvm = compile_and_run_relay(output_relay, {"input": input_np})
        self.assert_allclose(outputs_tvm[0], expected_np)

    def test_inline_with_scoped_tracing(self):
        # This test is computationally identical to `test_inline_and_symbolic`.
        # The `torch.jit._trace._trace_module_map` setup and teardown are PyTorch JIT internals
        # and do not have direct equivalents in TVM's graph construction or execution.
        # We only translate the functional computation.

        input_np = np.ones(1, dtype=np.float32) + 0.1 # Ensure positive for log

        # Build Relay graph (same as test_inline_and_symbolic)
        input_var = relay.var("input", shape=input_np.shape, dtype=str(input_np.dtype))
        exp_result = relay.op.exp(input_var)
        output_relay = relay.op.log(relay.op.log(exp_result))

        # Compute expected result using NumPy
        expected_np = np.log(np.log(np.exp(input_np)))

        # Compile and run with TVM
        outputs_tvm = compile_and_run_relay(output_relay, {"input": input_np})
        self.assert_allclose(outputs_tvm[0], expected_np)


if __name__ == "__main__":
    pytest.main([__file__])
