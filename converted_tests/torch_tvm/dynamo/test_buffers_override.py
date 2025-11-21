# Owner(s): ["module: dynamo"]

import unittest
import tvm
import tvm.relay as relay
import numpy as np
import tvm.testing
import tvm.runtime


# In TVM, the concept of `nn.Module` and its `register_buffer` is part of
# frontend conversion. We'll simulate the structure using Relay components.
# `torch._dynamo.test_case.TestCase` becomes `unittest.TestCase`.
class TestBuffersOverride(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # For dynamic global state management like `g_counter` in other tests,
        # we'd reset it here. Not applicable for this specific test file.
        pass

    def _build_and_run_relay_model(self, relay_func, params, input_data):
        # Helper function to compile and run a Relay function
        target = "llvm" # Or "cuda" if HAS_CUDA_AND_TRITON is true in the original context
        dev = tvm.cpu(0) # Or tvm.cuda(0)

        mod = tvm.IRModule.from_expr(relay_func)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        runtime = tvm.runtime.GraphModule(lib["default"](dev))
        for i, (name, data) in enumerate(input_data.items()):
            runtime.set_input(name, tvm.nd.array(data))
        runtime.run()
        return runtime.get_output(0).numpy()


    def test_buffers_override(self):
        class SomeModelRelay:
            def __init__(self):
                # In PyTorch, A is a buffer. In Relay, it would be a parameter
                # to the function or a constant.
                self.A_val = np.ones((3, 3), dtype="float32")
                self.A = relay.const(self.A_val)
                # self.buffers is a PyTorch-specific override that TVM Relay doesn't directly map.
                # It's related to how PyTorch modules manage internal state.
                self.buffers = [] # This line is kept for structural similarity but has no direct TVM effect.

            def forward_relay(self):
                x = relay.var("x", shape=(1, 1), dtype="float32")
                # Equivalent of self.A * torch.zeros(1, 1)
                # `torch.zeros(1,1)` is a constant in Relay
                zero_tensor = relay.const(np.zeros((1, 1), dtype="float32"))
                output = relay.op.multiply(self.A, zero_tensor)
                return relay.Function([x], output) # x is not used, but a graph needs input vars

        model = SomeModelRelay()
        # Original: compiled_model = torch.compile(model)
        # In TVM, we directly get the Relay function and compile it.
        relay_func = model.forward_relay()
        
        # `compiled_model.A` refers to the original PyTorch buffer.
        # Here we verify the value of `model.A_val` (the underlying numpy array for the Relay const).
        tvm.testing.assert_allclose(model.A_val, np.ones((3, 3), dtype="float32"))
        
        # Original: compiled_model()
        # To run the Relay graph, we need an input. The forward_relay defined above
        # uses a dummy input `x` that is not actually used in the computation,
        # but a Relay function usually expects inputs.
        dummy_input = {"x": np.zeros((1, 1), dtype="float32")}
        result_np = self._build_and_run_relay_model(relay_func, {}, dummy_input) # No extra params beyond constants
        
        # Assert against the expected output of `torch.ones(3,3) * torch.zeros(1,1)`
        expected_output = np.zeros((3, 3), dtype="float32")
        tvm.testing.assert_allclose(result_np, expected_output)


    def test_named_buffers_override(self):
        class SomeModelRelay:
            def __init__(self):
                self.B_val = np.ones((3, 3), dtype="float32")
                self.B = relay.const(self.B_val)
                # self.named_buffers is a PyTorch-specific override
                self.named_buffers = [] # This line is kept for structural similarity but has no direct TVM effect.

            def forward_relay(self):
                x = relay.var("x", shape=(1, 1), dtype="float32")
                zero_tensor = relay.const(np.zeros((1, 1), dtype="float32"))
                output = relay.op.multiply(self.B, zero_tensor)
                return relay.Function([x], output)

        model = SomeModelRelay()
        relay_func = model.forward_relay()

        tvm.testing.assert_allclose(model.B_val, np.ones((3, 3), dtype="float32"))

        dummy_input = {"x": np.zeros((1, 1), dtype="float32")}
        result_np = self._build_and_run_relay_model(
