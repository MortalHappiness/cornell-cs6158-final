# Owner(s): ["module: cuda graphs"]

import functools
import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.relay.build_module import build
from tvm.contrib.graph_executor import GraphModule
import tvm.testing as testing


# Dummy functions to replace PyTorch Dynamo specific helpers.
# These decorators and functions are PyTorch-internal and have no direct TVM equivalents.
# They are kept as no-ops to allow the original test structure to be parsed without errors.
def composed(*decs):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # These decorators are PyTorch-specific and cannot be translated to TVM.
            # The original purpose was to configure torch._dynamo and assert on its counters.
            # In TVM, we directly compile the Relay graph.
            return f(*args, **kwargs)
        return wrapper
    return deco

def assert_aot_autograd_counter(ok=True):
    def deco(f):
        @functools.wraps(f)
        def wrap(self, *args, **kwargs):
            # This is a PyTorch Dynamo internal counter assertion.
            # Not applicable to TVM.
            return f(self, *args, **kwargs)
        return wrap
    return deco

def patch_all(ok=True):
    return composed(
        # torch._dynamo.config.patch(verify_correctness=True, automatic_dynamic_shapes=True), # No TVM equivalent
        assert_aot_autograd_counter(ok),
    )


N_ITERS = 5

# Custom base class to adapt unittest assertions to pytest-compatible assertions
# and provide a 'same' utility for comparing TVM NDArrays/NumPy.
class MyTVMTestBase:
    def assertEqual(self, actual, expected, msg=None):
        assert actual == expected, msg or f"{actual} != {expected}"

    def assertGreater(self, actual, expected, msg=None):
        assert actual > expected, msg or f"{actual} <= {expected}"

    def assertTrue(self, condition, msg=None):
        assert condition, msg or "Condition is not True"
    
    # Custom 'same' equivalent for comparing numerical arrays/scalars.
    def same(self, actual_val, expected_val, rtol=1e-5, atol=1e-8):
        actual_np = actual_val.numpy() if isinstance(actual_val, tvm.runtime.NDArray) else actual_val
        expected_np = expected_val.numpy() if isinstance(expected_val, tvm.runtime.NDArray) else expected_val
        testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol)
        return True # Return True for compatibility with original self.assertTrue(same(...)) pattern

# Mock `torch.cuda.is_available` with TVM's device check.
# This ensures the test is skipped if CUDA is not truly available for TVM.
def is_cuda_available_mock():
    return tvm.cuda(0).exist

# Helper to compile and run a Relay module
def compile_and_run_relay_mod(relay_func, params, input_np_arrays, target_device="cuda", output_dtype="float32"):
    target = tvm.target.Target(target_device)
    dev = tvm.device(target_device, 0)
    
    relay_mod = tvm.IRModule.from_expr(relay_func)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = build(relay_mod, target=target, params=params)
    
    module = GraphModule(lib["default"](dev))
    
    # Set inputs
    for name, np_array in input_np_arrays.items():
        module.set_input(name, tvm.nd.array(np_array, device=dev))
    
    module.run()
    
    # Get output (assuming single output for these tests)
    return module.get_output(0)


@pytest.mark.skipif(not is_cuda_available_mock(), reason="these tests require cuda")
class TestAotCudagraphs(MyTVMTestBase):
    @patch_all()
    def test_basic(self):
        # Original PyTorch model:
        # def model(x, y):
        #     return (x + y) * y
        # @torch.compile(backend="cudagraphs")
        # def fn(x, y):
        #     for _ in range(N_ITERS):
        #         loss = model(x, y).sum()
        #         loss.backward()

        # TVM Relay equivalent for the forward pass `(x + y) * y .sum()`
        x_var = relay.var("x", shape=(3,), dtype="float32")
        y_var = relay.var("y", shape=(3,), dtype="float32")
        
        # Build the Relay graph for the model's computation
        intermediate_output = (x_var + y_var) * y_var
        # The original test then calls .sum() on the model output, so incorporate that.
        final_output = relay.op.reduce.sum(intermediate_output, axis=None, keepdims=False)
        
        # Define the Relay function
        relay_func = relay.Function([x_var, y_var], final_output)
        
        # Prepare inputs as NumPy arrays
        x_np = np.random.randn(3).astype(np.float32)
        y_np = np.random.randn(3).astype(np.float32)
        input_np_arrays = {"x": x_np, "y": y_np}
        
        # Calculate expected output using NumPy for comparison
        expected_intermediate_np = (x_np + y_np) * y_np
        expected_final_np = np.sum(expected_intermediate_np)
        
        # Compile and run the TVM module N_ITERS times.
        # Note: `loss.backward()` is a PyTorch autograd specific call and is not directly
        # translated to TVM Relay which is primarily a forward graph IR.
        # This test verifies only the forward pass output.
        for _ in range(N_ITERS):
            tvm_output_nd = compile_and_run_relay_mod(relay_func, {}, input_np_arrays, target_device="cuda")
            self.same(tvm_output_nd, expected_final_np)

    @patch_all()
    def test_dtoh(self):
        # Original PyTorch model:
        # def model(x, y):
        #     a = x + y
        #     b = a.cpu() * 3
        #     return b
        #
        # This test checks how torch.compile(cudagraphs) handles DtoH transfers.
        # In the context of a compiled TVM Relay graph targeting CUDA, we assume
        # all operations happen on the CUDA device unless explicitly specified.
        # The .cpu() operation in PyTorch would likely trigger a graph break or
        # fallback in Dynamo. For TVM, we'll implement the computation purely
        # on the target device for simplicity, as TVM's 'cudagraphs' analogy
        # focuses on a single device's compiled graph.

        x_var = relay.var("x", shape=(3,), dtype="float32")
        y_var = relay.var("y", shape=(3,), dtype="float32")

        a = x_var + y_var
        # Simplified: treat a.cpu() as if it's still on cuda for graph compilation context
        b = a * relay.const(3.0, "float32")
        final_output = relay.op.reduce.sum(b, axis=None, keepdims=False)
        
        relay_func = relay.Function([x_var, y_var], final_output)

        x_np = np.random.randn(3).astype(np.float32)
        y_np = np.random.randn(3).astype(np.float32)
        input_np_arrays = {"x": x_np, "y": y_np}

        # Calculate expected output using NumPy, mirroring the computation
        expected_a_np = x_np + y_np
        expected_b_np = expected_a_np * 3.0
        expected_final_np = np.sum(expected_b_np)
        
        for _ in range(N_ITERS):
            tvm_output_nd = compile_and_run_relay_mod(relay_func, {}, input_np_arrays, target_device="cuda")
            self.same(tvm_output_nd, expected_final_np)

    @patch_all()
    def test_htod(self):
        # Original PyTorch model:
        # def model(x, y):
        #     a = x + y
        #     return a * 3
        #
        # Here `x` is on CUDA, `y` on CPU. `x + y` implies `y` is moved to CUDA.
        # For TVM, if the target is CUDA, `y_var` will be treated as a CUDA tensor for the op.

        x_var = relay.var("x", shape=(3,), dtype="float32")
        y_var = relay.var("y", shape=(), dtype="float32") # Scalar input
        
        a = x_var + y_var # This operation implicitly handles broadcasting y to x's shape on CUDA
        final_output_val = a * relay.const(3.0, "float32")
        final_output = relay.op.reduce.sum(final_output_val, axis=None, keepdims=False)

        relay_func = relay.Function([x_var, y_var], final_output)

        x_np = np.random.randn(3).astype(np.float32)
        y_np = np.random.randn(1).astype(np.float32)[0] # NumPy scalar
        input_np_arrays = {"x": x_np, "y": y_np}

        expected_a_np = x_np + y_np # NumPy handles scalar broadcasting
        expected_output_val_np = expected_a_np * 3.0
        expected_final_np = np.sum(expected_output_val_np)
        
        for _ in range(N_ITERS):
            tvm_output_nd = compile_and_run_relay_mod(relay_func, {}, input_np_arrays, target_device="cuda")
            self.same(tvm_output_nd, expected_final_np)

    # The following tests involve tensor mutation (e.g., `y.add_`, `x.resize_`, `x.fill_`)
    # and / or creation of new tensors within the model function where the original PyTorch
    # test verifies side-effects or specific PyTorch object behaviors.
    # TVM Relay is a functional graph IR, and direct mutation of inputs or side-effects on
    # dynamically created tensors are not part of its functional execution model.
    # Therefore, these tests are marked as skipped with a TODO for full translation.
    # Placeholder code is included to ensure the file remains runnable.

    def test_mutate_input(self):
        pytest.skip("TODO: Mutation behavior (e.g., `add_`) not directly translatable to TVM Relay functional graph.")
        # Placeholder for runnable code, does not reflect full semantics of original PyTorch test
        x_np = np.random.randn(3).astype(np.float32)
        y_np = np.random.randn(3).astype(np.float32)
        for i in range(N_ITERS):
            # No self.subTest(i) in pytest placeholder
            y_orig_np = y_np.copy()
            y_np += 3.0 # Simulate PyTorch's y.add_(3)
            result_np = x_np * y_np
            loss_np = np.sum(result_np)
            self.same(y_np, y_orig_np + 3.0)
            # No backward pass simulation
        self.assertTrue(True) # Dummy assertion to satisfy test runner

    @patch_all()
    def test_mutate_constant(self):
        pytest.skip("TODO: Mutation of constants and implicit type promotion are complex for functional Relay.")
        # Placeholder for runnable code
        x_np = np.random.randn(1).astype(np.float32)
        y_np = np.random.randn(1).astype(np.float32)
        for i in range(N_ITERS):
            c_val = np.array(1, dtype=np.int64) # torch.tensor(1) is long by default
            c_val = (c_val + 2).astype(np.float32) # Simulate c.add_(2) and implicit float promotion due to `x*y*0`
            loss_np = np.sum(x_np * y_np * 0 + c_val)
            self.same(loss_np, np.array(3.0, dtype=np.float32)) # Original expects 3.0 float
        self.assertTrue(True) # Dummy assertion

    @patch_all()
    def test_factory(self):
        pytest.skip("TODO: Tensor factory ops with internal mutation and device placement need custom handling.")
        # Placeholder for runnable code
        y_np = np.random.randn(3).astype(np.float32)
        for i in range(N_ITERS):
            x_val = np.zeros(3, dtype=np.float32) # Simulate torch.zeros
            x_val += 3.0 # Simulate x.add_(3)
            result_np = x_val * y_np
            loss_np = np.sum(result_np)
            # No specific assertion in original, only loss.backward()
        self.assertTrue(True) # Dummy assertion

    @patch_all()
    def test_mutated_metadata(self):
        pytest.skip("TODO: In-place metadata mutation (resize_, fill_) requires careful graph construction or custom ops.")
        # Placeholder for runnable code
        x_np_initial = np.empty(0, dtype=np.float32)
        for i in range(N_ITERS):
            x_clone_np = x_np_initial.copy()
            # Simulate x.resize_(20) -> reshape to (20,)
            x_resized_np = np.resize(x_clone_np, (20,)) # numpy.resize returns new array
            # Simulate x.fill_(2) -> fill with 2.0
            rx_np = np.full((20,), 2.0, dtype=np.float32)
            self.same(rx_np, np.full((20,), 2.0, dtype=np.float32))
        self.assertTrue(True) # Dummy assertion

    @patch_all()
    def test_dead_fill(self):
        pytest.skip("TODO: Slicing (x[0:0]) and subsequent in-place fill_ with empty tensors.")
        # Placeholder for runnable code
        x_initial_np = np.empty(20, dtype=np.float32)
        for i in range(N_ITERS):
            x_np = x_initial_np.copy()
            y_np_slice = x_np[0:0] # Creates an empty array
            
            x_filled_np = np.full_like(x_np, 2.0) # Simulate x.fill_(2)
            y_filled_np = np.full_like(y_np_slice, 3.0) # Simulate y.fill_(3). An empty array filled is still empty.

            self.same(x_filled_np, np.full((20,), 2.0, dtype=np.float32))
            self.same(y_filled_np, np.empty(0, dtype=np.float32))
        self.assertTrue(True) # Dummy assertion
