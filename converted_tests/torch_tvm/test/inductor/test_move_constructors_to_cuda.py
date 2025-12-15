import functools
import unittest
import numpy as np
import tvm
from tvm import relay
from tvm.relay import op
from tvm.ir import IRModule
import pytest # Using pytest.skip for conditional skips
from typing import Tuple, Union, List

# Constants for TVM target (can be overridden per test if needed)
DEFAULT_TVM_TARGET = "llvm" # Default to CPU
DEFAULT_TVM_DEVICE = tvm.cpu(0)

# Mock FileCheck as it's for TorchInductor's internal code.
# For TVM, checking generated code is different. Will skip for now with a TODO.
class FileCheck:
    def check(self, pattern):
        # TODO: TVM code inspection equivalent for TorchInductor backend patterns.
        # The original TorchInductor checks were for specific backend code generation
        # (e.g., "cpp_fused" for C++ code, "triton.jit" for Triton kernels).
        # These patterns are not directly applicable to TVM's generated code.
        return self
    def check_not(self, pattern):
        # TODO: TVM code inspection equivalent for TorchInductor backend patterns.
        return self
    def run(self, code_string):
        # For now, just pass. In a real scenario, this would involve parsing TVM's
        # generated low-level code (LLVM, CUDA, etc.) and looking for relevant patterns.
        pass

class ConvertedTestCase(unittest.TestCase):
    def _run_tvm_graph(self, relay_graph_builder, inputs_np, target_name):
        relay_input_vars = []
        tvm_input_ndarrays = []
        
        for i, inp_np in enumerate(inputs_np):
            if isinstance(inp_np, (int, float, bool, np.number)): # np.number for scalar numpy types
                dtype_str = str(np.array(inp_np).dtype)
                if isinstance(inp_np, bool) or isinstance(inp_np, np.bool_): dtype_str = "bool"
                
                # For scalars, we use a 0-dim tensor for Relay var and NDArray
                relay_input_vars.append(relay.var(f"p{i}", shape=(), dtype=dtype_str))
                tvm_input_ndarrays.append(tvm.nd.array(np.array(inp_np), device=DEFAULT_TVM_DEVICE))
            else: # Assume it's a numpy array or compatible
                relay_input_vars.append(relay.var(f"p{i}", shape=inp_np.shape, dtype=str(inp_np.dtype)))
                tvm_input_ndarrays.append(tvm.nd.array(inp_np, device=DEFAULT_TVM_DEVICE))
        
        relay_expr = relay_graph_builder(*relay_input_vars)
        
        # Ensure the output is a tuple if the builder returns multiple values,
        # so IRModule.from_expr can wrap it in a `relay.Tuple` automatically.
        if isinstance(relay_expr, (list, tuple)):
            relay_func = relay.Function(relay_input_vars, relay.Tuple(list(relay_expr)))
        else:
            relay_func = relay.Function(relay_input_vars, relay_expr)

        mod = tvm.IRModule.from_expr(relay_func)

        target = tvm.target.Target(target_name)
        device = tvm.cuda(0) if "cuda" in target_name else tvm.cpu(0)

        # Check device availability
        if "cuda" in target_name and not device.exist:
            pytest.skip(f"Skipping test, CUDA device not available.")
        if "llvm" in target_name and not device.exist:
            pytest.skip(f"Skipping test, CPU device not available.")

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target)

        rt_mod = tvm.runtime.GraphModule(lib["default"](device))
        for i, tvm_input in enumerate(tvm_input_ndarrays):
            rt_mod.set_input(f"p{i}", tvm_input)
        rt_mod.run()

        num_outputs = rt_mod.get_num_outputs()
        if num_outputs > 1:
            outputs_tvm_nd = [rt_mod.get_output(i).copyto(tvm.cpu(0)) for i in range(num_outputs)]
            outputs_np = [out.numpy() for out in outputs_tvm_nd]
        else:
            outputs_tvm_nd = rt_mod.get_output(0).copyto(tvm.cpu(0))
            outputs_np = outputs_tvm_nd.numpy()
        
        code = ["TODO: TVM generated code inspection is different from TorchInductor FileCheck"]
        return outputs_np, code

    def _check_fn(self, numpy_ref_func, relay_graph_builder, expect_cpu, *inputs_np_args):
        # 1. Get eager reference output (using numpy equivalent)
        out_eager = numpy_ref_func(*inputs_np_args)
        if not isinstance(out_eager, (tuple, list)):
            out_eager = (out_eager,)
        
        # Determine target based on global default, which can be set per test case
        # In PyTorch, device of first input often determines device.
        # For this conversion, we explicitly set DEFAULT_TVM_TARGET/DEVICE in individual tests.
        current_tvm_target = DEFAULT_TVM_TARGET
        
        # 2. Run compiled TVM graph
        out_compiled_raw, code = self._run_tvm_graph(relay_graph_builder, inputs_np_args, target_name=current_tvm_target)
        if not isinstance(out_compiled_raw, (tuple, list)):
            out_compiled_raw = (out_compiled_raw,)

        # 3. Assert equality
        self.assertEqual(len(out_eager), len(out_compiled_raw))
        for i, (eager_val, compiled_val) in enumerate(zip(out_eager, out_compiled_raw)):
            tvm.testing.assert_allclose(eager_val, compiled_val, rtol=1e-5, atol=1e-5)

        # 4. Handle code checks (placeholder)
        # The `expect_cpu` flag is specific to TorchInductor's CPU fallback mechanism
        # and its code generation patterns. This is not directly convertible to TVM.
        # We will just satisfy the `FileCheck` call structure as a no-op.
        FileCheck().run(code[0]) # Placeholder


# TVM-specific test utilities
def requires_multigpu(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if tvm.cuda().exist and tvm.runtime.device.num_devices("cuda") > 1:
            return f(self, *args, **kwargs)
        else:
            pytest.skip("Requires multiple CUDA devices for TVM.")
    return wrapper

# These are PyTorch-specific and not used in TVM context.
IS_LINUX = True
HAS_CUDA_AND_TRITON = True # Assume true for environment checks, but relevance is for Inductor
TEST_MULTIGPU = True # Assume true for multi-gpu tests to be enabled if available

class TestMoveConstructorsToCuda(ConvertedTestCase):
    def setUp(self):
        # Store original global defaults
        self._original_tvm_target = DEFAULT_TVM_TARGET
        self._original_tvm_device = DEFAULT_TVM_DEVICE

    def tearDown(self):
        # Restore global defaults
        global DEFAULT_TVM_TARGET, DEFAULT_TVM_DEVICE
        DEFAULT_TVM_TARGET = self._original_tvm_target
        DEFAULT_TVM_DEVICE = self._original_tvm_device

    def test_simple(self):
        def numpy_foo(x_np):
            return x_np[np.arange(x_np.shape[0])]

        def relay_foo(x_relay):
            # PyTorch `x[torch.arange(x.shape[0])]` maps to `relay.take(x, indices, axis=0)`
            return relay.take(x_relay, relay.arange(op.shape(x_relay)[0], dtype="int64"), axis=0)

        inp_np = np.random.rand(32, 77, 512).astype(np.float32)

        global DEFAULT_TVM_TARGET, DEFAULT_TVM_DEVICE
        DEFAULT_TVM_TARGET = "cuda"
        DEFAULT_TVM_DEVICE = tvm.cuda(0)

        self._check_fn(numpy_foo, relay_foo, False, inp_np)

    def test_output_failure(self):
        def numpy_foo(x_np):
            tmp1 = np.arange(x_np.shape[0])
            return tmp1, x_np[tmp1]

        def relay_foo(x_relay):
            tmp1_relay = relay.arange(op.shape(x_relay)[0], dtype="int64")
            out1 = tmp1_relay
            out2 = relay.take(x_relay, tmp1_relay, axis=0)
            return out1, out2

        inp_np = np.random.rand(32, 77, 512).astype(np.float32)

        global DEFAULT_TVM_TARGET, DEFAULT_TVM_DEVICE
        DEFAULT_TVM_TARGET = "cuda"
        DEFAULT_TVM_DEVICE = tvm.cuda(0)

        self._check_fn(numpy_foo, relay_foo, True, inp_np)

    def test_non_convertable_op_failure(self):
        def numpy_foo(x_np):
            y_np = np.arange(x_np.shape[0])
            return x_np + y_np, np.ones([4], dtype=np.float32)

        def relay_foo(x_relay):
            y_relay = relay.arange(op.shape(x_relay)[0], dtype=str(x_relay.dtype))
            
            # Ensure dtypes match for element-wise ops, or rely on TVM's implicit casting
            y_relay_casted = relay.cast(y_relay, x_relay.dtype)
            
            out1 = relay.add(x_relay, y_relay_casted)
            out2 = relay.ones(shape=(4,), dtype="float32") # torch.ones([4], device="cuda")

            return out1, out2

        # PyTorch `inp = torch.rand([100])` defaults to CPU and float32.
        inp_np = np.random.rand(100).astype(np.float32)
        
        global DEFAULT_TVM_TARGET, DEFAULT_TVM_DEVICE
        DEFAULT_TVM_TARGET = "llvm" # Ensure CPU target
        DEFAULT_TVM_DEVICE = tvm.cpu(0)
        
        self._check_fn(numpy_foo, relay_foo, True, inp_np)

    def test_multiple_constructors(self):
        def numpy_foo(x_np):
            tmp1 = np.arange(x_np.shape[0])
            o1_np = x_np[tmp1] 
            
            tmp2_np_indices = np.arange(x_np.shape[1]) # [0, 1, ..., 199]
            # PyTorch `x[tmp2]` where `x` is (R, C) and `tmp2` is (1, C_idx) results in (1, C_idx, C)
            # In numpy, this is equivalent to `x_np[tmp2_np_indices]` (which is `x_np` if `tmp2_np_indices` is `0..R-1`)
            # and then `np.expand_dims` to add the leading dimension from `tmp2`'s original shape `(1, ...)`
            o2_indexed_np = x_np[tmp2_np_indices] # shape (200, 200)
            o2_np = np.expand_dims(o2_indexed_np, axis=0) # shape (1, 200, 200)
            
            # Ensure broadcasting for addition works
            # o1_np is (200, 200), o2_np is (1, 200, 200). 
            # NumPy will broadcast o1_np to (1, 200, 200) for addition.
            return o1_np, o2_np, o1_np + o2_np
            
        def relay_foo(x_relay):
            shape0 = op.shape(x_relay)[0]
            shape1 = op.shape(x_relay)[1]

            tmp1_relay = relay.arange(shape0, dtype="int64") # shape (S0,)
            o1_relay = relay.take(x_relay, tmp1_relay, axis=0) # shape (S0, S1)

            tmp2_relay_indices_1d = relay.arange(shape1, dtype="int64") # shape (S1,)
            o2_indexed_relay = relay.take(x_relay, tmp2_relay_indices_1d, axis=0) # shape (S1, S1)
            
            # Add back the leading dimension `1` to match PyTorch `x[tmp2]` semantics.
            o2_relay = relay.expand_dims(o2_indexed_relay, axis=0) # shape (1, S1, S1)

            # Ensure dtypes are float32 for addition based on original `torch.rand` input
            o1_relay_f = relay.cast(o1_relay, "float32")
            o2_relay_f = relay.cast(o2_relay, "float32")
            
            return o1_relay_f, o2_relay_f, relay.add(o1_relay_f, o2_relay_f)

        inp_np = np.random.rand(200, 200).astype(np.float32)

        global DEFAULT_TVM_TARGET, DEFAULT_TVM_DEVICE
        DEFAULT_TVM_TARGET = "cuda"
        DEFAULT_TVM_DEVICE = tvm.cuda(0)

        self._check_fn(numpy_foo, relay_foo, True, inp_np)

    def test_sets_equiv(self):
        def numpy_foo(x_np):
            c1_np = np.ones([4], dtype=np.int64)
            c2_np = np.arange(-1, 3) # [-1, 0, 1, 2]
            
            # c1_np + c2_np = [0, 1, 2, 3]
            out1 = x_np[c1_np + c2_np] # Indexing float32 tensor with int64 indices
            
            # c2_np - 4 * 2 = [-1, 0, 1, 2] - 8 = [-9, -8, -7, -6]
            out2 = c2_np - 8
            return out1, out2

        def relay_foo(x_relay):
            c1_relay = relay.ones(shape=(4,), dtype="int64")
            c2_relay = relay.arange(-1, 3, 1, dtype="int64") # arange(start, stop, step)
            
            indices_relay = relay.add(c1_relay, c2_relay) 
            out1_relay = relay.take(x_relay, indices_relay, axis=0)
            
            const_8_relay = relay.const(8, dtype="int64")
            out2_relay = relay.subtract(c2_relay, const_8_relay)
            
            return out1_relay, out2_relay

        inp_np = np.random.rand(4).astype(np.float32)

        global DEFAULT_TVM_TARGET, DEFAULT_TVM_DEVICE
        DEFAULT_TVM_TARGET = "cuda"
        DEFAULT_TVM_DEVICE = tvm.cuda(0)

        # The PyTorch test calls `torch.compile` twice. We emulate this by calling `_check_fn` twice.
        self._check_fn(numpy_foo, relay_foo, False, inp_np)
        self._check_fn(numpy_foo, relay_foo, False, inp_np)


    @requires_multigpu
    # @unittest.skip("https://github.com/pytorch/pytorch/issues/139520")
    # The original PyTorch issue indicates multi-GPU for constructors was NYI.
    # For TVM, the relay graph is compiled for a single target device.
    # We will run this on CUDA device 0. The `expect_cpu=True` from PyTorch
    # implies a fallback, which is specific to Inductor's internal logic.
    # For TVM, if target is CUDA, it will run on CUDA. So the `expect_cpu`
    # check cannot be directly mapped.
    def test_multi_gpu(self):
        def numpy_foo(x_np):
            # Device information is not in numpy.
            # This is effectively creating numpy arrays from data.
            return x_np[np.arange(x_np.shape[0])], np.ones([4], dtype=np.float32), np.ones([4], dtype=np.float32)

        def relay_foo(x_relay):
            out1 = relay.take(x_relay, relay.arange(op.shape(x_relay)[0], dtype="int64"), axis=0)
            
            # Relay ops like `ones` do not specify device. Device is for the whole compiled graph.
            # So `cuda:0` and `cuda:1` distinction won't happen inside the graph for these ops.
            # If the graph runs on CUDA, both `ones` results will be on that CUDA device.
            out2 = relay.ones(shape=(4,), dtype="float32")
            out3 = relay.ones(shape=(4,), dtype="float32")
            return out1, out2, out3

        inp_np = np.random.rand(100).astype(np.float32)

        global DEFAULT_TVM_TARGET, DEFAULT_TVM_DEVICE
        DEFAULT_TVM_TARGET = "cuda"
        DEFAULT_TVM_DEVICE = tvm.cuda(0) # This will run the whole graph on cuda:0

        # The original `expect_cpu=True` from PyTorch indicates a CPU fallback
        # for `torch.ones([4], device="cuda:0")` and `torch.ones([4], device="cuda:1")`
        # when running with Inductor multi-GPU NYI.
        # TVM will simply compile and run these on a single CUDA device if available.
        # Thus, `expect_cpu=True` doesn't directly translate as a check for TVM.
        # We pass it as `True` to match the original signature, but its meaning is lost.
        self._check_fn(numpy_foo, relay_foo, True, inp_np)

    def test_no_gpu(self):
        def numpy_foo(x_np):
            return x_np[np.arange(x_np.shape[0])]

        def relay_foo(x_relay):
            return relay.take(x_relay, relay.arange(op.shape(x_relay)[0], dtype="int64"), axis=0)

        # PyTorch `inp = torch.rand([100])` defaults to CPU and float32.
        inp_np = np.random.rand(100).astype(np.float32)

        global DEFAULT_TVM_TARGET, DEFAULT_TVM_DEVICE
        DEFAULT_TVM_TARGET = "llvm"
        DEFAULT_TVM_DEVICE = tvm.cpu(0)

        # Original expects `expect_cpu=True` because input is on CPU and no other device is specified.
        # For TVM, we explicitly target CPU, so this is functionally consistent with original intent.
        self._check_fn(numpy_foo, relay_foo, True, inp_np)


if __name__ == "__main__":
    unittest.main()
