import contextlib
import sys
import pytest
import numpy as np
import tvm
from tvm import relay
from tvm.relay import op as relay_op
from tvm.relay.op import nn as relay_nn
from tvm.relay.op import tensor as relay_tensor
from tvm.relay.op import reduce as relay_reduce
from tvm.relay.op import transform as relay_transform
from tvm.relay.op import algorithm as relay_algorithm
from tvm import runtime
from tvm import testing


# Mock PyTorch-specific components
class MockDynamoConfig:
    @contextlib.contextmanager
    def patch(self, *args, **kwargs):
        yield
class MockDynamo:
    config = MockDynamoConfig()
    def mark_dynamic(self, *args, **kwargs):
        pass # No direct TVM Relay counterpart at this level unless explicit DynTensorType is used.

class MockInductorConfig:
    @contextlib.contextmanager
    def patch(self, *args, **kwargs):
        yield
class MockInductorMetrics:
    def reset(self):
        pass
    @property
    def generated_kernel_count(self):
        return 0 # Always return 0 as TVM doesn't have this metric

# Mock the entire `torch` module that `_inductor` would refer to
class MockTorch:
    _dynamo = MockDynamo()
    _inductor = MockInductor()
    @contextlib.contextmanager
    def no_grad(self):
        yield

# Helper to convert a Python dtype object (like np.float32) to TVM string
def _numpy_to_tvm_dtype_str(dtype_obj):
    if isinstance(dtype_obj, str):
        return dtype_obj
    if dtype_obj == np.float32: return "float32"
    if dtype_obj == np.float64: return "float64"
    if dtype_obj == np.int64: return "int64"
    if dtype_obj == np.bool_: return "bool"
    raise ValueError(f"Unsupported numpy dtype: {dtype_obj}")

def _get_tvm_device_from_target_str(target_str):
    if "cuda" in target_str:
        return tvm.cuda(0)
    elif "llvm" in target_str or "cpu" in target_str: # Default for CPU is llvm
        return tvm.cpu(0)
    else:
        raise ValueError(f"Unsupported target: {target_str}")

# This helper function replaces `check_model_cuda` and `check_model`.
# It takes a function that *builds* the Relay graph and initial NumPy inputs.
def _check_tvm_model(relay_graph_builder_func, numpy_inputs, target_str="cuda"):
    relay_input_vars = []
    tvm_inputs_nd = []
    device = _get_tvm_device_from_target_str(target_str)

    # 1. Prepare Relay input variables and TVM NDArrays from NumPy inputs
    for i, np_inp in enumerate(numpy_inputs):
        relay_input_vars.append(
            relay.var(f"p{i}", shape=np_inp.shape, dtype=_numpy_to_tvm_dtype_str(np_inp.dtype))
        )
        tvm_inputs_nd.append(tvm.nd.array(np_inp, device=device))

    # 2. Build the Relay graph using the provided builder function
    relay_body = relay_graph_builder_func(*relay_input_vars)

    # Ensure output is a tuple for consistency if it's a single expression
    if not isinstance(relay_body, (tuple, list, tvm.ir.Array)):
        relay_body = (relay_body,)
    relay_body = relay.Tuple(list(relay_body)) # Ensure it's a relay.Tuple

    # 3. Create Relay Function and IRModule
    relay_func = relay.Function(relay_input_vars, relay_body)
    mod = tvm.IRModule.from_expr(relay_func)

    # 4. Compile the Relay IRModule
    target = tvm.target.Target(target_str)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    # 5. Execute the compiled module
    vm = tvm.runtime.vm.VirtualMachine(lib, device=device)
    tvm_output_raw = vm.invoke("main", *tvm_inputs_nd)

    # 6. Convert TVM output to NumPy
    if isinstance(tvm_output_raw, tvm.runtime.ndarray.NDArray):
        return (tvm_output_raw.numpy(),) # Wrap single output in tuple
    elif isinstance(tvm_output_raw, tvm.runtime.container.ADT):
        return tuple(o.numpy() for o in tvm_output_raw)
    else:
        # Scalar result, wrap in tuple for consistency
        return (np.array(tvm_output_raw),)


# Helper for @requires_cuda_and_triton
_HAS_CUDA = tvm.runtime.ndarray.gpu_enabled()
requires_cuda_and_triton = pytest.mark.skipif(not _HAS_CUDA, reason="Requires CUDA")

# Replacing TestCase from torch.testing._internal.common_utils
class TestCase:
    def assertEqual(self, actual, expected, rtol=1e-5, atol=1e-8):
        # Handle cases where actual or expected might be single non-tuple items
        if not isinstance(actual, (tuple, list)):
            actual = (actual,)
        if not isinstance(expected, (tuple, list)):
            expected = (expected,)
        
        assert len(actual) == len(expected), f"Length mismatch: {len(actual)} vs {len(expected)}"
        for a, e in zip(actual, expected):
            testing.assert_allclose(actual=a, desired=e, rtol=rtol, atol=atol)
    
    def assertTrue(self, expr):
        assert expr
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass


class ComboKernelTests(TestCase):
    def setUp(self):
        super().setUp()
        MockInductorMetrics().reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            MockInductorConfig().patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        MockInductorMetrics().reset()
        super().tearDown()

    @requires_cuda_and_triton
    def test_activation_functions(self):
        def test_activations_eager_np(a_np, b_np, c_np):
            a1 = np.maximum(0, a_np)
            b1 = 1 / (1 + np.exp(-b_np))
            c1 = np.tanh(c_np)
            return a1, b1, c1

        def test_activations_relay(a_relay, b_relay, c_relay):
            a1 = relay_nn.relu(a_relay)
            b1 = relay_tensor.sigmoid(b_relay)
            c1 = relay_tensor.tanh(c_relay)
            return a1, b1, c1

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
        ]

        out_eager_np = test_activations_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_activations_relay, inps_np, target_str="cuda")

        self.assertEqual(out_compiled_np, out_eager_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1) # REMOVED

    @requires_cuda_and_triton
    def test_reduce_functions(self):
        def test_reduce_eager_np(a_np, b_np, c_np, d_np):
            a1 = np.sum(a_np, axis=0)
            b1_values = np.max(b_np, axis=0) # PyTorch max(input, dim) returns (values, indices), but the test only uses values
            c1_values = np.min(c_np, axis=0) # PyTorch min(input, dim) returns (values, indices), but the test only uses values
            d1 = np.tanh(d_np)
            return a1, b1_values, c1_values, d1

        def test_reduce_relay(a_relay, b_relay, c_relay, d_relay):
            a1 = relay_reduce.sum(a_relay, axis=0, keepdims=False)
            b1 = relay_reduce.max(b_relay, axis=0, keepdims=False)
            c1 = relay_reduce.min(c_relay, axis=0, keepdims=False)
            d1 = relay_tensor.tanh(d_relay)
            return a1, b1, c1, d1

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(30, 8).astype(np.float32),
        ]

        out_eager_np = test_reduce_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_reduce_relay, inps_np, target_str="cuda")

        self.assertEqual(out_compiled_np, out_eager_np)
        # self.assertTrue(torch._inductor.metrics.generated_kernel_count <= 2) # REMOVED

    @requires_cuda_and_triton
    def test_mutated_args(self):
        def test_mutated_eager_np(a_np, b_np, c_np, d_np):
            # In-place ops are translated to new assignments in functional NumPy/Relay
            a_np = a_np + 1
            b_np = 1 / (1 + np.exp(-b_np))
            c_np = c_np + 5
            d_np = np.tanh(d_np)
            return a_np, b_np, c_np, d_np

        def test_mutated_relay(a_relay, b_relay, c_relay, d_relay):
            a_new = relay_tensor.add(a_relay, relay.const(1, dtype=_numpy_to_tvm_dtype_str(a_relay.checked_type.dtype)))
            b_new = relay_tensor.sigmoid(b_relay)
            c_new = relay_tensor.add(c_relay, relay.const(5, dtype=_numpy_to_tvm_dtype_str(c_relay.checked_type.dtype)))
            d_new = relay_tensor.tanh(d_relay)
            return a_new, b_new, c_new, d_new

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(30, 8).astype(np.float32),
        ]

        out_eager_np = test_mutated_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_mutated_relay, inps_np, target_str="cuda")

        self.assertEqual(out_compiled_np, out_eager_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1) # REMOVED

    @requires_cuda_and_triton
    def test_reduce_split(self):
        def fn_eager_np(a_np, b_np):
            # torch.linalg.vector_norm(a) -> default L2 norm over all dimensions
            a1 = np.sqrt(np.sum(a_np * a_np))
            b1 = np.sum(b_np, axis=0)
            return a1, b1

        def fn_relay(a_relay, b_relay):
            a1 = relay_tensor.sqrt(relay_reduce.sum(relay_tensor.multiply(a_relay, a_relay), axis=None, keepdims=False))
            b1 = relay_reduce.sum(b_relay, axis=0, keepdims=False)
            return a1, b1

        inps_np = [
            np.random.rand(2048, 512).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
        ]
        out_eager_np = fn_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(fn_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)

    @requires_cuda_and_triton
    def test_2d_blocking_partitioning(self):
        def fn_eager_np(a0_np, a1_np, a2_np, b0_np, b1_np, b2_np):
            c0 = a0_np + b0_np
            c1 = a1_np + b1_np
            c2 = a2_np + b2_np
            return c0, c1, c2

        def fn_relay(a0_relay, a1_relay, a2_relay, b0_relay, b1_relay, b2_relay):
            c0 = relay_tensor.add(a0_relay, b0_relay)
            c1 = relay_tensor.add(a1_relay, b1_relay)
            c2 = relay_tensor.add(a2_relay, b2_relay)
            return c0, c1, c2

        inps_np = (
            np.random.rand(30, 20).astype(np.float32),
            np.random.rand(40, 30).astype(np.float32),
            np.random.rand(36, 40).astype(np.float32),
            np.random.rand(30, 20).astype(np.float32),
            np.random.rand(30, 40).astype(np.float32).T, # .t() for transpose
            np.random.rand(40, 36).astype(np.float32).T, # .t() for transpose
        )
        out_eager_np = fn_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(fn_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2) # REMOVED


class ComboKernelBenchmarkTests(TestCase):
    def setUp(self):
        super().setUp()
        MockInductorMetrics().reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            MockInductorConfig().patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        MockInductorMetrics().reset()
        super().tearDown()

    @requires_cuda_and_triton
    def test_activation_benchmark(self):
        def test_activations_eager_np(a_np, b_np, c_np):
            a1 = np.maximum(0, a_np)
            b1 = 1 / (1 + np.exp(-b_np))
            c1 = np.tanh(c_np)
            return a1, b1, c1

        def test_activations_relay(a_relay, b_relay, c_relay):
            a1 = relay_nn.relu(a_relay)
            b1 = relay_tensor.sigmoid(b_relay)
            c1 = relay_tensor.tanh(c_relay)
            return a1, b1, c1

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
        ]

        out_eager_np = test_activations_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_activations_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5) # REMOVED

    @requires_cuda_and_triton
    def test_reduce_benchmark(self):
        def test_reduce_eager_np(a_np, b_np, c_np, d_np):
            a1 = np.sum(a_np, axis=0)
            b1_values = np.max(b_np, axis=0)
            c1_values = np.min(c_np, axis=0)
            d1 = np.tanh(d_np)
            return a1, b1_values, c1_values, d1

        def test_reduce_relay(a_relay, b_relay, c_relay, d_relay):
            a1 = relay_reduce.sum(a_relay, axis=0, keepdims=False)
            b1 = relay_reduce.max(b_relay, axis=0, keepdims=False)
            c1 = relay_reduce.min(c_relay, axis=0, keepdims=False)
            d1 = relay_tensor.tanh(d_relay)
            return a1, b1, c1, d1

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(30, 8).astype(np.float32),
        ]

        out_eager_np = test_reduce_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_reduce_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10) # REMOVED

    @requires_cuda_and_triton
    def test_mutated_benchmark(self):
        def test_mutated_eager_np(a_np, b_np, c_np, d_np):
            a_np = a_np + 1
            b_np = 1 / (1 + np.exp(-b_np))
            c_np = c_np + 5
            d_np = np.tanh(d_np)
            return a_np, b_np, c_np, d_np

        def test_mutated_relay(a_relay, b_relay, c_relay, d_relay):
            a_new = relay_tensor.add(a_relay, relay.const(1, dtype=_numpy_to_tvm_dtype_str(a_relay.checked_type.dtype)))
            b_new = relay_tensor.sigmoid(b_relay)
            c_new = relay_tensor.add(c_relay, relay.const(5, dtype=_numpy_to_tvm_dtype_str(c_relay.checked_type.dtype)))
            d_new = relay_tensor.tanh(d_relay)
            return a_new, b_new, c_new, d_new

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(30, 8).astype(np.float32),
        ]

        out_eager_np = test_mutated_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_mutated_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertTrue(torch._inductor.metrics.generated_kernel_count in [6, 9]) # REMOVED

    @requires_cuda_and_triton
    def test_round_robin_dispatch(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated_eager_np(a_np, b_np, c_np, d_np):
            a_np = a_np + 1
            b_np = 1 / (1 + np.exp(-b_np))
            c_np = c_np + 5
            d_np = np.tanh(d_np)
            return a_np, b_np, c_np, d_np

        def test_mutated_relay(a_relay, b_relay, c_relay, d_relay):
            a_new = relay_tensor.add(a_relay, relay.const(1, dtype=_numpy_to_tvm_dtype_str(a_relay.checked_type.dtype)))
            b_new = relay_tensor.sigmoid(b_relay)
            c_new = relay_tensor.add(c_relay, relay.const(5, dtype=_numpy_to_tvm_dtype_str(c_relay.checked_type.dtype)))
            d_new = relay_tensor.tanh(d_relay)
            return a_new, b_new, c_new, d_new

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 5).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(5, 18).astype(np.float32),
        ]

        out_eager_np = test_mutated_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_mutated_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6) # REMOVED

    @requires_cuda_and_triton
    def test_2d_blocking_benchmark(self):
        def fn_eager_np(a0_np, a1_np, a2_np, b0_np, b1_np, b2_np):
            c0 = a0_np + b0_np
            c1 = a1_np + b1_np
            c2 = a2_np + b2_np
            return c0, c1, c2

        def fn_relay(a0_relay, a1_relay, a2_relay, b0_relay, b1_relay, b2_relay):
            c0 = relay_tensor.add(a0_relay, b0_relay)
            c1 = relay_tensor.add(a1_relay, b1_relay)
            c2 = relay_tensor.add(a2_relay, b2_relay)
            return c0, c1, c2

        inps_np = (
            np.random.rand(30, 20).astype(np.float32),
            np.random.rand(40, 30).astype(np.float32),
            np.random.rand(36, 40).astype(np.float32),
            np.random.rand(30, 20).astype(np.float32),
            np.random.rand(30, 40).astype(np.float32).T,
            np.random.rand(40, 36).astype(np.float32).T,
        )
        out_eager_np = fn_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(fn_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertTrue(7 <= torch._inductor.metrics.generated_kernel_count <= 8) # REMOVED

    @requires_cuda_and_triton
    def test_persistent_reduction_no_x_dim(self):
        def fn_eager_np(x_np, y_np):
            return np.sum(x_np, axis=1), np.sum(y_np, axis=1)

        def fn_relay(x_relay, y_relay):
            return relay_reduce.sum(x_relay, axis=1), relay_reduce.sum(y_relay, axis=1)

        inps_np = (
            np.random.rand(16, 256).astype(np.float32),
            np.random.rand(32, 256).astype(np.float32),
        )
        MockDynamo().mark_dynamic(inps_np[0], 0, min=1, max=256)
        MockDynamo().mark_dynamic(inps_np[1], 0, min=1, max=256)
        out_eager_np = fn_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(fn_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4) # REMOVED


class ComboKernelDynamicShapesTests(TestCase):
    def setUp(self):
        super().setUp()
        MockInductorMetrics().reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            MockInductorConfig().patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                }
            )
        )
        self._test_stack.enter_context(
            MockDynamo().config.patch(
                {
                    "automatic_dynamic_shapes": False,
                    "assume_static_by_default": False,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        MockInductorMetrics().reset()
        super().tearDown()

    @requires_cuda_and_triton
    def test_dynamic_shapes_activations(self):
        def test_activations_eager_np(a_np, b_np, c_np):
            a1 = np.maximum(0, a_np)
            b1 = 1 / (1 + np.exp(-b_np))
            c1 = np.tanh(c_np)
            return a1, b1, c1

        def test_activations_relay(a_relay, b_relay, c_relay):
            a1 = relay_nn.relu(a_relay)
            b1 = relay_tensor.sigmoid(b_relay)
            c1 = relay_tensor.tanh(c_relay)
            return a1, b1, c1

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
        ]

        out_eager_np = test_activations_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_activations_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5) # REMOVED

    @requires_cuda_and_triton
    def test_dynamic_shapes_2d_blocking(self):
        def fn_eager_np(a0_np, a1_np, a2_np, b0_np, b1_np, b2_np):
            c0 = a0_np + b0_np
            c1 = a1_np + b1_np
            c2 = a2_np + b2_np
            return c0, c1, c2

        def fn_relay(a0_relay, a1_relay, a2_relay, b0_relay, b1_relay, b2_relay):
            c0 = relay_tensor.add(a0_relay, b0_relay)
            c1 = relay_tensor.add(a1_relay, b1_relay)
            c2 = relay_tensor.add(a2_relay, b2_relay)
            return c0, c1, c2

        inps_np = (
            np.random.rand(30, 20).astype(np.float32),
            np.random.rand(40, 30).astype(np.float32),
            np.random.rand(36, 40).astype(np.float32),
            np.random.rand(30, 20).astype(np.float32),
            np.random.rand(30, 40).astype(np.float32).T,
            np.random.rand(40, 36).astype(np.float32).T,
        )

        out_eager_np = fn_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(fn_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertTrue(7 <= torch._inductor.metrics.generated_kernel_count <= 8) # REMOVED

    @requires_cuda_and_triton
    def test_dynamic_shapes_reduce(self):
        def test_reduce_eager_np(a_np, b_np, c_np, d_np):
            a1 = np.sum(a_np, axis=0)
            b1_values = np.max(b_np, axis=0)
            c1_values = np.min(c_np, axis=0)
            d1 = np.tanh(d_np)
            return a1, b1_values, c1_values, d1

        def test_reduce_relay(a_relay, b_relay, c_relay, d_relay):
            a1 = relay_reduce.sum(a_relay, axis=0, keepdims=False)
            b1 = relay_reduce.max(b_relay, axis=0, keepdims=False)
            c1 = relay_reduce.min(c_relay, axis=0, keepdims=False)
            d1 = relay_tensor.tanh(d_relay)
            return a1, b1, c1, d1

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(30, 8).astype(np.float32),
        ]

        out_eager_np = test_reduce_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_reduce_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10) # REMOVED

    @requires_cuda_and_triton
    def test_dynamic_shapes_mutated(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated_eager_np(a_np, b_np, c_np, d_np):
            a_np = a_np + 1
            b_np = 1 / (1 + np.exp(-b_np))
            c_np = c_np + 5
            d_np = np.tanh(d_np)
            return a_np, b_np, c_np, d_np

        def test_mutated_relay(a_relay, b_relay, c_relay, d_relay):
            a_new = relay_tensor.add(a_relay, relay.const(1, dtype=_numpy_to_tvm_dtype_str(a_relay.checked_type.dtype)))
            b_new = relay_tensor.sigmoid(b_relay)
            c_new = relay_tensor.add(c_relay, relay.const(5, dtype=_numpy_to_tvm_dtype_str(c_relay.checked_type.dtype)))
            d_new = relay_tensor.tanh(d_relay)
            return a_new, b_new, c_new, d_new

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 5).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(5, 18).astype(np.float32),
        ]

        out_eager_np = test_mutated_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_mutated_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6) # REMOVED

    @requires_cuda_and_triton
    @MockInductorConfig().patch("combo_kernels_autotune", 0)
    def test_dynamic_shapes_activations_no_autotune(self):
        def test_activations_eager_np(a_np, b_np, c_np):
            a1 = np.maximum(0, a_np)
            b1 = 1 / (1 + np.exp(-b_np))
            c1 = np.tanh(c_np)
            return a1, b1, c1

        def test_activations_relay(a_relay, b_relay, c_relay):
            a1 = relay_nn.relu(a_relay)
            b1 = relay_tensor.sigmoid(b_relay)
            c1 = relay_tensor.tanh(c_relay)
            return a1, b1, c1

        inps_np = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
        ]

        out_eager_np = test_activations_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(test_activations_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5) # REMOVED

    @requires_cuda_and_triton
    @MockDynamo().config.patch("automatic_dynamic_shapes", True)
    @MockDynamo().config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim(self):
        def fn_eager_np(x_np, y_np):
            return np.sum(x_np, axis=1), np.sum(y_np, axis=1)

        def fn_relay(x_relay, y_relay):
            return relay_reduce.sum(x_relay, axis=1), relay_reduce.sum(y_relay, axis=1)

        inps_np = (
            np.random.rand(16, 256).astype(np.float32),
            np.random.rand(32, 256).astype(np.float32),
        )
        MockDynamo().mark_dynamic(inps_np[0], 0, min=1, max=256)
        MockDynamo().mark_dynamic(inps_np[1], 0, min=1, max=256)
        out_eager_np = fn_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(fn_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4) # REMOVED

    @requires_cuda_and_triton
    @MockDynamo().config.patch("automatic_dynamic_shapes", True)
    @MockDynamo().config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim_2(self):
        def fn_eager_np(x_np, y_np):
            return np.sum(x_np, axis=2), np.sum(y_np, axis=2)

        def fn_relay(x_relay, y_relay):
            return relay_reduce.sum(x_relay, axis=2), relay_reduce.sum(y_relay, axis=2)

        inps_np = (
            np.random.rand(8, 16, 256).astype(np.float32),
            np.random.rand(8, 32, 256).astype(np.float32),
        )
        MockDynamo().mark_dynamic(inps_np[0], (0, 1), min=1, max=256)
        MockDynamo().mark_dynamic(inps_np[1], (0, 1), min=1, max=256)
        out_eager_np = fn_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(fn_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)
        # self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4) # REMOVED

    @requires_cuda_and_triton
    @MockDynamo().config.patch("automatic_dynamic_shapes", True)
    @MockDynamo().config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_2d_blocking_round_robin(self):
        def fn_eager_np(a0_np, a1_np, a2_np, b0_np, b1_np, b2_np):
            c0 = a0_np + b0_np
            c1 = a1_np + b1_np
            c2 = a2_np + b2_np
            return c0, c1, c2

        def fn_relay(a0_relay, a1_relay, a2_relay, b0_relay, b1_relay, b2_relay):
            c0 = relay_tensor.add(a0_relay, b0_relay)
            c1 = relay_tensor.add(a1_relay, b1_relay)
            c2 = relay_tensor.add(a2_relay, b2_relay)
            return c0, c1, c2

        inps_np_1 = (
            np.random.rand(20, 30).astype(np.float32),
            np.random.rand(30, 30).astype(np.float32),
            np.random.rand(40, 32).astype(np.float32),
            np.random.rand(30, 20).astype(np.float32).T,
            np.random.rand(30, 30).astype(np.float32).T,
            np.random.rand(32, 40).astype(np.float32).T,
        )

        out_eager_np_1 = fn_eager_np(*inps_np_1)
        compiled_np_1 = _check_tvm_model(fn_relay, inps_np_1, target_str="cuda")
        self.assertEqual(out_eager_np_1, compiled_np_1)
        # self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6) # REMOVED
        MockInductorMetrics().reset() # Reset metrics for the next run (mocked)

        inps_np_2 = (
            np.random.rand(24, 30).astype(np.float32),
            np.random.rand(32, 30).astype(np.float32),
            np.random.rand(48, 32).astype(np.float32),
            np.random.rand(30, 24).astype(np.float32).T,
            np.random.rand(30, 32).astype(np.float32).T,
            np.random.rand(32, 48).astype(np.float32).T,
        )
        out_eager_np_2 = fn_eager_np(*inps_np_2)
        # The original code re-uses 'compiled', implying the same graph, but with dynamic shapes.
        # For our TVM model, _check_tvm_model re-compiles each time, but the graph structure is the same.
        compiled_np_2 = _check_tvm_model(fn_relay, inps_np_2, target_str="cuda")
        self.assertEqual(out_eager_np_2, compiled_np_2)
        # self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6) # REMOVED

    @requires_cuda_and_triton
    @MockDynamo().config.patch("automatic_dynamic_shapes", True)
    @MockDynamo().config.patch("assume_static_by_default", True)
    @MockInductorConfig().patch("triton.autotune_at_compile_time", True)
    def test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda(self):
        def fn_eager_np(x_np, y_np, z_np):
            return np.sum(x_np, axis=1), np.mean(y_np, axis=1), np.max(z_np, axis=1)

        def fn_relay(x_relay, y_relay, z_relay):
            return (
                relay_reduce.sum(x_relay, axis=1),
                relay_reduce.mean(y_relay, axis=1),
                relay_reduce.max(z_relay, axis=1),
            )

        inps_np = (
            np.random.rand(16, 128).astype(np.float32),
            np.random.rand(32, 128).astype(np.float32),
            np.random.rand(32, 256).astype(np.float32),
        )
        MockDynamo().mark_dynamic(inps_np[0], 0, min=1, max=256)
        MockDynamo().mark_dynamic(inps_np[1], 0, min=1, max=256)
        MockDynamo().mark_dynamic(inps_np[2], 0, min=1, max=256)
        out_eager_np = fn_eager_np(*inps_np)
        out_compiled_np = _check_tvm_model(fn_relay, inps_np, target_str="cuda")

        self.assertEqual(out_eager_np, out_compiled_np)


if __name__ == "__main__":
    _HAS_CPU = tvm.runtime.ndarray.cpu_enabled()
    if _HAS_CPU or _HAS_CUDA:
        pytest.main([__file__])
    else:
        sys.exit(0)
