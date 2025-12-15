import contextlib
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.contrib import graph_executor
import pytest # Often used with TVM tests

# Global TVM device setup
TVM_GPU_TARGET = "cuda"
TVM_CPU_TARGET = "llvm"
TVM_HAS_GPU = tvm.cuda().exist
TVM_DEVICE_TYPE = TVM_GPU_TARGET if TVM_HAS_GPU else TVM_CPU_TARGET
TVM_DEVICE = tvm.device(TVM_DEVICE_TYPE, 0)

# Dummy/mock objects for PyTorch-specific components
class DummyConfig:
    def patch(self, *args, **kwargs):
        # In TVM context, these patches are for PyTorch's internal compilation,
        # so they become no-ops.
        return contextlib.nullcontext()

config = DummyConfig()

# PyCodeCache is specific to TorchInductor's generated code. No direct TVM equivalent.
class DummyPyCodeCache:
    def __init__(self):
        self.modules = []
    def cache_clear(self):
        self.modules = []
PyCodeCache = DummyPyCodeCache()

# _dynamo.mark_dynamic is for dynamic shapes in PyTorch/TorchDynamo.
# TVM Relay handles dynamic shapes explicitly via `relay.TensorType(shape=None)`.
# For these tests, we'll assume static shapes are passed as concrete values
# or mark variables as dynamic in Relay if needed, but direct mark_dynamic on tensors
# outside Relay graph construction is not applicable.
class TorchDynamoMock:
    def mark_dynamic(self, *args, **kwargs):
        pass
_dynamo = TorchDynamoMock()

# FileCheck is for verifying generated code/log output patterns.
# This specific test uses it to verify bandwidth info in Triton kernel output.
# This is not directly mappable to TVM's compilation artifacts or focus.
# We will replace `FileCheck` assertions with direct numerical result checks.
class FileCheck:
    def check_count(self, pattern, count, exactly=1):
        # Placeholder: In a real migration, if needed, this would check `tvm.build` output logs.
        # For these specific tests, it's tied to Triton kernel benchmarking output.
        # We're removing the kernel benchmarking aspect.
        return self
    def run(self, text):
        pass

# `xfailIfSM89` and `is_big_gpu` are PyTorch/GPU-specific heuristics.
# Removing `xfailIfSM89` as its condition is specific to PyTorch's Triton integration.
# `is_big_gpu` also removed, assume general GPU capability if GPU is present.
class InductorUtilsMock:
    def is_big_gpu(self):
        return TVM_HAS_GPU # Simplified assumption for TVM context
_inductor_utils = InductorUtilsMock()


class TestKernelBenchmark(unittest.TestCase):
    # Original test used GPU_TYPE, which maps to TVM_DEVICE_TYPE for TVM.
    # However, for TVM, we'll use `tvm.cuda()` or `tvm.cpu()` directly for device context.
    # The tests should reflect TVM's execution environment.

    # python_path not directly needed for TVM Python-only tests.
    python_path = "" # Retained for structure, but effectively unused

    @classmethod
    def setUpClass(cls):
        # Mimic context management; relevant parts are config patches.
        # Other parts like `subprocess` env setup are for TorchInductor's internal
        # benchmarking scripts.
        cls.exit_stack = contextlib.ExitStack()
        cls.exit_stack.enter_context(config.patch(benchmark_kernel=True)) # Keyed argument for patch

    @classmethod
    def tearDownClass(cls):
        cls.exit_stack.close()

    def setUp(self):
        super().setUp()
        PyCodeCache.cache_clear()

    # The original `get_compiled_module` is for TorchInductor's internal module caching.
    # For TVM, we will construct and compile a Relay module for each test.
    # This helper is effectively removed and replaced by in-test compilation logic.
    def get_compiled_module(self):
        raise NotImplementedError("`get_compiled_module` is specific to TorchInductor and not mappable to TVM.")

    # These verification functions are about inspecting TorchInductor's generated Triton kernels
    # and their performance metrics (GB/s). This is not directly mappable to TVM.
    # We will replace these with direct assertions on numerical correctness of the TVM output.
    def verify_compiled_kernels(self, GB_count=1):
        raise NotImplementedError("`verify_compiled_kernels` is specific to TorchInductor's kernel benchmarking and not mappable to TVM.")

    def verify_remove_inductor_deps(self, compiled_module):
        raise NotImplementedError("`verify_remove_inductor_deps` is specific to TorchInductor's generated kernel code and not mappable to TVM.")

    def check_bandwidth(self, compiled_module, num_gb):
        raise NotImplementedError("`check_bandwidth` is specific to TorchInductor's kernel benchmarking and not mappable to TVM.")

    # Helper function to compile and run a Relay module
    def _compile_and_run(self, relay_mod_func, inputs):
        # Create a Relay function
        # Determine shapes and dtypes from NumPy inputs
        arg_vars = []
        for i, inp in enumerate(inputs):
            # Ensure scalar NumPy arrays get a shape of () for Relay.
            # Otherwise, shape is (len(inp.shape),)
            shape = inp.shape if inp.shape else ()
            arg_vars.append(relay.var(f"p{i}", type_annotation=relay.TensorType(shape, str(inp.dtype))))

        # Apply the Relay function logic to the defined variables
        body = relay_mod_func(*arg_vars)
        relay_func = relay.Function(arg_vars, body)
        mod = tvm.IRModule.from_expr(relay_func)
        
        # Build and run with GraphExecutor (common for TVM tests)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=TVM_DEVICE_TYPE)
        
        runtime = graph_executor.GraphModule(lib["default"](TVM_DEVICE))
        
        # Set inputs
        for i, inp_data in enumerate(inputs):
            runtime.set_input(i, tvm.nd.array(inp_data, TVM_DEVICE))
        
        # Execute
        runtime.run()
        
        # Get output
        # Assuming single output for these tests
        output_data = runtime.get_output(0).numpy()
        return output_data

    def test_pw_kernel_benchmark(self):
        # Original:
        # @torch.compile
        # def f(x):
        #     return torch.sin(x) + torch.cos(x)
        # inp = torch.rand(2, 3).to(device=GPU_TYPE)
        # out = f(inp)
        # self.verify_compiled_kernels()

        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")

        def f_relay(x_var):
            return relay.op.add(relay.op.tensor.sin(x_var), relay.op.tensor.cos(x_var))

        np_inp = np.random.rand(2, 3).astype(np.float32)
        tvm_out = self._compile_and_run(f_relay, [np_inp])

        # Calculate expected output with NumPy (emulating PyTorch)
        expected_out = np.sin(np_inp) + np.cos(np_inp)

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-5, atol=1e-5)
        # Removed self.verify_compiled_kernels() as it's not mappable.

    # TODO: Currently the Triton mm template +  relu fusion causes slowdown on XPU,
    # Need to refine the template and config for XPU.
    # @config.patch(
    #     max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    # )
    # @unittest.skipIf(
    #     not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    # )
    # @fresh_cache() # fresh_cache is Inductor-specific.
    def test_matmul_triton_kernel_benchmark(self):
        # Skipping condition mapping to TVM specific context
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Skipping IS_BIG_GPU and max_autotune config patches as they are TorchInductor-specific.
        # This test relies on Triton kernel benchmarking, which is not directly mappable.
        # We will convert the functional part only.

        M = 12544
        N = 256
        K = 64
        
        # Original: a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        # Original: b = torch.rand(N, K, dtype=torch.float16, device=GPU_TYPE).t()
        np_a = np.random.rand(M, K).astype(np.float16)
        np_b = np.random.rand(N, K).astype(np.float16).T

        # Original:
        # @torch.compile
        # def f(a, b):
        #     return torch.relu(a @ b)
        def f_relay(a_var, b_var):
            return relay.op.nn.relu(relay.op.nn.matmul(a_var, b_var))

        tvm_out = self._compile_and_run(f_relay, [np_a, np_b])
        expected_out = np.maximum(0, np.dot(np_a, np_b))

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2) # float16 precision

    # @config.patch(
    #     max_autotune=True, max_autotune_gemm_backends="TRITON", shape_padding=False
    # )
    # @fresh_cache()
    def test_mm_triton_kernel_benchmark(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Skipping config patches as they are TorchInductor-specific.

        M = 2048
        N = 2432
        K = 1949
        K_2 = 3581
        
        # Original: a = rand_strided((M, K_2), (K_2, 1), device=GPU_TYPE, dtype=torch.float16)
        # Original: b = rand_strided((K, N), (1, K), device=GPU_TYPE, dtype=torch.float16)
        # `rand_strided` not directly mappable for performance implications, use normal rand.
        np_a = np.random.rand(M, K_2).astype(np.float16)
        np_b = np.random.rand(K, N).astype(np.float16)

        # Original:
        # @torch.compile
        # def f(a, b):
        #     a_1 = torch.narrow(a, 1, 0, K)
        #     c = torch.mm(a_1, b)
        #     return c
        def f_relay(a_var, b_var):
            # narrow(input, dim, start, length) -> slice(input, axes=[dim], begin=[start], end=[start+length])
            a_1 = relay.op.transform.slice(a_var, axes=[1], begin=[0], end=[K])
            c = relay.op.nn.matmul(a_1, b_var)
            return c

        tvm_out = self._compile_and_run(f_relay, [np_a, np_b])

        np_a_1 = np_a[:, 0:K]
        expected_out = np.dot(np_a_1, np_b)

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.verify_compiled_kernels(GB_count=1)

    def test_matmul_bandwidth_computation(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # `torch.set_float32_matmul_precision("high")` is PyTorch specific. Removed.
        # This test's core assertion is about bandwidth, which is not directly mappable.
        # We will only convert the functional correctness.

        # Original:
        # @torch.compile
        # def f(x, y):
        #     z = x @ y
        #     w = z * z
        #     return w
        def f_relay(x_var, y_var):
            z = relay.op.nn.matmul(x_var, y_var)
            w = relay.op.tensor.multiply(z, z) # Assuming element-wise multiply
            return w

        M, N, K = 1000, 1000, 10
        np_x = np.random.rand(M, K).astype(np.float32)
        np_y = np.random.rand(K, N).astype(np.float32)

        tvm_out = self._compile_and_run(f_relay, [np_x, np_y])

        np_z = np.dot(np_x, np_y)
        expected_out = np_z * np_z

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-5, atol=1e-5)
        # Removed self.check_bandwidth(compiled_module, 0.008)

    def test_unused_input_bandwidth_computation(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Bandwidth computation is not directly mappable. Functional part only.

        M, N = 5, 1000000

        # Original:
        # @torch.compile
        # def f(a, b, c):
        #     return a + c
        def f_relay(a_var, b_var, c_var): # b_var is unused, matching original
            return relay.op.add(a_var, c_var)

        np_a = np.random.rand(M, N).astype(np.float16)
        np_b = np.random.rand(M, N).astype(np.float16)
        np_c = np.random.rand(M, N).astype(np.float16)
        
        # `torch._dynamo.mark_dynamic` is PyTorch-specific. Removed.
        inputs = (np_a, np_b, np_c)
        tvm_out = self._compile_and_run(f_relay, inputs)

        expected_out = np_a + np_c

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.check_bandwidth(compiled_module, "0.030")

    def test_reduction_bandwidth_computation(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Bandwidth computation not directly mappable. Functional part only.

        # Original:
        # @torch.compile
        # def f(a):
        #     return torch.sum(a, dim=1)
        def f_relay(a_var):
            return relay.op.reduce.sum(a_var, axis=[1], keepdims=False)

        np_a = np.random.rand(1000, 20, 1000).astype(np.float16)
        inputs = (np_a,)
        tvm_out = self._compile_and_run(f_relay, inputs)

        expected_out = np.sum(np_a, axis=1)

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.check_bandwidth(compiled_module, "0.042")

    # @config.patch(max_autotune=True)
    def test_fused_layernorm_bandwidth_computation(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Skipping config patch. Bandwidth not mappable. Functional part only.

        M, N = 10, 1000000

        # Original:
        # @torch.compile
        # def f(a, b, c, d):
        #     x0 = a + b
        #     x1 = torch.nn.functional.layer_norm(
        #         x0, normalized_shape=(N,), weight=c, bias=d, eps=1e-05
        #     )
        #     x2 = torch.sigmoid(x1)
        #     return x0 * x2
        def f_relay(a_var, b_var, c_var, d_var):
            x0 = relay.op.add(a_var, b_var)
            
            # LayerNorm needs axis from normalized_shape
            # Assuming `a_var` has the input rank.
            # normalized_shape=(N,) means normalizing over the last dimension.
            input_ndim = len(a_var.checked_type.shape) # assuming a_var and x0 have same rank
            normalized_shape_len = 1 # for (N,)
            layernorm_axes = tuple(range(input_ndim - normalized_shape_len, input_ndim))

            x1 = relay.op.nn.layer_norm(
                x0, gamma=c_var, beta=d_var, axis=layernorm_axes, epsilon=1e-05, center=True, scale=True
            )
            x2 = relay.op.tensor.sigmoid(x1)
            return relay.op.tensor.multiply(x0, x2)

        np_a = np.random.rand(M, N).astype(np.float16)
        np_b = np.random.rand(N).astype(np.float16)
        np_c = np.random.rand(N).astype(np.float16)
        np_d = np.random.rand(N).astype(np.float16)
        inputs = (np_a, np_b, np_c, np_d)
        
        tvm_out = self._compile_and_run(f_relay, inputs)

        # Emulate PyTorch layer_norm with NumPy
        # PyTorch layer_norm applies over trailing `normalized_shape` dimensions.
        # For `normalized_shape=(N,)`, it normalizes along the last dimension.
        np_x0 = np_a + np_b # broadcasting
        
        # Calculate mean and variance over the last axis
        mean_x0 = np.mean(np_x0, axis=-1, keepdims=True)
        var_x0 = np.var(np_x0, axis=-1, keepdims=True)
        
        # Normalize
        std_x0 = np.sqrt(var_x0 + 1e-05)
        np_x1 = (np_x0 - mean_x0) / std_x0
        
        # Apply gamma (weight) and beta (bias)
        np_x1 = np_x1 * np_c + np_d # c and d broadcast along axis -1
        
        np_x2 = 1 / (1 + np.exp(-np_x1)) # Sigmoid
        expected_out = np_x0 * np_x2

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.check_bandwidth(compiled_module, "0.046")

    def test_slice_add_cat_bandwidth_computation(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Bandwidth not mappable. Functional part only.

        M, N = 5, 1000000

        # Original:
        # @torch.compile
        # def f(a, b, c):
        #     x0 = torch.narrow(b, 1, N, N)
        #     # broadcasting
        #     x1 = x0 + c
        #     return torch.cat([a, x1], dim=1)
        
        def f_relay(a_var, b_var, c_var):
            # narrow(input, dim, start, length) -> slice(input, axes=[dim], begin=[start], end=[start+length])
            x0 = relay.op.transform.slice(b_var, axes=[1], begin=[N], end=[N + N])
            x1 = relay.op.add(x0, c_var)
            return relay.op.tensor.concatenate([a_var, x1], axis=1)

        np_a = np.random.rand(M, N).astype(np.float16)
        np_b = np.random.rand(M, N * 5).astype(np.float16)
        np_c = np.random.rand(N).astype(np.float16)
        # `torch._dynamo.mark_dynamic` is PyTorch-specific. Removed.
        inputs = (np_a, np_b, np_c)
        tvm_out = self._compile_and_run(f_relay, inputs)

        np_x0 = np_b[:, N : N + N] # Equivalent to narrow
        np_x1 = np_x0 + np_c
        expected_out = np.concatenate([np_a, np_x1], axis=1)

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.check_bandwidth(compiled_module, "0.052")

    def test_slice_add_bandwidth_computation(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Bandwidth not mappable. Functional part only.

        M, N = 5, 1000000

        # Original:
        # @torch.compile
        # def f(a, b, c):
        #     x0 = torch.narrow(b, 1, N, N)
        #     return a + x0 + c
        def f_relay(a_var, b_var, c_var):
            x0 = relay.op.transform.slice(b_var, axes=[1], begin=[N], end=[N + N])
            return relay.op.add(relay.op.add(a_var, x0), c_var)

        np_a = np.random.rand(M, N).astype(np.float16)
        np_b = np.random.rand(M, N * 5).astype(np.float16)
        np_c = np.random.rand(N).astype(np.float16)
        # `torch._dynamo.mark_dynamic` is PyTorch-specific. Removed.
        inputs = (np_a, np_b, np_c)
        tvm_out = self._compile_and_run(f_relay, inputs)

        np_x0 = np_b[:, N : N + N]
        expected_out = np_a + np_x0 + np_c

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.check_bandwidth(compiled_module, "0.032")

    def test_mm_slice_add_bandwidth_computation(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Bandwidth not mappable. Functional part only.

        M, N, K = 1000, 1000, 30

        # Original:
        # @torch.compile
        # def f(a, b, c):
        #     x0 = torch.mm(a, b)
        #     x1 = torch.narrow(c, 1, 20 * N, N)
        #     x2 = torch.narrow(c, 1, 21 * N, N)
        #     return x0 + x1 + x2
        def f_relay(a_var, b_var, c_var):
            x0 = relay.op.nn.matmul(a_var, b_var)
            x1 = relay.op.transform.slice(c_var, axes=[1], begin=[20 * N], end=[20 * N + N])
            x2 = relay.op.transform.slice(c_var, axes=[1], begin=[21 * N], end=[21 * N + N])
            return relay.op.add(relay.op.add(x0, x1), x2)

        np_a = np.random.rand(M, K).astype(np.float16)
        np_b = np.random.rand(K, N).astype(np.float16)
        np_c = np.random.rand(N, N * 100).astype(np.float16)
        inputs = (np_a, np_b, np_c)
        tvm_out = self._compile_and_run(f_relay, inputs)

        np_x0 = np.dot(np_a, np_b)
        np_x1 = np_c[:, 20 * N : 20 * N + N]
        np_x2 = np_c[:, 21 * N : 21 * N + N]
        expected_out = np_x0 + np_x1 + np_x2

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.check_bandwidth(compiled_module, num_gb)

    def test_mm_slice_add_bandwidth_computation_2(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Bandwidth not mappable. Functional part only.

        M, N, K = 1000, 1000, 30

        # Original:
        # @torch.compile
        # def f(a, b, c):
        #     x0 = torch.mm(a, b)
        #     x1 = torch.narrow(c, 1, 20 * N, N)
        #     x2 = torch.narrow(c, 1, 20 * N, N) # Note: same slice as x1
        #     return x0 + x1 + x2
        def f_relay(a_var, b_var, c_var):
            x0 = relay.op.nn.matmul(a_var, b_var)
            x1 = relay.op.transform.slice(c_var, axes=[1], begin=[20 * N], end=[20 * N + N])
            x2 = relay.op.transform.slice(c_var, axes=[1], begin=[20 * N], end=[20 * N + N])
            return relay.op.add(relay.op.add(x0, x1), x2)

        np_a = np.random.rand(M, K).astype(np.float16)
        np_b = np.random.rand(K, N).astype(np.float16)
        np_c = np.random.rand(N, N * 100).astype(np.float16)
        inputs = (np_a, np_b, np_c)
        tvm_out = self._compile_and_run(f_relay, inputs)

        np_x0 = np.dot(np_a, np_b)
        np_x1 = np_c[:, 20 * N : 20 * N + N]
        np_x2 = np_c[:, 20 * N : 20 * N + N]
        expected_out = np_x0 + np_x1 + np_x2

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.check_bandwidth(compiled_module, "0.006")

    # @xfailIfSM89 # PyTorch specific, removed.
    # @config.patch(
    #     max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    # )
    def test_slice_mm_bandwidth_computation(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # `_inductor.utils.is_big_gpu()` is PyTorch specific.
        # This test relies on Triton kernel benchmarking. Functional part only.

        M, N, K = 1000, 2000, 3000

        # Original:
        # @torch.compile
        # def f(a, b):
        #     x = torch.narrow(a, 1, K, K)
        #     return torch.mm(x, b)
        def f_relay(a_var, b_var):
            x = relay.op.transform.slice(a_var, axes=[1], begin=[K], end=[K + K])
            return relay.op.nn.matmul(x, b_var)

        np_a = np.random.rand(M, 3 * K).astype(np.float16)
        np_b = np.random.rand(K, N).astype(np.float16)
        # `torch._dynamo.mark_dynamic` is PyTorch-specific. Removed.
        inputs = (np_a, np_b)
        tvm_out = self._compile_and_run(f_relay, inputs)

        np_x = np_a[:, K : K + K]
        expected_out = np.dot(np_x, np_b)

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-2, atol=1e-2)
        # Removed self.check_bandwidth(compiled_module, "0.022")

    def test_star_dep(self):
        # This test uses `a[b] = 3.0` which is an indexed assignment.
        # TVM Relay is functional, so direct indexed assignment is not a primitive op.
        # This requires a composite approach, likely involving `relay.scatter_nd`.
        # `scatter_nd(data, indices, updates)` is the closest.
        # `indices` must be an integer tensor. `updates` must have a shape that
        # when combined with `indices` matches a portion of `data`.

        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        
        # Original:
        # @torch.compile
        # def f(a, b):
        #     a[b] = 3.0 # Indexed assignment
        def f_relay(a_var, b_var):
            # `a` is (10000, 5000) float32
            # `b` is (20000, 1) int32 (indices for dim 0)
            # `a[b] = 3.0` means update rows specified by b, across all columns, with 3.0
            
            # `updates` tensor shape needs to be (b_rows, a_cols) = (20000, 5000)
            three_val_scalar = relay.const(3.0, dtype='float32')
            
            # This requires knowing shapes dynamically, which is typically handled by Relay's type inference.
            # For this test, we hardcode to match the example logic based on input shapes.
            # The shape_of returns a 1-D tensor, so we need to extract scalar values.
            # Alternatively, we rely on the type inference if `b_var.checked_type.shape[0]` etc. are available at this stage.
            updates_shape_expr = relay.op.concatenate([
                relay.op.take(relay.shape_of(b_var), relay.const([0], dtype="int64")), # b_var.shape[0]
                relay.op.take(relay.shape_of(a_var), relay.const([1], dtype="int64"))  # a_var.shape[1]
            ], axis=0)
            
            updates_tensor = relay.op.transform.full(three_val_scalar, shape=updates_shape_expr, dtype='float32')

            # `b_var` as indices directly. `axis=1` means scatter along dim 0.
            return relay.op.transform.scatter_nd(a_var, b_var, updates_tensor)

        np_a_initial = np.random.rand(10000, 5000).astype(np.float32)
        np_b = np.random.randint(0, 10000, [20000], dtype=np.int32)[:, np.newaxis] # unsqueeze(1)
        
        inputs = (np_a_initial, np_b)
        tvm_out = self._compile_and_run(f_relay, inputs)

        # Calculate expected output with NumPy
        expected_out = np.copy(np_a_initial)
        # NumPy's advanced indexing will handle the broadcast assignment
        expected_out[np_b[:, 0]] = 3.0 # Index along axis 0 with `b` values, broadcast 3.0 to all columns

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-5, atol=1e-5)
        # Removed self.check_bandwidth(compiled_module, "0.200")

    def test_split_scan(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Bandwidth not mappable. Functional part only.

        # Original:
        # @torch.compile
        # def f(a):
        #     return a.cumsum(-1) # This implies calling on a flattened tensor: a.reshape(-1).cumsum(-1)

        def f_relay(a_var):
            # The test input `a.reshape(-1)` means a 1D tensor
            # cumsum(-1) on a 1D tensor is just cumsum(axis=0)
            return relay.op.transform.cumsum(a_var, axis=-1, exclusive=False) # Default inclusive

        np_a_original = np.random.rand(10000, 5000).astype(np.float32)
        np_a_flat = np_a_original.reshape(-1) # Input to the function `f` is flattened

        tvm_out = self._compile_and_run(f_relay, [np_a_flat])

        expected_out = np.cumsum(np_a_flat, axis=-1)

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-5, atol=1e-5)
        # Removed self.check_bandwidth(compiled_module, "0.400")

    # @config.patch("triton.unique_kernel_names", True)
    # @config.patch(benchmark_kernel=False)
    # @config.patch(compile_threads=1)
    def test_remove_inductor_deps(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # All config patches are TorchInductor-specific.
        # The `verify_remove_inductor_deps` checks generated Triton code structure,
        # which is not applicable to TVM. Functional part only.

        # Original:
        # @torch.compile
        # def f(a):
        #     return a.cos().sin()
        def f_relay(a_var):
            return relay.op.tensor.sin(relay.op.tensor.cos(a_var))

        np_a = np.random.randn(5).astype(np.float32)
        tvm_out = self._compile_and_run(f_relay, [np_a])

        expected_out = np.sin(np.cos(np_a))

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-5, atol=1e-5)
        # Removed `self.verify_remove_inductor_deps(compiled_module)`

    # @config.patch("triton.unique_kernel_names", True)
    # @config.patch(benchmark_kernel=False)
    # @config.patch(compile_threads=1)
    def test_remove_inductor_deps_multiple_kernels(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # All config patches are TorchInductor-specific.
        # The `verify_remove_inductor_deps` checks generated Triton code structure,
        # which is not applicable to TVM. Functional part only.

        # Original:
        # @torch.compile
        # def f(a):
        #     a = torch.mm(a, a)
        #     a = a.cos().sin()
        #     a = torch.mm(a, a)
        #     a = torch.softmax(a, dim=-1)
        #     return a
        def f_relay(a_var):
            a_intermediate = relay.op.nn.matmul(a_var, a_var)
            a_intermediate = relay.op.tensor.sin(relay.op.tensor.cos(a_intermediate))
            a_intermediate = relay.op.nn.matmul(a_intermediate, a_intermediate)
            a_intermediate = relay.op.nn.softmax(a_intermediate, axis=-1)
            return a_intermediate

        np_a = np.random.randn(5, 5).astype(np.float32)
        tvm_out = self._compile_and_run(f_relay, [np_a])

        np_a_intermediate = np.dot(np_a, np_a)
        np_a_intermediate = np.sin(np.cos(np_a_intermediate))
        np_a_intermediate = np.dot(np_a_intermediate, np_a_intermediate)
        
        # NumPy softmax
        e_x = np.exp(np_a_intermediate - np.max(np_a_intermediate, axis=-1, keepdims=True))
        expected_out = e_x / np.sum(e_x, axis=-1, keepdims=True)

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-5, atol=1e-5)
        # Removed `self.verify_remove_inductor_deps(compiled_module)`

    # @unittest.skipIf(
    #     not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    # )
    # @config.patch("triton.unique_kernel_names", True)
    # @config.patch("triton.unique_kernel_names", True) # Duplicate patch, likely artifact
    # @config.patch(benchmark_kernel=False)
    # @config.patch(compile_threads=1)
    # @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    def test_remove_inductor_deps_templates(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # Skipping config patches and IS_BIG_GPU as they are TorchInductor-specific.
        # The `verify_remove_inductor_deps` checks generated Triton code structure,
        # which is not applicable to TVM. Functional part only.

        # Original:
        # @torch.compile
        # def f(a):
        #     a = torch.mm(a, a)
        #     a = a.cos()
        #     a = torch.mm(a, a)
        #     a = a.sin()
        #     return a
        def f_relay(a_var):
            a_intermediate = relay.op.nn.matmul(a_var, a_var)
            a_intermediate = relay.op.tensor.cos(a_intermediate)
            a_intermediate = relay.op.nn.matmul(a_intermediate, a_intermediate)
            a_intermediate = relay.op.tensor.sin(a_intermediate)
            return a_intermediate

        np_a = np.random.randn(128, 128).astype(np.float32)
        tvm_out = self._compile_and_run(f_relay, [np_a])

        np_a_intermediate = np.dot(np_a, np.copy(np_a)) # Use copy to ensure correct intermediate for numpy
        np_a_intermediate = np.cos(np_a_intermediate)
        np_a_intermediate = np.dot(np_a_intermediate, np_a_intermediate)
        expected_out = np.sin(np_a_intermediate)

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-5, atol=1e-5)
        # Removed `self.verify_remove_inductor_deps(compiled_module)`

    # @config.patch("triton.unique_kernel_names", True)
    # @config.patch(benchmark_kernel=False)
    # @config.patch(compile_threads=1)
    def test_remove_inductor_deps_scalar(self):
        if not TVM_HAS_GPU:
            pytest.skip("Test requires GPU")
        # All config patches are TorchInductor-specific.
        # The `verify_remove_inductor_deps` checks generated Triton code structure,
        # which is not applicable to TVM. Functional part only.

        # Original:
        # @torch.compile
        # def f(a, b):
        #     return a + b
        def f_relay(a_var, b_var):
            return relay.op.add(a_var, b_var)

        np_a = np.array(1.0, dtype=np.float32) # Scalar
        np_b = np.array(2.0, dtype=np.float32) # Scalar
        
        tvm_out = self._compile_and_run(f_relay, [np_a, np_b])

        expected_out = np_a + np_b

        tvm.testing.assert_allclose(tvm_out, expected_out, rtol=1e-5, atol=1e-5)
        # Removed `self.verify_remove_inductor_deps(compiled_module)`


if __name__ == "__main__":
    unittest.main()
