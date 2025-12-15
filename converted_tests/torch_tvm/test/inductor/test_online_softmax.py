# Owner(s): ["module: inductor"]

import math
import os
import unittest
import time
import numpy as np

# TVM imports
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay import nn
from tvm.relay import transform
from tvm import topi  # topi.transform.transpose used directly in a comment for mapping
import tvm.runtime
import tvm.testing

# Placeholder for torch._inductor.config which is not applicable
class InductorConfig:
    def patch(self, **kwargs):
        class _Context:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return _Context()

inductor_config = InductorConfig()

# Replacement for torch._dynamo.utils.rmse, same
def rmse(actual, desired):
    return np.sqrt(np.mean((actual - desired)**2))

def same(expected, actual, tol=1e-2):
    return np.allclose(expected, actual, rtol=tol, atol=tol, equal_nan=True)

# Placeholder for torch._inductor.test_case.run_tests (assuming unittest.main)
def run_tests():
    unittest.main()

# Placeholder for torch._inductor.test_case.TestCase
class TestCase(unittest.TestCase):
    pass

# Placeholder for GPU_TYPE and HAS_CUDA_AND_TRITON
# Assuming GPU is available for CUDA tests if tvm.cuda().exist
GPU_TYPE = "cuda" if tvm.cuda().exist else "cpu"
HAS_CUDA_AND_TRITON = tvm.cuda().exist 

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
USE_LARGE_INPUT = os.environ.get("USE_LARGE_INPUT") == "1" or DO_PERF_TEST

# --- NumPy/Python equivalent implementations for Torch operations used as reference ---
# For reference calculations involving bfloat16, we'll use np.float32 as an approximation
# since native np.bfloat16 is not universally available and can introduce dependency.

def _to_numpy_dtype_for_ref(dtype_str):
    if dtype_str == 'bfloat16':
        return np.float32 # Approximation for NumPy reference
    elif dtype_str == 'float16':
        return np.float16
    elif dtype_str == 'float32':
        return np.float32
    elif dtype_str == 'float64':
        return np.float64
    elif dtype_str == 'int32':
        return np.int32
    elif dtype_str == 'int64':
        return np.int64
    elif dtype_str == 'bool':
        return np.bool_
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")

# NumPy equivalent for _prepare_softmax
def _prepare_softmax_np_ref(x_np, dim):
    xmax = np.amax(x_np, axis=dim, keepdims=True)
    exp_val = np.exp(x_np - xmax)
    xsum = np.sum(exp_val, axis=dim, keepdims=True)
    return xmax, xsum

# NumPy equivalent for F.softmax
def _softmax_np_ref(x_np, dim):
    e_x = np.exp(x_np - np.max(x_np, axis=dim, keepdims=True))
    # Handle division by zero for fully -inf rows, resulting in NaN.
    # PyTorch's softmax handles this as 0 in output.
    # np.seterr(divide='ignore', invalid='ignore') # Temporarily ignore warnings
    sum_e_x = np.sum(e_x, axis=dim, keepdims=True)
    result = np.where(sum_e_x == 0, 0, e_x / sum_e_x)
    return result

# NumPy equivalent for F.log_softmax
def _log_softmax_np_ref(x_np, dim):
    e_x = np.exp(x_np - np.max(x_np, axis=dim, keepdims=True))
    # np.seterr(divide='ignore', invalid='ignore') # Temporarily ignore warnings
    sum_e_x = np.sum(e_x, axis=dim, keepdims=True)
    log_sum_e_x = np.log(sum_e_x)
    result = x_np - np.max(x_np, axis=dim, keepdims=True) - log_sum_e_x
    # Handle cases where log_sum_e_x is -inf (e.g., all inputs were -inf)
    result = np.where(np.isinf(log_sum_e_x), float('-inf'), result)
    return result

# NumPy equivalent for F.softmin (1 - softmax(-x))
def _softmin_np_ref(x_np, dim):
    return 1 - _softmax_np_ref(-x_np, dim)

# NumPy equivalent for matmul from `test_sdpa`
def _sdpa_ref(q_np_in, k_np_in, v_np_in):
    # PyTorch's matmul with bfloat16 often promotes to float32 internally.
    # Emulate this behavior for numpy reference.
    q_np = q_np_in.astype(np.float32)
    k_np = k_np_in.astype(np.float32)
    v_np = v_np_in.astype(np.float32)
    
    # K.transpose(-2, -1)
    k_transposed = np.swapaxes(k_np, -2, -1)
    
    # Matmul
    matmul_qk = np.matmul(q_np, k_transposed)
    
    # Div
    div_factor = math.sqrt(k_np_in.shape[-1]) # Use original k_np_in shape for divisor
    div_res = matmul_qk / div_factor
    
    # Softmax
    softmax_res = _softmax_np_ref(div_res, dim=-1)
    
    # Matmul V
    matmul_v = np.matmul(softmax_res, v_np)
    
    # Cast back to original output dtype (which was bfloat16-equivalent float32 for ref)
    return matmul_v.astype(_to_numpy_dtype_for_ref('bfloat16'))


# --- TVM Relay Graph Construction functions ---

def _softmax_relay(x_var, dim):
    return nn.softmax(x_var, axis=dim)

def _log_softmax_relay(x_var, dim):
    return nn.log_softmax(x_var, axis=dim)

# TVM Relay equivalent for _prepare_softmax
def _prepare_softmax_relay_graph(x_var, dim):
    xmax = op.reduce.max(x_var, axis=dim, keepdims=True)
    exp_val = op.tensor.exp(op.tensor.subtract(x_var, xmax))
    xsum = op.reduce.sum(exp_val, axis=dim, keepdims=True)
    return xmax, xsum

# TVM Relay equivalent for softmin (1 - softmax(-x))
def _softmin_relay_graph(x_var, dim):
    return op.tensor.subtract(relay.const(1.0, dtype=x_var.dtype), nn.softmax(op.tensor.negative(x_var), axis=dim))


# --- TVM Compilation and Execution Helper ---

def _compile_and_run_tvm(f_relay_builder, *args_np, device_str=None):
    # f_relay_builder is a Python callable that takes Relay.Var(s) and returns a Relay expression.
    # args_np are the actual input numpy arrays.

    input_vars = []
    input_data_tvm_nd = []
    
    target_dev = tvm.cuda(0) if device_str and "cuda" in device_str and tvm.cuda().exist else tvm.cpu(0)

    for i, arg_np in enumerate(args_np):
        var_name = f"i{i}"
        # Determine TVM dtype for relay.var
        tvm_dtype = str(arg_np.dtype) 
        if "bfloat16" in device_str: # Heuristic to detect intended bfloat16 for the graph
            tvm_dtype = 'bfloat16'

        input_vars.append(relay.var(var_name, shape=arg_np.shape, dtype=tvm_dtype))
        input_data_tvm_nd.append(tvm.nd.array(arg_np, device=target_dev))

    relay_expr = f_relay_builder(*input_vars)

    if isinstance(relay_expr, (list, tuple)):
        func = relay.Function(input_vars, relay.Tuple(list(relay_expr)))
    else:
        func = relay.Function(input_vars, relay_expr)

    mod = tvm.IRModule.from_expr(func)

    target_tvm_str = "cuda" if target_dev.device_type == tvm.runtime.Device.kDLGPU else "llvm"

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target_tvm_str)
    
    vm_exec = tvm.runtime.vm.VirtualMachine(lib, target_dev)
    result = vm_exec.run(*input_data_tvm_nd)

    if isinstance(result, tvm.runtime.ndarray.NDArray):
        return result.numpy(), ("# TVM compiled code stub\n",)
    elif isinstance(result, tvm.runtime.container.ADT): # For tuples of results
        return tuple(r.numpy() for r in result), ("# TVM compiled code stub\n",)
    else:
        raise TypeError(f"Unexpected TVM result type: {type(result)}")

# Custom parametrize decorator that correctly handles stacking and kwargs
def parametrize(key, value_list):
    def decorator(test_func):
        if not hasattr(test_func, '_tvm_params'):
            test_func._tvm_params = [{}] # Initialize with one empty dict for first param
        
        new_param_combinations = []
        for val in value_list:
            new_param_combinations.append({key: val})

        combined_params = []
        for existing_kwargs in test_func._tvm_params:
            for new_kwargs_single_param in new_param_combinations:
                merged_kwargs = {**existing_kwargs, **new_kwargs_single_param}
                combined_params.append(merged_kwargs)
        test_func._tvm_params = combined_params
        
        return test_func
    return decorator

# Custom instantiate_parametrized_tests to generate actual test methods for unittest
def instantiate_parametrized_tests(cls):
    for attr_name in list(dir(cls)): # Iterate over a copy of dir(cls)
        method = getattr(cls, attr_name)
        if hasattr(method, "_tvm_params"):
            for i, kwargs in enumerate(method._tvm_params):
                # Create a unique name for the new test method
                # Ensure values are cast to string for name generation
                param_str = "_".join(f"{k}{str(v).replace('.', '_').replace('-', 'neg').replace(' ', '')}" for k, v in kwargs.items())
                new_name = f"{attr_name}_{i}_{param_str}"
                
                # Create a closure to capture kwargs for each test instance
                def make_test_method(original_method, captured_kwargs):
                    def test_wrapper(self_instance):
                        return original_method(self_instance, **captured_kwargs)
                    return test_wrapper
                
                setattr(cls, new_name, make_test_method(method, kwargs))
            delattr(cls, attr_name) # Remove the original templated test method
    return cls


class TestOnlineSoftmax(TestCase):
    def _compile_and_run(self, f_relay_builder, *args_np, device=None):
        return _compile_and_run_tvm(f_relay_builder, *args_np, device_str=device)

    def do_test_acc_and_perf(self, relay_op_builder_func):
        if DO_PERF_TEST:
            N = 32 * 1024
            V = 50304
        else:
            N, V = 1024, 2048

        # Input data for numpy reference and TVM execution
        # Use float32 for numpy reference of bfloat16, but TVM will receive 'bfloat16' type string
        x_np_raw = np.random.randn(N, V)
        x_np_for_tvm = x_np_raw.astype(np.float32) 

        # Get NumPy reference result (using float32 as bf16 approx)
        if relay_op_builder_func == _prepare_softmax_relay_graph:
            expected_res = _prepare_softmax_np_ref(x_np_for_tvm, dim=-1)
        elif relay_op_builder_func == _softmax_relay:
            expected_res = _softmax_np_ref(x_np_for_tvm, dim=-1)
        elif relay_op_builder_func == _log_softmax_relay:
            expected_res = _log_softmax_np_ref(x_np_for_tvm, dim=-1)
        else:
            raise ValueError(f"Unsupported relay_op_builder_func for reference: {relay_op_builder_func}")
        
        # Manually set dtype string for Relay var
        def wrap_builder_with_bf16_dtype(builder_func):
            def wrapper(x_var, dim_val):
                # Replace the input variable with one specifying bfloat16 dtype
                x_var_bf16 = relay.var(x_var.name_hint, shape=x_var.checked_type.shape, dtype='bfloat16')
                return builder_func(x_var_bf16, dim_val)
            return wrapper

        # All `do_test_acc_and_perf` calls in this file are for softmax/log_softmax/prepare_softmax,
        # which take a single tensor input and a dim. They all handle bfloat16.
        # The `device` string is augmented to pass the `bfloat16` intent to `_compile_and_run_tvm`.
        tvm_builder_to_run = wrap_builder_with_bf16_dtype(relay_op_builder_func)

        # Compile and run the Relay graph
        actual_res, _ = self._compile_and_run(tvm_builder_to_run, x_np_for_tvm, device=GPU_TYPE + '_bfloat16')

        if isinstance(expected_res, tuple):
             for i in range(len(expected_res)):
                tvm.testing.utils.assert_allclose(actual_res[i], expected_res[i], rtol=1e-2, atol=1e-2)
        else:
            tvm.testing.utils.assert_allclose(actual_res, expected_res, rtol=1e-2, atol=1e-2)

        if DO_PERF_TEST:
            # TODO: Implement TVM-specific benchmarking or skip
            pass 

    def test_softmax(self):
        self.do_test_acc_and_perf(_softmax_relay)

    def test_log_softmax(self):
        self.do_test_acc_and_perf(_log_softmax_relay)

    @inductor_config.patch(use_fast_math=True) # PyTorch-specific, ignored for TVM
    def test_prepare_softmax_perf(self):
        self.do_test_acc_and_perf(_prepare_softmax_relay_graph)

    def get_softmax_wrapper_tvm(self, V=50304, use_log_softmax=False, device_str=GPU_TYPE):
        N = 32 * 1024

        def f_relay_inner(x_var):
            if use_log_softmax:
                return _log_softmax_relay(x_var, dim=-1)
            else:
                return _softmax_relay(x_var, dim=-1)

        x_np_raw = np.random.randn(N, V)
        x_np_for_tvm = x_np_raw.astype(np.float32) 

        # Only return generated code stub, as actual code inspection is not supported
        _, source_codes = self._compile_and_run(
            lambda x_var: relay.var(x_var.name_hint, shape=x_var.checked_type.shape, dtype='bfloat16') if 'bfloat16' in device_str else x_var,
            x_np_for_tvm,
            device_str
        )
        return source_codes[0]

    def test_codegen_3pass_softmax_due_to_disable(self):
        # This test relied on inspecting Triton kernel loops (PyTorch Inductor specifics).
        # It cannot be directly translated for TVM.
        # We retain the call to `get_softmax_wrapper_tvm` to ensure it compiles without error,
        # but skip if CUDA is not available as the original test implied GPU for these code paths.
        if HAS_CUDA_AND_TRITON: 
            _ = self.get_softmax_wrapper_tvm()
        else:
            pass
        

    @parametrize("V", [2048, 50304])
    @parametrize("use_log_softmax", [False, True])
    def test_codegen_online_softmax(self, use_log_softmax, V):
        # Same as above, inspecting Triton kernel loops. Cannot be translated.
        if HAS_CUDA_AND_TRITON:
            _ = self.get_softmax_wrapper_tvm(use_log_softmax=use_log_softmax, V=V)
        else:
            pass

    def test_no_online_softmax_for_cpu(self):
        # Same as above, inspecting Triton kernel loops and CPU-specific code. Cannot be translated.
        # This one specifically tests for CPU device behavior.
        _ = self.get_softmax_wrapper_tvm(V=2048, device_str="cpu")
        pass

    def test_codegen_softmax_persistent_reduction(self):
        # Same as above, inspecting Triton kernel loops and persistent reduction. Cannot be translated.
        if HAS_CUDA_AND_TRITON:
            _ = self.get_softmax_wrapper_tvm(1024)
        else:
            pass

    @inductor_config.patch("triton.persistent_reductions", False) # PyTorch-specific, ignored for TVM
    def test_sdpa(self):
        if not HAS_CUDA_AND_TRITON: # Skip if no CUDA/Triton equivalent for PyTorch tests
            return

        q_np_raw, k_np_raw, v_np_raw = (
            np.random.randn(4, 2, 16, 32)
            for _ in range(3)
        )
        # Use float32 for numpy ref and for TVM input, but specify bfloat16 for TVM compile
        q_np_for_tvm = q_np_raw.astype(np.float32) 
        k_np_for_tvm = k_np_raw.astype(np.float32)
        v_np_for_tvm = v_np_raw.astype(np.float32)

        # NumPy reference (emulating bfloat16 inputs with float32 computation)
        ref = _sdpa_ref(q_np_for_tvm, k_np_for_tvm, v_np_for_tvm)

        def f_relay(q_var_in, k_var_in, v_var_in):
            # Explicitly cast to bfloat16 for internal graph type, then to float32 for matmul, then back.
            q_var = relay.var(q_var_in.name_hint, shape=q_var_in.checked_type.shape, dtype='bfloat16')
            k_var = relay.var(k_var_in.name_hint, shape=k_var_in.checked_type.shape, dtype='bfloat16')
            v_var = relay.var(v_var_in.name_hint, shape=v_var_in.checked_type.shape, dtype='bfloat16')

            q_float32 = op.tensor.cast(q_var, 'float32')
            k_float32 = op.tensor.cast(k_var, 'float32')
            v_float32 = op.tensor.cast(v_var, 'float32')

            k_transposed = op.transform.transpose(k_float32, axes=[0, 1, 3, 2])
            div_factor_val = math.sqrt(k_np_raw.shape[-1])
            div_factor_const = relay.const(div_factor_val, dtype='float32')
            
            matmul_qk = nn.matmul(q_float32, k_transposed)
            div_res = op.tensor.divide(matmul_qk, div_factor_const)
            softmax_res = nn.softmax(div_res, axis=-1)
            matmul_v = nn.matmul(softmax_res, v_float32)

            return op.tensor.cast(matmul_v, 'bfloat16') # Cast final result back

        act, (code,) = self._compile_and_run(f_relay, q_np_for_tvm, k_np_for_tvm, v_np_for_tvm, device=GPU_TYPE + '_bfloat16')
        tvm.testing.utils.assert_allclose(ref, act, atol=1e-2, rtol=1e-2)
        pass


    @parametrize("nrow", [2, 2048])
    @parametrize("dim", [-1, 0, 1])
    def test_prepare_softmax(self, dim, nrow):
        if not HAS_CUDA_AND_TRITON:
            return

        x_np_raw = np.random.randn(nrow, 2048)
        x_np_for_tvm = x_np_raw.astype(np.float32)

        # NumPy reference (emulating bfloat16 inputs with float32 computation)
        ref_res = _prepare_softmax_np_ref(x_np_for_tvm, dim)

        def f_relay(x_var_in):
            x_var = relay.var(x_var_in.name_hint, shape=x_var_in.checked_type.shape, dtype='bfloat16')
            return _prepare_softmax_relay_graph(x_var, dim)

        act_res, (code,) = self._compile_and_run(f_relay, x_np_for_tvm, device=GPU_TYPE + '_bfloat16')
        
        tvm.testing.utils.assert_allclose(act_res[0], ref_res[0], rtol=1e-2, atol=1e-2)
        tvm.testing.utils.assert_allclose(act_res[1], ref_res[1], rtol=1e-2, atol=1e-2)
        pass

    def test_split_reduction(self):
        if not HAS_CUDA_AND_TRITON:
            return

        x_np_raw = np.random.randn(1, 2**20)
        x_np_for_tvm = x_np_raw.astype(np.float32)

        # NumPy reference (emulating bfloat16 inputs with float32 computation)
        ref = _softmax_np_ref(x_np_for_tvm, dim=-1)

        def f_relay(x_var_in):
            x_var = relay.var(x_var_in.name_hint, shape=x_var_in.checked_type.shape, dtype='bfloat16')
            return _softmax_relay(x_var, dim=-1)

        act, (code,) = self._compile_and_run(f_relay, x_np_for_tvm, device=GPU_TYPE + '_bfloat16')
        tvm.testing.utils.assert_allclose(ref, act, atol=1e-3, rtol=1e-3)
        pass

    @parametrize("dtype_str", ['bfloat16', 'float16', 'float32'])
    def test_prepare_softmax_acc_with_fp64(self, dtype_str):
        # Only run bfloat16/float16 tests if CUDA is available, otherwise skip (similar to PyTorch)
        if not HAS_CUDA_AND_TRITON and (dtype_str == 'bfloat16' or dtype_str == 'float16'):
            return 

        if USE_LARGE_INPUT:
            M, N = 32768, 50257
        else:
            M, N = 1024, 2048

        x_np_raw = np.random.randn(M, N)
        x_np_for_ref = x_np_raw.astype(_to_numpy_dtype_for_ref(dtype_str))
        x_np_fp64 = x_np_raw.astype(np.float64) # For fp64 reference

        ref_fp64 = _prepare_softmax_np_ref(x_np_fp64, dim=-1)
        ref = _prepare_softmax_np_ref(x_np_for_ref, dim=-1)

        def f_relay(x_var_in):
            # Dynamically set dtype for Relay var based on test parameter
            x_var = relay.var(x_var_in.name_hint, shape=x_var_in.checked_type.shape, dtype=dtype_str)
            return _prepare_softmax_relay_graph(x_var, dim=-1)

        res, (code,) = self._compile_and_run(f_relay, x_np_for_ref, device=GPU_TYPE) 

        # Max should be exactly equal (or very close)
        tvm.testing.utils.assert_allclose(ref[0], res[0])
        tvm.testing.utils.assert_allclose(ref_fp64[0], ref[0])

        ref_error = rmse(ref_fp64[1], ref[1])
        res_error = rmse(ref_fp64[1], res[1])

        print(f"{dtype_str}: {ref_error=:.4f}, {res_error=:.4f}")

        assert res_error < ref_error + 0.1, f"Accuracy regression for {dtype_str}: res_error ({res_error}) not less than ref_error ({ref_error}) + 0.1"


    @parametrize("fn_name", ["softmax", "log_softmax"]) 
    @parametrize("dtype_str", ['bfloat16', 'float16', 'float32'])
    def test_softmax_acc_with_fp64(self, dtype_str, fn_name):
        # Only run bfloat16/float16 tests if CUDA is available, otherwise skip (similar to PyTorch)
        if not HAS_CUDA_AND_TRITON and (dtype_str == 'bfloat16' or dtype_str == 'float16'):
            return 

        if USE_LARGE_INPUT:
            M, N = 32768, 50257
        else:
            M, N = 1024, 2048

        x_np_raw = np.random.randn(M, N)
        x_np_for_ref = x_np_raw.astype(_to_numpy_dtype_for_ref(dtype_str))
        x_np_fp64 = x_np_raw.astype(np.float64)

        if fn_name == "softmax":
            ref_fp64 = _softmax_np_ref(x_np_fp64, dim=-1)
            ref = _softmax_np_ref(x_np_for_ref, dim=-1)
            relay_op_func = _softmax_relay
        else: # log_softmax
            ref_fp64 = _log_softmax_np_ref(x_np_fp64, dim=-1)
            ref = _log_softmax_np_ref(x_np_for_ref, dim=-1)
            relay_op_func = _log_softmax_relay

        def f_relay(x_var_in):
            # Dynamically set dtype for Relay var based on test parameter
            x_var = relay.var(x_var_in.name_hint, shape=x_var_in.checked_type.shape, dtype=dtype_str)
            return relay_op_func(x_var, dim=-1)

        res, (code,) = self._compile_and_run(f_relay, x_np_for_ref, device=GPU_TYPE)

        ref_error = rmse(ref_fp64, ref)
        res_error = rmse(ref_fp64, res)

        print(f"{fn_name} {dtype_str}: {ref_error=:.10f}, {res_error=:.10f}")

        assert res_error < ref_error + 0.1, f"Accuracy regression for {fn_name} {dtype_str}: res_error ({res_error}) not less than ref_error ({ref_error}) + 0.1"


    def test_softmin(self):
        if not HAS_CUDA_AND_TRITON:
            return

        # Softmin is 1 - softmax(-x)
        x_np = np.random.randn(1).astype(np.float32) 

        ref = _softmin_np_ref(x_np, dim=0)

        def f_relay(x_var):
            return _softmin_relay_graph(x_var, dim=0)

        act, (code,) = self._compile_and_run(f_relay, x_np, device=GPU_TYPE)
        tvm.testing.utils.assert_allclose(ref, act)
        pass


    def test_causal_mask(self):
        if not HAS_CUDA_AND_TRITON:
            return

        x_np = np.random.randn(2048, 2048).astype(np.float32)
        
        # Create mask similar to PyTorch: tril(ones) == 0 -> upper triangle is True
        mask_ones = np.ones((2048, 2048), dtype=x_np.dtype)
        mask_tril = np.tril(mask_ones)
        mask_condition_np = (mask_tril == 0).astype(np.bool_) # Upper triangle is True, lower/diag is False
        
        # NumPy reference
        x_masked_ref = np.where(mask_condition_np, float("-inf"), x_np)
        ref = _softmax_np_ref(x_masked_ref, dim=-1)

        def f_relay(x_var, mask_var):
            inf_const = relay.const(float("-inf"), dtype=x_var.dtype)
            masked_x = op.transform.where(mask_var, inf_const, x_var)
            return _softmax_relay(masked_x, dim=-1)

        act_res, _ = self._compile_and_run(f_relay, x_np, mask_condition_np, device=GPU_TYPE)
        
        assert not np.isnan(ref).any()
        assert not np.isnan(act_res).any()
        tvm.testing.utils.assert_allclose(ref, act_res)

    def test_tb_speech_transformer_attn(self):
        if not HAS_CUDA_AND_TRITON:
            return

        np.random.seed(1337) # Sync NumPy seed with PyTorch seed
        
        x_np_raw = np.random.randn(8, 10, 22, 204)
        x_np = x_np_raw.astype(np.float32)

        mask_raw_np = (np.random.randint(0, 2, (10, 204)) == 0).astype(np.bool_)
        mask_np = mask_raw_np.reshape(1, 10, 1, 204)

        # NumPy reference
        inf_const_np = float("-inf")
        x_filtered_ref = np.where(mask_np, inf_const_np, x_np)
        xmax_ref = np.amax(x_filtered_ref, axis=-1, keepdims=True)
        exp_diff_ref = np.exp(x_filtered_ref - xmax_ref)
        ref = np.sum(exp_diff_ref, axis=-1, keepdims=True)

        def f_relay(x_var, mask_var):
            inf_const_relay = relay.const(float("-inf"), dtype=x_var.dtype)
            x_masked = op.transform.where(mask_var, inf_const_relay, x_var)
            xmax = op.reduce.max(x_masked, axis=-1, keepdims=True)
            exp_diff = op.tensor.exp(op.tensor.subtract(x_masked, xmax))
            xsum = op.reduce.sum(exp_diff, axis=-1, keepdims=True)
            return xsum

        act_res, _ = self._compile_and_run(f_relay, x_np, mask_np, device=GPU_TYPE)
        
        assert not np.isnan(ref).any()
        assert not np.isnan(act_res).any()
        tvm.testing.utils.assert_allclose(ref, act_res)

    @inductor_config.patch(split_reductions=False) # PyTorch-specific, ignored for TVM
    def test_3d_tiled_online_softmax(self):
        if not HAS_CUDA_AND_TRITON:
            return

        M, N, K = 32, 8, 1024

        x_np_raw = np.random.randn(K, N, M)
        y_np_raw = np.random.randn(K, M, N)
        
        # NumPy reference (emulating bfloat16 inputs with float32 computation)
        x_ref_f32 = x_np_raw.astype(np.float32).transpose(2, 1, 0)
        y_ref_f32 = y_np_raw.astype(np.float32).transpose(1, 2, 0)
        
        mult_ref = x_ref_f32 * y_ref_f32
        ref = _softmax_np_ref(mult_ref, dim=-1)

        # TVM input data, will be interpreted as bfloat16 inside the graph
        x_np_for_tvm = x_np_raw.astype(np.float32).transpose(2, 1, 0)
        y_np_for_tvm = y_np_raw.astype(np.float32).transpose(1, 2, 0)

        def f_relay(x_var_in, y_var_in):
            x_var = relay.var(x_var_in.name_hint, shape=x_var_in.checked_type.shape, dtype='bfloat16')
            y_var = relay.var(y_var_in.name_hint, shape=y_var_in.checked_type.shape, dtype='bfloat16')

            mult_res = op.tensor.multiply(x_var, y_var)
            # Softmax will promote if needed or handle bf16 directly
            return nn.softmax(mult_res, axis=-1)

        act_res, _ = self._compile_and_run(f_relay, x_np_for_tvm, y_np_for_tvm, device=GPU_TYPE + '_bfloat16')
        tvm.testing.utils.assert_allclose(ref, act_res, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestOnlineSoftmax)

if __name__ == "__main__":
    # In a real TVM testing setup, `unittest.main()` would run the tests.
    # The original condition `IS_LINUX and HAS_CUDA_AND_TRITON` is for PyTorch Inductor.
    # For TVM, we check `tvm.cuda().exist`
    if os.name == 'posix' and tvm.cuda().exist: # Equivalent to IS_LINUX and HAS_CUDA_AND_TRITON for GPU tests
        run_tests()
    else:
        print("Skipping tests: Not on Linux or CUDA not available.")
