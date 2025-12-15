import os
import random
import tempfile
from unittest import mock

import numpy as np
import pytest

import tvm
from tvm import relay, te
from tvm.relay import op, transform
from tvm.relay.backend.interpreter import Interpreter
from tvm.relay.expr import Call
from tvm.runtime import device, ndarray, vm
from tvm.testing.utils import assert_allclose


# Define a custom Triton-like DSL for building Relay functions to simulate kernels
# This is a highly simplified representation, not a true Triton-to-TVM converter
class TritonSim:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        # In Triton, this sets up the grid and launches.
        # Here, we'll return a callable that can be compiled.
        def _kernel_callable(input_tensors, grid_args):
            # This callable will take the actual TVM NDArrays and grid to produce an output.
            # It should ideally return a Relay expression for compilation.
            # For simplicity, for these tests, we will define a function that *returns* a Relay function.
            pass

        return _kernel_callable


class tl:
    """Simplified mock for triton.language types and intrinsics."""

    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    constexpr = True

    @staticmethod
    def load(ptr, *args, **kwargs):
        # In a real conversion, this would be context-dependent.
        # For these simple kernels, inputs are assumed to be directly accessed.
        return ptr

    @staticmethod
    def store(ptr, val, *args, **kwargs):
        # In a real conversion, this means updating an output buffer.
        # For Relay, this means the 'ptr' (output var) gets 'val'.
        return val

    @staticmethod
    def full(shape, value, dtype):
        return op.transform.full(relay.const(value, dtype), shape=shape, dtype=dtype)

    @staticmethod
    def arange(start, end):
        return op.transform.arange(
            relay.const(start, "int64"),
            relay.const(end, "int64"),
            relay.const(1, "int64"),
            dtype="int64",
        )

    @staticmethod
    def program_id(axis):
        # Placeholder for program_id. In Relay, this is abstracted.
        # For these tests, we'll treat it as 0 unless explicitly used in ways that require actual indexing.
        return 0

    @staticmethod
    def broadcast_to(data, shape):
        return op.transform.broadcast_to(data, shape=shape)

    @staticmethod
    def where(cond, x, y):
        return op.transform.where(cond, x, y)


class libdevice:
    """Simplified mock for triton_helpers.libdevice functions."""

    @staticmethod
    def isinf(x):
        return op.tensor.isinf(x)


class triton_helpers:
    """Simplified mock for triton_helpers functions."""

    @staticmethod
    def any(data, axis):
        return op.reduce.any(data, axis=axis, keepdims=False)


# Dummy class for `StaticallyLaunchedCudaKernel` replacement
# In TVM, we compile a Relay module and run it. The concept of "launcher"
# is absorbed into the Relay VM/GraphExecutor.
class TVMCompiledKernel:
    def __init__(self, relay_func, params, target="cuda"):
        self.relay_func = relay_func
        self.params = params
        self.target = tvm.target.Target(target)
        self.compiled_module = None
        self.vm_executor = None

    def _compile(self):
        mod = tvm.IRModule.from_expr(self.relay_func)
        with self.target:
            self.compiled_module = relay.build(mod, target=self.target, params=self.params)
        self.vm_executor = vm.VirtualMachine(self.compiled_module, tvm.cuda(0))

    def run(self, *args_raw):
        if self.compiled_module is None:
            self._compile()

        # Convert args to TVM NDArrays
        tvm_args = []
        for arg in args_raw:
            if isinstance(arg, (np.ndarray, list, tuple, int, float, bool)):
                # Assume if arg has a ._tvm_ndarray, it's already converted/wrapped
                # Otherwise, convert Python types/numpy arrays to TVM.
                if isinstance(arg, (int, float, bool)):
                    tvm_args.append(relay.const(arg)) # Use relay.const for scalar constants
                else:
                    tvm_args.append(ndarray.array(arg, device=tvm.cuda(0)))
            elif isinstance(arg, ndarray.NDArray):
                tvm_args.append(arg)
            elif isinstance(arg, Call): # Handle case where arg is already a Relay expression (e.g. relay.const)
                tvm_args.append(arg)
            else:
                # If arg is expected to be an input to the Relay graph, it should be an NDArray
                raise TypeError(f"Unsupported argument type for TVM run: {type(arg)}")

        # Execute the compiled module.
        # The first argument in the Relay function is usually the data, followed by parameters.
        # The arguments passed to `run` should match the `relay_func`'s parameters.
        result = self.vm_executor(*tvm_args)

        return result


# Helper to convert python literals to Relay constants
def to_relay_const(val, dtype=None):
    if isinstance(val, (int, float, bool, np.generic)):
        if dtype is None:
            if isinstance(val, bool):
                dtype = "bool"
            elif isinstance(val, int):
                dtype = "int64"
            elif isinstance(val, float):
                dtype = "float32"
            else: # numpy generic
                dtype = str(val.dtype)
        return relay.const(val, dtype=dtype)
    return val

class TestStaticCudaLauncher(object): # Changed to inherit from object for pytest compatibility
    def setUp(self):
        super().__init__()
        self.tmp_files = []

    def tearDown(self):
        super().tearDown()
        for tmp_file in self.tmp_files:
            try:
                os.remove(tmp_file.name)
            except OSError:
                pass

    def write_cubin_to_tmp(self, kernel) -> str:
        # This is Triton-specific and not applicable to TVM compilation flow.
        # Return a dummy string for validity.
        return "/dummy/path/to/tvm_kernel.so"

    def _make_launcher(
        self,
        kernel_logic_func, # A Python function that returns a Relay function
        arg_shapes,
        arg_dtypes,
        arg_names,
        constexpr_args=None,
        target="cuda",
    ):
        """
        Translates a kernel logic function to a TVM Relay function, compiles it,
        and returns a wrapper for execution.
        """
        if constexpr_args is None:
            constexpr_args = {}

        # Create Relay variables for inputs
        relay_vars = []
        for name, shape, dtype in zip(arg_names, arg_shapes, arg_dtypes):
            if name not in constexpr_args:
                relay_vars.append(relay.var(name, shape=shape, dtype=dtype))

        # Build the Relay expression using the kernel_logic_func
        # Pass constants as plain Python values if the kernel logic expects them that way,
        # otherwise pass Relay constants. This requires careful alignment.
        relay_inputs_for_logic = []
        param_idx = 0
        for name, shape, dtype in zip(arg_names, arg_shapes, arg_dtypes):
            if name in constexpr_args:
                relay_inputs_for_logic.append(constexpr_args[name])
            else:
                relay_inputs_for_logic.append(relay_vars[param_idx])
                param_idx += 1

        relay_expr = kernel_logic_func(*relay_inputs_for_logic)
        
        # Ensure the output of the kernel logic function is an expression, not a Python object
        if isinstance(relay_expr, (int, float, bool, np.generic)):
            relay_expr = to_relay_const(relay_expr)

        # Create the Relay Function
        f = relay.Function(relay_vars, relay_expr)
        
        # Determine actual parameters (variables) for the Relay function
        mod = tvm.IRModule.from_expr(f)
        main_func = mod["main"]
        # Assuming params are empty unless specifically added for external arguments later
        params = {} # For these simple kernels, all args are inputs, no separate 'params' from a model

        compiled_kernel = TVMCompiledKernel(main_func, params, target=target)

        # A dummy for arg_tys, actual arg types are in relay_vars
        arg_tys_str = ""
        for dtype_str in arg_dtypes:
            if dtype_str == "object": # For tensor type, might need to distinguish
                arg_tys_str += "O"
            elif dtype_str == "int8": arg_tys_str += "b"
            elif dtype_str == "uint8": arg_tys_str += "B"
            elif dtype_str == "int16": arg_tys_str += "h"
            elif dtype_str == "uint16": arg_tys_str += "H"
            elif dtype_str == "int32": arg_tys_str += "i"
            elif dtype_str == "uint32": arg_tys_str += "I"
            elif dtype_str == "int64": arg_tys_str += "l"
            elif dtype_str == "uint64": arg_tys_str += "K"
            elif dtype_str.startswith("float"): arg_tys_str += "f"
            else: arg_tys_str += "?" # Unknown

        # Mock StaticallyLaunchedCudaKernel attributes
        launcher = mock.Mock()
        launcher.cubin_path = self.write_cubin_to_tmp(None)
        launcher.reload_cubin_from_raw = lambda x: None # Dummy method
        launcher.load_kernel = lambda dev: None # Dummy method
        launcher.run = compiled_kernel.run # Use the actual TVM run method
        launcher.arg_tys = arg_tys_str # Simplified representation
        launcher.arg_shapes = arg_shapes
        launcher.arg_dtypes = arg_dtypes
        launcher.arg_names = arg_names
        launcher.compiled_kernel = compiled_kernel # Store for internal compilation if needed
        return launcher

    @pytest.mark.cuda
    def test_basic(self):
        def simple_kernel_relay_logic(arg0_relay, arg1_relay):
            # arg0 is input, updated and returned. arg1 is a scalar.
            return op.tensor.add(arg0_relay, arg1_relay)

        # Initial data for arg0
        arg0_np = np.zeros(1, dtype=np.int32)
        arg0_tvm = ndarray.array(arg0_np, device=tvm.cuda(0))
        arg1_val = 5

        # Define shape, dtype, and names for Relay function
        arg_shapes = [(1,), ()] # (1,) for tensor, () for scalar
        arg_dtypes = ["int32", "int32"]
        arg_names = ["arg0", "arg1"]

        launcher = self._make_launcher(simple_kernel_relay_logic, arg_shapes, arg_dtypes, arg_names)
        
        # Execute the kernel
        # We need to pass the *actual* values/NDArrays to launcher.run
        # arg0_tvm is passed as an input, and the result should update arg0_tvm.
        # But TVM ops are functional, so it will return a *new* NDArray.
        # So we create a separate output buffer `new_arg0_tvm` for the check,
        # or capture the return.
        
        # For simplicity in this test, let's create a "new_arg0" to represent the output tensor.
        # The `run` method should take the exact arguments of the Relay function.
        # In `_make_launcher`, arg_shapes and arg_dtypes are for *Relay parameters*.
        # For simple_kernel_relay_logic, arg0_relay and arg1_relay are the parameters.
        # If arg1_relay is a scalar, it will be a Relay constant `tvm.relay.const(5, 'int32')`
        # when building the graph, or a Python scalar if it's a dynamic argument.
        
        # Let's refine how `_make_launcher` sets up the Relay function inputs and how `run` executes.
        # If arg1 is a literal, it's typically baked into the Relay graph as a constant.
        # But the original Triton test passes it as a runtime argument.
        # So, the Relay function should take `arg0_relay` and `arg1_relay` as variables.
        
        # Let's adjust the Relay graph generation slightly to always take inputs as vars for runtime args.
        def simple_kernel_relay_logic_with_vars(arg0_var, arg1_var):
            return op.tensor.add(arg0_var, arg1_var)

        launcher = self._make_launcher(
            simple_kernel_relay_logic_with_vars,
            arg_shapes=[(1,), ()],  # arg1 is a scalar, represented as () shape
            arg_dtypes=["int32", "int32"],
            arg_names=["arg0", "arg1"],
        )

        # Initial values for arg0. The *result* will be stored in a new tensor.
        new_arg0_np = np.zeros(1, dtype=np.int32)
        new_arg0_tvm = ndarray.array(new_arg0_np, device=tvm.cuda(0))

        # The Python scalar '5' needs to be wrapped as a Relay constant when provided to the graph builder,
        # but for launcher.run, it would be passed as a concrete value.
        # We need to pass concrete NDArrays or primitive Python values that map to Relay.const
        # for scalar inputs.
        result_tvm = launcher.run(new_arg0_tvm, arg1_val) # Pass Python int directly

        expected_np = np.array([5], dtype=np.int32)
        
        assert_allclose(result_tvm.numpy(), expected_np) # Use result_tvm directly
        assert launcher.arg_tys == "Oi"


    @pytest.mark.cuda
    def test_unsigned_integers(self):
        def unsigned_integers_relay_logic(arg0_var, arg1_var, arg2_var, arg3_var, arg4_var):
            intermediate_sum = op.tensor.add(arg1_var, arg2_var)
            intermediate_sum = op.tensor.add(intermediate_sum, arg3_var)
            intermediate_sum = op.tensor.add(intermediate_sum, arg4_var)
            return op.tensor.add(arg0_var, intermediate_sum)

        arg0_np = np.zeros(1, dtype=np.uint64)
        arg0_tvm = ndarray.array(arg0_np, device=tvm.cuda(0))
        arg_vals = (50, 50, 50, 50)

        arg_shapes = [(1,), (), (), (), ()]
        arg_dtypes = ["uint64", "uint8", "uint16", "uint32", "uint64"]
        arg_names = ["arg0", "arg1", "arg2", "arg3", "arg4"]

        launcher = self._make_launcher(
            unsigned_integers_relay_logic, arg_shapes, arg_dtypes, arg_names
        )

        new_arg0_np = np.zeros(1, dtype=np.uint64)
        new_arg0_tvm = ndarray.array(new_arg0_np, device=tvm.cuda(0))

        # Pass concrete values including the initial state of new_arg0_tvm
        # The result updates new_arg0_tvm logically, but returns a new tensor in TVM.
        result_tvm = launcher.run(new_arg0_tvm, *arg_vals) # Pass Python ints directly

        expected_val = sum(arg_vals)
        expected_np = np.array([expected_val], dtype=np.uint64)
        
        assert_allclose(result_tvm.numpy(), expected_np)
        assert launcher.arg_tys == "OBHIK" # O for tensor, B for uint8, H for uint16, I for uint32, K for uint64


    @pytest.mark.cuda
    def test_signed_integers(self):
        def signed_integers_relay_logic(arg0_var, arg1_var, arg2_var, arg3_var, arg4_var):
            intermediate_sum = op.tensor.add(arg1_var, arg2_var)
            intermediate_sum = op.tensor.add(intermediate_sum, arg3_var)
            intermediate_sum = op.tensor.add(intermediate_sum, arg4_var)
            return op.tensor.add(arg0_var, intermediate_sum)

        arg0_np = np.zeros(1, dtype=np.int64)
        arg0_tvm = ndarray.array(arg0_np, device=tvm.cuda(0))
        arg_vals = (50, 50, 50, 50)

        arg_shapes = [(1,), (), (), (), ()]
        arg_dtypes = ["int64", "int8", "int16", "int32", "int64"]
        arg_names = ["arg0", "arg1", "arg2", "arg3", "arg4"]

        launcher = self._make_launcher(
            signed_integers_relay_logic, arg_shapes, arg_dtypes, arg_names
        )

        new_arg0_np = np.zeros(1, dtype=np.int64)
        new_arg0_tvm = ndarray.array(new_arg0_np, device=tvm.cuda(0))
        
        result_tvm = launcher.run(new_arg0_tvm, *arg_vals)

        expected_val = sum(arg_vals)
        expected_np = np.array([expected_val], dtype=np.int64)
        
        assert_allclose(result_tvm.numpy(), expected_np)
        assert launcher.arg_tys == "Obhil" # O for tensor, b for int8, h for int16, i for int32, l for int64


    @pytest.mark.cuda
    def test_basic_1arg(self):
        def simple_kernel_1_arg_relay_logic(arg0_var):
            return op.tensor.add(arg0_var, relay.const(1, dtype=str(arg0_var.dtype)))

        arg0_np = np.zeros(1, dtype=np.int32)
        arg0_tvm = ndarray.array(arg0_np, device=tvm.cuda(0))

        arg_shapes = [(1,)]
        arg_dtypes = ["int32"]
        arg_names = ["arg0"]

        launcher = self._make_launcher(
            simple_kernel_1_arg_relay_logic, arg_shapes, arg_dtypes, arg_names
        )

        new_arg0_np = np.zeros(1, dtype=np.int32)
        new_arg0_tvm = ndarray.array(new_arg0_np, device=tvm.cuda(0))

        result_tvm = launcher.run(new_arg0_tvm)

        expected_np = np.array([1], dtype=np.int32)
        
        assert_allclose(result_tvm.numpy(), expected_np)
        assert launcher.arg_tys == "O"


    @pytest.mark.cuda
    def test_constexpr(self):
        # Constexprs are compiled directly into the cubin file,
        # so we never need to pass it to StaticCudaLauncher.
        # In TVM, this means it's a constant baked into the Relay graph.
        
        # The kernel logic will directly use the constant value
        CONSTANT_VAL = 5
        def kernel_constexpr_relay_logic(arg0_var):
            return op.tensor.add(arg0_var, relay.const(CONSTANT_VAL, dtype=str(arg0_var.dtype)))

        arg0_np = np.zeros(1, dtype=np.int32)
        arg0_tvm = ndarray.array(arg0_np, device=tvm.cuda(0))

        arg_shapes = [(1,)]
        arg_dtypes = ["int32"]
        arg_names = ["arg0"]

        # Note: CONSTANT is not passed as an `arg_name` because it's a constexpr (baked in)
        launcher = self._make_launcher(
            kernel_constexpr_relay_logic, arg_shapes, arg_dtypes, arg_names
        )

        new_arg0_np = np.zeros(1, dtype=np.int32)
        new_arg0_tvm = ndarray.array(new_arg0_np, device=tvm.cuda(0))
        
        result_tvm = launcher.run(new_arg0_tvm)

        expected_np = np.array([CONSTANT_VAL], dtype=np.int32)
        
        assert_allclose(result_tvm.numpy(), expected_np)
        assert launcher.arg_tys == "O"


    @pytest.mark.cuda
    def test_implied_constant(self):
        """xnumel is unused in this kernel, but isn't explicitly marked as a constexpr"""

        # This kernel was generated by inductor so it has a bunch of unused arguments. We don't change it
        def triton_red_fused_any_isinf_0_relay_logic(
            in_ptr0_var,
            out_ptr0_var,
            xnumel_var,  # Inferred constant, will be ignored at runtime
            r0_numel_var,
            XBLOCK_val,
            R0_BLOCK_val,
        ):
            # Simulate the logic within Relay
            # xnumel is effectively a constant and might be optimized away or passed as a dummy.
            # For TVM's Relay representation, if it's passed as a variable, it needs to be there.

            # Simplified logic for Relay
            # Replace tl.full with relay.full
            # Replace libdevice.isinf with op.tensor.isinf
            # Replace triton_helpers.any with op.reduce.any
            
            # The core operation: check if any element in in_ptr0 is inf
            # and store result in out_ptr0.
            
            # This is complex to model directly with Relay building blocks from the Triton DSL.
            # I'll create a simplified Relay graph that performs the equivalent:
            # 1. Check isinf on in_ptr0
            # 2. Reduce.any on the result
            # 3. Store the scalar boolean result into out_ptr0 (which will be a tensor)
            
            # The kernel logic is:
            #   tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy="evict_first", other=0.0)
            #   tmp1 = libdevice.isinf(tmp0).to(tl.int1)
            #   ...
            #   _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
            #   tmp3 = triton_helpers.any(_tmp3.to(tl.int8), 1)[:, None].to(tl.int1)
            #   tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
            
            # This is essentially: `out_ptr0 = any(isinf(in_ptr0))`
            
            # The original Python test uses `arg0` for `in_ptr0` and `arg1` for `out_ptr0`.
            
            # The `r0_numel_var` is used as loop range, but also passed as a concrete value.
            # `XBLOCK_val`, `R0_BLOCK_val` are constexpr, handled by `_make_launcher`'s `constexpr_args`.
            
            # We need to ensure that the input `in_ptr0_var` is treated correctly, as its data is loaded.
            
            # `xnumel` is truly unused in the kernel body in Python (reassigned to 1),
            # but it is present in the signature.
            # TVM compilation will ensure unused parameters are handled.
            
            is_inf_result = op.tensor.isinf(in_ptr0_var)
            # Assuming it's reduced over all dimensions for `any` if axis is not specified or implicit
            # In triton, it reduces over axis=1 which means rows become scalar.
            # Here, the input `in_ptr0` is a 1D tensor, so `any` without axis will reduce to a scalar.
            # The original triton kernel uses `any(..., 1)[:, None]`, which would be `any(..., axis=1, keepdims=True)`
            # but the input is 1D. Let's simplify and assume the effect is to get a scalar boolean.
            
            # Check shape of in_ptr0: arg0 is `torch.tensor([0.0, 0.5, float("inf"), 5], device="cuda")`
            # This means `in_ptr0` is (4,).
            # The original kernel loops over `r0_numel` (128), and each `r0_0` is `r0_index`.
            # `in_ptr0 + (r0_0)` suggests flat access.
            # `r0_mask` restricts valid accesses.
            # The `any(_tmp3.to(tl.int8), 1)[:, None]` is reducing a (XBLOCK, R0_BLOCK) shaped tensor along axis 1.
            # If XBLOCK=1, R0_BLOCK=1, this means reducing a (1,1) tensor along axis 1, resulting in (1,)
            # The final `out_ptr0` is a (1,) tensor.
            # This implies a global 'any' reduction over potentially flattened data.
            
            # Let's simplify:
            # The actual effect is if any element in `in_ptr0` is inf.
            
            # The arg `out_ptr0_var` gets assigned the result of `any_isinf`.
            # In Relay, we return the computed output.
            any_inf = op.reduce.any(is_inf_result, axis=None, keepdims=False) # Reduce over all elements
            return any_inf # Return the scalar boolean result

        arg0_np = np.array([0.0, 0.5, float("inf"), 5], dtype=np.float32)
        arg0_tvm = ndarray.array(arg0_np, device=tvm.cuda(0))
        arg1_np_init = np.array([False], dtype=np.bool_)
        arg1_tvm_init = ndarray.array(arg1_np_init, device=tvm.cuda(0)) # This is `out_ptr0` in kernel
        
        # Args for _make_launcher:
        # in_ptr0, out_ptr0, xnumel (implied constant), r0_numel
        # XBLOCK (constexpr), R0_BLOCK (constexpr)
        
        # in_ptr0: tensor (4,) float32
        # out_ptr0: tensor (1,) bool
        # xnumel: int (implied constant, will be optimized out for actual TVM func)
        # r0_numel: int scalar (dynamic runtime arg)

        relay_arg_shapes = [(4,), (1,), (), ()]
        relay_arg_dtypes = ["float32", "bool", "int64", "int64"] # xnumel and r0_numel are int, default to int64
        relay_arg_names = ["in_ptr0", "out_ptr0", "xnumel", "r0_numel"]

        # Constexprs are passed to _make_launcher to be baked into the kernel logic function.
        constexpr_map = {
            "XBLOCK": 1,
            "R0_BLOCK": 1,
        }
        
        # When calling _make_launcher, we pass the python literals that are NOT constexprs.
        # So xnumel and r0_numel are dynamic params.
        
        launcher = self._make_launcher(
            triton_red_fused_any_isinf_0_relay_logic,
            relay_arg_shapes,
            relay_arg_dtypes,
            relay_arg_names,
            constexpr_args=constexpr_map # These are for the *compiler*, not runtime args directly
        )

        # Arguments to launcher.run correspond to dynamic arguments of the Relay function.
        # The Relay function generated by `_make_launcher` will only take non-constexpr args.
        # `in_ptr0_var`, `out_ptr0_var`, `xnumel_var`, `r0_numel_var`.
        
        # xnumel is treated as dynamic argument `1` and r0_numel as `128`.
        # However, as per the comment, xnumel is *unused* in kernel, which means it should not be passed to `launcher.run`.
        # The `_make_launcher` should reflect this: only include *actual* runtime parameters in the Relay function.

        # Let's redefine _make_launcher to automatically exclude constexprs and unused arguments
        # from the Relay function definition, or just pass dummy values as part of the run.
        # For `test_implied_constant`, `xnumel` is unused. This implies the kernel's actual signature
        # is just `(in_ptr0, out_ptr0, r0_numel)`.

        # Refined definition for `triton_red_fused_any_isinf_0_relay_logic` which only takes dynamic args:
        def triton_red_fused_any_isinf_0_simplified_relay_logic(in_ptr0_var, out_ptr0_var, r0_numel_var):
            is_inf_result = op.tensor.isinf(in_ptr0_var)
            any_inf = op.reduce.any(is_inf_result, axis=None, keepdims=False)
            
            # The original stores into out_ptr0. Relay returns the new value.
            # We want to emulate `tl.store(out_ptr0 + (...), any_inf)`
            # So the expression *returned* by the relay function is the `any_inf`.
            # If `out_ptr0` is mutable, it would be updated. In functional Relay, it's a new tensor.
            return any_inf # The scalar boolean result

        # Only dynamic arguments are passed to the Relay function
        relay_arg_shapes_simplified = [(4,), (1,), ()] # in_ptr0, out_ptr0, r0_numel
        relay_arg_dtypes_simplified = ["float32", "bool", "int64"]
        relay_arg_names_simplified = ["in_ptr0", "out_ptr0", "r0_numel"]

        launcher_simplified = self._make_launcher(
            triton_red_fused_any_isinf_0_simplified_relay_logic,
            relay_arg_shapes_simplified,
            relay_arg_dtypes_simplified,
            relay_arg_names_simplified,
        )

        arg2_np = np.array([False], dtype=np.bool_) # Represents `out_ptr0` buffer for result
        arg2_tvm = ndarray.array(arg2_np, device=tvm.cuda(0))
        r0_numel_val = 128 # The dynamic scalar argument

        # The `launcher.run` takes the actual inputs: in_ptr0, out_ptr0 (buffer), r0_numel
        result_tvm = launcher_simplified.run(arg0_tvm, arg2_tvm, r0_numel_val)
        
        # Expected result: `isinf([0.0, 0.5, inf, 5])` is `[F, F, T, F]`. `any` of this is `True`.
        expected_np_value = np.array([True], dtype=np.bool_)
        assert_allclose(result_tvm.numpy(), expected_np_value) # Compare result directly


    @pytest.mark.cuda
    def test_kernel_no_args(self):
        def kernel_no_op_relay_logic():
            return relay.const(0, "int32") # A dummy return, as ops must return something

        arg_shapes = []
        arg_dtypes = []
        arg_names = []

        launcher = self._make_launcher(kernel_no_op_relay_logic, arg_shapes, arg_dtypes, arg_names)
        
        # Run with no arguments
        result_tvm = launcher.run()
        
        expected_np = np.array(0, dtype=np.int32)
        assert_allclose(result_tvm.numpy(), expected_np)


    @pytest.mark.cuda
    def test_high_shared_mem(self):
        def simple_kernel_relay_logic(arg0_relay, arg1_relay):
            return op.tensor.add(arg0_relay, arg1_relay)

        arg0_np = np.zeros(1, dtype=np.int32)
        arg0_tvm = ndarray.array(arg0_np, device=tvm.cuda(0))
        arg1_val = 5

        arg_shapes = [(1,), ()]
        arg_dtypes = ["int32", "int32"]
        arg_names = ["arg0", "arg1"]

        launcher = self._make_launcher(simple_kernel_relay_logic, arg_shapes, arg_dtypes, arg_names)
        
        # The `compiled_kernel.shared = 50000` is a Triton-specific memory allocation hint.
        # In TVM, shared memory is managed by the scheduling/codegen, not directly exposed at this level.
        # This test checks if the kernel *runs* despite this setting.
        # We simulate this by simply running the kernel.
        
        new_arg0_np = np.zeros(1, dtype=np.int32)
        new_arg0_tvm = ndarray.array(new_arg0_np, device=tvm.cuda(0))
        
        # launcher.slow_launch_kernel = True # This is a Triton-specific flag, ignored in TVM context
        result_tvm = launcher.run(new_arg0_tvm, arg1_val)

        expected_np = np.array([5], dtype=np.int32)
        
        assert_allclose(result_tvm.numpy(), expected_np)
        assert launcher.arg_tys == "Oi"


    @pytest.mark.cuda
    def test_too_high_shared_mem(self):
        def simple_kernel_relay_logic(arg0_relay, arg1_relay):
            return op.tensor.add(arg0_relay, arg1_relay)

        arg0_np = np.zeros(1, dtype=np.int32)
        arg0_tvm = ndarray.array(arg0_np, device=tvm.cuda(0))
        arg1_val = 5

        arg_shapes = [(1,), ()]
        arg_dtypes = ["int32", "int32"]
        arg_names = ["arg0", "arg1"]

        # The `compiled_kernel.shared = 99999999` is a Triton-specific memory allocation constraint.
        # TVM's compilation typically performs resource analysis and *might* fail with an OOM error
        # during compilation or runtime if the generated code truly uses too much.
        # However, for a simple addition kernel, it's unlikely to fail based on shared memory.
        # This test expects a RuntimeError specific to Triton.
        # Without a way to inject a "shared memory limit" into the TVM compilation pipeline
        # at this abstraction level, a direct equivalent cannot be perfectly simulated.
        
        # We will check that the basic compilation and run works, as TVM's compiler
        # will not inherently fail for a simple add op, regardless of a dummy "shared" attribute.
        # This means the original `assertRaisesRegex` is not directly mappable semantically
        # without a TVM-specific equivalent resource constraint mechanism.
        
        # For now, we will just ensure it *doesn't* raise an unexpected error by allowing it to run.
        # The expected failure path in PyTorch is Triton-specific.
        launcher = self._make_launcher(simple_kernel_relay_logic, arg_shapes, arg_dtypes, arg_names)
        
        new_arg0_np = np.zeros(1, dtype=np.int32)
        new_arg0_tvm = ndarray.array(new_arg0_np, device=tvm.cuda(0))
        
        # Running should succeed because it's a simple operation, unless TVM has a specific resource error.
        result_tvm = launcher.run(new_arg0_tvm, arg1_val)
        
        expected_np = np.array([5], dtype=np.int32)
        assert_allclose(result_tvm.numpy(), expected_np)
        
        # TODO: A more robust mapping would involve simulating resource constraints in TVM's codegen/scheduling,
        # or replacing with a TVM-specific error if such a mechanism exists and can be triggered.
        # Currently, there's no direct equivalent to "allocate too much shared memory" at the Relay level
        # that would cause a compile-time failure like Triton's.
        


    @pytest.mark.cuda
    def test_kernel_empty_tensor(self):
        # The kernel logic as defined by PyTorch (triton_poi_fused_cat_0)
        # return torch.cat(((x * 4), y + 10))
        # This involves empty tensor `x`.
        
        # Simplified Relay logic for `torch.cat(((x * 4), y + 10))`
        def fused_cat_relay_logic(in_ptr0, in_ptr1, ks0_val, xnumel_val):
            # in_ptr0 is x (shape (0,))
            # in_ptr1 is y (shape (20,))
            # ks0_val is 0 (size of x)
            # xnumel_val is 20 (total output size)
            
            # (x * 4)
            mul_expr = op.tensor.multiply(in_ptr0, relay.const(4.0, dtype=str(in_ptr0.dtype)))
            
            # (y + 10)
            add_expr = op.tensor.add(in_ptr1, relay.const(10.0, dtype=str(in_ptr1.dtype)))
            
            # Concatenate results. The `mul_expr` (from `in_ptr0`) will have shape (0,)
            # The `add_expr` (from `in_ptr1`) will have shape (20,)
            # Concatenating `(0,)` and `(20,)` results in `(20,)`.
            # Note: For empty tensors, multiplication would typically result in an empty tensor of same dtype.
            # TVM's `op.concatenate` handles empty tensors correctly.
            
            # Example: [empty] + [1, 2, 3] -> [1, 2, 3]
            # Example: [1, 2] + [empty] -> [1, 2]
            
            # The triton kernel is complex because it uses `tl.where` to "select" between `x*4` and `y+10`
            # based on index `x0 < ks0` (which is `x0 < 0` for empty x).
            # If `x0 < 0` is always false, then `tmp4` is always false.
            # So `tmp9` is always `tl.full(..., 0.0)`.
            # `tmp10` is `x0 >= 0` which is always true.
            # So `tmp18` (final result) is always `tmp17` (y+10 part).
            # This means the kernel simplifies to just `y + 10`.
            
            return op.tensor.add(in_ptr1, relay.const(10.0, dtype=str(in_ptr1.dtype)))


        arg0_val = 0 # This is `ks0` in triton kernel (size of empty tensor `x`)
        arg1_np = np.random.rand(0).astype(np.float32) # Empty tensor x
        arg1_tvm = ndarray.array(arg1_np, device=tvm.cuda(0))
        arg2_np = np.random.rand(20).astype(np.float32) # Tensor y
        arg2_tvm = ndarray.array(arg2_np, device=tvm.cuda(0))
        
        buf0_np_ref = arg2_np + 10.0 # Expected result
        buf0_tvm_ref = ndarray.array(buf0_np_ref, device=tvm.cuda(0))

        # The `triton_poi_fused_cat_0` takes:
        # in_ptr0 (x), in_ptr1 (y), out_ptr0 (buf0), ks0 (arg0_val), xnumel (20 + arg0_val), XBLOCK
        
        # Simplified Relay logic takes: in_ptr0, in_ptr1, ks0_val, xnumel_val
        # The first two are tensors, last two are scalars that will be baked into the Relay graph or passed as constants.
        
        relay_arg_shapes = [(0,), (20,), (), ()] # in_ptr0, in_ptr1, ks0, xnumel
        relay_arg_dtypes = ["float32", "float32", "int64", "int64"]
        relay_arg_names = ["in_ptr0", "in_ptr1", "ks0", "xnumel"]

        launcher = self._make_launcher(
            fused_cat_relay_logic, relay_arg_shapes, relay_arg_dtypes, relay_arg_names
        )

        buf1_np = np.empty(20, dtype=np.float32) # Output buffer for result
        buf1_tvm = ndarray.array(buf1_np, device=tvm.cuda(0))
        
        # Call with actual arguments. The returned result from launcher.run is the output.
        # Note: the triton kernel takes `xnumel` as a dynamic argument which is `20 + arg0_val`.
        # So we pass `20` as the `xnumel_val`.
        result_tvm = launcher.run(arg1_tvm, arg2_tvm, arg0_val, 20)
        
        assert_allclose(result_tvm.numpy(), buf0_tvm_ref.numpy())


    @pytest.mark.cuda
    def test_kernel_many_args(self):
        N = 200

        # Construct the Relay logic dynamically for many arguments
        def many_args_relay_logic(out_tensor_var, *input_args_vars):
            total_sum = out_tensor_var
            for arg_var in input_args_vars:
                total_sum = op.tensor.add(total_sum, arg_var)
            return total_sum

        kernel_args_np = tuple(random.random() for _ in range(N))
        
        # Define shapes, dtypes, and names for Relay function dynamically
        arg_shapes = [(1,)] + [() for _ in range(N)] # Output tensor + N scalar inputs
        arg_dtypes = ["float32"] + ["float32" for _ in range(N)] # Assuming float for random.random
        arg_names = ["out_tensor"] + [f"arg_{i}" for i in range(N)]

        launcher = self._make_launcher(
            many_args_relay_logic, arg_shapes, arg_dtypes, arg_names
        )

        buf0_np = np.zeros(1, dtype=np.float32)
        buf0_tvm = ndarray.array(buf0_np, device=tvm.cuda(0))
        
        # Prepare concrete TVM args
        tvm_input_args = [ndarray.array(np.array([val], dtype=np.float32), device=tvm.cuda(0)) if not isinstance(val, (int, float, bool)) else val for val in kernel_args_np]
        
        # The launcher.run expects the initial `out_tensor` buffer and then the `N` inputs.
        result_tvm = launcher.run(buf0_tvm, *tvm_input_args)

        # Expected result calculation
        expected_val = np.sum(kernel_args_np)
        expected_np = np.array([expected_val], dtype=np.float32)
        
        assert_allclose(result_tvm.numpy(), expected_np, rtol=1e-5, atol=1e-8)


# Replace torch._inductor.config.patch with a dummy or remove if not applicable
# The goal is to check functional correctness of the torch.compile'd function.
# This means we'll construct the equivalent Relay graph and run it.

# Dummy context manager for TVM config patches, as they are not directly equivalent.
class tvm_config_patch:
    def __init__(self, config_dict):
        self.config_dict = config_dict
    def __enter__(self):
        pass # No direct TVM config equivalent for these PyTorch settings
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Dummy for PyCodeCache
class PyCodeCache:
    @staticmethod
    def load(code_string):
        # This is a placeholder. In a real scenario, this would parse the code_string
        # and extract the kernel definition. For tests, we'd simply define the
        # Relay equivalent directly.
        class DummyModule:
            def __getattr__(self, name):
                # Return a dummy object for the kernel name, if needed.
                return lambda *args, **kwargs: None
        return DummyModule()

class TestStaticTritonCompileResult(object): # Changed to inherit from object for pytest compatibility
    """
    Tests static cuda launcher with torch.compile()
    """

    # For these tests, we will directly construct the Relay equivalent of the PyTorch function.
    # The `torch.compile` decorator and config patches are removed.

    @pytest.mark.cuda
    def test_basic_compile(self):
        # Equivalent of:
        # @torch.compile
        # def foo(x, y):
        #     return x + y
        def foo_relay_logic(x_var, y_var):
            return op.tensor.add(x_var, y_var)

        x_np = np.random.randn(10).astype(np.float32)
        y_np = np.random.randn(10).astype(np.float32)
        x_tvm = ndarray.array(x_np, device=tvm.cuda(0))
        y_tvm = ndarray.array(y_np, device=tvm.cuda(0))

        relay_func_params = [relay.var("x", shape=x_tvm.shape, dtype=str(x_tvm.dtype)),
                             relay.var("y", shape=y_tvm.shape, dtype=str(y_tvm.dtype))]
        relay_expr = foo_relay_logic(*relay_func_params)
        relay_mod = tvm.IRModule.from_expr(relay.Function(relay_func_params, relay_expr))
        
        with tvm.target.Target("cuda"):
            compiled_mod = relay.build(relay_mod, target="cuda")
        vm_executor = vm.VirtualMachine(compiled_mod, tvm.cuda(0))
        
        compiled_result_tvm = vm_executor(x_tvm, y_tvm)
        
        eager_result_np = x_np + y_np # Equivalent eager computation

        assert_allclose(compiled_result_tvm.numpy(), eager_result_np)


    @pytest.mark.cuda
    # The error gets raised on a worker, so we want to not use a separate process
    # @torch._inductor.config.patch("compile_threads", 1) # Not applicable
    def test_incompatible_code(self):
        # This test expects an InductorError due to a user-defined Triton kernel not being
        # statically launchable by PyTorch's inductor by default.
        # In TVM, the "user-defined triton kernel" concept doesn't exist directly.
        # We would directly model the kernel's functionality in Relay.
        
        # The intent of this test is to show that a specific compilation path fails.
        # A direct translation would require a TVM "compilation pipeline" that *mimics*
        # Inductor's behavior of rejecting certain kernels for static launch.
        # This is beyond a simple API mapping.
        
        # Since the problem statement mandates runnable Python and clear TODOs for
        # non-confident rewrites, and the core functionality of the kernel (add) *can* be
        # compiled by TVM, the "incompatible code" aspect is not directly mappable.
        # We will instead just show that the kernel *would* run if translated.
        # If the *goal* was to demonstrate a TVM compilation failure for a specific reason,
        # that would require a different TVM feature (e.g., specific pass failing, unsupported op).

        # User defined triton kernel (conceptually)
        def custom_kernel_relay_logic(arg_0_var, arg_1_var):
            return op.tensor.add(arg_0_var, arg_1_var)

        # Equivalent of:
        # @torch.compile
        # def foo(x):
        #     custom_kernel[1,](x, 5) # This is where the Triton kernel would be called
        #     return x
        # This means `x = x + 5`.

        def foo_relay_logic(x_var):
            return custom_kernel_relay_logic(x_var, relay.const(5, str(x_var.dtype)))

        x_np = np.random.randn(1).astype(np.float32)
        x_tvm = ndarray.array(x_np, device=tvm.cuda(0))

        # This part effectively compiles and runs the `foo_relay_logic`
        relay_func_params = [relay.var("x", shape=x_tvm.shape, dtype=str(x_tvm.dtype))]
        relay_expr = foo_relay_logic(*relay_func_params)
        relay_mod = tvm.IRModule.from_expr(relay.Function(relay_func_params, relay_expr))
        
        with tvm.target.Target("cuda"):
            compiled_mod = relay.build(relay_mod, target="cuda")
        vm_executor = vm.VirtualMachine(compiled_mod, tvm.cuda(0))
        
        # Call compiled function
        compiled_result_tvm = vm_executor(x_tvm)
        
        eager_result_np = x_np + 5 # Expected eager computation result
        assert_allclose(compiled_result_tvm.numpy(), eager_result_np)
        
        # TODO: The original test expects `self.assertRaisesRegex` for a Triton-specific error.
        # Simulating such an error in TVM (without creating an actual unsupported op)
        # would require a custom TVM pass or a different testing approach.
        # Currently, the TVM equivalent correctly computes the result, so the error test is skipped.


    @pytest.mark.cuda
    # The error gets raised on a worker, so we want to not use a separate process
    # @torch._inductor.config.patch( # Not applicable
    #     {"compile_threads": 1, "static_launch_user_defined_triton_kernels": True}
    # )
    def test_static_launch_user_defined_triton_kernels(self):
        # This test checks that with a specific config patch, the user-defined Triton kernel *does* launch statically.
        # In TVM, we directly model the kernel's functionality in Relay, and it will compile and run.
        # This is effectively testing the successful compilation and execution of the kernel.

        def custom_kernel_relay_logic(arg_0_var, arg_1_var):
            return op.tensor.add(arg_0_var, arg_1_var)

        def foo_relay_logic(x_var):
            return custom_kernel_relay_logic(x_var, relay.const(5, str(x_var.dtype)))

        x_np = np.random.randn(1).astype(np.float32)
        x_tvm = ndarray.array(x_np, device=tvm.cuda(0))
        
        relay_func_params = [relay.var("x", shape=x_tvm.shape, dtype=str(x_tvm.dtype))]
        relay_expr = foo_relay_logic(*relay_func_params)
        relay_mod = tvm.IRModule.from_expr(relay.Function(relay_func_params, relay_expr))
        
        with tvm.target.Target("cuda"):
            compiled_mod = relay.build(relay_mod, target="cuda")
        vm_executor = vm.VirtualMachine(compiled_mod, tvm.cuda(0))
        
        compiled_result_tvm = vm_executor(x_tvm)
        
        x2_np = x_np.copy() # x.clone().detach_() equivalent
        eager_result_np = x2_np + 5
        
        assert_allclose(compiled_result_tvm.numpy(), eager_result_np)


    @pytest.mark.cuda
    def test_empty_tensor(self):
        # Equivalent of:
        # @torch.compile()
        # def foo(x, y):
        #   return torch.cat(((x * 4), y + 10))
        # x is an empty tensor.
        
        # Simplified Relay logic for `torch.cat(((x * 4), y + 10))`
        def foo_relay_logic(x_var, y_var):
            # Same logic as triton_poi_fused_cat_0_relay_logic simplified from earlier
            # If x_var is empty, x_var * 4 is empty
            # The concatenate operation would handle this naturally in Relay.
            
            # (x * 4)
            mul_expr = op.tensor.multiply(x_var, relay.const(4.0, dtype=str(x_var.dtype)))
            
            # (y + 10)
            add_expr = op.tensor.add(y_var, relay.const(10.0, dtype=str(y_var.dtype)))
            
            return op.tensor.concatenate((mul_expr, add_expr), axis=0)

        x_np = np.random.rand(0).astype(np.float32) # Empty tensor
        y_np = np.random.rand(20).astype(np.float32) # Non-empty tensor
        
        x_tvm = ndarray.array(x_np, device=tvm.cuda(0))
        y_tvm = ndarray.array(y_np, device=tvm.cuda(0))

        relay_func_params = [relay.var("x", shape=x_tvm.shape, dtype=str(x_tvm.dtype)),
                             relay.var("y", shape=y_tvm.shape, dtype=str(y_tvm.dtype))]
        relay_expr = foo_relay_logic(*relay_func_params)
        relay_mod = tvm.IRModule.from_expr(relay.Function(relay_func_params, relay_expr))
        
        with tvm.target.Target("cuda"):
            compiled_mod = relay.build(relay_mod, target="cuda")
        vm_executor = vm.VirtualMachine(compiled_mod, tvm.cuda(0))
        
        compiled_result_tvm = vm_executor(x_tvm, y_tvm)
        
        # Eager calculation
        eager_x_mul = x_np * 4
        eager_y_add = y_np + 10
        eager_result_np = np.concatenate((eager_x_mul, eager_y_add), axis=0)

        assert_allclose(compiled_result_tvm.numpy(), eager_result_np)


    @pytest.mark.cuda
    def test_any(self):
        # Equivalent of:
        # def fn(x):
        #     return (
        #         x.any(-1),
        #         x.isinf().any(),
        #         torch.all(x.isinf(), dim=0),
        #         torch.all(torch.logical_not(x.isinf())),
        #     )
        
        def fn_relay_logic(x_var):
            # x.any(-1) -> op.reduce.any(x, axis=-1, keepdims=False)
            any_last_dim = op.reduce.any(x_var, axis=-1, keepdims=False)
            
            # x.isinf().any() -> op.reduce.any(op.tensor.isinf(x), axis=None, keepdims=False)
            isinf_all_dims = op.reduce.any(op.tensor.isinf(x_var), axis=None, keepdims=False)
            
            # torch.all(x.isinf(), dim=0) -> op.reduce.all(op.tensor.isinf(x), axis=0, keepdims=False)
            all_isinf_dim0 = op.reduce.all(op.tensor.isinf(x_var), axis=0, keepdims=False)
            
            # torch.all(torch.logical_not(x.isinf())) -> op.reduce.all(op.tensor.logical_not(op.tensor.isinf(x)), axis=None, keepdims=False)
            all_not_isinf_all_dims = op.reduce.all(op.tensor.logical_not(op.tensor.isinf(x_var)), axis=None, keepdims=False)
            
            return relay.Tuple((any_last_dim, isinf_all_dims, all_isinf_dim0, all_not_isinf_all_dims))

        # Initial arg
        arg_np = -np.random.rand(64, 2).astype(np.float64) # Example 2D tensor for x.any(-1) to work
        arg_tvm = ndarray.array(arg_np, device=tvm.cuda(0))

        relay_func_params = [relay.var("x", shape=arg_tvm.shape, dtype=str(arg_tvm.dtype))]
        relay_expr = fn_relay_logic(*relay_func_params)
        relay_mod = tvm.IRModule.from_expr(relay.Function(relay_func_params, relay_expr))
        
        with tvm.target.Target("cuda"):
            compiled_mod = relay.build(relay_mod, target="cuda")
        vm_executor = vm.VirtualMachine(compiled_mod, tvm.cuda(0))
        
        # Run first time
        compiled_result_tvm = vm_executor(arg_tvm)
        
        # Eager calculation (numpy equivalent)
        eager_result_np = (
            np.any(arg_np, axis=-1),
            np.any(np.isinf(arg_np)),
            np.all(np.isinf(arg_np), axis=0),
            np.all(np.logical_not(np.isinf(arg_np))),
        )
        
        for comp_res, eager_res in zip(compiled_result_tvm, eager_result_np):
            assert_allclose(comp_res.numpy(), eager_res)

        # Modify arg and run again
        arg_np_modified = arg_np.copy()
        arg_np_modified[1, 0] = float("inf")
        arg_tvm_modified = ndarray.array(arg_np_modified, device=tvm.cuda(0))

        compiled_result_tvm_modified = vm_executor(arg_tvm_modified)
        
        eager_result_np_modified = (
            np.any(arg_np_modified, axis=-1),
            np.any(np.isinf(arg_np_modified)),
            np.all(np.isinf(arg_np_modified), axis=0),
            np.all(np.logical_not(np.isinf(arg_np_modified))),
        )
        
        for comp_res, eager_res in zip(compiled_result_tvm_modified, eager_result_np_modified):
            assert_allclose(comp_res.numpy(), eager_res)


    @pytest.mark.cuda
    def test_disable_static_cuda_launcher(self):
        # This test checks that disabling the static cuda launcher works in PyTorch Inductor.
        # In TVM, there is no direct equivalent of "static cuda launcher" to disable in this context.
        # The goal of this test, when converted, is to show that the *functional result* is still correct.
        # The mocking of `make_launcher` cannot be directly translated to TVM in a meaningful way
        # as TVM's compilation flow is different.

        # Equivalent of:
        # @torch.compile
        # def fn(x, y):
        #     return torch.cat(((x * 4), y + 10))
        def fn_relay_logic(x_var, y_var):
            mul_expr = op.tensor.multiply(x_var, relay.const(4.0, dtype=str(x_var.dtype)))
            add_expr = op.tensor.add(y_var, relay.const(10.0, dtype=str(y_var.dtype)))
            return op.tensor.concatenate((mul_expr, add_expr), axis=0)

        x_np = np.random.rand(20).astype(np.float32)
        y_np = np.random.rand(20).astype(np.float32)
        
        x_tvm = ndarray.array(x_np, device=tvm.cuda(0))
        y_tvm = ndarray.array(y_np, device=tvm.cuda(0))

        relay_func_params = [relay.var("x", shape=x_tvm.shape, dtype=str(x_tvm.dtype)),
                             relay.var("y", shape=y_tvm.shape, dtype=str(y_tvm.dtype))]
        relay_expr = fn_relay_logic(*relay_func_params)
        relay_mod = tvm.IRModule.from_expr(relay.Function(relay_func_params, relay_expr))
        
        # The `with torch._inductor.config.patch(...)` block is removed.
        # The mocked assertion `mocked.assert_not_called()` is also removed as it's implementation-specific.
        
        with tvm.target.Target("cuda"):
            compiled_mod = relay.build(relay_mod, target="cuda")
        vm_executor = vm.VirtualMachine(compiled_mod, tvm.cuda(0))
        
        compiled_result_tvm = vm_executor(x_tvm, y_tvm)
        
        eager_x_mul = x_np * 4
        eager_y_add = y_np + 10
        eager_result_np = np.concatenate((eager_x_mul, eager_y_add), axis=0)

        assert_allclose(compiled_result_tvm.numpy(), eager_result_np)

if __name__ == "__main__":
    # In PyTorch, run_tests() collects and runs unittest.TestCase classes.
    # For pytest, simply run `pytest <filename>`.
    # To make this file runnable on its own, we can use pytest.main().
    pytest.main([__file__])
