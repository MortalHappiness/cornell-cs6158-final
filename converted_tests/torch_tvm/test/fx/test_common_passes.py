import sys
import itertools
import numpy as np
import pytest

import tvm
import tvm.relay as relay
import tvm.testing
import tvm.topi as topi # Imported for potential future use or consistency, though not strictly used by ops here
from tvm.relay.op.random import threefry_key, uniform, normal # Imported for consistency, though not used by ops here
from tvm.transform import PassContext

# --- Helper functions for TVM Relay conversion and execution ---

def _get_tvm_device(device_str):
    if device_str == "cpu":
        return tvm.cpu(0)
    elif device_str == "cuda":
        return tvm.cuda(0)
    else:
        raise ValueError(f"Unsupported device: {device_str}")


# This class wraps TVM's EliminateCommonSubexpr pass to fit the test structure
class TVMCSEPass:
    def __call__(self, mod):
        with PassContext(opt_level=3): # Apply with optimization level that includes CSE
            return tvm.relay.transform.EliminateCommonSubexpr()(mod)


def make_relay_module_from_test_func(f_name, f_builder, sample_inputs_np, is_factory_test):
    """
    Converts a Python function `f_builder` (which constructs Relay expressions) into a tvm.IRModule.
    It uses `sample_inputs_np` to infer shapes and dtypes for Relay input variables.
    """
    input_vars = []
    
    # For factory tests, the last element in sample_inputs_np is the device string,
    # which is passed to the *original* PyTorch function, but is a dummy for Relay var definitions.
    pure_inputs_for_var_def = sample_inputs_np[:-1] if is_factory_test else sample_inputs_np

    for i, inp_np in enumerate(pure_inputs_for_var_def):
        var_name = f"input_{i}"
        relay_var = relay.var(var_name, shape=inp_np.shape, dtype=str(inp_np.dtype))
        input_vars.append(relay_var)

    # Call the Relay graph-building function with Relay variables.
    # The `device` argument for factory functions is a placeholder during graph construction.
    if is_factory_test:
        dummy_device_str = sample_inputs_np[-1] # This is the device string like "cpu" or "cuda"
        result_expr = f_builder(*input_vars, device=dummy_device_str)
    else:
        result_expr = f_builder(*input_vars)

    # Ensure the result is a tuple of expressions if it's a single expression
    if not isinstance(result_expr, (tuple, list)):
        result_expr = (result_expr,)
    
    # Ensure all elements in result_expr are actual Relay expressions
    final_results = []
    for expr in result_expr:
        if not isinstance(expr, relay.Expr):
            # If the test function f_builder directly returns Python primitives (e.g., from relay.const),
            # wrap them in a relay.Constant if they are numeric.
            if isinstance(expr, (int, float, bool, np.generic)):
                expr = relay.const(expr)
            else:
                raise TypeError(f"Function {f_name} returned non-Relay expression: {type(expr)}, value: {expr}")
        final_results.append(expr)

    relay_func = relay.Function(input_vars, relay.Tuple(final_results))
    mod = tvm.IRModule({f_name: relay_func})
    return mod


# --- Rewritten PyTorch test functions to return TVM Relay expressions ---
# These functions now take relay.Var inputs and return relay.Expr outputs.

def relay_FactoryFunctionCall(x, device): # `device` is a dummy argument for graph building
    # original: y = torch.full(x.shape, 3, device=device)
    # Assuming the fill value 3 would lead to a float32 tensor if x is float32 (from randn)
    y = relay.op.transform.full(relay.const(3.0, "float32"), shape=x.shape, dtype="float32")
    # original: z = torch.add(y, x)
    z = relay.op.tensor.add(y, x)
    return z


def relay_TorchTensorCall(x):
    # original: y = torch.tensor(3) defaults to int64 in PyTorch
    y_const = relay.const(3, "int64")
    # original: return x + y. If x is float32, y_const is promoted to float32.
    return relay.op.tensor.add(x, relay.op.cast(y_const, x.dtype))


def relay_TakeList(x):
    # original: z = torch.cat([x, x])
    z = relay.op.tensor.concatenate([x, x], axis=0)
    return z


def relay_ReturnList(x):
    # original: a = torch.arange(10).reshape(5, 2)
    # torch.arange(10) defaults to int64 in PyTorch
    a_range = relay.op.transform.arange(relay.const(0, "int64"), relay.const(10, "int64"), relay.const(1, "int64"), "int64")
    a = relay.op.transform.reshape(a_range, (5, 2))
    # original: z = torch.split(a, [1, 4])
    # The mapping table and PyTorch behavior for a list of sizes: (1, 4) means split into chunks of these sizes.
    z = relay.op.transform.split(a, (1, 4), axis=0)
    return z


def relay_Mutation(x):
    # original: y = x + 2
    y = relay.op.tensor.add(x, relay.const(2, x.dtype))
    # original: y.add_(1) -> functional equivalent: y = y + 1
    y_mut = relay.op.tensor.add(y, relay.const(1, y.dtype))
    # original: return x + y -> return x + y_mut
    return relay.op.tensor.add(x, y_mut)


def relay_MutationInput(x):
    # original: x.add_(1) -> functional equivalent: x_new = x + 1
    x_new = relay.op.tensor.add(x, relay.const(1, x.dtype))
    # original: y = x + 2 -> y = x_new + 2
    y = relay.op.tensor.add(x_new, relay.const(2, x_new.dtype))
    # original: return x + y -> return x_new + y
    return relay.op.tensor.add(x_new, y)


def relay_MutationFactory(x, device): # `device` is a dummy argument for graph building
    # original: y = torch.full(x.shape, 3, device=device)
    y = relay.op.transform.full(relay.const(3.0, "float32"), shape=x.shape, dtype="float32")
    # original: y.add_(1) -> functional equivalent: y = y + 1
    y_mut = relay.op.tensor.add(y, relay.const(1, y.dtype))
    # original: return x + y -> return x + y_mut
    return relay.op.tensor.add(x, y_mut)


def relay_MutationTorchTensorCall(x):
    # original: y = torch.tensor(3) defaults to int64 in PyTorch
    y_const = relay.const(3, "int64")
    # original: y.add_(1) -> functional equivalent: y = y + 1
    y_mut = relay.op.tensor.add(y_const, relay.const(1, y_const.dtype))
    # original: return x + y -> return x + y_mut. If x is float32, y_mut is promoted to float32.
    return relay.op.tensor.add(x, relay.op.cast(y_mut, x.dtype))


def relay_MutationMetadata(x):
    # original: x.resize_(2)
    # Assumes original x is (10,) and target resize is (2,) based on PyTorch context.
    # This is effectively a reshape.
    x_reshaped = relay.op.transform.reshape(x, (2,))
    # original: return x -> return x_reshaped
    return x_reshaped


# Map of original PyTorch function names to their TVM Relay graph-building equivalents
Relay_Test_Functions = {
    "FactoryFunctionCall": relay_FactoryFunctionCall,
    "TorchTensorCall": relay_TorchTensorCall,
    "TakeList": relay_TakeList,
    "ReturnList": relay_ReturnList,
    "Mutation": relay_Mutation,
    "MutationInput": relay_MutationInput,
    "MutationMetadata": relay_MutationMetadata,
    "MutationTorchTensorCall": relay_MutationTorchTensorCall,
    "MutationFactory": relay_MutationFactory,
}

# The passes to test
Passes = [TVMCSEPass]

# Original test cases (these names correspond to the original PyTorch functions)
Test_Cases = [
    "TakeList",
    "TorchTensorCall",
    "ReturnList",
    "Mutation",
    "MutationInput",
    "MutationMetadata",
    "MutationTorchTensorCall",
]

# Factory test cases (which take an additional 'device' argument)
Factory_Test_Cases = ["FactoryFunctionCall", "MutationFactory"]

# Devices to test on
Devices = ["cpu"]
# Check if CUDA is available for TVM
if tvm.cuda().exist:
    Devices.append("cuda")


# Helper for naming pytest parametrized tests
def name_fn(common_pass_cls, f_name, device_str):
    """Names parametrized test cases."""
    return f"{common_pass_cls.__name__}_{f_name}_{device_str}"


class TestCommonPass:
    @pytest.mark.parametrize(
        "common_pass_cls,f_name,device_str",
        itertools.product(Passes, Test_Cases, Devices),
        ids=name_fn,
    )
    def test_correctness(self, common_pass_cls, f_name, device_str):
        # 1. Prepare input data (numpy for ground truth and for TVM conversion)
        inp_np = np.random.randn(10).astype("float32") # Default to float32 consistent with torch.randn

        # 2. Get expected result from original PyTorch function logic
        # Temporarily import torch to run the original Python function for ground truth
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed, cannot get ground truth for correctness test.")
            return

        torch_device = torch.device(device_str)
        torch_inp = torch.tensor(inp_np, device=torch_device)
        
        # Access original PyTorch function definition using globals()
        original_py_func = globals()[f_name]
        
        expected_torch_output = original_py_func(torch_inp)
        if isinstance(expected_torch_output, (tuple, list)):
            expected_numpy_output = [t.cpu().numpy() for t in expected_torch_output]
        else:
            expected_numpy_output = expected_torch_output.cpu().numpy()
        
        # 3. Create Relay module from the converted function
        relay_func_builder = Relay_Test_Functions[f_name]
        mod = make_relay_module_from_test_func(f_name, relay_func_builder, [inp_np], False)
        
        # 4. Apply the TVM pass (e.g., CSE)
        tvm_pass = common_pass_cls()
        modified_mod = tvm_pass(mod)
        assert isinstance(modified_mod, tvm.IRModule)

        # 5. Compile and execute the modified Relay module
        target_host = "llvm" if device_str == "cpu" else None
        tvm_device = _get_tvm_device(device_str)
        
        executor = relay.build(modified_mod, target=tvm.target.Target(device_str), target_host=target_host)
        vm = tvm.runtime.vm.VirtualMachine(executor, tvm_device)

        # Convert numpy inputs to TVM NDArrays. Only one input for these tests.
        tvm_inputs = [tvm.nd.array(inp_np, device=tvm_device)]
        
        # Execute and get result
        result_tvm = vm.run(*tvm_inputs)

        # 6. Compare results
        if isinstance(result_tvm, tvm.runtime.container.ADT): # Relay tuple output
            result_arrays = [r.numpy() for r in result_tvm]
        else: # Single tensor output
            result_arrays = result_tvm.numpy()

        tvm.testing.assert_allclose(result_arrays, expected_numpy_output, rtol=1e-5, atol=1e-5)


    @pytest.mark.parametrize(
        "common_pass_cls,f_name,device_str",
        itertools.product(Passes, Factory_Test_Cases, Devices),
        ids=name_fn,
    )
    def test_correctness_factory(self, common_pass_cls, f_name, device_str):
        # 1. Prepare input data (numpy for ground truth and for TVM conversion)
        inp_np = np.random.randn(10).astype("float32")

        # 2. Get expected result from original PyTorch function logic
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed, cannot get ground truth for correctness test.")
            return

        torch_device = torch.device(device_str)
        torch_inp = torch.tensor(inp_np, device=torch_device)
        
        original_py_func = globals()[f_name] # Access original PyTorch function
        
        expected_torch_output = original_py_func(torch_inp, torch_device)
        if isinstance(expected_torch_output, (tuple, list)):
            expected_numpy_output = [t.cpu().numpy() for t in expected_torch_output]
        else:
            expected_numpy_output = expected_torch_output.cpu().numpy()
        
        # 3. Create Relay module from the converted function
        relay_func_builder = Relay_Test_Functions[f_name]
        # For factory tests, the make_relay_module_from_test_func helper needs the actual `device_str`
        # as it passes it to the `f_builder` function (e.g. `relay_FactoryFunctionCall`)
        mod = make_relay_module_from_test_func(f_name, relay_func_builder, [inp_np, device_str], True)
        
        # 4. Apply the TVM pass (e.g., CSE)
        tvm_pass = common_pass_cls()
        modified_mod = tvm_pass(mod)
        assert isinstance(modified_mod, tvm.IRModule)

        # 5. Compile and execute the modified Relay module
        target_host = "llvm" if device_str == "cpu" else None
        tvm_device = _get_tvm_device(device_str)

        executor = relay.build(modified_mod, target=tvm.target.Target(device_str), target_host=target_host)
        vm = tvm.runtime.vm.VirtualMachine(executor, tvm_device)

        # Convert numpy inputs to TVM NDArrays. Only the tensor inputs are passed to vm.run.
        tvm_inputs = [tvm.nd.array(inp_np, device=tvm_device)]
        
        # Execute and get result
        result_tvm = vm.run(*tvm_inputs)

        # 6. Compare results
        if isinstance(result_tvm, tvm.runtime.container.ADT): # Relay tuple output
            result_arrays = [r.numpy() for r in result_tvm]
        else: # Single tensor output
            result_arrays = result_tvm.numpy()

        tvm.testing.assert_allclose(result_arrays, expected_numpy_output, rtol=1e-5, atol=1e-5)


# --- Original PyTorch test functions (included for ground truth calculation) ---
# These functions are called by the test methods to get the expected outputs.
# They are placed at the global scope so `globals()[f_name]` can find them.
# Each function includes its own `import torch` to avoid a global dependency.

# Owner(s): ["oncall: fx"]
# Original imports for these functions are implicitly handled by local `import torch`
# within each function if needed.

def FactoryFunctionCall(x, device):
    import torch
    y = torch.full(x.shape, 3, device=device)
    z = torch.add(y, x)
    return z

def TorchTensorCall(x):
    import torch
    y = torch.tensor(3)
    return x + y

def TakeList(x):
    import torch
    z = torch.cat([x, x])
    return z

def ReturnList(x):
    import torch
    a = torch.arange(10).reshape(5, 2)
    z = torch.split(a, [1, 4])
    return z

def Mutation(x):
    import torch
    y = x + 2
    y.add_(1) # In-place mutation
    return x + y

def MutationInput(x):
    import torch
    x.add_(1) # In-place mutation
    y = x + 2
    return x + y

def MutationFactory(x, device):
    import torch
    y = torch.full(x.shape, 3, device=device)
    y.add_(1) # In-place mutation
    return x + y

def MutationTorchTensorCall(x):
    import torch
    y = torch.tensor(3)
    y.add_(1) # In-place mutation
    return x + y

def MutationMetadata(x):
    import torch
    # x.resize_ changes the view of the tensor, here it truncates
    # The original test code might imply specific side effects that make_fx handles.
    # For correctness, we must replicate the observable output.
    # PyTorch resize_ modifies shape. If size=2 is given for a tensor of size 10,
    # it is resized to size 2. This is effectively a slice + shallow copy.
    # We simulate this behavior here, as it's the *result* that matters for `return x`.
    # Original tensor `x` (size 10) becomes a view of size 2.
    # In TVM, we produce a new tensor that is `x_reshaped`.
    x_cloned = x.clone() # Clone to avoid modifying original torch_inp outside this func
    x_cloned.resize_(2)
    return x_cloned

# The original `if __name__ == "__main__": raise_on_run_directly("test/test_fx.py")`
# is removed as pytest automatically discovers and runs tests.
# If direct execution prevention outside pytest is strictly needed, a boilerplate
# `if __name__ == "__main__": pass` or `pytest.main([__file__])` could be added.
