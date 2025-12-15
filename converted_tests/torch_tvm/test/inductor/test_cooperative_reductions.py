import pytest
import numpy as np
import tvm
import tvm.relay as relay
import tvm.relay.op.nn
import tvm.relay.op.tensor
import tvm.relay.op.reduce
import tvm.relay.op.algorithm
import tvm.relay.transform
import tvm.runtime as rt
import tvm.testing
import math
from typing import Any, Callable, List, Tuple, Union

# Helper to map PyTorch dtypes to TVM/NumPy dtypes
def map_dtype(pytorch_dtype):
    if pytorch_dtype == 'float16':
        return 'float16'
    elif pytorch_dtype == 'float32':
        return 'float32'
    elif pytorch_dtype == 'float64':
        return 'float64'
    elif pytorch_dtype == 'int64':
        return 'int64'
    elif pytorch_dtype == 'bool':
        return 'bool'
    else:
        # Fallback for other dtypes, might need more specific handling
        return str(pytorch_dtype)

# Helper for the common execution logic
def compile_and_run_tvm(
    relay_graph_builder: Callable[..., Union[relay.Expr, Tuple[relay.Expr, ...]]],
    inputs_np: List[np.ndarray],
    target: str = "cuda",
    device_id: int = 0,
    output_dtype: Union[str, None] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Compiles a Relay function built by `relay_graph_builder` and runs it with NumPy inputs.

    Args:
        relay_graph_builder: A Python function that takes tvm.relay.Var inputs and constructs
                             a Relay expression (or tuple of expressions).
        inputs_np: A list of NumPy arrays representing the input data.
        target: The TVM target string (e.g., "cuda", "llvm").
        device_id: The device ID.
        output_dtype: Optional. The expected output dtype (TVM string format).
                      If None, output dtype is inferred or matches input for elementary ops.

    Returns:
        The result of the execution as a NumPy array or a tuple of NumPy arrays.
    """
    if target == "cuda":
        dev = tvm.cuda(device_id)
    elif target == "llvm":
        dev = tvm.cpu(device_id)
    else:
        raise ValueError(f"Unsupported target: {target}")

    input_vars = []
    for i, arr in enumerate(inputs_np):
        input_vars.append(relay.var(f"x_{i}", shape=arr.shape, dtype=arr.dtype))

    # Construct the Relay graph
    relay_output_expr = relay_graph_builder(*input_vars)

    if not isinstance(relay_output_expr, (list, tuple)):
        relay_output_expr = (relay_output_expr,)

    # Apply output dtype cast if specified and needed
    casted_output_exprs = []
    for out_expr in relay_output_expr:
        if output_dtype and str(out_expr.checked_type.dtype) != output_dtype:
            casted_output_exprs.append(relay.cast(out_expr, output_dtype))
        else:
            casted_output_exprs.append(out_expr)

    if len(casted_output_exprs) == 1:
        main_func_body = casted_output_exprs[0]
    else:
        main_func_body = relay.expr.Tuple(casted_output_exprs)

    mod = tvm.IRModule.from_expr(relay.Function(input_vars, main_func_body))
    
    # Compile the module
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    # Create a TVM runtime module and execute
    m = rt.GraphModule(lib["default"](dev))

    # Set inputs
    for i, arr in enumerate(inputs_np):
        m.set_input(f"x_{i}", tvm.nd.array(arr, dev))

    # Execute
    m.run()

    # Get output
    num_outputs = m.get_num_outputs()
    if num_outputs == 1:
        return m.get_output(0).numpy()
    else:
        return tuple(m.get_output(i).numpy() for i in range(num_outputs))


@pytest.mark.parametrize(
    "name",
    [
        "sum",
        "mean",
        "prod",
        "amin",
        "amax",
        "min",
        "max",
        "var_mean",
        "std",
        "softmax",
    ],
)
@pytest.mark.parametrize("dtype", ['float16', 'float32', 'float64'])
class TestCooperativeReductionFns:
    # `setUp` is replaced by pytest fixture if needed. `autouse=True` on `setup_method` makes it run before each test.
    # No direct TVM equivalent for `torch._inductor.metrics.generated_kernel_count` or `torch._dynamo.reset()`
    # These internal inductor mechanisms are removed from the TVM port.
    @pytest.fixture(autouse=True)
    def setup_method(self):
        pass # No specific setup needed for TVM tests that translates from PyTorch's Inductor metrics/reset

    def run_and_check(
        self,
        numpy_ref_func: Callable[..., Union[np.ndarray, Tuple[np.ndarray, ...]]],
        relay_graph_builder: Callable[..., Union[relay.Expr, Tuple[relay.Expr, ...]]],
        args_np: List[np.ndarray],
        dtype_str: Union[str, None] = None,
        *,
        expect_kernel_count=1 # Not used in TVM version
    ):
        # Define fixed tolerances
        RTOL = 1e-5
        ATOL = 1e-6

        # calculate reference value in higher precision when input dtype is float16
        ref_dtype = dtype_str
        if dtype_str == 'float16':
            ref_dtype = 'float64'

        # Prepare arguments for NumPy reference (possibly higher precision)
        args_for_numpy_ref = [arr.astype(ref_dtype) for arr in args_np]
        
        # Calculate expected output using the NumPy reference function
        expected = numpy_ref_func(*args_for_numpy_ref)

        # Apply output dtype to expected result for comparison if specified
        if dtype_str:
            if isinstance(expected, (tuple, list)):
                expected = tuple(t.astype(dtype_str) if isinstance(t, np.ndarray) else t for t in expected)
            elif isinstance(expected, np.ndarray):
                expected = expected.astype(dtype_str)
        else: # Default to float64 for numeric results if no specific dtype
            if isinstance(expected, (tuple, list)):
                expected = tuple(t.astype('float64') if isinstance(t, np.ndarray) and not np.issubdtype(t.dtype, np.bool_) else t for t in expected)
            elif isinstance(expected, np.ndarray) and not np.issubdtype(expected.dtype, np.bool_):
                expected = expected.astype('float64')


        # Compile and run the TVM module
        result = compile_and_run_tvm(relay_graph_builder, args_np, output_dtype=dtype_str)

        # For comparison, ensure result is also a tuple/list if expected is
        if isinstance(expected, (tuple, list)):
            if isinstance(result, np.ndarray):
                result = (result,)
            elif not isinstance(result, type(expected)):
                result = type(expected)(result)

            # Ensure dtype of results matches for comparison
            temp_result = []
            for r_item in result:
                if isinstance(r_item, np.ndarray) and dtype_str and str(r_item.dtype) != dtype_str:
                    temp_result.append(r_item.astype(dtype_str))
                else:
                    temp_result.append(r_item)
            result = tuple(temp_result) if isinstance(result, tuple) else temp_result
        else:
            if isinstance(result, np.ndarray) and dtype_str and str(result.dtype) != dtype_str:
                result = result.astype(dtype_str)
            elif isinstance(result, np.ndarray) and not dtype_str and not np.issubdtype(result.dtype, np.bool_):
                result = result.astype('float64')


        # Apply assert_close with fixed tolerances for tensor comparisons
        if isinstance(result, np.ndarray) and isinstance(expected, np.ndarray):
            tvm.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)
        elif isinstance(result, (tuple, list)) and isinstance(expected, (tuple, list)):
            # Iterate through elements for comparison
            for r_item, e_item in zip(result, expected):
                if isinstance(r_item, np.ndarray) and isinstance(e_item, np.ndarray):
                    tvm.testing.assert_allclose(r_item, e_item, rtol=RTOL, atol=ATOL)
                else:
                    assert r_item == e_item, f"Mismatch: {r_item} != {e_item}"
        else:
            assert result == expected, f"Mismatch: {result} != {expected}"

        # Source code inspection specific to Inductor is removed.

    @pytest.mark.skipif(not tvm.testing.device_enabled("cuda"), reason="Requires CUDA")
    def test_reduction_fns(self, name, dtype):
        # Replaced IS_SM89 with generic CUDA check.
        # This skip condition is translated directly from the PyTorch test.
        if dtype == 'float64' and name in ["std", "var_mean"]:
            pytest.skip("Skipping test due to potential timeouts or known issues with float64 for std/var_mean")

        # Define the NumPy equivalent for reference
        def numpy_ref_fn(x_np, y_np):
            combined_np = x_np + y_np
            if name == "var_mean":
                mean_np = np.mean(combined_np, axis=-1)
                var_np = np.var(combined_np, axis=-1, ddof=1) # PyTorch torch.var/std default unbiased=True (ddof=1)
                return var_np, mean_np
            elif name == "std":
                return np.std(combined_np, axis=-1, ddof=1) # PyTorch torch.std default unbiased=True (ddof=1)
            elif name == "softmax":
                # Numerically stable softmax for NumPy
                shifted_x = combined_np - np.max(combined_np, axis=-1, keepdims=True)
                exp_x = np.exp(shifted_x)
                return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
            elif name == "amin": return np.min(combined_np, axis=-1) # np.amin is alias to np.min
            elif name == "amax": return np.max(combined_np, axis=-1) # np.amax is alias to np.max
            elif name == "min": return np.min(combined_np, axis=-1)
            elif name == "max": return np.max(combined_np, axis=-1)
            elif name == "sum": return np.sum(combined_np, axis=-1)
            elif name == "mean": return np.mean(combined_np, axis=-1)
            elif name == "prod": return np.prod(combined_np, axis=-1)
            raise ValueError(f"Unsupported reduction function for NumPy: {name}")

        # Define the Relay graph builder function
        def relay_graph_builder(x_var, y_var):
            combined_relay = relay.op.tensor.add(x_var, y_var)
            if name == "var_mean":
                std_relay = relay.op.reduce.std(combined_relay, axis=-1, keepdims=False, unbiased=True)
                mean_relay = relay.op.reduce.mean(combined_relay, axis=-1, keepdims=False)
                var_relay = relay.op.tensor.multiply(std_relay, std_relay) # Variance is std_dev squared
                return var_relay, mean_relay
            elif name == "std":
                return relay.op.reduce.std(combined_relay, axis=-1, keepdims=False, unbiased=True)
            elif name == "softmax":
                return relay.op.nn.softmax(combined_relay, axis=-1)
            elif name == "amin": return relay.op.reduce.min(combined_relay, axis=-1, keepdims=False)
            elif name == "amax": return relay.op.reduce.max(combined_relay, axis=-1, keepdims=False)
            elif name == "min": return relay.op.reduce.min(combined_relay, axis=-1, keepdims=False)
            elif name == "max": return relay.op.reduce.max(combined_relay, axis=-1, keepdims=False)
            elif name == "sum": return relay.op.reduce.sum(combined_relay, axis=-1, keepdims=False)
            elif name == "mean": return relay.op.reduce.mean(combined_relay, axis=-1, keepdims=False)
            elif name == "prod": return relay.op.reduce.prod(combined_relay, axis=-1, keepdims=False)
            raise ValueError(f"Unsupported reduction function for Relay: {name}")

        args_np = [np.random.randn(1, 1024**2).astype(map_dtype(dtype)) for _ in range(2)]
        self.run_and_check(numpy_ref_fn, relay_graph_builder, args_np, dtype)

    @pytest.mark.skipif(not tvm.testing.device_enabled("cuda"), reason="Requires CUDA")
    def test_bool_reduction_fns(self):
        # Numpy reference function
        def numpy_ref_fn(x_np, y_np):
            return [
                np.any(x_np == y_np),
                np.all(x_np == y_np),
                np.any(x_np != y_np),
                np.all(x_np != y_np),
                np.any(x_np < y_np),
                np.all(x_np > y_np),
            ]

        # Relay graph builder function
        def relay_graph_builder(x_var, y_var):
            eq = relay.op.tensor.equal(x_var, y_var)
            ne = relay.op.tensor.not_equal(x_var, y_var)
            lt = relay.op.tensor.less(x_var, y_var)
            gt = relay.op.tensor.greater(x_var, y_var)
            return [
                relay.op.reduce.any(eq, keepdims=False),
                relay.op.reduce.all(eq, keepdims=False),
                relay.op.reduce.any(ne, keepdims=False),
                relay.op.reduce.all(ne, keepdims=False),
                relay.op.reduce.any(lt, keepdims=False),
                relay.op.reduce.all(gt, keepdims=False),
            ]

        args_np = [np.random.randn(1024).astype('float32') for _ in range(2)]
        self.run_and_check(numpy_ref_fn, relay_graph_builder, args_np, dtype_str='bool')

    @pytest.mark.parametrize("bs", [1, 2, 5, 15])
    @pytest.mark.parametrize("count", [1024**2 + 1, 1024**2 - 1, 1024])
    @pytest.mark.skipif(not tvm.testing.device_enabled("cuda"), reason="Requires CUDA")
    def test_non_power_of_2(self, bs, count):
        # Numpy reference function
        def numpy_ref_fn(x_np):
            mean_np = np.mean(x_np)
            std_np = np.std(x_np, ddof=1) # PyTorch default unbiased=True (ddof=1)
            min_np = np.min(x_np)
            return mean_np, std_np + min_np

        # Relay graph builder function
        def relay_graph_builder(x_var):
            mean_relay = relay.op.reduce.mean(x_var, keepdims=False)
            std_relay = relay.op.reduce.std(x_var, keepdims=False, unbiased=True)
            min_relay = relay.op.reduce.min(x_var, keepdims=False)
            return mean_relay, relay.op.tensor.add(std_relay, min_relay)

        args_np = [np.random.randn(bs, count).astype('float32')]
        self.run_and_check(numpy_ref_fn, relay_graph_builder, args_np)

    @pytest.mark.skipif(not tvm.testing.device_enabled("cuda"), reason="Requires CUDA")
    def test_chained_reductions(self):
        # Numpy reference function
        def numpy_ref_fn(x_np):
            for _ in range(8):
                shifted_x = x_np - np.max(x_np, axis=1, keepdims=True)
                softmax_np = np.exp(shifted_x) / np.sum(np.exp(shifted_x), axis=1, keepdims=True)
                x_np = x_np + softmax_np
            return x_np

        # Relay graph builder function
        def relay_graph_builder(x_var):
            for _ in range(8):
                softmax_relay = relay.op.nn.softmax(x_var, axis=1)
                x_var = relay.op.tensor.add(x_var, softmax_relay)
            return x_var

        args_np = [np.random.randn(4, 100000).astype('float32')]
        self.run_and_check(numpy_ref_fn, relay_graph_builder, args_np)

    @pytest.mark.skipif(not tvm.testing.device_enabled("cuda"), reason="Requires CUDA")
    def test_reduce_split(self):
        # Numpy reference function
        def numpy_ref_fn(a_np, b_np):
            # torch.linalg.vector_norm(a) maps to sqrt(sum(a*a)) for l2 norm
            a1_np = np.sqrt(np.sum(a_np * a_np))
            b1_np = np.sum(b_np, axis=0)
            return a1_np, b1_np

        # Relay graph builder function
        def relay_graph_builder(a_var, b_var):
            a_sq = relay.op.tensor.multiply(a_var, a_var)
            a_sum = relay.op.reduce.sum(a_sq, keepdims=False)
            a1_relay = relay.op.tensor.sqrt(a_sum)

            b1_relay = relay.op.reduce.sum(b_var, axis=0, keepdims=False)
            return a1_relay, b1_relay

        inps_np = [
            np.random.rand(2048, 512).astype('float32'),
            np.random.rand(20, 20).astype('float32'),
        ]
        self.run_and_check(numpy_ref_fn, relay_graph_builder, inps_np)


# All the TestFixedConfigs and associated logic are specific to TorchInductor's internal
# heuristics and configuration patching mechanisms, which do not have direct, portable
# equivalents in TVM's compilation stack. Thus, they are not translated.


if __name__ == "__main__":
    pytest.main([__file__])
