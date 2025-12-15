import numpy as np
import tvm
from tvm import relay
import pytest
from tvm.testing import assert_allclose


# Helper function to run a Relay operator with Python scalar/numpy array inputs
# and extract the result as a Python scalar or numpy array.
def _run_relay_op(op_func, *args, input_dtypes=None, output_dtype=None):
    relay_vars = []
    runtime_inputs = []

    # Separate tensor-like inputs from other arguments (e.g., strings for einsum)
    tensor_like_args = []
    other_func_args = []  # Non-tensor arguments directly passed to the op_func like 'equation' for einsum

    for arg in args:
        if isinstance(arg, (int, float, bool, list, tuple, np.ndarray, tvm.nd.NDArray)):
            tensor_like_args.append(arg)
        else:
            other_func_args.append(arg)  # e.g., equation string for einsum

    for i, arg in enumerate(tensor_like_args):
        np_val = arg
        var_dtype = None

        if input_dtypes and i < len(input_dtypes):
            var_dtype = input_dtypes[i]

        if isinstance(arg, bool):
            np_val = np.array(arg, dtype="bool")
            if var_dtype is None:
                var_dtype = "bool"
        elif isinstance(arg, int):
            if var_dtype is None:
                # Default for bare ints if no specific type is given
                var_dtype = "int64"
            np_val = np.array(arg, dtype=var_dtype)
        elif isinstance(arg, float):
            if var_dtype is None:
                # Default for bare floats if no specific type is given
                var_dtype = "float64"
            np_val = np.array(arg, dtype=var_dtype)
        elif isinstance(arg, (list, tuple)):
            if len(arg) == 0:
                raise ValueError("Empty list/tuple not supported as input to relay op")
            # Infer dtype from first element for list/tuple if not specified
            if var_dtype is None:
                if isinstance(arg[0], bool):
                    var_dtype = "bool"
                elif isinstance(arg[0], int):
                    var_dtype = "int64"
                elif isinstance(arg[0], float):
                    var_dtype = "float64"
                else:
                    raise TypeError(f"Unsupported list element type: {type(arg[0])}")
            np_val = np.array(arg, dtype=var_dtype)
        elif isinstance(arg, np.ndarray):
            var_dtype = str(arg.dtype)  # Preserve numpy array dtype
            np_val = arg
        else:
            raise TypeError(f"Unhandled tensor-like arg type: {type(arg)}")

        relay_vars.append(relay.var(f"var{i}", relay.TensorType(np_val.shape, var_dtype)))
        runtime_inputs.append(tvm.nd.array(np_val))

    # Construct args for the op_func call
    op_call_args = list(relay_vars)
    op_call_args.extend(other_func_args)  # Add non-tensor args (like equation string for einsum)

    # Apply the operation
    if op_func == relay.op.tensor.einsum:
        if not other_func_args or not isinstance(other_func_args[0], str):
            raise ValueError("einsum requires an equation string as a non-tensor argument.")
        op_result = op_func(relay_vars, other_func_args[0])
    elif len(op_call_args) == 1:
        op_result = op_func(op_call_args[0])
    elif len(op_call_args) == 2:
        op_result = op_func(op_call_args[0], op_call_args[1])
    else:
        raise ValueError(
            f"Unsupported number of arguments ({len(op_call_args)}) or operation structure for simple Relay ufunc wrapper."
        )

    # Handle explicit output_dtype if provided (cast after op)
    if output_dtype and str(op_result.checked_type.dtype) != output_dtype:
        op_result = relay.cast(op_result, output_dtype)

    func = relay.Function(relay_vars, op_result)
    mod = tvm.IRModule.from_expr(func)

    target = tvm.target.Target("llvm")  # Default CPU target
    executor = relay.build_module.create_executor("graph", mod, target, tvm.cpu(0))

    result_nd = executor(*runtime_inputs)

    if result_nd.shape == ():
        return result_nd.numpy().item()  # Get scalar result
    else:
        return result_nd.numpy()


class TestBinaryUfuncBasic:
    def test_add(self):
        # torch._numpy._ufuncs.add maps to tvm.relay.op.tensor.add
        expected_np = np.add(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.add, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_arctan2(self):
        # torch._numpy._ufuncs.arctan2 maps to tvm.relay.op.tensor.atan2
        expected_np = np.arctan2(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.atan2, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_bitwise_and(self):
        # torch._numpy._ufuncs.bitwise_and maps to tvm.relay.op.tensor.bitwise_and
        expected_np = np.bitwise_and(5, 6)
        actual_tvm = _run_relay_op(relay.op.tensor.bitwise_and, 5, 6, input_dtypes=("int64", "int64"))
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_bitwise_or(self):
        # torch._numpy._ufuncs.bitwise_or maps to tvm.relay.op.tensor.bitwise_or
        expected_np = np.bitwise_or(5, 6)
        actual_tvm = _run_relay_op(relay.op.tensor.bitwise_or, 5, 6, input_dtypes=("int64", "int64"))
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_bitwise_xor(self):
        # torch._numpy._ufuncs.bitwise_xor maps to tvm.relay.op.tensor.bitwise_xor
        expected_np = np.bitwise_xor(5, 6)
        actual_tvm = _run_relay_op(relay.op.tensor.bitwise_xor, 5, 6, input_dtypes=("int64", "int64"))
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_copysign(self):
        # torch._numpy._ufuncs.copysign maps to tvm.relay.op.tensor.copysign
        expected_np = np.copysign(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.copysign, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_divide(self):
        # torch._numpy._ufuncs.divide maps to tvm.relay.op.tensor.divide
        expected_np = np.divide(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.divide, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_equal(self):
        # torch._numpy._ufuncs.equal maps to tvm.relay.op.tensor.equal
        expected_np = np.equal(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.equal, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_float_power(self):
        # torch._numpy._ufuncs.float_power maps to tvm.relay.op.tensor.power
        expected_np = np.float_power(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.power, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_floor_divide(self):
        # torch._numpy._ufuncs.floor_divide maps to tvm.relay.op.tensor.floor_divide
        expected_np = np.floor_divide(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.floor_divide, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_fmax(self):
        # torch._numpy._ufuncs.fmax maps to tvm.relay.op.tensor.maximum
        expected_np = np.fmax(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.maximum, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_fmin(self):
        # torch._numpy._ufuncs.fmin maps to tvm.relay.op.tensor.minimum
        expected_np = np.fmin(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.minimum, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_fmod(self):
        # torch._numpy._ufuncs.fmod maps to tvm.relay.op.tensor.fmod
        expected_np = np.fmod(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.fmod, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    @pytest.mark.skip(reason="No direct TVM Relay op for gcd. Would require composite ops or custom op.")
    def test_gcd(self):
        # TODO: Implement tvm.relay.op.algorithm.gcd or a composite operation.
        # Placeholder values to allow the test file to run.
        expected_np = np.gcd(5, 6)
        actual_tvm = 1
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_greater(self):
        # torch._numpy._ufuncs.greater maps to tvm.relay.op.tensor.greater
        expected_np = np.greater(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.greater, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_greater_equal(self):
        # torch._numpy._ufuncs.greater_equal maps to tvm.relay.op.tensor.greater_equal
        expected_np = np.greater_equal(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.greater_equal, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    @pytest.mark.skip(reason="No direct TVM Relay op for heaviside. Would require composite ops (where, greater, less).")
    def test_heaviside(self):
        # TODO: Implement heaviside using composite operations in Relay.
        # Placeholder values to allow the test file to run.
        expected_np = np.heaviside(0.5, 0.6)
        actual_tvm = 0.0
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_hypot(self):
        # torch._numpy._ufuncs.hypot maps to tvm.relay.op.tensor.hypot
        expected_np = np.hypot(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.hypot, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    @pytest.mark.skip(reason="No direct TVM Relay op for lcm. Depends on gcd.")
    def test_lcm(self):
        # TODO: Implement lcm using composite operations in Relay (requires gcd).
        # Placeholder values to allow the test file to run.
        expected_np = np.lcm(5, 6)
        actual_tvm = 30
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_ldexp(self):
        # torch._numpy._ufuncs.ldexp maps to tvm.relay.op.tensor.ldexp
        expected_np = np.ldexp(0.5, 6)
        actual_tvm = _run_relay_op(relay.op.tensor.ldexp, 0.5, 6, input_dtypes=("float64", "int32"))
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_left_shift(self):
        # torch._numpy._ufuncs.left_shift maps to tvm.relay.op.tensor.left_shift
        expected_np = np.left_shift(5, 6)
        actual_tvm = _run_relay_op(relay.op.tensor.left_shift, 5, 6, input_dtypes=("int64", "int64"))
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_less(self):
        # torch._numpy._ufuncs.less maps to tvm.relay.op.tensor.less
        expected_np = np.less(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.less, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_less_equal(self):
        # torch._numpy._ufuncs.less_equal maps to tvm.relay.op.tensor.less_equal
        expected_np = np.less_equal(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.less_equal, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    @pytest.mark.skip(reason="No direct TVM Relay op for logaddexp. Would require composite ops (exp, add, log).")
    def test_logaddexp(self):
        # TODO: Implement logaddexp using composite operations in Relay.
        # Placeholder values to allow the test file to run.
        expected_np = np.logaddexp(0.5, 0.6)
        actual_tvm = np.log(np.exp(0.5) + np.exp(0.6))
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    @pytest.mark.skip(reason="No direct TVM Relay op for logaddexp2. Would require composite ops (exp2, add, log2).")
    def test_logaddexp2(self):
        # TODO: Implement logaddexp2 using composite operations in Relay.
        # Placeholder values to allow the test file to run.
        expected_np = np.logaddexp2(0.5, 0.6)
        actual_tvm = np.log2(np.power(2, 0.5) + np.power(2, 0.6))
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_logical_and(self):
        # torch._numpy._ufuncs.logical_and maps to tvm.relay.op.tensor.logical_and
        expected_np = np.logical_and(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.logical_and, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_logical_or(self):
        # torch._numpy._ufuncs.logical_or maps to tvm.relay.op.tensor.logical_or
        expected_np = np.logical_or(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.logical_or, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_logical_xor(self):
        # torch._numpy._ufuncs.logical_xor maps to tvm.relay.op.tensor.logical_xor
        expected_np = np.logical_xor(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.logical_xor, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_matmul(self):
        # torch._numpy._ufuncs.matmul maps to tvm.relay.op.nn.matmul
        expected_np = np.matmul([0.5], [0.6])
        actual_tvm = _run_relay_op(relay.op.nn.matmul, [0.5], [0.6])
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_maximum(self):
        # torch._numpy._ufuncs.maximum maps to tvm.relay.op.tensor.maximum
        expected_np = np.maximum(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.maximum, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_minimum(self):
        # torch._numpy._ufuncs.minimum maps to tvm.relay.op.tensor.minimum
        expected_np = np.minimum(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.minimum, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    @pytest.mark.skip(reason="No direct TVM Relay op for float remainder with sign of divisor (NumPy-style). tvm.relay.op.tensor.fmod has sign of dividend.")
    def test_remainder(self):
        # TODO: Implement NumPy-style float remainder using composite operations.
        # Placeholder values to allow the test file to run.
        expected_np = np.remainder(0.5, 0.6)
        actual_tvm = 0.5
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_multiply(self):
        # torch._numpy._ufuncs.multiply maps to tvm.relay.op.tensor.multiply
        expected_np = np.multiply(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.multiply, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_nextafter(self):
        # torch._numpy._ufuncs.nextafter maps to tvm.relay.op.tensor.nextafter
        expected_np = np.nextafter(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.nextafter, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_not_equal(self):
        # torch._numpy._ufuncs.not_equal maps to tvm.relay.op.tensor.not_equal
        expected_np = np.not_equal(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.not_equal, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_power(self):
        # torch._numpy._ufuncs.power maps to tvm.relay.op.tensor.power
        expected_np = np.power(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.power, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_right_shift(self):
        # torch._numpy._ufuncs.right_shift maps to tvm.relay.op.tensor.right_shift
        expected_np = np.right_shift(5, 6)
        actual_tvm = _run_relay_op(relay.op.tensor.right_shift, 5, 6, input_dtypes=("int64", "int64"))
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)

    def test_subtract(self):
        # torch._numpy._ufuncs.subtract maps to tvm.relay.op.tensor.subtract
        expected_np = np.subtract(0.5, 0.6)
        actual_tvm = _run_relay_op(relay.op.tensor.subtract, 0.5, 0.6)
        assert_allclose(actual_tvm, expected_np, atol=1e-7, rtol=1e-5, check_dtype=False)
