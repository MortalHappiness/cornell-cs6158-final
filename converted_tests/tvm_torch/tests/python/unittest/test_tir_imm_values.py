import math
import random
import numpy as np
import torch
import pytest

# Helper to map TVM dtype strings to PyTorch dtypes
def _str_to_torch_dtype(dtype_str):
    if dtype_str == "int8":
        return torch.int8
    elif dtype_str == "uint8":
        return torch.uint8
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "uint32":
        # PyTorch does not have uint32, using int64 for a wider range.
        # Note: Arithmetic overflow behavior and negative value handling might differ from TVM's uint32.
        return torch.int64
    elif dtype_str == "int64":
        return torch.int64
    elif dtype_str == "uint64":
        # PyTorch does not have uint64, using int64 for a wider range.
        # Note: Arithmetic overflow behavior and negative value handling might differ from TVM's uint64.
        return torch.int64
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    # Custom TVM dtypes not in PyTorch, skip these cases
    elif dtype_str in ["int4", "int40"]:
        pytest.skip(f"PyTorch does not natively support custom dtype: {dtype_str}")
        return None  # Should not be reached due to skip
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

# Helper to get info (equivalent of numpy.iinfo / numpy.finfo)
def _get_info(torch_dtype):
    if torch_dtype.is_floating_point:
        return torch.finfo(torch_dtype)
    elif torch_dtype == torch.uint8: # torch.iinfo only covers signed types and uint8
        return torch.iinfo(torch_dtype)
    elif torch_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.iinfo(torch_dtype)
    else:
        # For emulated types like uint32/uint64 (mapped to int64)
        # return info for the underlying PyTorch type.
        return torch.iinfo(torch.int64)

# Custom arithmetic functions to match TVM's behavior more closely
def _torch_floordiv(x, y):
    # PyTorch's floor_divide matches NumPy's and C++'s behavior for float, and floor(x/y) for integers
    return torch.floor_divide(x, y)

def _torch_truncdiv(x, y):
    # PyTorch's div with rounding_mode='trunc' matches C-style integer division
    return torch.div(x, y, rounding_mode='trunc')

def _torch_floormod(x, y):
    # Python's % operator matches floor division based modulo
    return x % y

def _torch_truncmod(x, y):
    # PyTorch's remainder matches C-style remainder (sign of dividend)
    return torch.remainder(x, y)


@pytest.mark.parametrize(
    "dtype, literals",
    [
        ["int8", [-128, 0, 127]],
        ["uint8", [0, 255]],
        ["int32", [-2147483648, 2147483647]],
        # For uint32, use values fitting in int64 for PyTorch emulation
        ["uint32", [0, 4294967295]], # max 2^32-1
        ["int64", [-9223372036854775808, 9223372036854775807]],
        # For uint64, use values fitting in int64 for PyTorch emulation
        ["uint64", [0, 9223372036854775807]], # max 2^63-1
    ],
)
def test_tir_make_intimm(dtype, literals):
    for l in literals:
        t_dtype = _str_to_torch_dtype(dtype)
        imm = torch.tensor(l, dtype=t_dtype)
        assert imm.item() == l, imm


@pytest.mark.parametrize(
    "dtype, literals",
    [
        ["int8", [-129, 128]], # Exceeds [-128, 127]
        ["uint8", [-1, 256]],  # Exceeds [0, 255]
        ["int32", [-2147483650, 2147483648]], # Exceeds [-2^31, 2^31-1]
        ["uint32", [-1, 4294967296]], # -1 is invalid for uint, 2^32 is > int64 max
        ["uint64", [-1, 18446744073709551616]], # -1 is invalid for uint, 2^64 is > int64 max
    ],
)
def test_tir_invalid_intimm(dtype, literals):
    for l in literals:
        t_dtype = _str_to_torch_dtype(dtype)
        # Handle cases where PyTorch's int64 can represent values that TVM's uint types cannot
        if t_dtype == torch.int64 and dtype.startswith("u"):
            if l < 0:
                pytest.fail(
                    f"PyTorch {t_dtype} can represent negative {l}, "
                    f"but TVM {dtype} would raise error. Semantic mismatch. (TODO)"
                )
            if l > 9223372036854775807 and l <= 18446744073709551615: # Value fits in uint64 but not signed int64
                 pytest.fail(
                    f"PyTorch {t_dtype} cannot represent {l} directly as unsigned. "
                    f"Semantic mismatch. (TODO)"
                )
        with pytest.raises(RuntimeError): # torch.tensor raises RuntimeError for out-of-range
            torch.tensor(l, dtype=t_dtype)


@pytest.mark.parametrize(
    "dtype, literals",
    [
        [
            "uint64",
            {
                9223372036854775807: 9223372036854775807, # fits in int64
                18446744073709551615: 18446744073709551615, # This is 2^64 - 1
            },
        ],
    ],
)
def test_tir_large_py_int_literals(dtype, literals):
    """
    For large uint value, use LargeUIntImm intrin,
    """
    # PyTorch does not have a direct equivalent to TVM's LargeUIntImm for representing
    # values larger than torch.int64 max (2^63 - 1) as a single integer tensor type.
    # When trying to create a torch.tensor with such a large literal, it would typically
    # result in a RuntimeError due to overflow.
    # The original TVM test checks internal IR representation (x.args[1] << 32) + x.args[0]
    # which has no direct PyTorch analogue.
    pytest.skip("PyTorch does not support `LargeUIntImm` for integers exceeding `int64` max. (TODO)")

def test_tir_intimm_overflow():
    # PyTorch integer tensor arithmetic performs wraparound, similar to C fixed-size integers.
    # Custom int types (int4, int40) are not supported and will be skipped by _str_to_torch_dtype.
    # Note: `2**32` for uint64 is `18446744073709551616`. `torch.int64` can represent `2**32`.
    # `2**32 * 2**32 = 2**64`. `torch.int64` typically wraps for `2**64` to `0` when multiplied.
    assert (torch.tensor(255, dtype=torch.uint8) + torch.tensor(1, dtype=torch.uint8)).item() == 0
    assert (torch.tensor(2**31 - 1, dtype=torch.int32) + torch.tensor(1, dtype=torch.int32)).item() == -(2**31)
    assert (torch.tensor(2**32 - 1, dtype=_str_to_torch_dtype("uint32")) + torch.tensor(1, dtype=_str_to_torch_dtype("uint32"))).item() == 0 # Will wrap in int64 too.
    assert (torch.tensor(2**63 - 1, dtype=torch.int64) + torch.tensor(1, dtype=torch.int64)).item() == -(2**63)
    assert (torch.tensor(2**32, dtype=_str_to_torch_dtype("uint64")) * torch.tensor(2**32, dtype=_str_to_torch_dtype("uint64"))).item() == 0 # `2^64` wraps to `0` in int64.
    
    # customized int types - skip
    pytest.skip("PyTorch does not support custom int types like int4 or int40. (TODO)")
    # assert (torch.tensor(7, dtype=_str_to_torch_dtype("int4")) + torch.tensor(1, dtype=_str_to_torch_dtype("int4"))).item() == -8
    # assert (torch.tensor(2**39 - 1, dtype=_str_to_torch_dtype("int40")) + torch.tensor(1, dtype=_str_to_torch_dtype("int40"))).item() == -(2**39)


def compare_float_value(value, expect, msg):
    if math.isfinite(value):
        assert np.abs(value - expect) < 1e-5, f"{value} vs {expect}, {msg}"
    elif math.isnan(value):
        assert math.isnan(expect), f"{value} vs {expect}, {msg}"
    elif math.isinf(value):
        assert math.isinf(expect), f"{value} vs {expect}, {msg}"


@pytest.mark.parametrize(
    "dtype, literals",
    [
        ["float16", [-65504.0, 3.14, 65504.0, float('inf'), float('nan')]],
        ["bfloat16", [-3.38953139e38, 3.38953139e38, 3.14, float('inf'), float('nan')]], # Adding inf/nan for completeness
        ["float32", [torch.finfo(torch.float32).min, 3.14, torch.finfo(torch.float32).max, float('inf'), float('nan')]],
        ["float64", [torch.finfo(torch.float64).min, 3.14, torch.finfo(torch.float64).max, float('inf'), float('nan')]],
    ],
)
def test_tir_make_floatimm(dtype, literals):
    for l in literals:
        t_dtype = _str_to_torch_dtype(dtype)
        imm = torch.tensor(l, dtype=t_dtype)
        compare_float_value(imm.item(), l, "imm value should match feed value")


@pytest.mark.parametrize(
    "dtype, literals",
    [
        ["float16", [-65505.0, 65505.0]], # Exceeds float16 range (~6.55e4)
        ["float32", [-3.402e39, 3.402e39]], # Exceeds float32 range (~3.4028e38)
    ],
)
def test_tir_invalid_floatimm(dtype, literals):
    """Currently only fp16 and fp32 have range check."""
    for l in literals:
        t_dtype = _str_to_torch_dtype(dtype)
        with pytest.raises(RuntimeError): # torch.tensor raises RuntimeError for out-of-range floats
            torch.tensor(l, dtype=t_dtype)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("literal", [3.14, float('nan'), float('inf')])
def test_tir_special_floatimms(dtype, literal):
    t_dtype = _str_to_torch_dtype(dtype)
    x = torch.tensor(literal, dtype=t_dtype)
    compare_float_value(x.item(), literal, "imm value should match feed value")


def test_tir_too_large_literal_f64():
    # Behavior check: if literal f64 value is out of dtype range, the
    # object is still constructed, and eval to infinity.
    # PyTorch's `torch.tensor` will convert literal to nearest representable value,
    # which for 1.7976e309 (larger than float64 max) is infinity.
    assert math.isinf(torch.tensor(1.7976e309, dtype=torch.float64).item())


@pytest.mark.parametrize(
    "literal, expect_dtype_str",
    [
        (256, "int64"),  # PyTorch defaults to int64 for Python int literals
        (2147483647, "int64"),
        (-2147483648, "int64"),
        (2147483648, "int64"),
        (-2147483649, "int64"),
        (3.14159, "float32"), # PyTorch defaults to float32 for Python float literals
        (torch.finfo(torch.float32).min, "float32"),
        (torch.finfo(torch.float32).max, "float32"),
        # Large float literals default to float64
        (-3.402e39, "float64"),
        (3.402e39, "float64"),
    ],
)
def test_tir_const_auto_dtype(literal, expect_dtype_str):
    x = torch.tensor(literal)
    expected_torch_dtype = _str_to_torch_dtype(expect_dtype_str)
    assert x.dtype == expected_torch_dtype
    # Note: For float comparison, use compare_float_value to handle NaN/Inf and precision.
    if isinstance(literal, float) or math.isnan(literal) or math.isinf(literal):
        compare_float_value(x.item(), literal, "auto-typed float value mismatch")
    else:
        assert x.item() == literal


def check_tir_const_fold(
    dtype, foldf, calcf, x_range=None, y_range=None, expect=None, skip_overflow=False
):
    """Helper to check constant folding behavior

    Parameters
    ----------
    dtype: str
        Datatype of constants

    foldf: (x, y) -> z
        Folding function to call (takes torch.Tensor, returns torch.Tensor)

    calcf: (x, y) -> z
        Compiled calculation function to call (takes Python scalars, returns Python scalar)

    x_range: Union[int, float, tuple]
        Single value or value range [min, max]

    y_range: Union[int, float, tuple]
        Single value or value range [min, max]

    expect: Union[int, float]
        Expected calculation result

    skip_overflow: bool
        Skip assertion if the Python (arbitrary precision) result would overflow
        the fixed-point PyTorch type.
    """
    seed = random.randint(0, 2147483648)
    np.random.seed(seed)
    
    py_dtype = _str_to_torch_dtype(dtype)
    ninfo = _get_info(py_dtype)

    if x_range is None:
        if py_dtype.is_floating_point:
            x_range = (ninfo.min, ninfo.max)
        elif py_dtype == torch.uint8:
            x_range = (0, 255)
        else: # For other integer types
            x_range = (ninfo.min, ninfo.max)

    if isinstance(x_range, (int, float)):
        x_val = x_range
    elif py_dtype.is_floating_point:
        x_val = np.random.uniform(x_range[0], x_range[1])
    else: # For integer types
        x_val = np.random.randint(x_range[0], x_range[1] + 1, dtype=dtype)


    if y_range is None:
        if py_dtype.is_floating_point:
            y_range = (ninfo.min, ninfo.max)
        elif py_dtype == torch.uint8:
            y_range = (0, 255)
        else: # For other integer types
            y_range = (ninfo.min, ninfo.max)

    if isinstance(y_range, (int, float)):
        y_val = y_range
    elif py_dtype.is_floating_point:
        y_val = np.random.uniform(y_range[0], y_range[1])
    else: # For integer types
        y_val = np.random.randint(y_range[0], y_range[1] + 1, dtype=dtype)

    # Calculate Python's arbitrary precision result first (for skip_overflow)
    # The `foldf` lambda takes torch.Tensors, but Python's scalars are needed here.
    # This assumes `foldf` performs a basic arithmetic op that can be replicated by Python scalars.
    if skip_overflow:
        python_scalar_op = None
        if foldf.__name__ == '<lambda>':
            # Extract basic operation from lambda (e.g. x * y)
            # This is a heuristic and might need to be smarter for complex lambdas.
            if '*' in foldf.__code__.co_code.decode(): python_scalar_op = lambda a, b: a * b
            elif '+' in foldf.__code__.co_code.decode(): python_scalar_op = lambda a, b: a + b
            elif '-' in foldf.__code__.co_code.decode(): python_scalar_op = lambda a, b: a - b
            # Specific intrinsic wrappers:
            elif 'floordiv' in foldf.__qualname__: python_scalar_op = lambda a, b: math.floor(a / b)
            elif 'truncdiv' in foldf.__qualname__: python_scalar_op = lambda a, b: int(a / b)
            elif 'floormod' in foldf.__qualname__: python_scalar_op = lambda a, b: a % b
            elif 'truncmod' in foldf.__qualname__: python_scalar_op = lambda a, b: a - int(a / b) * b
        if python_scalar_op:
            try:
                py_res_scalar = python_scalar_op(x_val, y_val)
            except ZeroDivisionError:
                # If Python raises ZeroDivisionError, it means it's not a normal overflow case,
                # the test specific logic for ZeroDivisionError will catch it.
                pass
            else:
                if py_dtype.is_floating_point:
                     # For floats, overflow means inf, not really "skip" unless NaN
                    if math.isinf(py_res_scalar) and not math.isinf(expect): return
                    if math.isnan(py_res_scalar): return # Skip if Python calc is NaN, as it propagates.
                else:
                    # Check if the arbitrary precision Python int result fits in the PyTorch fixed-size int type
                    # For uint types, map to int64 max for overflow detection
                    max_val_check = ninfo.max
                    min_val_check = ninfo.min
                    if dtype.startswith("uint") and py_dtype == torch.int64:
                        # For TVM's uint32/uint64 (emulated by torch.int64), check against original uint max
                        if dtype == "uint32":
                            max_val_check = 4294967295
                            min_val_check = 0
                        elif dtype == "uint64":
                            max_val_check = 18446744073709551615
                            min_val_check = 0
                        
                    if not (min_val_check <= py_res_scalar <= max_val_check):
                        return # Skip if arbitrary precision result would overflow the target type.

    # Perform folding with torch.tensor
    fold_res = foldf(torch.tensor(x_val, dtype=py_dtype), torch.tensor(y_val, dtype=py_dtype))
    
    # Execute calculation via the compiled PyTorch equivalent function (returns scalar)
    calc_res = calcf(x_val, y_val)

    flaky_msg = (
        f"{dtype} ({x_val}, {y_val}, {expect}) const folding check failed.\n"
        + "This test is intentionally non-deterministic, "
        + f"if it fails please report it in github issue together with this seed {seed}\n"
    )
    
    if py_dtype.is_floating_point:
        compare_float_value(calc_res, fold_res.item(), flaky_msg)
        if expect is not None:
            compare_float_value(expect, calc_res, flaky_msg)
    else:
        assert calc_res == fold_res.item(), flaky_msg
        if expect is not None:
            assert expect == calc_res, flaky_msg


def test_tir_floatimm_const_fold():
    """Behavior check: folding fp32 match platform f32 arithmetic"""
    
    # For PyTorch, we define Python functions that simulate the "compiled" behavior.
    # PyTorch's tensor operations directly perform the arithmetic.
    def fmul_py(x, y):
        return (torch.tensor(x, dtype=torch.float32) * torch.tensor(y, dtype=torch.float32)).item()
    def fadd_py(x, y):
        return (torch.tensor(x, dtype=torch.float32) + torch.tensor(y, dtype=torch.float32)).item()
    def fsub_py(x, y):
        return (torch.tensor(x, dtype=torch.float32) - torch.tensor(y, dtype=torch.float32)).item()
    def fdiv_py(x, y):
        return (torch.tensor(x, dtype=torch.float32) / torch.tensor(y, dtype=torch.float32)).item()

    # overflow
    # For PyTorch, float division by zero typically results in inf/nan, not an error.
    # The original TVM test expected TVMError, which is a semantic difference.
    # We adapt to expect inf.
    check_tir_const_fold("float32", lambda x, y: x * y, fmul_py, 3.0e30, 3.0e30, float('inf'))
    check_tir_const_fold("float32", lambda x, y: x * y, fmul_py, 3.0e30, -3.0e30, float('-inf'))
    check_tir_const_fold("float32", lambda x, y: x / y, fdiv_py, 3.0e30, 3.0e-30, float('inf'))

    # divide by zero
    # PyTorch float division by zero returns inf/nan, not raises.
    # The original TVM test expects an error here, which is a semantic difference.
    # Replaced with a direct assertion for expected PyTorch behavior.
    # TODO: Semantic difference - TVM raises error for float division by zero, PyTorch returns inf/nan.
    assert math.isinf(fdiv_py(1.0, 0.0))
    
    # nan and inf
    check_tir_const_fold("float32", lambda x, y: x + y, fadd_py, 1.0, float('nan'), float('nan'))
    check_tir_const_fold("float32", lambda x, y: x + y, fadd_py, 1.0, float('inf'), float('inf'))
    check_tir_const_fold("float32", lambda x, y: x + y, fadd_py, 1.0, float('-inf'), float('-inf'))

    # randomized check
    check_tir_const_fold("float32", lambda x, y: x * y, fmul_py)
    check_tir_const_fold("float32", lambda x, y: x + y, fadd_py)
    check_tir_const_fold("float32", lambda x, y: x - y, fsub_py)
    check_tir_const_fold(
        "float32", lambda x, y: x / y, fdiv_py, y_range=(0.01, torch.finfo(torch.float32).max)
    )


def test_tir_int8_const_fold():
    """Behavior check: folding i8 operation match platform i8 arithmetic"""
    
    def fmul_py(x, y): return (torch.tensor(x, dtype=torch.int8) * torch.tensor(y, dtype=torch.int8)).item()
    def fadd_py(x, y): return (torch.tensor(x, dtype=torch.int8) + torch.tensor(y, dtype=torch.int8)).item()
    def fsub_py(x, y): return (torch.tensor(x, dtype=torch.int8) - torch.tensor(y, dtype=torch.int8)).item()
    def ftruncdiv_py(x, y): return _torch_truncdiv(torch.tensor(x, dtype=torch.int8), torch.tensor(y, dtype=torch.int8)).item()
    def ffloordiv_py(x, y): return _torch_floordiv(torch.tensor(x, dtype=torch.int8), torch.tensor(y, dtype=torch.int8)).item()
    def ftruncmod_py(x, y): return _torch_truncmod(torch.tensor(x, dtype=torch.int8), torch.tensor(y, dtype=torch.int8)).item()
    def ffloormod_py(x, y): return _torch_floormod(torch.tensor(x, dtype=torch.int8), torch.tensor(y, dtype=torch.int8)).item()

    # overflow (PyTorch int8 wraps around)
    check_tir_const_fold("int8", lambda x, y: x + y, fadd_py, 127, 1, -128)
    # The expected value for 127 * 127 in int8 wraps around multiple times.
    # Python int: 127 * 127 = 16129. int8 max is 127.
    # 16129 % 256 = 25. If signed, then 25 - 128 = -103?
    # PyTorch `(torch.tensor(127, dtype=torch.int8) * torch.tensor(127, dtype=torch.int8)).item()` is 1.
    # This matches the TVM test result.
    check_tir_const_fold("int8", lambda x, y: x * y, fmul_py, 127, 127, 1)

    # divide by zero
    with pytest.raises(ZeroDivisionError): # PyTorch raises ZeroDivisionError for int division by zero
        check_tir_const_fold("int8", _torch_floordiv, ffloordiv_py, 1, 0)
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("int8", _torch_truncdiv, ftruncdiv_py, 1, 0)

    # i8 mod folding is implemented in PyTorch
    assert _torch_floormod(torch.tensor(7, dtype=torch.int8), torch.tensor(3, dtype=torch.int8)).item() == 1
    assert _torch_truncmod(torch.tensor(7, dtype=torch.int8), torch.tensor(3, dtype=torch.int8)).item() == 1

    # randomized check
    check_tir_const_fold("int8", lambda x, y: x * y, fmul_py)
    check_tir_const_fold("int8", lambda x, y: x + y, fadd_py)
    check_tir_const_fold("int8", lambda x, y: x - y, fsub_py)
    check_tir_const_fold(
        "int8", _torch_floordiv, ffloordiv_py, y_range=(1, torch.iinfo(torch.int8).max)
    )
    check_tir_const_fold(
        "int8", _torch_truncdiv, ftruncdiv_py, y_range=(1, torch.iinfo(torch.int8).max)
    )


def test_tir_uint8_const_fold():
    """Behavior check: folding u8 operation match platform u8 arithmetic"""

    def fmul_py(x, y): return (torch.tensor(x, dtype=torch.uint8) * torch.tensor(y, dtype=torch.uint8)).item()
    def fadd_py(x, y): return (torch.tensor(x, dtype=torch.uint8) + torch.tensor(y, dtype=torch.uint8)).item()
    def fsub_py(x, y): return (torch.tensor(x, dtype=torch.uint8) - torch.tensor(y, dtype=torch.uint8)).item()
    def ftruncdiv_py(x, y): return _torch_truncdiv(torch.tensor(x, dtype=torch.uint8), torch.tensor(y, dtype=torch.uint8)).item()
    def ffloordiv_py(x, y): return _torch_floordiv(torch.tensor(x, dtype=torch.uint8), torch.tensor(y, dtype=torch.uint8)).item()
    def ftruncmod_py(x, y): return _torch_truncmod(torch.tensor(x, dtype=torch.uint8), torch.tensor(y, dtype=torch.uint8)).item()
    def ffloormod_py(x, y): return _torch_floormod(torch.tensor(x, dtype=torch.uint8), torch.tensor(y, dtype=torch.uint8)).item()

    # overflow (PyTorch uint8 wraps around)
    check_tir_const_fold("uint8", lambda x, y: x + y, fadd_py, 255, 1, 0)

    # zero sub (PyTorch uint8 underflow to wrap-around)
    # The original TVM test expected an error (`pytest.raises(tvm.TVMError)`).
    # PyTorch uint8 arithmetic wraps around, so 0 - 10 would be 246.
    # This is a semantic difference from TVM.
    # TODO: Semantic difference - TVM raises error for uint underflow, PyTorch wraps.
    assert fsub_py(0, 10) == 246

    # divide by zero
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("uint8", _torch_floordiv, ffloordiv_py, 1, 0)
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("uint8", _torch_truncdiv, ftruncdiv_py, 1, 0)

    # u8 mod folding is implemented in PyTorch
    assert _torch_floormod(torch.tensor(7, dtype=torch.uint8), torch.tensor(3, dtype=torch.uint8)).item() == 1
    assert _torch_truncmod(torch.tensor(7, dtype=torch.uint8), torch.tensor(3, dtype=torch.uint8)).item() == 1

    # randomized check
    check_tir_const_fold("uint8", lambda x, y: x * y, fmul_py)
    check_tir_const_fold("uint8", lambda x, y: x + y, fadd_py)
    # y_range for sub, to ensure x >= y to avoid underflow if the test wants that.
    check_tir_const_fold("uint8", lambda x, y: x - y, fsub_py, x_range=(10, 255), y_range=(0, 9)) 
    check_tir_const_fold(
        "uint8", _torch_floordiv, ffloordiv_py, y_range=(1, torch.iinfo(torch.uint8).max)
    )
    check_tir_const_fold(
        "uint8", _torch_truncdiv, ftruncdiv_py, y_range=(1, torch.iinfo(torch.uint8).max)
    )


def test_tir_int32_const_fold():
    """Behavior check: folding i32 operation match platform i32 arithmetic"""
    
    def fmul_py(x, y): return (torch.tensor(x, dtype=torch.int32) * torch.tensor(y, dtype=torch.int32)).item()
    def fadd_py(x, y): return (torch.tensor(x, dtype=torch.int32) + torch.tensor(y, dtype=torch.int32)).item()
    def fsub_py(x, y): return (torch.tensor(x, dtype=torch.int32) - torch.tensor(y, dtype=torch.int32)).item()
    def ftruncdiv_py(x, y): return _torch_truncdiv(torch.tensor(x, dtype=torch.int32), torch.tensor(y, dtype=torch.int32)).item()
    def ftruncmod_py(x, y): return _torch_truncmod(torch.tensor(x, dtype=torch.int32), torch.tensor(y, dtype=torch.int32)).item()
    def ffloordiv_py(x, y): return _torch_floordiv(torch.tensor(x, dtype=torch.int32), torch.tensor(y, dtype=torch.int32)).item()
    def ffloormod_py(x, y): return _torch_floormod(torch.tensor(x, dtype=torch.int32), torch.tensor(y, dtype=torch.int32)).item()

    # i32 overflow is not specified, only check for range. PyTorch int32 wraps around.
    assert -(2**31) <= (torch.tensor(2**31 - 1, dtype=torch.int32) + torch.tensor(1, dtype=torch.int32)).item() < 2**31
    assert -(2**31) <= (torch.tensor(-(2**31), dtype=torch.int32) - torch.tensor(1, dtype=torch.int32)).item() < 2**31

    # divide by zero
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("int32", _torch_floordiv, ffloordiv_py, 1, 0)
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("int32", _torch_floormod, ffloormod_py, 1, 0)
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("int32", _torch_truncdiv, ftruncdiv_py, 1, 0)
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("int32", _torch_truncmod, ftruncmod_py, 1, 0)

    # randomized check
    check_tir_const_fold("int32", lambda x, y: x * y, fmul_py, skip_overflow=True)
    check_tir_const_fold("int32", lambda x, y: x + y, fadd_py, skip_overflow=True)
    check_tir_const_fold("int32", lambda x, y: x - y, fsub_py, skip_overflow=True)
    check_tir_const_fold(
        "int32",
        _torch_floordiv,
        ffloordiv_py,
        y_range=(1, torch.iinfo(torch.int32).max),
        skip_overflow=True,
    )
    check_tir_const_fold(
        "int32",
        _torch_truncdiv,
        ftruncdiv_py,
        y_range=(1, torch.iinfo(torch.int32).max),
        skip_overflow=True,
    )
    check_tir_const_fold(
        "int32",
        _torch_floormod,
        ffloormod_py,
        y_range=(1, torch.iinfo(torch.int32).max),
        skip_overflow=True,
    )
    check_tir_const_fold(
        "int32",
        _torch_truncmod,
        ftruncmod_py,
        y_range=(1, torch.iinfo(torch.int32).max),
        skip_overflow=True,
    )


def test_tir_uint32_const_fold():
    """Behavior check: folding u32 operation match platform u32 arithmetic"""
    # Using int64 for PyTorch's emulation of uint32 due to lack of native uint32 type.
    # Overflow/wraparound behavior will be based on int64 semantics, which might
    # differ from strict uint32 behavior in some edge cases for very large numbers
    # or conversion from/to negative numbers.
    py_uint32_dtype = _str_to_torch_dtype("uint32") # This will be torch.int64

    def fmul_py(x, y): return (torch.tensor(x, dtype=py_uint32_dtype) * torch.tensor(y, dtype=py_uint32_dtype)).item()
    def fadd_py(x, y): return (torch.tensor(x, dtype=py_uint32_dtype) + torch.tensor(y, dtype=py_uint32_dtype)).item()
    def fsub_py(x, y): return (torch.tensor(x, dtype=py_uint32_dtype) - torch.tensor(y, dtype=py_uint32_dtype)).item()
    def ftruncdiv_py(x, y): return _torch_truncdiv(torch.tensor(x, dtype=py_uint32_dtype), torch.tensor(y, dtype=py_uint32_dtype)).item()
    def ffloordiv_py(x, y): return _torch_floordiv(torch.tensor(x, dtype=py_uint32_dtype), torch.tensor(y, dtype=py_uint32_dtype)).item()
    def ftruncmod_py(x, y): return _torch_truncmod(torch.tensor(x, dtype=py_uint32_dtype), torch.tensor(y, dtype=py_uint32_dtype)).item()
    def ffloormod_py(x, y): return _torch_floormod(torch.tensor(x, dtype=py_uint32_dtype), torch.tensor(y, dtype=py_uint32_dtype)).item()


    # u32 overflow is not specified, only check for range. PyTorch int64 wraps around.
    # Max uint32 is 4294967295. Adding 1 should wrap to 0.
    assert 0 <= (torch.tensor(4294967295, dtype=py_uint32_dtype) + torch.tensor(1, dtype=py_uint32_dtype)).item() < 2**32

    # divide by zero
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("uint32", _torch_floordiv, ffloordiv_py, 1, 0)
    with pytest.raises(ZeroDivisionError):
        check_tir_const_fold("uint32", _torch_truncdiv, ftruncdiv_py, 1, 0)

    # u32 mod folding is implemented in PyTorch
    assert _torch_floormod(torch.tensor(7, dtype=py_uint32_dtype), torch.tensor(3, dtype=py_uint32_dtype)).item() == 1
    assert _torch_truncmod(torch.tensor(7, dtype=py_uint32_dtype), torch.tensor(3, dtype=py_uint32_dtype)).item() == 1

    # randomized check
    check_tir_const_fold("uint32", lambda x, y: x * y, fmul_py, skip_overflow=True)
    check_tir_const_fold("uint32", lambda x, y: x + y, fadd_py, skip_overflow=True)
    check_tir_const_fold("uint32", lambda x, y: x - y, fsub_py, skip_overflow=True)
    check_tir_const_fold(
        "uint32",
        _torch_floordiv,
        ffloordiv_py,
        y_range=(1, 4294967295), # use uint32 max for range check
        skip_overflow=True,
    )
    check_tir_const_fold(
        "uint32",
        _torch_truncdiv,
        ftruncdiv_py,
        y_range=(1, 4294967295), # use uint32 max for range check
        skip_overflow=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
