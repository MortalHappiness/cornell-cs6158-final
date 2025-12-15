import numpy as np
import tvm
from tvm import relay
import pytest
from tvm.testing.utils import assert_allclose
import math
from unittest import TestCase

# Unified helper to run a unary scalar Relay op
def _run_unary_scalar_relay_op(op_func_expr_builder, scalar_input, input_dtype="float32"):
    x = relay.var("x", shape=(), dtype=input_dtype)
    
    # Build the Relay function by applying the builder to the input variable
    expr = op_func_expr_builder(x)
    
    func = relay.Function([x], expr)
    mod = tvm.IRModule.from_expr(func)
    
    # Target and device (assuming CPU for general scalar tests)
    target = "llvm" 
    dev = tvm.device(target, 0)

    # Compile and run the module
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    
    vm_exec = tvm.runtime.vm.VirtualMachine(lib, dev)
    
    # Prepare input as a scalar NDArray
    input_np = np.array(scalar_input, dtype=input_dtype)
    input_tvm = tvm.nd.array(input_np, dev)
    
    result_tvm = vm_exec.invoke("main", input_tvm)
    return result_tvm.numpy().item() # Extract Python scalar


# TVM equivalent ufuncs, wrapping Relay ops through the helper
def absolute(x): return _run_unary_scalar_relay_op(relay.op.tensor.abs, x)
def arccos(x): return _run_unary_scalar_relay_op(relay.op.tensor.acos, x)
def arccosh(x): return _run_unary_scalar_relay_op(relay.op.tensor.acosh, x)
def arcsin(x): return _run_unary_scalar_relay_op(relay.op.tensor.asin, x)
def arcsinh(x): return _run_unary_scalar_relay_op(relay.op.tensor.asinh, x)
def arctan(x): return _run_unary_scalar_relay_op(relay.op.tensor.atan, x)
def arctanh(x): return _run_unary_scalar_relay_op(relay.op.tensor.atanh, x)
def cbrt(x): return _run_unary_scalar_relay_op(lambda y: relay.op.power(y, relay.const(1/3, y.dtype)), x)
def ceil(x): return _run_unary_scalar_relay_op(relay.op.tensor.ceil, x)
# For conjugate of a real number, it's just the number itself.
def conjugate(x): return x 
def cos(x): return _run_unary_scalar_relay_op(relay.op.tensor.cos, x)
def cosh(x): return _run_unary_scalar_relay_op(relay.op.tensor.cosh, x)
def deg2rad(x): return _run_unary_scalar_relay_op(lambda y: relay.op.multiply(y, relay.const(math.pi / 180.0, y.dtype)), x)
def degrees(x): return _run_unary_scalar_relay_op(lambda y: relay.op.multiply(y, relay.const(180.0 / math.pi, y.dtype)), x)
def exp(x): return _run_unary_scalar_relay_op(relay.op.tensor.exp, x)
def exp2(x): return _run_unary_scalar_relay_op(lambda y: relay.op.power(relay.const(2.0, y.dtype), y), x)
def expm1(x): return _run_unary_scalar_relay_op(lambda y: relay.op.subtract(relay.op.exp(y), relay.const(1.0, y.dtype)), x)
def fabs(x): return absolute(x) # For float, fabs is same as abs
def floor(x): return _run_unary_scalar_relay_op(relay.op.tensor.floor, x)
def isfinite(x): return _run_unary_scalar_relay_op(relay.op.tensor.isfinite, x) # Input is float, output bool inferred
def isinf(x): return _run_unary_scalar_relay_op(relay.op.tensor.isinf, x)     # Input is float, output bool inferred
def isnan(x): return _run_unary_scalar_relay_op(relay.op.tensor.isnan, x)     # Input is float, output bool inferred
def log(x): return _run_unary_scalar_relay_op(relay.op.tensor.log, x)
def log10(x): return _run_unary_scalar_relay_op(relay.op.tensor.log10, x)
def log1p(x): return _run_unary_scalar_relay_op(lambda y: relay.op.log(relay.op.add(y, relay.const(1.0, y.dtype))), x)
def log2(x): return _run_unary_scalar_relay_op(relay.op.tensor.log2, x)
def logical_not(x):
    # numpy.logical_not(non_zero_float) is False. This implies casting to bool first.
    return _run_unary_scalar_relay_op(lambda y: relay.op.logical_not(relay.op.not_equal(y, relay.const(0.0, y.dtype))), x)
def negative(x): return _run_unary_scalar_relay_op(relay.op.negative, x)
def positive(x): return x # Identity operation for positive
def rad2deg(x): return degrees(x)
def radians(x): return deg2rad(x)
def reciprocal(x): return _run_unary_scalar_relay_op(lambda y: relay.op.divide(relay.const(1.0, y.dtype), y), x)
def rint(x): return _run_unary_scalar_relay_op(relay.op.tensor.round, x) # Assumes round-half-to-even which is typical for IEEE 754 round to nearest even.
def sign(x): return _run_unary_scalar_relay_op(relay.op.tensor.sign, x)
def signbit(x): return _run_unary_scalar_relay_op(lambda y: relay.op.less(y, relay.const(0.0, y.dtype)), x) # Input is float, output bool inferred
def sin(x): return _run_unary_scalar_relay_op(relay.op.tensor.sin, x)
def sinh(x): return _run_unary_scalar_relay_op(relay.op.tensor.sinh, x)
def sqrt(x): return _run_unary_scalar_relay_op(relay.op.tensor.sqrt, x)
def square(x): return _run_unary_scalar_relay_op(lambda y: relay.op.multiply(y, y), x)
def tan(x): return _run_unary_scalar_relay_op(relay.op.tensor.tan, x)
def tanh(x): return _run_unary_scalar_relay_op(relay.op.tensor.tanh, x)
def trunc(x): return _run_unary_scalar_relay_op(relay.op.tensor.trunc, x)


class TestUnaryUfuncs(TestCase):
    def test_absolute(self):
        assert_allclose(np.absolute(0.5), absolute(0.5), atol=1e-14)

    def test_arccos(self):
        assert_allclose(np.arccos(0.5), arccos(0.5), atol=1e-14)

    def test_arccosh(self):
        assert_allclose(np.arccosh(1.5), arccosh(1.5), atol=1e-14)

    def test_arcsin(self):
        assert_allclose(np.arcsin(0.5), arcsin(0.5), atol=1e-14)

    def test_arcsinh(self):
        assert_allclose(np.arcsinh(0.5), arcsinh(0.5), atol=1e-14)

    def test_arctan(self):
        assert_allclose(np.arctan(0.5), arctan(0.5), atol=1e-14)

    def test_arctanh(self):
        assert_allclose(np.arctanh(0.5), arctanh(0.5), atol=1e-14)

    def test_cbrt(self):
        assert_allclose(np.cbrt(0.5), cbrt(0.5), atol=1e-14)

    def test_ceil(self):
        assert_allclose(np.ceil(0.5), ceil(0.5), atol=1e-14)

    def test_conjugate(self):
        assert_allclose(np.conjugate(0.5), conjugate(0.5), atol=1e-14)

    def test_cos(self):
        assert_allclose(np.cos(0.5), cos(0.5), atol=1e-14)

    def test_cosh(self):
        assert_allclose(np.cosh(0.5), cosh(0.5), atol=1e-14)

    def test_deg2rad(self):
        assert_allclose(np.deg2rad(0.5), deg2rad(0.5), atol=1e-14)

    def test_degrees(self):
        assert_allclose(np.degrees(0.5), degrees(0.5), atol=1e-14)

    def test_exp(self):
        assert_allclose(np.exp(0.5), exp(0.5), atol=1e-14)

    def test_exp2(self):
        assert_allclose(np.exp2(0.5), exp2(0.5), atol=1e-14)

    def test_expm1(self):
        assert_allclose(np.expm1(0.5), expm1(0.5), atol=1e-14)

    def test_fabs(self):
        assert_allclose(np.fabs(0.5), fabs(0.5), atol=1e-14)

    def test_floor(self):
        assert_allclose(np.floor(0.5), floor(0.5), atol=1e-14)

    def test_isfinite(self):
        assert_allclose(np.isfinite(0.5), isfinite(0.5), atol=1e-14)

    def test_isinf(self):
        assert_allclose(np.isinf(0.5), isinf(0.5), atol=1e-14)

    def test_isnan(self):
        assert_allclose(np.isnan(0.5), isnan(0.5), atol=1e-14)

    def test_log(self):
        assert_allclose(np.log(0.5), log(0.5), atol=1e-14)

    def test_log10(self):
        assert_allclose(np.log10(0.5), log10(0.5), atol=1e-14)

    def test_log1p(self):
        assert_allclose(np.log1p(0.5), log1p(0.5), atol=1e-14)

    def test_log2(self):
        assert_allclose(np.log2(0.5), log2(0.5), atol=1e-14)

    def test_logical_not(self):
        assert_allclose(np.logical_not(0.5), logical_not(0.5), atol=1e-14)

    def test_negative(self):
        assert_allclose(np.negative(0.5), negative(0.5), atol=1e-14)

    def test_positive(self):
        assert_allclose(np.positive(0.5), positive(0.5), atol=1e-14)

    def test_rad2deg(self):
        assert_allclose(np.rad2deg(0.5), rad2deg(0.5), atol=1e-14)

    def test_radians(self):
        assert_allclose(np.radians(0.5), radians(0.5), atol=1e-14)

    def test_reciprocal(self):
        assert_allclose(np.reciprocal(0.5), reciprocal(0.5), atol=1e-14)

    def test_rint(self):
        assert_allclose(np.rint(0.5), rint(0.5), atol=1e-14)

    def test_sign(self):
        assert_allclose(np.sign(0.5), sign(0.5), atol=1e-14)

    def test_signbit(self):
        assert_allclose(np.signbit(0.5), signbit(0.5), atol=1e-14)

    def test_sin(self):
        assert_allclose(np.sin(0.5), sin(0.5), atol=1e-14)

    def test_sinh(self):
        assert_allclose(np.sinh(0.5), sinh(0.5), atol=1e-14)

    def test_sqrt(self):
        assert_allclose(np.sqrt(0.5), sqrt(0.5), atol=1e-14)

    def test_square(self):
        assert_allclose(np.square(0.5), square(0.5), atol=1e-14)

    def test_tan(self):
        assert_allclose(np.tan(0.5), tan(0.5), atol=1e-14)

    def test_tanh(self):
        assert_allclose(np.tanh(0.5), tanh(0.5), atol=1e-14)

    def test_trunc(self):
        assert_allclose(np.trunc(0.5), trunc(0.5), atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__])
