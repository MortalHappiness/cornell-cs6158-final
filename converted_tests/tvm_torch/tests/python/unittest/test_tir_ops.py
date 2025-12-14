# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import torch
import numpy as np
import pytest
import functools # For tvm.tir.all/any composite mapping

# Custom dtype mapping for TVM string dtypes to PyTorch dtypes
_TVM_DTYPE_TO_TORCH = {
    "bool": torch.bool,
    "uint1": torch.bool, # TVM uint1 is often used for boolean logic
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "uint16": torch.int16, # PyTorch doesn't have uint16, default to int16
    "int32": torch.int32,
    "uint32": torch.int32, # PyTorch doesn't have uint32, default to int32 (or int64 for safety)
    "int64": torch.int64,
    "uint64": torch.int64, # PyTorch doesn't have uint64, default to int64
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

def convert_tvm_dtype_to_torch(tvm_dtype_str):
    return _TVM_DTYPE_TO_TORCH.get(tvm_dtype_str, None)

def check_throws(f, expected_exception=Exception):
    try:
        f()
    except expected_exception:
        pass
    else:
        raise AssertionError(f"Should have raised {expected_exception.__name__} but didn't.")


def test_const_fold():
    def check(f, *args):
        # In TVM, tvm.tir.const(x, "int32") creates an IR integer constant.
        # Python's native operators on integers (or scalar tensors) perform constant folding.
        # The 'check' function validates that the IR constant folding matches Python's native arithmetic.
        # For PyTorch, we can directly perform the operation on Python native types or scalar tensors
        # and compare with the expected result.

        # Simulate tvm.tir.const(x, "int32") by using PyTorch scalar tensors
        x_pt = f(*[torch.tensor(val, dtype=torch.int32) for val in args])
        # Original TVM 'y' was a Python integer (after evaluation), now directly computed.
        y_py = f(*args)

        # In TVM, x would be tvm.tir.IntImm and y would be a Python int.
        # We check if the PyTorch scalar tensor value equals the Python int value.
        assert isinstance(x_pt, torch.Tensor), "Result should be a PyTorch tensor"
        assert x_pt.ndim == 0, "Result should be a scalar tensor"
        assert x_pt.item() == int(y_py), f"Check error: {x_pt.item()} vs {y_py}"

    # Operators directly map in Python/PyTorch
    tmod = lambda a, b: torch.fmod(a, b) # tvm.tir.truncmod maps to torch.fmod for C-style remainder
    tdiv = lambda a, b: torch.div(a, b, rounding_mode='trunc') # tvm.tir.truncdiv maps to torch.div with trunc rounding

    check(lambda x, y: x + y, 3, 4)
    check(lambda x, y: x * y, 3, 12)
    check(lambda x, y: x * y - 10, 3, 12)
    check(lambda x, y: x - tmod(y, 10), 3, 12)
    check(lambda x, y: tdiv(x, y) + 10, 100, 12)
    check(lambda x, y: (x & y) + 10, 112, 128) # Bitwise AND is the same
    check(lambda x, y: x > y, 112, 128)
    check(lambda x, y: x < y, 112, 128)
    check(lambda x, y: x <= y, 112, 128)
    check(lambda x, y: x >= y, 112, 128)
    check(lambda x, y: (x | y) ^ 10, 112, 128) # Bitwise OR and XOR are the same


def test_const_fold2():
    # In TVM, te.var creates a symbolic variable.
    # .same_as() checks structural equality of IR nodes, not value equality.
    # PyTorch's eager mode does not have a direct equivalent for symbolic IR simplification checks.
    # These checks would typically involve a symbolic tracing system like torch.fx to compare graphs,
    # which is beyond a simple API mapping.
    # Therefore, we use dummy tensors for numeric evaluations and skip symbolic equality checks.

    # Approximating behavior where possible for numeric checks, but marking symbolic ones with TODO.
    x_val = torch.tensor(5, dtype=torch.int32)
    tmod = lambda a, b: torch.fmod(a, b)
    tdiv = lambda a, b: torch.div(a, b, rounding_mode='trunc')

    # (x + 0).same_as(x) -> check functional equivalence (value-based)
    assert torch.equal(x_val + 0, x_val)
    # (0 + x).same_as(x) -> check functional equivalence (value-based)
    assert torch.equal(0 + x_val, x_val)
    # (x - 0).same_as(x) -> check functional equivalence (value-based)
    assert torch.equal(x_val - 0, x_val)
    # tmod(x, 1).value == 0 -> check functional equivalence (value-based)
    assert tmod(x_val, 1).item() == 0

    # (x * 1).same_as(x) -> check functional equivalence (value-based)
    assert torch.equal(x_val * 1, x_val)
    # (1 * x).same_as(x) -> check functional equivalence (value-based)
    assert torch.equal(1 * x_val, x_val)

    # assert isinstance(tdiv(1, x), tvm.tir.Div)
    # This check is for the IR type of the division operation, which has no direct PyTorch eager equivalent.
    # In PyTorch, tdiv(1, x_val) would result in a torch.Tensor.
    assert isinstance(tdiv(torch.tensor(1, dtype=torch.int32), x_val), torch.Tensor)

    # Note: The original TVM test uses `same_as` for structural IR identity,
    # which is not directly comparable to PyTorch eager tensor operations.
    # The conversions above approximate the intent by checking value equality where appropriate.


def test_const_fold3():
    # Test that using ints with logic operations is forbidden
    # In PyTorch, logical ops like torch.logical_and/or expect boolean tensors.
    # TVM's `uint1` dtype typically maps to `torch.bool`.
    # When `val` (0 or 1) is converted to a `torch.bool` tensor, it becomes compatible.
    # So, unlike TVM's IR which might disallow specific integer types in boolean contexts,
    # PyTorch with `torch.bool` will not throw errors here.
    
    # x = te.var("x", "uint1") -> represented by a bool tensor for operations
    x_dummy_bool_tensor = torch.tensor(True, dtype=torch.bool)

    # Test const folding when both arguments are const
    for torch_func, py_func in [
        (torch.logical_and, lambda a, b: a and b),
        (torch.logical_or, lambda a, b: a or b),
    ]:
        for v1 in [0, 1]:
            for v2 in [0, 1]:
                # In TVM: tvm_func(tvm.tir.const(v1, "uint1"), tvm.tir.const(v2, "uint1"))
                # -> tvm.tir.const(py_func(v1, v2), "uint1")
                # In PyTorch:
                a_tensor = torch.tensor(v1, dtype=torch.bool)
                b_tensor = torch.tensor(v2, dtype=torch.bool)
                expected_val = py_func(bool(v1), bool(v2)) # Python's `and`/`or` on bools
                expected_tensor = torch.tensor(expected_val, dtype=torch.bool)
                actual_tensor = torch_func(a_tensor, b_tensor)
                assert torch.equal(actual_tensor, expected_tensor), \
                    f"Expected {expected_tensor} for {v1} {torch_func.__name__} {v2}, got {actual_tensor}"

    # x = te.var("x", "uint1")
    true_const = torch.tensor(True, dtype=torch.bool)
    false_const = torch.tensor(False, dtype=torch.bool)

    # The original TVM tests check for symbolic IR simplification using `.same_as()`.
    # In PyTorch eager mode, we check for functional equivalence (same output values for all inputs).
    for x_val_bool in [True, False]:
        x_val_tensor = torch.tensor(x_val_bool, dtype=torch.bool)

        # tvm.tir.all(x, true).same_as(x)  -> torch.logical_and(x_val_tensor, true_const) == x_val_tensor
        assert torch.equal(torch.logical_and(x_val_tensor, true_const), x_val_tensor)
        # tvm.tir.all(true, x).same_as(x)  -> torch.logical_and(true_const, x_val_tensor) == x_val_tensor
        assert torch.equal(torch.logical_and(true_const, x_val_tensor), x_val_tensor)
        # tvm.tir.any(x, false).same_as(x) -> torch.logical_or(x_val_tensor, false_const) == x_val_tensor
        assert torch.equal(torch.logical_or(x_val_tensor, false_const), x_val_tensor)
        # tvm.tir.any(false, x).same_as(x) -> torch.logical_or(false_const, x_val_tensor) == x_val_tensor
        assert torch.equal(torch.logical_or(false_const, x_val_tensor), x_val_tensor)

        # tvm.tir.all(x, false).same_as(false) -> torch.logical_and(x_val_tensor, false_const) == false_const
        assert torch.equal(torch.logical_and(x_val_tensor, false_const), false_const)
        # tvm.tir.all(false, x).same_as(false) -> torch.logical_and(false_const, x_val_tensor) == false_const
        assert torch.equal(torch.logical_and(false_const, x_val_tensor), false_const)
        # tvm.tir.any(x, true).same_as(true) -> torch.logical_or(x_val_tensor, true_const) == true_const
        assert torch.equal(torch.logical_or(x_val_tensor, true_const), true_const)
        # tvm.tir.any(true, x).same_as(true) -> torch.logical_or(true_const, x_val_tensor) == true_const
        assert torch.equal(torch.logical_or(true_const, x_val_tensor), true_const)


def test_const_fold4():
    x1 = torch.tensor(4, dtype=torch.int32)
    x2 = x1 + 5
    tdiv = lambda a, b: torch.div(a, b, rounding_mode='trunc')

    assert isinstance(x2, torch.Tensor) and x2.ndim == 0 and x2.item() == 9
    x3 = tdiv(x2, torch.tensor(3, dtype=torch.int32)) # Ensure 3 is also a tensor for operation
    assert isinstance(x3, torch.Tensor) and x3.ndim == 0 and x3.item() == 3 # `torch.div` with integers + trunc_mode
    x4 = x3 + 0.55
    assert isinstance(x4, torch.Tensor) and x4.ndim == 0 and abs(x4.item() - 3.55) < 1e-6
    x5 = torch.ceil(x4)
    assert isinstance(x5, torch.Tensor) and x5.ndim == 0 and x5.item() == 4.0
    x6 = x5.to(torch.int32) # .astype("int") in TVM context
    assert isinstance(x6, torch.Tensor) and x6.ndim == 0 and x6.item() == 4, f"x6={x6.item()}"

    # Complex expression from TVM
    val_6_5 = torch.tensor(6.5, dtype=torch.float32)
    expr_inner = (val_6_5 - 1) / 1.5 # (5.5 / 1.5) = 3.666...
    expr_rounded = torch.round(expr_inner) # 4.0 (rounds to nearest even for .5, but here 3.66... rounds to 4)
    expr_final = expr_rounded + 2 # 6.0
    y = expr_final.to(torch.int32)
    assert isinstance(y, torch.Tensor) and y.ndim == 0 and y.item() == 6


def test_binary_dtype_match():
    def verify_general_dtype_support(f, is_conditional=False):
        # TVM rules: (lhs_dtype, rhs_dtype) -> out_dtype
        # PyTorch has its own dtype promotion rules.
        # We try to match the *outcome* of the dtype promotion.
        rules = [
            [("bool", "int32"), "int32"], # torch.bool + torch.int32 -> torch.int32 (true=1)
            [("int32", "float32"), "float32"], # torch.int32 + torch.float32 -> torch.float32
            [("int32", "int64"), "int64"], # torch.int32 + torch.int64 -> torch.int64
            [("uint32", "int8"), "uint32"], # torch.int32 + torch.int8 -> torch.int32
            [("uint32", "int32"), "uint32"], # torch.int32 + torch.int32 -> torch.int32
        ]
        for (lhs_dtype_str, rhs_dtype_str), out_dtype_str in rules:
            lhs_torch_dtype = convert_tvm_dtype_to_torch(lhs_dtype_str)
            rhs_torch_dtype = convert_tvm_dtype_to_torch(rhs_dtype_str)
            out_torch_dtype_expected = convert_tvm_dtype_to_torch(out_dtype_str)

            # Create dummy tensors for operation
            # Using values that prevent overflow for mixed types if converted to lower precision
            lhs = torch.tensor(1, dtype=lhs_torch_dtype)
            rhs = torch.tensor(2, dtype=rhs_torch_dtype)

            out = f(lhs, rhs)
            
            # For conditional ops (e.g., comparison), PyTorch output dtype is always bool.
            if is_conditional:
                assert out.dtype == torch.bool, f"Expected bool for conditional op output, got {out.dtype}"
            else:
                # For non-conditional ops, check promoted dtype
                assert out.dtype == out_torch_dtype_expected, \
                    f"Dtype mismatch for op({lhs_dtype_str}, {rhs_dtype_str}): Expected {out_torch_dtype_expected}, got {out.dtype}"

            # The original TVM test checked `out.a.dtype` or `out.args[0].dtype`.
            # This is specific to TVM's IR node structure and internal type promotion.
            # In PyTorch, once the operation `f(lhs, rhs)` is performed, `out` is the result tensor.
            # The `out.dtype` already reflects the result of type promotion.

    def verify_callop_float_only(f):
        # In TVM, `te.power(int, int)` raises an error. PyTorch's `torch.pow` handles `int, int` correctly.
        # This part of the test verifies PyTorch's behavior (dtype promotion) rather than enforcing TVM's specific error.
        
        for lhs_dtype_str in ["int32", "float32", "float64"]:
            for rhs_dtype_str in ["int32", "float32", "float64"]:
                lhs_torch_dtype = convert_tvm_dtype_to_torch(lhs_dtype_str)
                rhs_torch_dtype = convert_tvm_dtype_to_torch(rhs_dtype_str)

                lhs = torch.tensor(2, dtype=lhs_torch_dtype)
                rhs = torch.tensor(3, dtype=rhs_torch_dtype)

                out = f(lhs, rhs)

                # PyTorch `torch.pow` dtype promotion rules:
                # If both are integral, result is integral (same as larger input type).
                # If one is float, result is float (promoted).
                expected_dtype = None
                if "float" in lhs_dtype_str or "float" in rhs_dtype_str:
                    if "float64" in [lhs_dtype_str, rhs_dtype_str]:
                        expected_dtype = torch.float64
                    elif "float32" in [lhs_dtype_str, rhs_dtype_str]:
                        expected_dtype = torch.float32
                    elif "float16" in [lhs_dtype_str, rhs_dtype_str]:
                        expected_dtype = torch.float16
                else: # Both are integral types
                    if lhs_torch_dtype == torch.int64 or rhs_torch_dtype == torch.int64:
                        expected_dtype = torch.int64
                    else:
                        expected_dtype = torch.int32 # Default int type

                assert out.dtype == expected_dtype, \
                    f"Power dtype mismatch for ({lhs_dtype_str}, {rhs_dtype_str}): Expected {expected_dtype}, got {out.dtype}"

    verify_general_dtype_support(lambda a, b: a + b)
    verify_general_dtype_support(lambda a, b: a * b)
    verify_general_dtype_support(lambda a, b: a >= b, is_conditional=True)
    verify_general_dtype_support(lambda a, b: a <= b, is_conditional=True)
    
    # Using torch.pow for `te.power`
    verify_callop_float_only(lambda a, b: torch.pow(a, b))

    # verify bool & int32 constant folding
    # In TVM, `tvm.tir.const(1)` and `tvm.tir.const(True, "uint1")` might be canonicalized to the same IR value.
    # In PyTorch, a raw Python `1 == True` is `True`.
    # For tensors, `torch.equal` checks both value and shape/dtype.
    # If we cast to a common dtype, then equality can be checked.
    assert torch.equal(torch.tensor(1, dtype=torch.int32), torch.tensor(True, dtype=torch.int32))
    assert not torch.equal(torch.tensor(2, dtype=torch.int32), torch.tensor(True, dtype=torch.int32))


def test_if_then_else():
    # TVM: tvm.tir.if_then_else(cond, true_val, false_val)
    # PyTorch: torch.where(condition, input, other)
    cases = [
        # (cond_input_type_marker, lhs_dtype, rhs_dtype), out_dtype
        [("symbolic_bool", "bool", "int32"), "int32"], # Symbolic condition, bool
        [(True, "int32", "float32"), "float32"], # True condition, values promoted
        [(False, "int32", "int64"), "int64"], # False condition, values promoted
        [("symbolic_bool", "uint32", "int32"), "uint32"], # Symbolic condition, uint32/int32 -> uint32
        [("symbolic_bool", "int32", "uint32"), "uint32"], # Symbolic condition, int32/uint32 -> uint32
        [("symbolic_non_bool", "uint32", "int32"), "uint32"], # Symbolic condition, non-bool in TVM
    ]
    
    for (cond_type_marker, lhs_dtype_str, rhs_dtype_str), out_dtype_str in cases:
        lhs_torch_dtype = convert_tvm_dtype_to_torch(lhs_dtype_str)
        rhs_torch_dtype = convert_tvm_dtype_to_torch(rhs_dtype_str)
        out_torch_dtype_expected = convert_tvm_dtype_to_torch(out_dtype_str)

        # Using float dummy values to allow for more flexible dtype conversions
        lhs = torch.tensor(1.0, dtype=lhs_torch_dtype)
        rhs = torch.tensor(2.0, dtype=rhs_torch_dtype)

        if cond_type_marker is True or cond_type_marker is False:
            # Constant folding for condition (Python bool)
            
            cond_tensor = torch.tensor(cond_type_marker, dtype=torch.bool)
            
            # Simulate out = tvm.tir.if_then_else(cond, lhs, rhs)
            actual_out = torch.where(cond_tensor, lhs, rhs)
            
            # Simulate out2 = tvm.tir.if_then_else(not cond, rhs, lhs)
            cond_not_tensor = torch.tensor(not cond_type_marker, dtype=torch.bool)
            actual_out2 = torch.where(cond_not_tensor, rhs, lhs)
            
            # Simulate out3 = tvm.tir.if_then_else(not cond, lhs, rhs)
            actual_out3 = torch.where(cond_not_tensor, lhs, rhs)

            # TVM: assert tvm.ir.structural_equal(out, out2) == 1
            # In PyTorch, check value equality and dtype.
            assert torch.equal(actual_out, actual_out2)
            assert actual_out.dtype == out_torch_dtype_expected

            if cond_type_marker: # If condition is True
                expected_out = lhs.to(out_torch_dtype_expected)
                expected_out3 = rhs.to(out_torch_dtype_expected)
                assert torch.equal(actual_out, expected_out)
                assert torch.equal(actual_out3, expected_out3)
            else: # If condition is False
                expected_out = rhs.to(out_torch_dtype_expected)
                expected_out3 = lhs.to(out_torch_dtype_expected)
                assert torch.equal(actual_out, expected_out)
                assert torch.equal(actual_out3, expected_out3)

        elif cond_type_marker == "symbolic_bool":
            # In TVM, `te.var("cond", dtype="bool")` is a symbolic boolean condition.
            # In PyTorch, we need a concrete boolean tensor for execution.
            # This part of the test focused on the output dtype, not specific execution.
            
            # Create a dummy concrete condition tensor for execution and dtype check
            concrete_cond = torch.tensor(True, dtype=torch.bool) # Arbitrary boolean value for execution
            out = torch.where(concrete_cond, lhs, rhs)
            
            assert out.dtype == out_torch_dtype_expected, \
                f"Conditional (symbolic bool) dtype mismatch: Expected {out_torch_dtype_expected}, got {out.dtype}"

        elif cond_type_marker == "symbolic_non_bool":
            # In TVM: `check_throws(lambda: tvm.tir.if_then_else(cond, lhs, rhs))`
            # if `cond.dtype != "bool"`.
            # PyTorch's `torch.where` expects a boolean `condition`. Passing a non-boolean tensor
            # (which is not implicitly convertible to bool, like float0 or int0/1) will raise a `RuntimeError`.
            non_bool_cond = torch.tensor(1, dtype=torch.int32) # A non-boolean tensor condition
            check_throws(lambda: torch.where(non_bool_cond, lhs, rhs), expected_exception=RuntimeError)
        else:
            raise ValueError(f"Unknown condition type marker: {cond_type_marker}")


if __name__ == "__main__":
    test_const_fold()
    test_const_fold2()
    test_const_fold3()
    test_const_fold4()
    test_binary_dtype_match()
    test_if_then_else()
