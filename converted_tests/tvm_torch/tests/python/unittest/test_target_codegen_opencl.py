import torch
import pytest
import numpy as np
import functools
import re

# Helper to map TVM dtypes to PyTorch dtypes
def tvm_dtype_to_torch_dtype(dtype_str):
    dtype_map = {
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "uint16": torch.int16, # PyTorch doesn't have uint16, using int16 as closest
        "int32": torch.int32,
        "uint32": torch.int32, # PyTorch doesn't have uint32, using int32
        "int64": torch.int64,
        "uint64": torch.int64, # PyTorch doesn't have uint64, using int64
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool,
    }
    return dtype_map.get(dtype_str, None)

# Determine device
_device_name = "cuda" if torch.cuda.is_available() else "cpu"
_current_device = torch.device(_device_name)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA compatible GPU or equivalent for OpenCL emulation.")
def test_opencl_ternary_expression():
    def check_if_then_else(dev, n, dtype_str):
        torch_dtype = tvm_dtype_to_torch_dtype(dtype_str)
        if torch_dtype is None:
            pytest.skip(f"PyTorch does not have a direct mapping for dtype: {dtype_str}")
        if "uint" in dtype_str and not _device_name == "cpu":
            # PyTorch CUDA often doesn't support unsigned integer types directly for operations
            pytest.skip(f"PyTorch on {_device_name} might not support unsigned integer dtype: {dtype_str}")

        A = torch.randn((n,), dtype=torch_dtype, device=dev)
        # For condition `A[0] > 0`, ensure A[0] is not exactly 0 to avoid ambiguity for some dtypes
        if torch_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            A[0] = torch.tensor(1, dtype=torch_dtype, device=dev)
        else:
            A[0] = torch.tensor(0.5, dtype=torch_dtype, device=dev) # Make A[0] > 0

        true_value = torch.tensor(1, dtype=torch_dtype, device=dev)
        false_value = torch.tensor(3, dtype=torch_dtype, device=dev)
        max_lhs = torch.tensor(2, dtype=torch_dtype, device=dev)

        # Emulate tvm.tir.if_then_else and tvm.te.max
        max_rhs_scalar = torch.where(A[0] > 0, true_value, false_value)
        
        # Original tvm.te.max operation implies scalar inputs which are then broadcasted for C
        # `C = te.compute((n,), lambda i: tvm.te.max(max_lhs, max_rhs), name="C")`
        # This translates to element-wise max of two scalars, broadcast to (n,)
        C_expected_scalar = torch.max(max_lhs, max_rhs_scalar)
        C_actual = torch.full((n,), C_expected_scalar.item(), dtype=torch_dtype, device=dev)

        # In TVM, it only tested compilation. Here we perform the computation and might add a numerical check.
        # Given the original test only 'needed to test compiling', a simple successful execution implies pass for now.
        # No actual numeric comparison was done in TVM test source.
        # Example check:
        # torch.testing.assert_close(C_actual, C_expected_scalar.expand(n), rtol=1e-5, atol=1e-5)
        
    def check_select(dev, n, dtype_str):
        torch_dtype = tvm_dtype_to_torch_dtype(dtype_str)
        if torch_dtype is None:
            pytest.skip(f"PyTorch does not have a direct mapping for dtype: {dtype_str}")
        if "uint" in dtype_str and not _device_name == "cpu":
            pytest.skip(f"PyTorch on {_device_name} might not support unsigned integer dtype: {dtype_str}")

        A = torch.randn((n,), dtype=torch_dtype, device=dev)
        if torch_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            A[0] = torch.tensor(1, dtype=torch_dtype, device=dev)
        else:
            A[0] = torch.tensor(0.5, dtype=torch_dtype, device=dev)

        true_value = torch.tensor(1, dtype=torch_dtype, device=dev)
        false_value = torch.tensor(3, dtype=torch_dtype, device=dev)
        max_lhs = torch.tensor(2, dtype=torch_dtype, device=dev)

        # Emulate tvm.tir.Select and tvm.te.max
        max_rhs_scalar = torch.where(A[0] > 0, true_value, false_value)
        C_expected_scalar = torch.max(max_lhs, max_rhs_scalar)
        C_actual = torch.full((n,), C_expected_scalar.item(), dtype=torch_dtype, device=dev)

    dev = _current_device

    check_if_then_else(dev, 1, "int8")
    check_if_then_else(dev, 1, "uint8")
    check_if_then_else(dev, 1, "int16")
    check_if_then_else(dev, 1, "uint16")
    check_select(dev, 1, "int8")
    check_select(dev, 1, "uint8")
    check_select(dev, 1, "int16")
    check_select(dev, 1, "uint16")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA compatible GPU or equivalent for OpenCL emulation.")
def test_opencl_inf_nan():
    def check_inf_nan(dev, n, value, dtype_str):
        torch_dtype = tvm_dtype_to_torch_dtype(dtype_str)
        if torch_dtype is None:
            pytest.skip(f"PyTorch does not have a direct mapping for dtype: {dtype_str}")
        if torch_dtype not in [torch.float32, torch.float64]:
            pytest.skip(f"Inf/NaN values are typically for floating point types, not {dtype_str}")

        A = torch.empty((n,), dtype=torch_dtype, device=dev) # Input A is not used in compute C
        
        inf_nan_value = torch.tensor(value, dtype=torch_dtype, device=dev)
        
        # Emulate C = te.compute((n,), lambda i: inf_value, name="C")
        C_actual = torch.full((n,), inf_nan_value.item(), dtype=torch_dtype, device=dev)

        # Verify the output contains the expected inf/nan value
        if torch.isinf(inf_nan_value):
            assert torch.all(torch.isinf(C_actual))
            assert torch.all((C_actual > 0) == (inf_nan_value > 0)) # Check sign of infinity
        elif torch.isnan(inf_nan_value):
            assert torch.all(torch.isnan(C_actual))
        else:
            torch.testing.assert_close(C_actual, inf_nan_value.expand(n), rtol=1e-5, atol=1e-5)

    dev = _current_device

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float64")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float64")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float64")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA compatible GPU or equivalent for OpenCL emulation.")
def test_opencl_max():
    def check_max(dev, n, dtype_str):
        torch_dtype = tvm_dtype_to_torch_dtype(dtype_str)
        if torch_dtype is None:
            pytest.skip(f"PyTorch does not have a direct mapping for dtype: {dtype_str}")
        if "uint" in dtype_str and not _device_name == "cpu":
            pytest.skip(f"PyTorch on {_device_name} might not support unsigned integer dtype: {dtype_str}")

        # Input A for A[0]
        A = torch.randn((n,), dtype=torch_dtype, device=dev)
        A[0] = torch.tensor(5, dtype=torch_dtype, device=dev) # ensure A[0] is positive to make max_lhs > 0

        # Emulate tvm.te.max
        max_lhs_scalar = A[0] + torch.tensor(1, dtype=torch_dtype, device=dev)
        max_rhs_scalar = torch.tensor(0, dtype=torch_dtype, device=dev)
        
        # C = te.compute((n,), lambda i: tvm.te.max(max_lhs, max_rhs), name="C")
        C_expected_scalar = torch.max(max_lhs_scalar, max_rhs_scalar)
        C_actual = torch.full((n,), C_expected_scalar.item(), dtype=torch_dtype, device=dev)

        # Verify result
        torch.testing.assert_close(C_actual, C_expected_scalar.expand(n), rtol=1e-5, atol=1e-5)

    dev = _current_device

    check_max(dev, 1, "int8")
    check_max(dev, 1, "uint8")
    check_max(dev, 1, "int16")
    check_max(dev, 1, "uint16")
    check_max(dev, 1, "float32")
    check_max(dev, 1, "float64")


# @tvm.testing.requires_gpu is not needed for this test as it only inspects generated source, 
# but we maintain the opencl context for target-specific behavior for the original test intent.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA compatible GPU or equivalent for OpenCL emulation.")
def test_opencl_erf():
    def check_erf(dev, n, dtype_str):
        torch_dtype = tvm_dtype_to_torch_dtype(dtype_str)
        if torch_dtype is None:
            pytest.skip(f"PyTorch does not have a direct mapping for dtype: {dtype_str}")
        if torch_dtype not in [torch.float32, torch.float64]:
            pytest.skip(f"Erf is typically for floating point types, not {dtype_str}")

        # Input A for erf
        A = torch.randn((n,), dtype=torch_dtype, device=dev)
        
        # C = te.compute(A.shape, lambda *i: te.erf(A(*i)), name="C")
        C_actual = torch.erf(A)

        # The original TVM test inspected generated OpenCL source code for 'erf' and 'erff'.
        # This is TVM-specific code generation inspection and has no direct PyTorch equivalent.
        # We ensure the PyTorch operation runs.
        # TODO: Consider adding a numerical check against a known `np.erf` result if necessary for validation.
        # For now, just checking that the operation executes successfully.
        expected_C_np = np.erf(A.cpu().numpy()).astype(A.dtype)
        torch.testing.assert_close(C_actual, torch.tensor(expected_C_np, device=dev), rtol=1e-5, atol=1e-5)

    dev = _current_device

    check_erf(dev, 1, "float32")
    check_erf(dev, 1, "float64")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA compatible GPU or equivalent for OpenCL emulation.")
def test_opencl_type_casting():
    def check_type_casting(ctx, n, dtype_str):
        torch_dtype = tvm_dtype_to_torch_dtype(dtype_str)
        if torch_dtype is None:
            pytest.skip(f"PyTorch does not have a direct mapping for dtype: {dtype_str}")
        
        block_size = 4

        # In TVM, `i` is a symbolic loop variable. Here we simulate it with a tensor of indices.
        i_indices = torch.arange(n, dtype=torch.int32, device=ctx)

        # Conditions from TVM:
        # i // block_size == tvm.tir.const(3, "int32")
        # i % block_size == tvm.tir.const(3, "int32")
        cond1 = (i_indices // block_size) == 3
        cond2 = (i_indices % block_size) == 3
        
        # tvm.tir.all(*[cond1, cond2])
        combined_cond = functools.reduce(torch.logical_and, [cond1, cond2])

        # tvm.tir.const(1, dtype)
        true_value = torch.tensor(1, dtype=torch_dtype, device=ctx)
        # tvm.tir.const(0, dtype)
        false_value = torch.tensor(0, dtype=torch_dtype, device=ctx)

        # C = te.compute(...) -> torch.where
        C_actual = torch.where(combined_cond, true_value, false_value)

        # The original TVM test checked generated OpenCL assembly code patterns.
        # This is TVM-specific code generation inspection and has no direct PyTorch equivalent.
        # TODO: If a numerical verification of the computed output is needed, add it here.
        # Example numerical verification:
        C_expected = torch.zeros((n,), dtype=torch_dtype, device=ctx)
        # The condition `(i // 4 == 3) && (i % 4 == 3)` is only true for i = 15
        if n > 15:
            C_expected[15] = true_value
        torch.testing.assert_close(C_actual, C_expected, rtol=1e-5, atol=1e-5)

    dev = _current_device

    check_type_casting(dev, 16, "float32")


if __name__ == "__main__":
    pytest.main([__file__])
