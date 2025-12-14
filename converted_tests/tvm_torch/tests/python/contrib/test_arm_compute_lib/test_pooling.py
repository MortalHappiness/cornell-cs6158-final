import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global device setting for all tests
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoolingModel(nn.Module):
    def __init__(self, typef, sizes, strides, dilation, padding, ceil_mode, count_include_pad):
        super().__init__()
        self.typef = typef
        self.sizes = sizes
        self.strides = strides
        self.dilation = dilation
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

        # For these test cases, TVM's padding argument is a 2-tuple (H_pad, W_pad) which maps directly to PyTorch's padding.
        self.pytorch_padding = self.padding

    def forward(self, x):
        # Input is NHWC, convert to NCHW for PyTorch ops
        # TVM tests use NHWC layout by default for these pooling ops in Relay.
        # PyTorch's F.pool2d ops expect NCHW.
        x_nchw = x.permute(0, 3, 1, 2) # N H W C -> N C H W

        # Functional pooling operations in PyTorch (F.max_pool2d, F.avg_pool2d) generally operate on floating-point types.
        # If the input tensor `x` is an integer type (uint8, int8), we convert it to float32 for computation.
        # The output will then be converted back, rounded, and clamped for comparison, to mimic integer arithmetic.
        if x_nchw.dtype in (torch.uint8, torch.int8):
            x_float = x_nchw.to(torch.float32)
        else:
            x_float = x_nchw

        if self.typef == "nn.max_pool2d":
            out_nchw = F.max_pool2d(
                x_float,
                kernel_size=self.sizes,
                stride=self.strides,
                dilation=self.dilation,
                padding=self.pytorch_padding,
                ceil_mode=self.ceil_mode,
            )
        elif self.typef == "nn.avg_pool2d":
            # PyTorch's F.avg_pool2d does not support dilation > 1.
            # The test cases for avg_pool2d that have dilation > 1 will be skipped.
            if self.dilation != (1, 1):
                # This branch should not be reached due to pytest.skip in the test function itself.
                raise NotImplementedError("PyTorch F.avg_pool2d does not support dilation > 1")
            out_nchw = F.avg_pool2d(
                x_float,
                kernel_size=self.sizes,
                stride=self.strides,
                padding=self.pytorch_padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
        elif self.typef == "nn.l2_pool2d":
            # Reimplement L2 pooling: sqrt(avg_pool2d(x^2))
            squared_x = x_float ** 2.0
            avg_pooled_squared = F.avg_pool2d(
                squared_x,
                kernel_size=self.sizes,
                stride=self.strides,
                padding=self.pytorch_padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
            out_nchw = torch.sqrt(avg_pooled_squared)
        else:
            raise ValueError(f"Pooling type {self.typef} not supported")

        # Convert output back to NHWC
        out_nhwc = out_nchw.permute(0, 2, 3, 1) # N C H W -> N H W C
        return out_nhwc


class GlobalPoolingModel(nn.Module):
    def __init__(self, typef):
        super().__init__()
        self.typef = typef

    def forward(self, x):
        # Input is NHWC, convert to NCHW for PyTorch ops
        x_nchw = x.permute(0, 3, 1, 2) # N H W C -> N C H W

        # Handle integer inputs for adaptive pooling as well, convert to float for computation
        if x_nchw.dtype in (torch.uint8, torch.int8):
            x_float = x_nchw.to(torch.float32)
        else:
            x_float = x_nchw

        if self.typef == "nn.global_max_pool2d":
            out_nchw = F.adaptive_max_pool2d(x_float, output_size=1)
        elif self.typef == "nn.global_avg_pool2d":
            out_nchw = F.adaptive_avg_pool2d(x_float, output_size=1)
        else:
            raise ValueError(f"Global pooling type {self.typef} not supported")

        # Convert output back to NHWC
        out_nhwc = out_nchw.permute(0, 2, 3, 1) # N C H W -> N H W C
        return out_nhwc

def _get_low_high_atol_rtol(dtype_str):
    if dtype_str == "float32":
        low, high, atol, rtol = (-127, 128, 0.001, 0.001)
    elif dtype_str == "uint8":
        low, high, atol, rtol = (0, 255, 1, 0) # atol=1 for integer-like comparison
    elif dtype_str == "int8":
        low, high, atol, rtol = (-127, 128, 1, 0) # atol=1 for integer-like comparison
    else:
        pytest.fail(f"dtype not expected: {dtype_str}")
    return low, high, atol, rtol

_TORCH_DTYPES = {
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int32": torch.int32,
}

# The original TVM tests verify integration with Arm Compute Library (ACL) and its codegen.
# This infrastructure is TVM-specific and cannot be directly converted to PyTorch.
# The tests below re-implement the pooling operations in PyTorch and verify their numerical
# correctness when compiled with TorchInductor, which is the analogous PyTorch optimization path.
# Tests for TVM's internal codegen details are not translated.

# fmt: off
@pytest.mark.parametrize(
     "typef,dtype_str,size,stride,dilation,pad,ceil_mode,count_include_pad,input_shape",
     [
        ("nn.max_pool2d", "float32",  (3, 3), (2, 2), (1, 1), (0, 0), False, False, (27, 27, 512)),
        ("nn.max_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 0), False, True,  (16, 16, 16)),
        ("nn.max_pool2d", "float32",  (3, 3), (2, 2), (1, 1), (1, 1), True,  True,  (15, 15, 16)),
        ("nn.max_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16)),
        ("nn.max_pool2d", "uint8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16)),
        ("nn.max_pool2d", "uint8", (2, 2), (2, 2), (1, 1), (1, 1), True,  True,  (15, 15, 16)),
        ("nn.max_pool2d", "uint8", (2, 2), (2, 2), (3, 2), (1, 1), True,  True,  (15, 15, 16)),
        ("nn.max_pool2d", "int8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16)),
        ("nn.max_pool2d", "int8", (2, 2), (2, 2), (1, 1), (1, 1), True,  True,  (15, 15, 16)),
        ("nn.max_pool2d", "int8", (2, 2), (2, 2), (3, 2), (1, 1), True,  True,  (15, 15, 16)),
        ("nn.avg_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (1, 1), False, False, (16, 16, 16)),
        ("nn.avg_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 0), False, True,  (16, 16, 16)),
        ("nn.avg_pool2d", "float32",  (3, 3), (2, 2), (3, 2), (0, 1), True,  False, (15, 15, 16)),
        ("nn.avg_pool2d", "uint8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16)),
        ("nn.avg_pool2d", "int8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16)),
        ("nn.l2_pool2d",  "float32",  (2, 2), (2, 2), (1, 1), (0, 1), True,  False, (16, 16, 16)),
        ("nn.l2_pool2d",  "float32",  (3, 3), (2, 2), (1, 1), (0, 0), False, False, (16, 16, 16)),
        ("nn.l2_pool2d",  "float32",  (2, 2), (2, 2), (1, 1), (1, 1), False,  True, (15, 15, 16)),
     ],
)
# fmt: on
def test_pooling(
    typef,
    dtype_str,
    size,
    stride,
    dilation,
    pad,
    ceil_mode,
    count_include_pad,
    input_shape,
):
    # PyTorch's F.avg_pool2d does not support a dilation argument.
    # We skip tests that use dilation > 1 for avg_pool2d.
    if typef == "nn.avg_pool2d" and dilation != (1, 1):
        pytest.skip("PyTorch F.avg_pool2d does not support dilation > 1")

    low, high, atol, rtol = _get_low_high_atol_rtol(dtype_str)
    torch_dtype = _TORCH_DTYPES[dtype_str]

    full_shape = (1, *input_shape)
    np.random.seed(0)

    # Generate input data as the original dtype.
    # The PoolingModel will handle conversion to float32 for computation if necessary.
    input_np = np.random.uniform(low, high, full_shape).astype(dtype_str)
    x = torch.tensor(input_np, dtype=torch_dtype, device=_DEVICE)

    model = PoolingModel(
        typef,
        size,
        stride,
        dilation,
        pad,
        ceil_mode,
        count_include_pad,
    ).to(_DEVICE)

    # Run native PyTorch model
    output_native = model(x)

    # Compile with TorchInductor
    compiled_model = torch.compile(model)
    output_compiled = compiled_model(x)

    # For uint8/int8 inputs, the internal pooling uses float32.
    # To compare the results and match TVM's likely integer-arithmetic behavior (implied by atol=1, rtol=0),
    # we round the float outputs and clamp them to the expected integer range.
    if dtype_str in ("uint8", "int8"):
        output_native_for_cmp = output_native.to(torch.float32).round().clamp(low, high).to(torch_dtype)
        output_compiled_for_cmp = output_compiled.to(torch.float32).round().clamp(low, high).to(torch_dtype)
    else:
        output_native_for_cmp = output_native
        output_compiled_for_cmp = output_compiled

    torch.testing.assert_allclose(output_native_for_cmp, output_compiled_for_cmp, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "typef,dtype_str,input_shape",
    [
        ["nn.global_max_pool2d", "float32", (8, 8, 16)],
        ["nn.global_max_pool2d", "float32", (9, 9, 16)],
        ["nn.global_max_pool2d", "uint8", (8, 8, 16)],
        ["nn.global_max_pool2d", "uint8", (9, 9, 16)],
        ["nn.global_max_pool2d", "int8", (8, 8, 16)],
        ["nn.global_max_pool2d", "int8", (9, 9, 16)],
        ["nn.global_avg_pool2d", "float32", (8, 8, 16)],
        ["nn.global_avg_pool2d", "float32", (9, 9, 16)],
        ["nn.global_avg_pool2d", "uint8", (8, 8, 16)],
        ["nn.global_avg_pool2d", "uint8", (9, 9, 16)],
        ["nn.global_avg_pool2d", "int8", (8, 8, 16)],
        ["nn.global_avg_pool2d", "int8", (9, 9, 16)],
    ],
)
def test_global_pooling(typef, dtype_str, input_shape):
    low, high, atol, rtol = _get_low_high_atol_rtol(dtype_str)
    torch_dtype = _TORCH_DTYPES[dtype_str]

    full_shape = (1, *input_shape)
    np.random.seed(0)

    # Generate input data as the original dtype.
    # The GlobalPoolingModel will handle conversion to float32 for computation if necessary.
    input_np = np.random.uniform(low, high, full_shape).astype(dtype_str)
    x = torch.tensor(input_np, dtype=torch_dtype, device=_DEVICE)

    model = GlobalPoolingModel(typef).to(_DEVICE)

    # Run native PyTorch model
    output_native = model(x)

    # Compile with TorchInductor
    compiled_model = torch.compile(model)
    output_compiled = compiled_model(x)
    
    # For uint8/int8 inputs, the internal pooling uses float32.
    # To compare the results and match TVM's likely integer-arithmetic behavior (implied by atol=1, rtol=0),
    # we round the float outputs and clamp them to the expected integer range.
    if dtype_str in ("uint8", "int8"):
        output_native_for_cmp = output_native.to(torch.float32).round().clamp(low, high).to(torch_dtype)
        output_compiled_for_cmp = output_compiled.to(torch.float32).round().clamp(low, high).to(torch_dtype)
    else:
        output_native_for_cmp = output_native
        output_compiled_for_cmp = output_compiled

    torch.testing.assert_allclose(output_native_for_cmp, output_compiled_for_cmp, rtol=rtol, atol=atol)

# The original `test_codegen_pooling` and `test_codegen_global_pooling` functions,
# along with the `_get_expected_*_codegen` helper functions,
# are specific to TVM's internal code generation and verification mechanisms.
# They do not have direct equivalents in the PyTorch/TorchInductor ecosystem and are therefore omitted.

# The `if __name__ == "__main__":` block is typically not used in pytest-based test suites
# as pytest discovers and runs tests automatically.
# It is omitted in the converted output.
