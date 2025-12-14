import torch
import torch.nn.functional as F
import pytest
import numpy as np

# PyTorch devices for testing
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}

def to_torch_dtype(dtype_str):
    return DTYPE_MAP.get(dtype_str, None)

def _get_pooling_op_func(
    typef, sizes, strides, padding, ceil_mode, count_include_pad
):
    """Return a PyTorch functional pooling operation."""
    # In TVM's original code, if padding was (ph, pw), it was internally expanded
    # to (ph, pw, ph, pw) for top, left, bottom, right.
    # For PyTorch's F.max_pool2d and F.avg_pool2d, `padding` expects
    # a single int or a tuple (ph, pw) for symmetric padding.
    # The `trials` in the test always provide `padding` as a 2-tuple (ph, pw),
    # which directly maps to PyTorch's symmetric padding argument.
    pytorch_padding = padding # (ph, pw) from trials

    if typef == "nn.max_pool2d":
        def func(input_tensor):
            return F.max_pool2d(
                input_tensor,
                kernel_size=sizes,
                stride=strides,
                padding=pytorch_padding,
                ceil_mode=ceil_mode,
            )
        return func
    elif typef == "nn.avg_pool2d":
        def func(input_tensor):
            return F.avg_pool2d(
                input_tensor,
                kernel_size=sizes,
                stride=strides,
                padding=pytorch_padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
        return func
    else:
        raise ValueError(f"Pooling function {typef} not supported")


def _get_global_pooling_op_func(typef):
    """Return a PyTorch functional global pooling operation."""
    if typef == "nn.global_max_pool2d":
        def func(input_tensor):
            return F.adaptive_max_pool2d(input_tensor, output_size=1)
        return func
    elif typef == "nn.global_avg_pool2d":
        def func(input_tensor):
            return F.adaptive_avg_pool2d(input_tensor, output_size=1)
        return func
    else:
        raise ValueError(f"Global pooling function {typef} not supported")


@pytest.mark.parametrize("device", DEVICES)
def test_pooling(device):
    np.random.seed(0)

    dtype = "float32"
    torch_dtype = to_torch_dtype(dtype)

    trials = [
        ["nn.max_pool2d", (3, 3), (2, 2), (0, 0), False, False, (27, 27, 512)],
        ["nn.max_pool2d", (2, 2), (2, 2), (0, 0), False, True, (16, 16, 16)], # count_include_pad has no effect on max_pool2d
        ["nn.max_pool2d", (3, 3), (2, 2), (1, 1), True, True, (15, 15, 16)],
        ["nn.max_pool2d", (2, 2), (2, 2), (0, 1), False, False, (16, 16, 16)],
        ["nn.avg_pool2d", (2, 2), (2, 2), (1, 1), False, False, (16, 16, 16)],
        ["nn.avg_pool2d", (2, 2), (2, 2), (0, 0), False, True, (16, 16, 16)],
        ["nn.avg_pool2d", (3, 3), (2, 2), (0, 1), True, False, (15, 15, 16)],
    ]

    for (
        typef,
        size,
        stride,
        pad,
        ceil_mode,
        count_include_pad,
        input_shape_hwc,
    ) in trials:
        # TVM's internal representation for conv/pool usually assumes NCHW,
        # where `input_shape_hwc` is (H, W, C).
        # Convert to NCHW for PyTorch: (N, C, H, W)
        shape_nchw = (1, input_shape_hwc[2], input_shape_hwc[0], input_shape_hwc[1])
        
        # Create numpy input
        np_input = np.random.uniform(-127, 128, shape_nchw).astype(dtype)
        
        # Create PyTorch input tensor
        input_tensor = torch.from_numpy(np_input).to(device=device, dtype=torch_dtype)

        # Get the functional pooling operation
        pooling_op_func = _get_pooling_op_func(
            typef, size, stride, pad, ceil_mode, count_include_pad
        )

        # Calculate reference output (uncompiled PyTorch)
        ref_output = pooling_op_func(input_tensor)

        # Calculate compiled output (TorchInductor)
        # Wrap the function for compilation
        compiled_pooling_op_func = torch.compile(pooling_op_func, backend="inductor")
        compiled_output = compiled_pooling_op_func(input_tensor)

        # Verify results
        # Use assert_close which handles precision robustly and is recommended
        torch.testing.assert_close(compiled_output, ref_output, rtol=0.001, atol=0.001)


@pytest.mark.parametrize("device", DEVICES)
def test_global_pooling(device):
    np.random.seed(0)

    dtype = "float32"
    torch_dtype = to_torch_dtype(dtype)

    trials = [
        ["nn.global_max_pool2d", (8, 8, 16)],
        ["nn.global_max_pool2d", (9, 9, 16)],
        ["nn.global_max_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (9, 9, 16)],
    ]

    for typef, input_shape_hwc in trials:
        # TVM's internal representation for conv/pool usually assumes NCHW,
        # where `input_shape_hwc` is (H, W, C).
        # Convert to NCHW for PyTorch: (N, C, H, W)
        shape_nchw = (1, input_shape_hwc[2], input_shape_hwc[0], input_shape_hwc[1])
        
        # Create numpy input
        np_input = np.random.uniform(-127, 128, shape_nchw).astype(dtype)

        # Create PyTorch input tensor
        input_tensor = torch.from_numpy(np_input).to(device=device, dtype=torch_dtype)

        # Get the functional global pooling operation
        global_pooling_op_func = _get_global_pooling_op_func(typef)

        # Calculate reference output (uncompiled PyTorch)
        ref_output = global_pooling_op_func(input_tensor)

        # Calculate compiled output (TorchInductor)
        compiled_global_pooling_op_func = torch.compile(global_pooling_op_func, backend="inductor")
        compiled_output = compiled_global_pooling_op_func(input_tensor)

        # Verify results
        torch.testing.assert_close(compiled_output, ref_output, rtol=0.001, atol=0.001)

# The original TVM codegen tests (test_codegen_pooling, test_codegen_global_pooling) and
# their associated helper functions (_calculate_output_shape, _get_expected_pooling_codegen,
# _get_expected_global_pooling_codegen) are specific to TVM's internal IR verification
# and do not have a direct equivalent in PyTorch/TorchInductor user-level testing.
# They are removed.

if __name__ == "__main__":
    pytest.main([__file__])
