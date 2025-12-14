import torch
import torch.nn.functional as F
import pytest
import numpy as np
import functools

# Helper function to generate random tensors
def _rand_tensor(shape, dtype, device="cpu"):
    if isinstance(dtype, str):
        dtype_map = {
            "float32": torch.float32,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
            # Add other dtypes as needed
        }
        dtype = dtype_map.get(dtype, torch.float32) # Default to float32
    return torch.randn(shape, dtype=dtype, device=device)


# For TVM Relay graph transformations, there isn't a direct PyTorch API
# for applying the transformation and then asserting the structure of the
# resulting computational graph.
# Instead, we define 'before' and 'expected' as Python functions that
# perform the computations on PyTorch tensors.
# The `tvm.ir.assert_structural_equal` check is replaced by a numerical
# comparison of outputs from the 'before' and 'expected' functions.

# These functions are placeholders for TVM-specific IR manipulation.
# They will be replaced by direct Python function/Module calls that
# operate on concrete PyTorch tensors for numerical comparison.
class CombineParallelDensePlaceholder:
    def __init__(self, min_num_branches, to_batch):
        pass # The transformation is implicitly applied by using 'expected' model
    def __call__(self, expr):
        # In PyTorch, we would run the `expr` model and expect the `expected` model
        # to produce the same result if the transformation was applied.
        # This placeholder does nothing as the 'expected' functions are pre-transformed.
        return expr

class InferTypePlaceholder:
    def __call__(self, mod):
        return mod # No-op in PyTorch functional context

# Placeholder for tvm.transform.Pass
class TVMTransformPassPlaceholder:
    def __call__(self, mod):
        return mod

# Placeholder for tvm.ir.assert_structural_equal
def assert_structural_equal_todo(actual_output, expected_output, rtol=1e-5, atol=1e-5):
    # TODO: Structural comparison of compilation IR is TVM-specific and not
    # directly translatable to PyTorch user-facing API for an arbitrary pass.
    # We instead compare numerical outputs for functional correctness.
    torch.testing.assert_close(actual_output, expected_output, rtol=rtol, atol=atol)


# We convert `run_opt_pass` to simply return the output of the PyTorch-equivalent functions.
def run_opt_pass(func, opt_pass_placeholder):
    return func


# Test 1: test_combine_parallel_dense
def test_combine_parallel_dense():
    """Simple testcase. One dense cannot be combined due to shape mismatch"""

    # TVM Relay graph definition translated to a Python function using PyTorch ops
    def before_pytorch(x, w1, w2, w3, w4):
        y1 = F.linear(x, w1)
        y2 = F.linear(x, w2)
        y3 = F.linear(x, w3) # y3 cannot be combined due to shape mismatch
        y4 = F.linear(x, w4)
        return y1, y2, y3, y4

    def expected_pytorch(x, w1, w2, w3, w4):
        # Stack x for batch_matmul. Needs to be (batch, input_features)
        # Assuming x is (i, k), w is (j, k)
        # Stacked x: (3, i, k) for (y1, y2, y4)
        x_stacked = torch.stack((x, x, x), dim=0) # (3, i, k)

        # Stack w for batch_matmul. Need to transpose weights to (k, j) for batch_matmul
        # Initial w is (j, k), so stack w1, w2, w4 as (3, j, k)
        # For batch_matmul, the second operand should be (B, K, J).
        # So w needs to be (3, k, j)
        w_stacked = torch.stack((w1, w2, w4), dim=0) # (3, j, k)
        
        # PyTorch F.linear implicitly does x @ w.T if w is (out, in)
        # So for bmm, if we want (N, i, j), we need (N, i, k) @ (N, k, j)
        # Our w_stacked is (N, j, k), so we need to transpose the last two dimensions to (N, k, j)
        y_batched = torch.bmm(x_stacked, w_stacked.transpose(1, 2)) # (3, i, j)

        # Split needs split_size_or_sections. Here, we split into 3 equal parts.
        y_split = torch.split(y_batched, 1, dim=0) # Each element (1, i, j)
        y1 = torch.squeeze(y_split[0], dim=0) # (i, j)
        y2 = torch.squeeze(y_split[1], dim=0) # (i, j)
        y4 = torch.squeeze(y_split[2], dim=0) # (i, j)

        # y3 is not combined
        y3 = F.linear(x, w3)
        return y1, y2, y3, y4

    def check(i, j, k):
        # Generate concrete tensors for comparison
        x_val = _rand_tensor((i, k), "float32")
        w1_val = _rand_tensor((j, k), "float32")
        w2_val = _rand_tensor((j, k), "float32")
        w3_val = _rand_tensor((j + 1, k), "float32") # Shape mismatch
        w4_val = _rand_tensor((j, k), "float32")

        # Run before and expected PyTorch functions
        output_before = before_pytorch(x_val, w1_val, w2_val, w3_val, w4_val)
        output_expected = expected_pytorch(x_val, w1_val, w2_val, w3_val, w4_val)

        # Compare numerical outputs
        # Note: We compare outputs directly since structural IR comparison is TVM-specific.
        for actual, expected in zip(output_before, output_expected):
            torch.testing.assert_close(actual, expected)

    check(3, 5, 4)
    check(100, 200, 300)


# Test 2: test_combine_parallel_dense_biasadd
def test_combine_parallel_dense_biasadd():
    """Testcase of combining dense + 1d biasadd"""

    def before_pytorch(x, w1, w2, b1, b2):
        y1 = F.linear(x, w1)
        y2 = F.linear(x, w2)
        y1 = y1 + b1
        y2 = y2 + b2
        return y1, y2

    def expected_pytorch(x, w1, w2, b1, b2, is_2d_bias):
        x_stacked = torch.stack((x, x), dim=0) # (2, i, k)
        w_stacked = torch.stack((w1, w2), dim=0) # (2, j, k)
        
        y_batched = torch.bmm(x_stacked, w_stacked.transpose(1, 2)) # (2, i, j)

        # Handle bias
        if not is_2d_bias:
            b1 = torch.unsqueeze(b1, dim=0) # (1, j)
            b2 = torch.unsqueeze(b2, dim=0) # (1, j)
        
        # If b1, b2 were (i, j), stack them to (2, i, j)
        # If b1, b2 were (j), unsqueeze to (1, j), then stack to (2, 1, j)
        # Then add to y_batched (2, i, j), which broadcasts (2, 1, j) to (2, i, j)
        b_stacked = torch.stack((b1, b2), dim=0) # (2, 1, j) or (2, i, j)
        y_batched = y_batched + b_stacked

        y_split = torch.split(y_batched, 1, dim=0)
        y1 = torch.squeeze(y_split[0], dim=0)
        y2 = torch.squeeze(y_split[1], dim=0)
        return y1, y2

    def check(i, j, k, is_2d_bias):
        x_val = _rand_tensor((i, k), "float32")
        w1_val = _rand_tensor((j, k), "float32")
        w2_val = _rand_tensor((j, k), "float32")

        if is_2d_bias:
            b1_val = _rand_tensor((i, j), "float32")
            b2_val = _rand_tensor((i, j), "float32")
        else:
            b1_val = _rand_tensor((j,), "float32")
            b2_val = _rand_tensor((j,), "float32")

        output_before = before_pytorch(x_val, w1_val, w2_val, b1_val, b2_val)
        output_expected = expected_pytorch(x_val, w1_val, w2_val, b1_val, b2_val, is_2d_bias)

        for actual, expected in zip(output_before, output_expected):
            torch.testing.assert_close(actual, expected)

    check(3, 5, 4, False)
    check(100, 200, 300, False)
    check(3, 5, 4, True)
    check(100, 200, 300, True)


# Test 3: test_combine_parallel_dense_biasadd_scale_reshape
def test_combine_parallel_dense_biasadd_scale_reshape():
    """Testcase of combining dense + 1d biasadd + multiply with non-fused reshape"""

    def before_pytorch(x, w1, w2, b1, b2, scale1, scale2, newshape):
        y1 = F.linear(x, w1)
        y2 = F.linear(x, w2)
        y1 = y1 + b1
        y2 = y2 + b2
        y1 = y1 * scale1
        y2 = y2 * scale2
        y1 = y1.reshape(newshape)
        y2 = y2.reshape(newshape)
        return y1, y2

    def expected_pytorch(x, w1, w2, b1, b2, scale1, scale2, newshape):
        x_stacked = torch.stack((x, x), dim=0) # (2, i, k)
        w_stacked = torch.stack((w1, w2), dim=0) # (2, j, k)
        
        y_batched = torch.bmm(x_stacked, w_stacked.transpose(1, 2)) # (2, i, j)

        # Bias handling (b1, b2 are (j,) originally)
        b1 = torch.unsqueeze(b1, dim=0) # (1, j)
        b2 = torch.unsqueeze(b2, dim=0) # (1, j)
        b_stacked = torch.stack((b1, b2), dim=0) # (2, 1, j)
        y_batched = y_batched + b_stacked

        # Scale handling (scale1, scale2 are (1,) originally)
        # The TVM `expand_dims(scale1, 0)` on `shape=(1,)` makes it `(1,1)`.
        # Then `stack` on `((1,1), (1,1))` makes it `(2,1,1)`.
        scale1_unsqueeze_0 = torch.unsqueeze(scale1, 0) # (1,1)
        scale2_unsqueeze_0 = torch.unsqueeze(scale2, 0) # (1,1)
        scale_stacked = torch.stack((scale1_unsqueeze_0, scale2_unsqueeze_0), dim=0) # (2, 1, 1)
        y_batched = y_batched * scale_stacked

        y_split = torch.split(y_batched, 1, dim=0)
        y1 = torch.squeeze(y_split[0], dim=0)
        y2 = torch.squeeze(y_split[1], dim=0)

        y1 = y1.reshape(newshape)
        y2 = y2.reshape(newshape)
        return y1, y2

    def check(i, j, k, scale1_const, scale2_const, newshape):
        x_val = _rand_tensor((i, k), "float32")
        w1_val = _rand_tensor((j, k), "float32")
        w2_val = _rand_tensor((j, k), "float32") # Corrected from (2*j, k) for batch_matmul

        # For this test, b1, b2 shapes are (j,)
        b1_val = _rand_tensor((j,), "float32")
        b2_val = _rand_tensor((j,), "float32")

        # scale1, scale2 shapes are (1,)
        scale1_val = _rand_tensor((1,), "float32") + scale1_const # Add a constant to ensure non-zero
        scale2_val = _rand_tensor((1,), "float32") + scale2_const

        output_before = before_pytorch(x_val, w1_val, w2_val, b1_val, b2_val, scale1_val, scale2_val, newshape)
        output_expected = expected_pytorch(x_val, w1_val, w2_val, b1_val, b2_val, scale1_val, scale2_val, newshape)

        for actual, expected in zip(output_before, output_expected):
            torch.testing.assert_close(actual, expected)

    check(3, 5, 4, 0.5, 0.25, (1, 1, 15))
    check(100, 200, 300, 0.5, 0.25, (1, 1, 20000))


# Test 4: test_combine_parallel_dense_flat
def test_combine_parallel_dense_flat():
    """Simple testcase. All matmul of different output dim can be combined"""

    def before_pytorch(x, w1, w2, w3):
        y1 = F.linear(x, w1)
        y2 = F.linear(x, w2)
        y3 = F.linear(x, w3)
        return y1, y2, y3

    def expected_pytorch(x, w1, w2, w3, j):
        # x: (i, k)
        # w1: (j, k)
        # w2: (2*j, k)
        # w3: (3*j, k)
        # Combined w_stacked: (j + 2*j + 3*j, k) = (6*j, k)
        w_stacked = torch.cat((w1, w2, w3), dim=0) # (6*j, k)
        y = F.linear(x, w_stacked) # (i, 6*j)

        # PyTorch slicing with slice_mode="size" logic:
        # y1 = y[:, 0 : 0 + j]
        # y2 = y[:, j : j + (2 * j)] = y[:, j : 3*j]
        # y3 = y[:, 3*j : 3*j + (3 * j)] = y[:, 3*j : 6*j]

        y1 = y[:, :j]
        y2 = y[:, j : 3 * j] 
        y3 = y[:, 3 * j : 6 * j] 
        return y1, y2, y3

    def check(i, j, k):
        x_val = _rand_tensor((i, k), "float32")
        w1_val = _rand_tensor((j, k), "float32")
        w2_val = _rand_tensor((2 * j, k), "float32")
        w3_val = _rand_tensor((3 * j, k), "float32")

        output_before = before_pytorch(x_val, w1_val, w2_val, w3_val)
        # The 'expected' pass is transform.CombineParallelDense(min_num_branches=3, to_batch=False)
        output_expected = expected_pytorch(x_val, w1_val, w2_val, w3_val, j)

        for actual, expected in zip(output_before, output_expected):
            torch.testing.assert_close(actual, expected)

    check(3, 5, 4)
    check(100, 200, 300)


# Test 5: test_combine_parallel_dense_flat_biasadd
def test_combine_parallel_dense_flat_biasadd():
    """Testcase of combining dense + 1d biasadd with different out dims"""

    def before_pytorch(x, w1, w2, b1, b2):
        y1 = F.linear(x, w1)
        y2 = F.linear(x, w2)
        y1 = y1 + b1
        y2 = y2 + b2
        return y1, y2

    def expected_pytorch(x, w1, w2, b1, b2, j, bias_shape1, bias_shape2):
        # x: (i, k)
        # w1: (j, k)
        # w2: (2*j, k)
        w_stacked = torch.cat((w1, w2), dim=0) # (3*j, k)
        y = F.linear(x, w_stacked) # (i, 3*j)

        # Bias handling
        # TVM `repeat(tensor, repeats, axis)` repeats *elements*.
        # `relay.repeat(relay.expand_dims(b1, -1), j, 0)` -> for b1=() implies scalar
        # expand_dims(scalar, -1) -> (1,)
        # repeat((1,), j, 0) -> (j,)
        # Same for b2=() and (1,)
        # `bias_shape1` and `bias_shape2` can be `()`, `(1,)`, `(j,)`, `(i,j)`.

        b1_processed = b1
        if len(bias_shape1) == 0: # scalar bias
            b1_processed = b1.reshape((1,)) # ensure it's a 1-D tensor if scalar
            b1_processed = b1_processed.repeat(j) # repeat j times to match dimension j
        elif bias_shape1[-1] == 1: # if last dim is 1, repeat along last dim
            b1_processed = b1_processed.repeat(*([1] * (len(bias_shape1) - 1)), j)
        
        b2_processed = b2
        if len(bias_shape2) == 0: # scalar bias
            b2_processed = b2.reshape((1,))
            b2_processed = b2_processed.repeat(2 * j)
        elif bias_shape2[-1] == 1: # if last dim is 1, repeat along last dim
            b2_processed = b2_processed.repeat(*([1] * (len(bias_shape2) - 1)), 2 * j)
        
        # Concat bias
        # Determine the dimension for concatenation. For scalar to 1D, axis=0. For ND, it's last dim.
        concat_dim_b1 = len(bias_shape1) - 1 if len(bias_shape1) > 0 else 0
        concat_dim_b2 = len(bias_shape2) - 1 if len(bias_shape2) > 0 else 0
        # Given TVM uses `max(0, len(bias_shape1) - 1)` for concat axis, it implies
        # concatenation on the *last* dimension where features are.
        b_concat = torch.cat((b1_processed, b2_processed), dim=max(concat_dim_b1, concat_dim_b2))
        
        y = y + b_concat

        # Slicing
        # `begin = [0 for _ in range(n_out_dims - 1)]`
        # `end = [-1 for _ in range(n_out_dims - 1)]`
        # `strides = [1 for _ in range(n_out_dims)]`
        # This implies `y[..., 0:j]` and `y[..., j:j+(2*j)]`
        y1 = y[..., :j]
        y2 = y[..., j : 3 * j] 

        return y1, y2

    def check(i, j, k, bias_shape1, bias_shape2):
        x_val = _rand_tensor((i, k), "float32")
        w1_val = _rand_tensor((j, k), "float32")
        w2_val = _rand_tensor((2 * j, k), "float32")

        b1_val = _rand_tensor(bias_shape1, "float32")
        b2_val = _rand_tensor(bias_shape2, "float32")

        output_before = before_pytorch(x_val, w1_val, w2_val, b1_val, b2_val)
        # The 'expected' pass is transform.CombineParallelDense(min_num_branches=2, to_batch=False)
        output_expected = expected_pytorch(x_val, w1_val, w2_val, b1_val, b2_val, j, bias_shape1, bias_shape2)

        for actual, expected in zip(output_before, output_expected):
            torch.testing.assert_close(actual, expected)

    check(3, 5, 4, (), ())
    check(3, 5, 4, (1,), (1,))
    check(3, 5, 4, (5,), (1,))
    check(3, 5, 4, (1,), (10,))
    check(3, 5, 4, (3, 1), (3, 1))
    check(3, 5, 4, (3, 5), (3, 10))
    check(3, 5, 4, (3, 1), (3, 10))
    check(3, 5, 4, (3, 5), (3, 1))
    check(3, 5, 4, (9, 3, 5), (9, 3, 10))
    check(3, 5, 4, (9, 3, 5), (9, 3, 1))
    check(3, 5, 4, (9, 3, 1), (9, 3, 10))


# Test 6: test_combine_parallel_dense_flat_biasadd_scale_reshape
def test_combine_parallel_dense_flat_biasadd_scale_reshape():
    """Testcase of combining dense with different out dims
    following bias add, scale, reshape ops
    """

    def before_pytorch(x, w1, w2, b1, b2, scale1, scale2, newshape1, newshape2):
        y1 = F.linear(x, w1)
        y2 = F.linear(x, w2)
        y1 = y1 + b1
        y2 = y2 + b2
        y1 = y1 * scale1
        y2 = y2 * scale2
        y1 = y1.reshape(newshape1)
        y2 = y2.reshape(newshape2)
        return y1, y2

    def expected_pytorch(x, w1, w2, b1, b2, scale1, scale2, newshape1, newshape2, j):
        # x: (i, k)
        # w1: (j, k)
        # w2: (2*j, k)
        w_stacked = torch.cat((w1, w2), dim=0) # (3*j, k)
        y = F.linear(x, w_stacked) # (i, 3*j)

        # Bias (b1: (j,), b2: (2*j,))
        b_concat = torch.cat((b1, b2), dim=0) # (3*j,)
        y = y + b_concat

        # Scale (scale1: (1,), scale2: (1,))
        # TVM `repeat(scale1, j, 0)` -> (j,)
        # TVM `repeat(scale2, 2 * j, 0)` -> (2*j,)
        scale1_repeated = scale1.repeat(j) # (j,)
        scale2_repeated = scale2.repeat(2 * j) # (2*j,)
        scale_concat = torch.cat((scale1_repeated, scale2_repeated), dim=0) # (3*j,)
        y = y * scale_concat

        # Slicing and reshape
        y1 = y[:, :j]
        y2 = y[:, j : 3 * j] 

        y1 = y1.reshape(newshape1)
        y2 = y2.reshape(newshape2)
        return y1, y2

    def check(i, j, k, scale_val1, scale_val2, newshape1, newshape2):
        x_val = _rand_tensor((i, k), "float32")
        w1_val = _rand_tensor((j, k), "float32")
        w2_val = _rand_tensor((2 * j, k), "float32")
        b1_val = _rand_tensor((j,), "float32")
        b2_val = _rand_tensor((2 * j,), "float32")
        scale1_val = _rand_tensor((1,), "float32") + scale_val1 # Add constant to prevent 0 scale
        scale2_val = _rand_tensor((1,), "float32") + scale_val2

        output_before = before_pytorch(x_val, w1_val, w2_val, b1_val, b2_val, scale1_val, scale2_val, newshape1, newshape2)
        # The 'expected' pass is transform.CombineParallelDense(min_num_branches=2, to_batch=False)
        output_expected = expected_pytorch(x_val, w1_val, w2_val, b1_val, b2_val, scale1_val, scale2_val, newshape1, newshape2, j)

        for actual, expected in zip(output_before, output_expected):
            torch.testing.assert_close(actual, expected)

    check(3, 5, 4, 0.5, 0.25, (1, 1, 15), (1, 1, 30)) 
    check(100, 200, 300, 0.5, 0.25, (1, 1, 20000), (1, 1, 40000))


# Test 7: test_combine_parallel_dense_expand_dims
def test_combine_parallel_dense_expand_dims():
    """Verify that the correct slice axis is selected after the combined dense."""

    def before_pytorch(x, w1, w2):
        y1 = F.linear(x, w1)
        y1 = torch.unsqueeze(y1, dim=2)

        y2 = F.linear(x, w2)
        y2 = torch.unsqueeze(y2, dim=2)
        return y1, y2

    def expected_pytorch(x, w1, w2):
        # x: (2, 32)
        # w1: (16, 32)
        # w2: (8, 32)
        w_stacked = torch.cat((w1, w2), dim=0) # (24, 32)
        y = F.linear(x, w_stacked) # (2, 24)
        y = torch.unsqueeze(y, dim=2) # (2, 24, 1)

        # Slicing based on original shapes
        # y1 needs to be (2, 16, 1)
        # y2 needs to be (2, 8, 1)

        # strides = [1, 1, 1]
        # y1 = y[:, 0 : 0 + 16, 0 : 0 + 1] -> y[:, :16, :1]
        # y2 = y[:, 16 : 16 + 8, 0 : 0 + 1] -> y[:, 16:24, :1]
        
        y1 = y[:, :16, :1]
        y2 = y[:, 16:24, :1]
        return y1, y2

    # Concrete tensor shapes from the original TVM test
    i = 2 # batch size
    k = 32 # input features
    j1 = 16 # output features for w1
    j2 = 8 # output features for w2

    x_val = _rand_tensor((i, k), "float32")
    w1_val = _rand_tensor((j1, k), "float32")
    w2_val = _rand_tensor((j2, k), "float32")

    output_before = before_pytorch(x_val, w1_val, w2_val)
    # The 'expected' pass is transform.CombineParallelDense(min_num_branches=2, to_batch=False)
    output_expected = expected_pytorch(x_val, w1_val, w2_val)

    for actual, expected in zip(output_before, output_expected):
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    pytest.main([__file__])
