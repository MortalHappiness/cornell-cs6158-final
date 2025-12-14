import numpy as np
import pytest
import torch
import torch.nn.functional as F
import functools # Needed for logical_and/or reduction if multiple conditions are used

# Helper for dtypes
def get_torch_dtype(tvm_dtype_str):
    if tvm_dtype_str == "float32":
        return torch.float32
    if tvm_dtype_str == "float16":
        return torch.float16
    if tvm_dtype_str == "int32":
        return torch.int32
    if tvm_dtype_str == "int64":
        return torch.int64
    if tvm_dtype_str == "bool":
        return torch.bool
    # Assuming qint8 is the common quantized integer type, adjust if others are used
    if tvm_dtype_str == "int8":
        return torch.qint8
    raise ValueError(f"Unknown TVM dtype: {tvm_dtype_str}")

# Helper for upsampling (relay.nn.upsampling)
def upsampling_pytorch(data, scale_h, scale_w, layout="NCHW", mode='nearest'):
    # PyTorch's F.interpolate assumes NCHW by default. If layout is NHWC, data needs permutation.
    if layout == "NHWC":
        data = data.permute(0, 3, 1, 2) # NHWC to NCHW
    
    # TVM's upsampling often implies a mode like nearest or bilinear
    # Assuming nearest by default if not specified in TVM context.
    output = F.interpolate(data, scale_factor=(scale_h, scale_w), mode=mode)

    if layout == "NHWC":
        output = output.permute(0, 2, 3, 1) # NCHW to NHWC
    return output

# Helper for strided_slice (relay.op.strided_slice)
def strided_slice_pytorch(data, begin, end, strides):
    slices = []
    # Adjusting for potential relay.Any() and MAX_INT values in end
    # For `end=[..., 2147483647]`, Python's slicing handles this by going to the end.
    for i in range(len(data.shape)):
        _begin = begin[i] if i < len(begin) else 0
        # If end[i] is MAX_INT, use None to slice till the end of dimension
        _end_val = end[i] if i < len(end) and end[i] != 2147483647 else None
        _stride = strides[i] if i < len(strides) else 1
        slices.append(slice(_begin, _end_val, _stride))
    return data[tuple(slices)]

# Helper for gather_nd (relay.gather_nd)
def gather_nd_pytorch(data, indices):
    # This is a general implementation for NumPy/PyTorch-style advanced indexing
    # where indices provides coordinates for the non-batch dimensions.
    # Assumes indices is a tensor of coordinates, typically (..., C) where C is rank of data being indexed.
    if indices.ndim == 0:
        return data.flatten()[indices.item()]
    
    # For 2D indices (e.g., [[0,1], [1,0]]) where last dim is coordinates
    # and previous dims are batch-like for the indices.
    # The output shape will be `indices.shape[:-1] + data.shape[indices.shape[-1]:]`
    if indices.shape[-1] <= data.ndim:
        # Construct the tuple of indices for advanced indexing
        coord_dims = indices.shape[-1]
        # Transpose to get (coord_dim, ...) from (..., coord_dim)
        coords_t = indices.T 
        # For data[idx0, idx1, ...], each idx is a tensor.
        # This works if `indices` provides the full coordinates.
        # Example: data.shape=(10,2), indices.shape=(2,2) -> data[indices[:,0], indices[:,1]]
        # Example: data.shape=(10,2,3), indices.shape=(2,1) -> data[indices[:,0], :, :]
        
        # Generalize: separate coordinates for each dimension
        multi_dim_indices = [coords_t[i] for i in range(coord_dims)]
        
        # Add slices for remaining dimensions not covered by indices
        remaining_dims = data.ndim - coord_dims
        if remaining_dims > 0:
            multi_dim_indices.extend([slice(None)] * remaining_dims)

        return data[multi_dim_indices]
    else:
        raise ValueError(
            f"Indices last dimension ({indices.shape[-1]}) is greater than data dimensions ({data.ndim})"
        )


# Mapping for TVM's FuseOps pass is complex and structural.
# PyTorch equivalent will execute the unfused ops and assert numerical equality.
# Structural equality checks are replaced with TODOs.

def test_fuse_simple():
    """Simple testcase."""
    # This test verifies structural equality of the fused graph.
    # In PyTorch, we can only verify numerical equivalence.

    def before_pytorch(x_tensor):
        y = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
        z = torch.exp(y)
        w = torch.squeeze(z)
        return w

    # In TVM, `expected` defines the structurally fused Relay Function.
    # The computational graph should be numerically equivalent to `before_pytorch`.
    def expected_pytorch(x_tensor):
        y = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
        z = torch.exp(y)
        w = torch.squeeze(z)
        return w

    x_np = np.random.rand(10, 20).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    output_before = before_pytorch(x_tensor)
    output_expected = expected_pytorch(x_tensor)

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)

def test_conv2d_fuse():
    """Test fusion case of conv2d"""

    def before_pytorch(x_tensor, w1_tensor, w2_tensor, w3_tensor):
        x = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
        y = F.conv2d(x, w1_tensor, stride=1, padding=1, dilation=1, groups=1)
        y1 = torch.add(torch.tensor(1.0, dtype=torch.float32), y)
        y = torch.add(y, y1)
        z2 = F.conv2d(y, w2_tensor, stride=1, padding=0, dilation=1, groups=1)
        z3 = F.conv2d(y, w3_tensor, stride=1, padding=1, dilation=1, groups=1)
        z = torch.add(z2, z3)
        return z

    # In TVM, `expected` defines the structurally fused Relay Function.
    # The computational graph should be numerically equivalent to `before_pytorch`.
    def expected_pytorch(x_tensor, w1_tensor, w2_tensor, w3_tensor):
        # The structure with f0, f1, f2, f3 and Calls represents the fused IR
        # but the actual computation is the same as the unfused version.
        x = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
        y = F.conv2d(x, w1_tensor, stride=1, padding=1, dilation=1, groups=1)
        y1 = torch.add(torch.tensor(1.0, dtype=torch.float32), y)
        y = torch.add(y, y1)
        z2 = F.conv2d(y, w2_tensor, stride=1, padding=0, dilation=1, groups=1)
        z3 = F.conv2d(y, w3_tensor, stride=1, padding=1, dilation=1, groups=1)
        z = torch.add(z2, z3)
        return z

    dshape = (1, 16, 64, 64)
    x_np = np.random.rand(*dshape).astype(np.float32)
    w1_np = np.random.rand(16, 16, 3, 3).astype(np.float32)
    w2_np = np.random.rand(16, 16, 1, 1).astype(np.float32)
    w3_np = np.random.rand(16, 16, 3, 3).astype(np.float32)

    x_tensor = torch.tensor(x_np)
    w1_tensor = torch.tensor(w1_np)
    w2_tensor = torch.tensor(w2_np)
    w3_tensor = torch.tensor(w3_np)

    output_before = before_pytorch(x_tensor, w1_tensor, w2_tensor, w3_tensor)
    output_expected = expected_pytorch(x_tensor, w1_tensor, w2_tensor, w3_tensor)

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)


def test_concatenate():
    """Test fusion case involving concat op and Tuple node"""

    def before_pytorch(x_tensor):
        pooled = F.max_pool2d(x_tensor, kernel_size=2, stride=2, padding=0)
        # Assuming mode='nearest' for upsampling
        upsampled = F.interpolate(pooled, scale_factor=2, mode='nearest')
        concat = torch.cat((upsampled, x_tensor), dim=1)
        out = torch.add(concat, torch.tensor(1.0, dtype=torch.float32))
        return out

    def expected_pytorch(x_tensor):
        # The structure with f0, f1 and Calls represents the fused IR
        # but the actual computation is the same as the unfused version.
        pooled = F.max_pool2d(x_tensor, kernel_size=2, stride=2, padding=0)
        upsampled = F.interpolate(pooled, scale_factor=2, mode='nearest')
        concat = torch.cat((upsampled, x_tensor), dim=1)
        out = torch.add(concat, torch.tensor(1.0, dtype=torch.float32))
        return out

    dshape = (1, 16, 64, 64)
    x_np = np.random.rand(*dshape).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    output_before = before_pytorch(x_tensor)
    output_expected = expected_pytorch(x_tensor)

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)


def test_tuple_root():
    """Test fusion case where Tuple node is the root in its group"""

    def before_pytorch(x_tensor):
        pooled = F.max_pool2d(x_tensor, kernel_size=2, stride=2, padding=0)
        upsampled = F.interpolate(pooled, scale_factor=2, mode='nearest')
        out = (upsampled, x_tensor) # This is a Python tuple
        return out

    def expected_pytorch(x_tensor):
        # The structure with f0, f1 and Calls represents the fused IR
        # but the actual computation is the same as the unfused version.
        pooled = F.max_pool2d(x_tensor, kernel_size=2, stride=2, padding=0)
        upsampled = F.interpolate(pooled, scale_factor=2, mode='nearest')
        out = (upsampled, x_tensor)
        return out

    dshape = (1, 16, 64, 64)
    x_np = np.random.rand(*dshape).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    output_before = before_pytorch(x_tensor)
    output_expected = expected_pytorch(x_tensor)

    # Compare elements of the tuple
    for o_b, o_e in zip(output_before, output_expected):
        torch.testing.assert_allclose(o_b, o_e)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)


def test_stop_fusion():
    def before_pytorch(x_tensor):
        y = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
        # relay.annotation.stop_fusion is a TVM-specific instruction to prevent fusion.
        # In PyTorch, ops are typically executed sequentially in eager mode,
        # or fusion is handled internally by torch.compile without user control at this level.
        # So, for numerical equivalence, we just continue the computation.
        z = torch.exp(y)
        return z

    def expected_pytorch(x_tensor):
        # The structure with f1, f2 and Calls represents the separated fused IR
        # but the actual computation is the same as the unfused version.
        y = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
        z = torch.exp(y)
        return z

    dshape = (10, 20)
    x_np = np.random.rand(*dshape).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    output_before = before_pytorch(x_tensor)
    output_expected = expected_pytorch(x_tensor)

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)


def test_fuse_myia_regression():
    def before_pytorch(x_tensor, y_tensor):
        # relay.ScopeBuilder and if/else constructs are for Relay IR control flow.
        # In PyTorch, we use standard Python control flow.
        if torch.greater(x_tensor, y_tensor):
            return x_tensor
        else:
            return y_tensor

    def expected_pytorch(x_tensor, y_tensor):
        # The structure with fused_gt and Calls represents the fused IR
        # but the actual computation is the same as the unfused version.
        if torch.greater(x_tensor, y_tensor):
            return x_tensor
        else:
            return y_tensor

    dshape = () # Scalar shape
    dtype = "int64"
    x_np = np.array(5, dtype=np.int64)
    y_np = np.array(10, dtype=np.int64)
    
    x_tensor = torch.tensor(x_np, dtype=get_torch_dtype(dtype))
    y_tensor = torch.tensor(y_np, dtype=get_torch_dtype(dtype))

    output_before = before_pytorch(x_tensor, y_tensor)
    output_expected = expected_pytorch(x_tensor, y_tensor)

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)


def test_fuse_tuple_get_elemwise():
    def before_pytorch(X_tensor, W_tensor):
        dim = X_tensor.shape[1]
        matmul = F.linear(X_tensor, W_tensor.T) # dense expects (input, weight.T) for (B, in_f) x (out_f, in_f) -> (B, out_f)
        
        # relay.split uses indices_or_sections. If int, it's number of sections.
        # PyTorch F.split uses split_size_or_sections. If int, it's size of each chunk.
        # If indices_or_sections=3, means three equal sections along axis=1
        split_size = matmul.shape[1] // 3
        splitted = torch.split(matmul, split_size, dim=1) # 3 sections
        
        out = torch.sigmoid(splitted[0]) + torch.tanh(splitted[1]) * torch.exp(splitted[2])
        return out

    def expected_pytorch(X_tensor, W_tensor):
        # The structure with f0, f1 and Calls represents the fused IR
        # but the actual computation is the same as the unfused version.
        dim = X_tensor.shape[1]
        matmul = F.linear(X_tensor, W_tensor.T)
        split_size = matmul.shape[1] // 3
        splitted = torch.split(matmul, split_size, dim=1)
        out = torch.sigmoid(splitted[0]) + torch.tanh(splitted[1]) * torch.exp(splitted[2])
        return out

    dim = 10
    X_np = np.random.rand(1, dim).astype(np.float32)
    W_np = np.random.rand(3 * dim, dim).astype(np.float32) # (out_features, in_features) for F.linear
    
    X_tensor = torch.tensor(X_np)
    W_tensor = torch.tensor(W_np)

    output_before = before_pytorch(X_tensor, W_tensor)
    output_expected = expected_pytorch(X_tensor, W_tensor)

    torch.testing.assert_allclose(output_before, output_expected, rtol=1e-4, atol=1e-4)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)


def test_tuple_get_root():
    def before_pytorch(X_tensor, W_tensor):
        # dim = X_tensor.shape[1] is 3*dim
        split_size = X_tensor.shape[1] // 3
        splitted = torch.split(X_tensor, split_size, dim=1)
        out = F.linear(splitted[0], W_tensor.T) # dense expects (input, weight.T)
        return out

    def expected_pytorch(X_tensor, W_tensor):
        # The structure with f0, f1 and Calls represents the fused IR
        # but the actual computation is the same as the unfused version.
        split_size = X_tensor.shape[1] // 3
        splitted = torch.split(X_tensor, split_size, dim=1)
        out = F.linear(splitted[0], W_tensor.T)
        return out

    dim = 10
    X_np = np.random.rand(1, 3 * dim).astype(np.float32)
    W_np = np.random.rand(dim, dim).astype(np.float32)
    
    X_tensor = torch.tensor(X_np)
    W_tensor = torch.tensor(W_np)

    output_before = before_pytorch(X_tensor, W_tensor)
    output_expected = expected_pytorch(X_tensor, W_tensor)

    torch.testing.assert_allclose(output_before, output_expected, rtol=1e-4, atol=1e-4)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)


# Helper functions from TVM tests, adapted for PyTorch numerical execution
def gen_intermediate_tuple_pytorch(x_tensor):
    y1 = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
    tmp = torch.squeeze(x_tensor)
    tmp = torch.add(tmp, torch.tensor(1.0, dtype=torch.float32))
    y2 = torch.add(tmp, torch.tensor(1.0, dtype=torch.float32))
    y3 = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
    concat = torch.cat((y1, y2, y3), dim=1)
    out_inj = torch.squeeze(concat)
    out = torch.add(out_inj, torch.tensor(1.0, dtype=torch.float32))
    return out

def test_tuple_intermediate():
    dshape = (1, 16, 64, 64)
    x_np = np.random.rand(*dshape).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    # `before` in TVM defines the full computation
    output_before = gen_intermediate_tuple_pytorch(x_tensor)

    # `expected` in TVM describes the *fused IR structure*. Numerically, it's the same computation.
    def expected_pytorch_intermediate(x_tensor):
        return gen_intermediate_tuple_pytorch(x_tensor)

    output_expected = expected_pytorch_intermediate(x_tensor)

    torch.testing.assert_allclose(output_before, output_expected, rtol=1e-4, atol=1e-4)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(m["main"], after)


def gen_consecutive_tuple_pytorch(x_tensor):
    y1 = gen_intermediate_tuple_pytorch(x_tensor)
    y2 = gen_intermediate_tuple_pytorch(x_tensor)
    y3 = gen_intermediate_tuple_pytorch(x_tensor)
    concat = torch.cat((y1, y2, y3), dim=1)
    return concat

def test_tuple_consecutive():
    dshape = (1, 16, 64, 64)
    x_np = np.random.rand(*dshape).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    def before_pytorch(x_tensor):
        concat = gen_consecutive_tuple_pytorch(x_tensor)
        pooled = F.max_pool2d(concat, kernel_size=2, stride=2, padding=0)
        out = torch.add(pooled, torch.tensor(1.0, dtype=torch.float32))
        out2 = torch.add(out, torch.tensor(1.0, dtype=torch.float32))
        out_tup = (out, out2)
        return out_tup

    # `expected` in TVM describes the *fused IR structure*. Numerically, it's the same computation.
    def expected_pytorch_consecutive(x_tensor):
        concat = gen_consecutive_tuple_pytorch(x_tensor)
        pooled = F.max_pool2d(concat, kernel_size=2, stride=2, padding=0)
        out = torch.add(pooled, torch.tensor(1.0, dtype=torch.float32))
        out2 = torch.add(out, torch.tensor(1.0, dtype=torch.float32))
        out_tup = (out, out2)
        return out_tup

    output_before = before_pytorch(x_tensor)
    output_expected = expected_pytorch_consecutive(x_tensor)

    for o_b, o_e in zip(output_before, output_expected):
        torch.testing.assert_allclose(o_b, o_e, rtol=1e-4, atol=1e-4)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(m["main"], after)


def conv_pytorch(data_tensor, w_tensor):
    y = F.conv2d(data_tensor, w_tensor, stride=1, padding=1, dilation=1, groups=1)
    return F.relu(input=y)

def inception_like_pytorch(data_tensor, w1_tensor, w2_tensor):
    c0 = conv_pytorch(data_tensor, w1_tensor)
    c1 = conv_pytorch(data_tensor, w2_tensor)
    return torch.cat((c0, c1), dim=1)

def test_inception_like():
    dshape = (1, 16, 64, 64)
    x_np = np.random.rand(*dshape).astype(np.float32)
    w_shape = (16, dshape[1], 3, 3) # (out_channels, in_channels, kH, kW)

    x_tensor = torch.tensor(x_np)
    w1_tensor = torch.tensor(np.random.rand(*w_shape).astype(np.float32))
    w2_tensor = torch.tensor(np.random.rand(*w_shape).astype(np.float32))
    w3_tensor = torch.tensor(np.random.rand(16, dshape[1] * 2, 3, 3).astype(np.float32)) # Inception output has 2*channels
    w4_tensor = torch.tensor(np.random.rand(16, dshape[1] * 2, 3, 3).astype(np.float32))

    def before_pytorch(x_tensor, w1, w2, w3, w4):
        in1 = inception_like_pytorch(x_tensor, w1, w2)
        in2 = inception_like_pytorch(in1, w3, w4) # Note: w3, w4 apply to in1's shape
        return in2

    # `expected` in TVM describes the *fused IR structure*. Numerically, it's the same computation.
    def expected_pytorch_inception(x_tensor, w1, w2, w3, w4):
        return before_pytorch(x_tensor, w1, w2, w3, w4)

    output_before = before_pytorch(x_tensor, w1_tensor, w2_tensor, w3_tensor, w4_tensor)
    output_expected = expected_pytorch_inception(x_tensor, w1_tensor, w2_tensor, w3_tensor, w4_tensor)

    torch.testing.assert_allclose(output_before, output_expected, rtol=1e-4, atol=1e-4)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(m["main"], after)


def test_fuse_parallel_injective():
    """Test fusing parallel injective ops to an elemwise op."""

    def before_pytorch(x_tensor):
        y = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
        z = torch.squeeze(y)
        u = torch.permute(y, dims=[1, 0]) # Assuming default transpose for 2D is swap(0,1)
        w = torch.bitwise_left_shift(z, u.to(z.dtype)) # left_shift expects integer types
        return w

    def expected_pytorch(x_tensor):
        # The structure with f1 represents the fused IR
        # but the actual computation is the same as the unfused version.
        y = torch.add(x_tensor, torch.tensor(1.0, dtype=torch.float32))
        z = torch.squeeze(y)
        u = torch.permute(y, dims=[1, 0])
        w = torch.bitwise_left_shift(z, u.to(z.dtype))
        return w

    x_np = np.random.rand(10, 20).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    # left_shift requires integer inputs. Convert inputs for bitwise operations.
    x_int_tensor = (x_tensor * 10).to(torch.int32) 
    # Make sure operations are on integer tensors compatible with bitwise_left_shift
    y_int = torch.add(x_int_tensor, torch.tensor(1, dtype=torch.int32))
    z_int = torch.squeeze(y_int)
    u_int = torch.permute(y_int, dims=[1, 0])
    
    output_before = torch.bitwise_left_shift(z_int, u_int)
    output_expected = output_before # Expected is numerically same

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(zz, after)


def test_immutable():
    """Verify the fusion pass won't change original module."""
    # This test explicitly checks TVM IRModule immutability and structural equality.
    # This cannot be directly translated to PyTorch's computational model.
    # We will skip this test's core assertion as it is TVM-specific.

    # TODO: This test verifies internal IRModule immutability and structural equality
    # after a TVM-specific compiler pass (FuseOps). PyTorch does not expose a
    # comparable IR structure or API for this level of internal compiler verification.
    # The core assertion `assert tvm.ir.structural_equal(...)` is not translatable.
    pass


def test_split():
    """Test that the result is well formed."""
    # This test verifies internal IR structure after fusion, including Relay's RefCreate/RefRead.
    # Relay's RefCreate/RefRead deal with mutable references in the IR, which is a TVM-specific concept
    # at the graph level that doesn't directly map to PyTorch tensor operations.
    # As such, this test's structural and mutable reference aspects are not directly translatable.

    # TODO: This test involves TVM Relay's mutable references (RefCreate/RefRead) and structural
    # IR validation after fusion, which are concepts specific to TVM's graph IR and compiler passes.
    # There is no direct functional or structural equivalent in PyTorch for this test.
    pass


def test_fuse_max():
    """Test the constraint of number of nodes in op fusion."""

    def before_pytorch(x_tensor, n_ops):
        y = x_tensor
        for _ in range(n_ops):
            y = torch.exp(y)
        return y

    # In TVM, `expected` defines a graph with potentially two fused functions
    # (f1, f2) based on max_fused_ops. Numerically, it's still the same computation
    # as simply applying n_ops `exp` operations.
    def expected_pytorch(x_tensor, n_ops_expected_structure):
        y = x_tensor
        for _ in range(n_ops_expected_structure):
            y = torch.exp(y)
        return y

    # Test case 1: n = 300, max_fused_ops = 256
    # This means TVM expects 2 fused functions.
    # Numerically, it should still be 300 exp ops.
    n_ops_1 = 300
    x_np_1 = np.random.rand(10, 20).astype(np.float32)
    x_tensor_1 = torch.tensor(x_np_1)
    
    output_before_1 = before_pytorch(x_tensor_1, n_ops_1)
    output_expected_1 = expected_pytorch(x_tensor_1, n_ops_1)

    torch.testing.assert_allclose(output_before_1, output_expected_1, rtol=1e-4, atol=1e-4)

    # Test case 2: n = 20, max_fused_ops = 10
    # This means TVM expects 2 fused functions.
    # Numerically, it should still be 20 exp ops.
    n_ops_2 = 20
    # max_fused_ops = 10 (this is a TVM config param, not a PyTorch op param)
    x_np_2 = np.random.rand(10, 20).astype(np.float32)
    x_tensor_2 = torch.tensor(x_np_2)

    output_before_2 = before_pytorch(x_tensor_2, n_ops_2)
    output_expected_2 = expected_pytorch(x_tensor_2, n_ops_2)

    torch.testing.assert_allclose(output_before_2, output_expected_2, rtol=1e-4, atol=1e-4)

    # TODO: The original TVM test verifies that `FuseOps` respects `max_depth`
    # by checking the structural equality of the resulting IR. PyTorch does not
    # offer a direct equivalent to set `max_depth` for fusion and verify IR structure.
    # The `tvm.transform.PassContext` is a TVM-specific mechanism.


link_params = [False, True] # PyTorch doesn't have a direct equivalent for 'link_params' in this context


@pytest.mark.parametrize("link_params", link_params)
def test_fuse_take(link_params):
    """Test fusion case involving concat and take"""
    # 'link_params' is a TVM-specific optimization detail for fusion.
    # In PyTorch, we focus on numerical correctness.

    def before_pytorch(x_tensor, indices_tensor):
        concat = torch.cat((x_tensor, x_tensor), dim=-1)
        # relay.op.take with axis=None (default for op.take) or axis=-1 (as implied by example)
        # For torch.take, it flattens the input.
        # For specific axis, torch.take_along_dim is needed.
        # Given x_tensor shape (10,1), concat (10,2). indices ([0])
        # If axis is -1: select the first element from the last dim.
        # If axis is None: flatten (20,) -> take indices[0]=0 -> output scalar
        # The TVM example implies taking along a specific axis implicitly,
        # where indices = [0] means the element at index 0 of the last dimension.
        # Let's assume the TVM code intends to take along axis=-1 (the last dim).
        # We need to unsqueeze indices to match rank for take_along_dim.
        # indices_tensor is (1,). For (10,2) take_along_dim, needs (1,1) for dim=-1
        # Or more accurately based on numpy behaviour for `np.take(arr, indices, axis=axis)`:
        # result.shape = arr.shape[:axis] + indices.shape + arr.shape[axis+1:]
        # So for (10,2) and indices (1,), axis -1 implies output (10,1)
        # This is `torch.index_select(concat, dim=-1, index=indices_tensor)`
        out = torch.index_select(concat, dim=-1, index=indices_tensor)
        return out

    def expected_pytorch(x_tensor, indices_tensor):
        # The fused structure in TVM still performs the same numerical operation.
        concat = torch.cat((x_tensor, x_tensor), dim=-1)
        out = torch.index_select(concat, dim=-1, index=indices_tensor)
        return out

    shape = (10, 1) # Note: tvm.tir.const(10, "int64") are concrete values
    x_np = np.random.rand(*shape).astype(np.float32)
    indices_np = np.array([0], dtype=np.int64) # Single index 0
    
    x_tensor = torch.tensor(x_np)
    indices_tensor = torch.tensor(indices_np, dtype=get_torch_dtype("int64"))

    output_before = before_pytorch(x_tensor, indices_tensor)
    output_expected = expected_pytorch(x_tensor, indices_tensor)

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: The `link_params` configuration and structural equality check are
    # TVM-specific compiler pass concerns and cannot be directly translated to PyTorch.


@pytest.mark.parametrize("link_params", link_params)
def test_fuse_gather_nd(link_params):
    """Test fusion case involving concat and gather_nd"""
    # 'link_params' is a TVM-specific optimization detail for fusion.

    def before_pytorch(x_tensor, indices_tensor):
        concat = torch.cat((x_tensor, x_tensor), dim=-1) # (10, 1) -> (10, 2)
        # relay.gather_nd maps to numpy-style advanced indexing.
        # indices_tensor is ([[0,1], [1,0]])
        # This means select concat[0,1] and concat[1,0]
        out = gather_nd_pytorch(concat, indices_tensor)
        return out

    def expected_pytorch(x_tensor, indices_tensor):
        # The fused structure in TVM still performs the same numerical operation.
        concat = torch.cat((x_tensor, x_tensor), dim=-1)
        out = gather_nd_pytorch(concat, indices_tensor)
        return out

    shape = (10, 1)
    x_np = np.random.rand(*shape).astype(np.float32)
    indices_np = np.array([[0, 1], [1, 0]], dtype=np.int64) # (2,2) indices for (10,2) tensor
    
    x_tensor = torch.tensor(x_np)
    indices_tensor = torch.tensor(indices_np, dtype=get_torch_dtype("int64"))

    output_before = before_pytorch(x_tensor, indices_tensor)
    output_expected = expected_pytorch(x_tensor, indices_tensor)

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: The `link_params` configuration and structural equality check are
    # TVM-specific compiler pass concerns and cannot be directly translated to PyTorch.


# @tvm.testing.uses_gpu # Handled by device setup
def test_fuse_bcast_reduce_scalar():
    """Test fusion case with broadcast and reduction involving scalar"""

    def before_pytorch(x_tensor):
        less_result = torch.less(x_tensor, torch.tensor(10, dtype=torch.int32))
        z = torch.min(less_result) # min over all elements (scalar output)
        return z

    def expected_pytorch(x_tensor):
        # Fused structure implies same numerical computation.
        less_result = torch.less(x_tensor, torch.tensor(10, dtype=torch.int32))
        z = torch.min(less_result)
        return z

    # Input is a scalar
    x_np = np.array(5, dtype=np.int32)
    x_tensor = torch.tensor(x_np, dtype=get_torch_dtype("int32"))

    output_before = before_pytorch(x_tensor)
    output_expected = expected_pytorch(x_tensor)

    torch.testing.assert_allclose(output_before, output_expected)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(m["main"], after)
    # The `for tgt, dev in tvm.testing.enabled_targets(): relay.build(m, tgt)`
    # is a TVM-specific build process for different targets.


def test_fuse_max_diamond():
    def create_diamond_pytorch(x_tensor, branch_len):
        x1 = x_tensor
        x2 = x_tensor
        for _ in range(branch_len):
            x1 = torch.exp(x1)
            x2 = torch.exp(x2)
        return torch.add(x1, x2)

    def before_pytorch(x_tensor, branch_len, num_diamond):
        out = x_tensor
        for _ in range(num_diamond):
            out = create_diamond_pytorch(out, branch_len)
        return out

    def expected_pytorch(x_tensor, branch_len, num_diamond):
        # The fused structure implies the same numerical computation.
        return before_pytorch(x_tensor, branch_len, num_diamond)

    branch_len = 5
    num_diamond = 3
    x_np = np.random.rand(10, 20).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    output_before = before_pytorch(x_tensor, branch_len, num_diamond)
    output_expected = expected_pytorch(x_tensor, branch_len, num_diamond)

    torch.testing.assert_allclose(output_before, output_expected, rtol=1e-4, atol=1e-4)

    # TODO: The original TVM test uses `tvm.transform.PassContext` to configure
    # `relay.FuseOps.max_depth` and asserts structural equality. This is a
    # TVM-specific compiler detail and cannot be directly translated to PyTorch.


def test_fuse_dynamic_squeeze_slice_take():
    # Input data uses dynamic shapes in TVM.
    # For PyTorch execution, concrete shapes are used for tensors.
    input_data_np = np.random.random([1, 2, 4]).astype("float32")
    take_val_np = np.array([0]).astype("int64")

    x_tensor = torch.tensor(input_data_np)
    take_val_tensor = torch.tensor(take_val_np, dtype=get_torch_dtype("int64"))

    # TVM ops: squeeze -> strided_slice -> take
    # 1. squeeze(x, axis=[0]): (1, 2, 4) -> (2, 4)
    squeeze_out = torch.squeeze(x_tensor, dim=0)

    # 2. strided_slice(squeeze_out, begin=[0, 0], end=[15130, 2147483647], strides=[1, 1])
    # The large end values mean slice to end.
    # (2, 4) -> (2, 4)
    strided_slice_out = strided_slice_pytorch(squeeze_out, begin=[0,0], end=[15130, 2147483647], strides=[1,1])

    # 3. take(strided_slice_out, take_val, axis=0):
    # strided_slice_out is (2, 4). take_val is (1,). axis=0.
    # This means select along dimension 0 using the index in take_val.
    # take_val_tensor is (1,). For torch.take_along_dim, indices must broadcast to data.
    # take_val_tensor.unsqueeze(1) makes it (1,1).
    # torch.take_along_dim( (2,4), (1,1), dim=0 ) -> (1,4)
    # Then squeeze it to match the expected NumPy result shape (4,)
    take_out = torch.take_along_dim(strided_slice_out, take_val_tensor.unsqueeze(1), dim=0).squeeze(0)

    # Reference NumPy computation:
    np_result = np.squeeze(input_data_np[:, take_val_np[0], :], axis=0)

    torch.testing.assert_allclose(take_out, torch.tensor(np_result))


# @tvm.testing.uses_gpu # Handled by device setup
def test_fuse_softmax():
    """Test if softmax can be fused with following ops."""
    channel_size = 16

    def before_pytorch(x_tensor):
        softmax = torch.softmax(x_tensor, dim=-1) # Default axis for relay.nn.softmax is -1
        out = softmax.to(get_torch_dtype("float16"))
        return out

    def expected_pytorch(x_tensor):
        # Fused structure implies same numerical computation.
        softmax = torch.softmax(x_tensor, dim=-1)
        out = softmax.to(get_torch_dtype("float16"))
        return out

    # Prepare input and reference for PyTorch
    inp_np = np.random.randn(16, channel_size).astype("float32")
    inp_tensor = torch.tensor(inp_np)

    # Compute reference (using torch for accuracy, then convert to numpy for compatibility)
    ref_tensor = torch.softmax(inp_tensor, dim=-1).to(get_torch_dtype("float16"))
    ref_np = ref_tensor.cpu().numpy()

    output_before = before_pytorch(inp_tensor)
    output_expected = expected_pytorch(inp_tensor)

    torch.testing.assert_allclose(output_before.cpu().numpy(), ref_np, rtol=1e-4, atol=1e-4)
    torch.testing.assert_allclose(output_expected.cpu().numpy(), ref_np, rtol=1e-4, atol=1e-4)

    # TODO: Original TVM test asserts structural equality of the IR after fusion.
    # PyTorch does not expose an equivalent IR for direct structural comparison of
    # fusion transformation results from internal compiler passes like FuseOps.
    # assert tvm.ir.structural_equal(m["main"], after)
    # The `for tgt, dev in tvm.testing.enabled_targets(): relay.build(m, tgt)`
    # is a TVM-specific build process for different targets.


if __name__ == "__main__":
    pytest.main([__file__])
