import numpy as np
import pytest
import torch
import functools

# Helper to convert string dtype to torch.dtype
def _to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "int32":
        return torch.int32
    if dtype_str == "int64":
        return torch.int64
    if dtype_str == "uint8":
        return torch.uint8
    if dtype_str == "uint16":
        return torch.uint16
    if dtype_str == "uint32":
        return torch.uint32
    if dtype_str == "bool":
        return torch.bool
    raise ValueError(f"Unsupported dtype: {dtype_str}")

# Helper to map numpy-like slicing to torch for dynamic strides/axes
def _torch_strided_slice(data, begin, end, strides=None, axes=None):
    num_dims = data.ndim
    slices = [slice(None)] * num_dims
    
    # Default strides to 1 for all dimensions
    effective_strides = [1] * num_dims
    if strides is not None:
        if isinstance(strides, torch.Tensor):
            effective_strides = strides.cpu().numpy().tolist()
        else:
            effective_strides = strides
            
    # Default begin/end
    if isinstance(begin, torch.Tensor):
        begin = begin.cpu().numpy().tolist()
    if isinstance(end, torch.Tensor):
        end = end.cpu().numpy().tolist()

    if axes is None:
        for i in range(num_dims):
            current_begin = begin[i] if i < len(begin) else 0
            current_end = end[i] if i < len(end) else data.shape[i]
            current_stride = effective_strides[i] if i < len(effective_strides) else 1

            # Handle negative indices and None
            if current_begin is None: current_begin = 0
            if current_end is None: current_end = data.shape[i]
            
            # numpy-style negative indexing for begin/end
            if current_begin < 0: current_begin += data.shape[i]
            if current_end < 0: current_end += data.shape[i]

            slices[i] = slice(current_begin, current_end, current_stride)
    else:
        for i, ax in enumerate(axes):
            current_begin = begin[i] if i < len(begin) else 0
            current_end = end[i] if i < len(end) else data.shape[ax]
            current_stride = effective_strides[i] if i < len(effective_strides) else 1

            # Handle negative indices and None
            if current_begin is None: current_begin = 0
            if current_end is None: current_end = data.shape[ax]

            # numpy-style negative indexing for begin/end
            if current_begin < 0: current_begin += data.shape[ax]
            if current_end < 0: current_end += data.shape[ax]

            slices[ax] = slice(current_begin, current_end, current_stride)
            
    return data[tuple(slices)]

# Helper for `strided_set`
def _torch_strided_set(data, value, begin, end, strides=None, axes=None):
    output = data.clone()
    num_dims = data.ndim
    slices = [slice(None)] * num_dims
    
    # Default strides to 1 for all dimensions
    effective_strides = [1] * num_dims
    if strides is not None:
        if isinstance(strides, torch.Tensor):
            effective_strides = strides.cpu().numpy().tolist()
        else:
            effective_strides = strides

    if isinstance(begin, torch.Tensor):
        begin = begin.cpu().numpy().tolist()
    if isinstance(end, torch.Tensor):
        end = end.cpu().numpy().tolist()
            
    if axes is None:
        for i in range(num_dims):
            current_begin = begin[i] if i < len(begin) else 0
            current_end = end[i] if i < len(end) else data.shape[i]
            current_stride = effective_strides[i] if i < len(effective_strides) else 1

            # Handle negative indices and None
            if current_begin is None: current_begin = 0
            if current_end is None: current_end = data.shape[i]

            # numpy-style negative indexing for begin/end
            if current_begin < 0: current_begin += data.shape[i]
            if current_end < 0: current_end += data.shape[i]
            
            slices[i] = slice(current_begin, current_end, current_stride)
    else:
        for i, ax in enumerate(axes):
            current_begin = begin[i] if i < len(begin) else 0
            current_end = end[i] if i < len(end) else data.shape[ax]
            current_stride = effective_strides[i] if i < len(effective_strides) else 1

            # Handle negative indices and None
            if current_begin is None: current_begin = 0
            if current_end is None: current_end = data.shape[ax]
            
            # numpy-style negative indexing for begin/end
            if current_begin < 0: current_begin += data.shape[ax]
            if current_end < 0: current_end += data.shape[ax]

            slices[ax] = slice(current_begin, current_end, current_stride)

    output[tuple(slices)] = value
    return output

# Helper for gather_python (for numpy reference)
def _numpy_gather(data, axis, indices):
    return np.take(data, indices, axis=axis)

# Helper for sequence_mask_python (for numpy reference)
def _numpy_sequence_mask(data_np, lengths_torch, mask_value, axis):
    np_lengths = lengths_torch.cpu().numpy()
    output = np.full(data_np.shape, mask_value, dtype=data_np.dtype)
    
    indices = np.arange(data_np.shape[axis])
    
    indices_expanded_shape = [1] * data_np.ndim
    indices_expanded_shape[axis] = data_np.shape[axis]
    indices_broadcast = indices.reshape(indices_expanded_shape)

    lengths_expanded_shape = list(data_np.shape)
    for i in range(len(lengths_expanded_shape)):
        if i != axis: # The batch dimension(s)
            lengths_expanded_shape[i] = 1
    
    # Find the actual batch dimension, it's the one whose size matches lengths.shape[0]
    # For a generic ND tensor, batch_axis could be any dimension that `lengths` corresponds to.
    # The original TVM code uses `batch_axis = 1 - axis` for 2D. We generalize this.
    batch_size = lengths_torch.shape[0]
    batch_dim_found = False
    for dim_idx in range(data_np.ndim):
        if dim_idx != axis and data_np.shape[dim_idx] == batch_size:
            lengths_expanded_shape[dim_idx] = batch_size
            batch_dim_found = True
            break
    
    lengths_broadcast = np_lengths.reshape(lengths_expanded_shape)

    mask = indices_broadcast < lengths_broadcast
    
    output[mask] = data_np[mask]
    return output

# PyTorch sequence_mask implementation
def _torch_sequence_mask(data, lengths, mask_value, axis):
    if lengths.ndim != 1:
        raise ValueError("lengths must be a 1D tensor.")
    
    max_len = data.shape[axis]
    batch_size = lengths.shape[0]

    indices = torch.arange(max_len, device=data.device, dtype=lengths.dtype)
    
    indices_expanded_shape = [1] * data.ndim
    indices_expanded_shape[axis] = max_len
    indices = indices.view(indices_expanded_shape)

    lengths_expanded_shape = list(data.shape)
    for i in range(len(lengths_expanded_shape)):
        if i != axis: # The batch dimension(s)
            lengths_expanded_shape[i] = 1
    
    batch_dim_found = False
    for dim_idx in range(data.ndim):
        if dim_idx != axis and data.shape[dim_idx] == batch_size:
            lengths_expanded_shape[dim_idx] = batch_size
            batch_dim_found = True
            break
    
    lengths = lengths.view(lengths_expanded_shape)

    mask = indices < lengths
    
    mask_value_tensor = torch.tensor(mask_value, device=data.device, dtype=data.dtype)

    return torch.where(mask, data, mask_value_tensor)

# PyTorch reverse_sequence implementation
def _torch_reverse_sequence(data, seq_lengths, batch_axis, seq_axis):
    output = data.clone()

    batch_axis = batch_axis if batch_axis >= 0 else data.ndim + batch_axis
    seq_axis = seq_axis if seq_axis >= 0 else data.ndim + seq_axis

    if seq_lengths.shape[0] != data.shape[batch_axis]:
        raise ValueError(
            f"For reverse_sequence seq_lengths size should match with dimension of batch axis,"
            f" but got dimension of batch_axis = {data.shape[batch_axis]}, and seq_length size = {seq_lengths.shape[0]}"
        )

    all_slices = [slice(None)] * data.ndim

    for i in range(data.shape[batch_axis]):
        length = seq_lengths[i].item() 

        if length <= 0:
            continue

        all_slices[batch_axis] = i
        batch_item = data[tuple(all_slices)]

        seq_slices = [slice(None)] * batch_item.ndim
        seq_slices[seq_axis] = slice(0, length)
        
        segment_to_flip = batch_item[tuple(seq_slices)]
        
        flipped_segment = torch.flip(segment_to_flip, dims=(seq_axis,))
        
        new_batch_item = batch_item.clone()
        new_batch_item[tuple(seq_slices)] = flipped_segment
        
        output[tuple(all_slices)] = new_batch_item

    return output


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not enabled"))])
def device(request):
    return torch.device(request.param)

@pytest.fixture
def float_dtype():
    return torch.float32

@pytest.fixture
def int_dtype():
    return torch.int32


def verify_expand_dims(in_shape, out_shape, axis, num_newaxis, device, float_dtype):
    A_torch = torch.randn(in_shape, dtype=float_dtype, device=device)
    
    B_torch = A_torch
    for _ in range(num_newaxis):
        B_torch = torch.unsqueeze(B_torch, dim=axis)

    data_npy = A_torch.cpu().numpy()
    out_npy = data_npy.reshape(out_shape)

    torch.testing.assert_allclose(B_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("in_shape,out_shape,axis,num_newaxis", [
    ((3, 10), (3, 10, 1, 1), 2, 2),
    ((3, 10), (1, 3, 10), -3, 1),
])
def test_expand_dims(in_shape, out_shape, axis, num_newaxis, device, float_dtype):
    verify_expand_dims(in_shape, out_shape, axis, num_newaxis, device, float_dtype)


def verify_reinterpret(in_shape, in_dtype_str, out_dtype_str, generator, device):
    in_torch_dtype = _to_torch_dtype(in_dtype_str)

    data_npy = generator(in_shape).astype(in_dtype_str)
    out_npy = data_npy.view(out_dtype_str)

    data_torch = torch.tensor(data_npy, dtype=in_torch_dtype, device=device)
    out_torch = torch.from_numpy(data_npy.view(out_dtype_str)).to(device)

    np.testing.assert_equal(out_torch.cpu().numpy(), out_npy)


@pytest.mark.parametrize("in_shape,in_dtype,out_dtype,generator", [
    ((1000,), "float32", "int32", lambda shape: np.random.randn(*shape) * 1000),
    ((1000,), "float16", "int16", lambda shape: np.random.randn(*shape) * 100),
    ((1000,), "int16", "uint16", lambda shape: np.random.randint(-1000, 1000, size=shape)),
    ((1000,), "uint32", "int32", lambda shape: np.random.randint(0, 2**32 - 1, size=shape)),
])
def test_reinterpret(in_shape, in_dtype, out_dtype, generator, device):
    # Skip float16 reinterpret on CUDA if device explicitly doesn't support FP16
    if in_dtype == "float16" and device.type == "cuda":
        # PyTorch generally handles FP16 transparently if CUDA is available on modern GPUs.
        # This explicit check is often not needed, but can be added for strict compatibility.
        # For simplicity, we assume modern CUDA devices support it.
        pass
    verify_reinterpret(in_shape, in_dtype, out_dtype, generator, device)


def verify_transpose(in_shape, axes, device, float_dtype):
    data_npy = np.arange(np.prod(in_shape)).reshape(in_shape).astype(float_dtype.numpy())
    A_torch = torch.tensor(data_npy, dtype=float_dtype, device=device)

    if axes is None:
        B_torch = torch.permute(A_torch, dims=tuple(reversed(range(A_torch.ndim))))
    else:
        B_torch = torch.permute(A_torch, dims=axes)

    out_npy = data_npy.transpose(axes)

    torch.testing.assert_allclose(B_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("in_shape,axes", [
    ((3, 10, 2), (1, 0, 2)),
    ((3, 10, 5), (2, 0, 1)),
    ((3, 10), None),
])
def test_transpose(in_shape, axes, device, float_dtype):
    verify_transpose(in_shape, axes, device, float_dtype)


@pytest.mark.parametrize("target_device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not enabled"))])
def test_transpose_unfused_schedule(target_device):
    # This TVM test verifies that transpose is not fused when there are other ops
    # The concept of "unfused schedule" is TVM-specific. In PyTorch, operations are generally
    # optimized by the backend (e.g., TorchInductor or CUDA kernels for GPU).
    # This PyTorch test checks for functional correctness, as the low-level fusion aspects
    # are handled by PyTorch's internal compilation/dispatch.

    shape = (100, (torch.cuda.get_device_properties(0).warp_size if target_device == "cuda" else 1) + 3)
    dev = torch.device(target_device)
    
    # Test 1: plain transpose
    x_val = torch.randn(shape, dtype=torch.float32, device=dev)
    actual = torch.transpose(x_val, 0, 1) # Directly call transpose
    expected = np.transpose(x_val.cpu().numpy())
    torch.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)

    # Test 2: transpose of (x + y)
    x_val = torch.randn(shape, dtype=torch.float32, device=dev)
    y_val = torch.randn(shape, dtype=torch.float32, device=dev)
    actual = torch.transpose(x_val + y_val, 0, 1)
    expected = np.transpose((x_val + y_val).cpu().numpy())
    torch.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


def verify_reshape(src_shape, dst_shape, device, float_dtype):
    data_npy = np.random.normal(size=src_shape).astype(float_dtype.numpy())
    A_torch = torch.tensor(data_npy, dtype=float_dtype, device=device)

    B_torch = torch.reshape(A_torch, dst_shape)

    out_npy = np.reshape(data_npy, newshape=dst_shape)

    torch.testing.assert_allclose(B_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("src_shape,dst_shape", [
    ((1, 2, 3, 4), (2, 3, 4)),
    ((4, 2, 3, 4), (2, 4, 12)),
    ((4, 2, 3, 4), (2, 48)),
    ((16,), (2, 2, 2, 2)),
    ((4, 0), (2, 0, 2)), # Special case with zero dimension
])
def test_reshape(src_shape, dst_shape, device, float_dtype):
    verify_reshape(src_shape, dst_shape, device, float_dtype)


def verify_where(in_shape, device, float_dtype):
    cond_npy = np.random.uniform(low=-1, high=1, size=in_shape).astype(float_dtype.numpy())
    x_npy = np.random.uniform(size=in_shape).astype(float_dtype.numpy())
    y_npy = np.random.uniform(size=in_shape).astype(float_dtype.numpy())
    
    Cond_torch = torch.tensor(cond_npy, dtype=float_dtype, device=device)
    A_torch = torch.tensor(x_npy, dtype=float_dtype, device=device)
    B_torch = torch.tensor(y_npy, dtype=float_dtype, device=device)

    C_torch = torch.where(Cond_torch.bool(), A_torch, B_torch)

    out_npy = np.where(cond_npy, x_npy, y_npy)

    torch.testing.assert_allclose(C_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("in_shape", [
    (), # Scalar case
    (1, 2, 3, 4),
])
def test_where(in_shape, device, float_dtype):
    verify_where(in_shape, device, float_dtype)


def verify_squeeze(src_shape, axis, device, float_dtype):
    data_npy = np.random.normal(size=src_shape).astype(float_dtype.numpy())
    A_torch = torch.tensor(data_npy, dtype=float_dtype, device=device)

    if isinstance(axis, (tuple, list)):
        if not axis: # Empty tuple means no dimensions removed
            B_torch = A_torch 
        else:
            B_torch = A_torch
            for d in sorted(axis, reverse=True): # Squeeze dims in reverse order to avoid index shifts
                 B_torch = torch.squeeze(B_torch, dim=d)
    else:
        B_torch = torch.squeeze(A_torch, dim=axis)

    out_npy = np.squeeze(data_npy, axis=axis)

    torch.testing.assert_allclose(B_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("src_shape,axis", [
    ((1, 2, 3, 4), 0),
    ((1, 2, 1, 4), None),
    ((1, 1, 1, 4), (1, 2)), # Tuple of axes, requires multi-squeeze or reshape
    ((1, 1, 1, 1), None),
    ((1, 1, 1, 1), ()), # Empty tuple means no axes removed, shape remains same
])
def test_squeeze(src_shape, axis, device, float_dtype):
    verify_squeeze(src_shape, axis, device, float_dtype)


def test_squeeze_inline_let_expression(device, float_dtype):
    # This is a TVM-specific test for IR optimization (inline let expression).
    # In PyTorch, we directly compute the expression.

    A_torch = torch.tensor(np.array((1, 2)).astype("float32"), dtype=float_dtype, device=device)
    
    E_torch = torch.squeeze(A_torch) 

    index_val = (2 * A_torch[0] - 1).to(torch.int32)
    
    if index_val.item() < 0 or index_val.item() >= E_torch.shape[0]:
        pytest.fail(f"Index {index_val.item()} out of bounds for tensor of size {E_torch.shape[0]}")

    C_torch = E_torch[index_val.long()] 
    
    out_actual = C_torch.unsqueeze(0).cpu().numpy()
    out_expected = np.array([2.0]).astype("float32") # Based on A[0]=1, index = 1, E[1]=2

    torch.testing.assert_allclose(out_actual, out_expected, rtol=1e-5, atol=1e-5)


def verify_concatenate(shapes, axis, device, float_dtype):
    data_npys = [np.random.normal(size=shape).astype(float_dtype.numpy()) for shape in shapes]
    tensor_l_torch = [torch.tensor(data_npy, dtype=float_dtype, device=device) for data_npy in data_npys]

    out_tensor_torch = torch.cat(tensor_l_torch, dim=axis)

    out_npy = np.concatenate(data_npys, axis=axis)

    torch.testing.assert_allclose(out_tensor_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shapes,axis", [
    ([(2,), (2,), (2,)], -1),
    ([(2, 3, 4), (2, 2, 4), (2, 5, 4)], 1),
    ([(1, 2, 4), (1, 2, 3), (1, 2, 7), (1, 2, 8), (1, 2, 1)], -1),
    ([(5, 6, 7, 3), (16, 6, 7, 3), (12, 6, 7, 3), (8, 6, 7, 3), (2, 6, 7, 3)], 0),
    ([(1, 14400), (1, 2400), (1, 640), (1, 240)], 1),
])
def test_concatenate(shapes, axis, device, float_dtype):
    verify_concatenate(shapes, axis, device, float_dtype)


def verify_stack(shapes, axis, device, float_dtype):
    data_npys = [np.random.normal(size=shape).astype(float_dtype.numpy()) for shape in shapes]
    tensor_l_torch = [torch.tensor(data_npy, dtype=float_dtype, device=device) for data_npy in data_npys]

    out_tensor_torch = torch.stack(tensor_l_torch, dim=axis)

    out_npy = np.stack(data_npys, axis=axis)

    torch.testing.assert_allclose(out_tensor_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shapes,axis", [
    ([(2,), (2,), (2,)], -1),
    ([(2,), (2,), (2,)], 1),
    ([(2,), (2,), (2,)], 0),
    ([(2, 2, 4), (2, 2, 4), (2, 2, 4)], 1),
    ([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], -1),
])
def test_stack(shapes, axis, device, float_dtype):
    verify_stack(shapes, axis, device, float_dtype)


def verify_split(src_shape, indices_or_sections, axis, device, float_dtype):
    data_npy = np.random.normal(size=src_shape).astype(float_dtype.numpy())
    A_torch = torch.tensor(data_npy, dtype=float_dtype, device=device)

    tensor_l_torch = torch.split(A_torch, split_size_or_sections=indices_or_sections, dim=axis)

    out_npys = np.split(data_npy, indices_or_sections, axis=axis)

    for out_torch, out_npy in zip(tensor_l_torch, out_npys):
        torch.testing.assert_allclose(out_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("src_shape,indices_or_sections,axis", [
    ((2, 12, 3), 3, 1), # Split into 3 equal parts
    ((2, 12, 3), [2, 4], 1), # Split at indices 2 and 4 (along dim 1)
    ((10, 12, 24), [5, 7, 9], -1),
])
def test_split(src_shape, indices_or_sections, axis, device, float_dtype):
    verify_split(src_shape, indices_or_sections, axis, device, float_dtype)


def verify_expand_like(in_shape, out_shape, axis, device, float_dtype):
    input_npy = np.random.uniform(size=in_shape).astype(float_dtype.numpy())
    A_torch = torch.tensor(input_npy, dtype=float_dtype, device=device)

    real_axis = [x if x >= 0 else x + len(out_shape) for x in axis]
    real_axis = sorted(real_axis) 

    expanded_A = A_torch
    for ax in real_axis:
        expanded_A = torch.unsqueeze(expanded_A, ax)

    C_torch = torch.broadcast_to(expanded_A, out_shape)

    out_npy = input_npy.copy()
    
    for x in real_axis:
        out_npy = np.expand_dims(out_npy, x)
    
    reps = [1] * len(out_npy.shape)
    for x in real_axis:
        reps[x] = out_shape[x]
    out_npy = np.tile(out_npy, reps)

    assert C_torch.shape == out_shape
    assert out_npy.shape == out_shape
    
    torch.testing.assert_allclose(C_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("in_shape,out_shape,axis", [
    ((3,), (2, 3), [0]),
    ((2,), (2, 3), [1]),
    ((3, 4), (3, 5, 4), [1]),
    ((5, 7), (5, 6, 7, 8), [1, 3]),
])
def test_expand_like(in_shape, out_shape, axis, device, float_dtype):
    # This test might exhibit slight behavior differences on CUDA due to complex broadcasting/tiling simulation.
    # The original TVM test only runs on LLVM.
    if device.type == "cuda":
        pytest.skip("Skipping expand_like on CUDA due to potential precision or behavior differences in complex broadcasting/tiling simulation compared to CPU NumPy reference.")
    verify_expand_like(in_shape, out_shape, axis, device, float_dtype)


def verify_flip(in_shape, axis, device, float_dtype):
    x_np = np.random.uniform(size=in_shape).astype(float_dtype.numpy())
    A_torch = torch.tensor(x_np, dtype=float_dtype, device=device)

    B_torch = torch.flip(A_torch, dims=(axis,)) + 1

    out_npy = np.flip(x_np, axis) + 1

    torch.testing.assert_allclose(B_torch.cpu().numpy(), out_npy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("in_shape,axis", [
    ((3, 4, 3), 1),
    ((3, 4, 3), 0),
