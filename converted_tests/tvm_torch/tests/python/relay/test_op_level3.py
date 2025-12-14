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
""" Support level3 operator test cases.
"""
import sys
import functools
import math
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Helper to convert TVM dtypes to torch dtypes
def _to_torch_dtype(tvm_dtype_str):
    if tvm_dtype_str == "float32":
        return torch.float32
    elif tvm_dtype_str == "float64":
        return torch.float64
    elif tvm_dtype_str == "int32":
        return torch.int32
    elif tvm_dtype_str == "int64":
        return torch.int64
    elif tvm_dtype_str == "uint8":
        return torch.uint8
    elif tvm_dtype_str == "uint16":
        return torch.int16 # PyTorch does not have uint16, mapping to int16
    elif tvm_dtype_str == "uint32":
        return torch.int32 # PyTorch does not have uint32, mapping to int32
    elif tvm_dtype_str == "bool":
        return torch.bool
    elif tvm_dtype_str == "int8":
        return torch.int8
    # Add other dtypes as needed
    raise ValueError(f"Unsupported TVM dtype: {tvm_dtype_str}")

# Helper for reshape_like to calculate the target shape
def _tvm_partial_reshape_shape(input_shape, like_shape, lhs_begin, lhs_end, rhs_begin, rhs_end):
    # This logic mimics TVM's behavior:
    # new shape is (input_shape[:lhs_begin] + like_shape[rhs_begin:rhs_end] + input_shape[lhs_end:])
    
    # In TVM `reshape_like` doc:
    # "The result will have shape `lhs_shape[:lhs_begin]` + `rhs_shape[rhs_begin:rhs_end]` + `lhs_shape[lhs_end:]`."
    # The current input `input_shape` is `lhs_shape`.
    # The current `like_shape` is `rhs_shape`.

    res_shape = []
    
    # Prefix from lhs
    for i in range(lhs_begin):
        res_shape.append(input_shape[i])

    # Middle from rhs
    for i in range(rhs_begin, rhs_end):
        res_shape.append(like_shape[i])

    # Suffix from lhs
    for i in range(lhs_end, len(input_shape)):
        res_shape.append(input_shape[i])

    return tuple(res_shape)


# Helper for cumsum/cumprod with exclusive flag
def _torch_cumsum_exclusive(data, dim, dtype=None, exclusive=False):
    if not exclusive:
        return torch.cumsum(data, dim=dim, dtype=dtype)
    
    # For exclusive cumsum: [0, a, a+b, a+b+c, ...]
    # Equivalent to rolling the inclusive cumsum and prepending a zero.
    
    # Create a tensor of zeros matching the slice that will be prepended
    slice_shape = list(data.shape)
    slice_shape[dim] = 1
    
    if dtype is None:
        if data.is_floating_point():
            zero_slice = torch.zeros(slice_shape, dtype=data.dtype, device=data.device)
        else: # For integer types, ensure `0` is within the type range
            zero_slice = torch.zeros(slice_shape, dtype=data.dtype, device=data.device)
    else:
        zero_slice = torch.zeros(slice_shape, dtype=dtype, device=data.device)
    
    # Calculate inclusive cumsum
    inclusive_cumsum = torch.cumsum(data, dim=dim, dtype=dtype)
    
    # Concatenate the zero slice and the inclusive cumsum, then remove the last element along dim
    result = torch.cat([zero_slice, inclusive_cumsum], dim=dim)
    result = result.narrow(dim=dim, start=0, length=data.shape[dim]) # Take original length
    return result

def _torch_cumprod_exclusive(data, dim, dtype=None, exclusive=False):
    if not exclusive:
        return torch.cumprod(data, dim=dim, dtype=dtype)
    
    # For exclusive cumprod: [1, x0, x0*x1, x0*x1*x2, ...]
    # Equivalent to rolling the inclusive cumprod and prepending a one.
    
    # Create a tensor of ones matching the slice that will be prepended
    slice_shape = list(data.shape)
    slice_shape[dim] = 1
    
    if dtype is None:
        one_slice = torch.ones(slice_shape, dtype=data.dtype, device=data.device)
    else:
        one_slice = torch.ones(slice_shape, dtype=dtype, device=data.device)
    
    # Calculate inclusive cumprod
    inclusive_cumprod = torch.cumprod(data, dim=dim, dtype=dtype)
    
    # Concatenate the one slice and the inclusive cumprod, then remove the last element along dim
    result = torch.cat([one_slice, inclusive_cumprod], dim=dim)
    result = result.narrow(dim=dim, start=0, length=data.shape[dim]) # Take original length
    return result

# numpy reference for scatter (from TVM test)
def _numpy_ref_scatter(data, indices, updates, axis=0):
    idx = np.indices(indices.shape).reshape(indices.ndim, -1)

    updated_idx = np.copy(idx)
    indices_flat = indices.reshape(-1)
    for i in range(len(indices_flat)):
        # Adjust axis to be positive if negative
        actual_axis = axis if axis >= 0 else data.ndim + axis
        updated_idx[actual_axis, i] = indices_flat[i]
    scattered = np.copy(data)
    scattered[tuple(updated_idx)] = updates[tuple(idx)]
    return scattered

# numpy reference for gather_nd (from TVM test's ref_funcs.gather_nd)
def _numpy_ref_gather_nd(data, indices, batch_dims=0):
    # This implementation matches the general logic of TF's gather_nd for simple cases.
    # It assumes indices are within bounds for correctness.
    
    # Ensure data and indices are numpy arrays for consistency in ref calculation
    data = np.asarray(data)
    indices = np.asarray(indices)

    # Output shape calculation
    output_shape = data.shape[:batch_dims] + indices.shape[batch_dims:-1] + data.shape[batch_dims + indices.shape[-1] :]
    output = np.zeros(output_shape, dtype=data.dtype)

    # Create an iterator over the batch dimensions and index dimensions
    batch_index_iterator = np.ndindex(*indices.shape[:-1])

    for batch_and_index_coords in batch_index_iterator:
        # Separate batch coordinates from indexing coordinates
        batch_coords = batch_and_index_coords[:batch_dims]
        index_coords_base = batch_and_index_coords[batch_dims:]
        
        # Get the actual index values for gathering (e.g., [[1,0], [0,1]] -> indices (1,0) or (0,1))
        # The last dimension of 'indices' array specifies the elements to gather
        gather_indices = indices[batch_and_index_coords]
        
        # Construct the full index tuple for `data`
        # This will be `(batch_coords, gather_indices, remaining_data_dims)`
        full_data_index_tuple = batch_coords + tuple(gather_indices)

        # Determine the slice of data to extract
        data_slice_to_extract = data[full_data_index_tuple]
        
        # Place this extracted slice into the output
        output_place_coords = batch_coords + index_coords_base
        output[output_place_coords] = data_slice_to_extract

    return output

# Helper to simulate PyTorch's scatter_nd behavior for reference
# This is a general helper for scatter_nd to match behavior.
# PyTorch equivalent is typically `index_put_`
def _torch_scatter_nd(data, indices, updates, mode="add"):
    # `indices` must be (num_indices, index_depth) or (batch_dims, num_indices, index_depth)
    # `updates` must be (num_indices, trailing_dims) or (batch_dims, num_indices, trailing_dims)

    # The actual indices to put, need to be reshaped to be compatible with index_put_
    if indices.ndim > 1:
        # Assuming indices are like [[idx0_d0, idx0_d1], [idx1_d0, idx1_d1]]
        # We need them as (idx0_d0, idx1_d0, ...)
        indices_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))
    else: # 1D indices, e.g., for flattening or single dimension indexing
        indices_tuple = (indices,)

    result = data.clone()
    if mode == "add":
        result.index_put_(indices_tuple, updates, accumulate=True)
    elif mode == "update":
        result.index_put_(indices_tuple, updates)
    else:
        raise ValueError(f"Unsupported scatter_nd mode: {mode}")
    return result

# numpy reference for unique
def _numpy_calc_unique(data, is_sorted=False, return_counts=False):
    # This matches the TVM test's custom numpy unique calc
    uniq, index, inverse, counts = np.unique(
        data, return_index=True, return_inverse=True, return_counts=True
    )
    num_uniq = np.array([len(uniq)]).astype("int32")
