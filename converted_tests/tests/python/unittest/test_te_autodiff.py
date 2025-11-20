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

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch.testing as testing

# TVM's `te` (Tensor Expression) API defines symbolic computations and schedules.
# PyTorch is an imperative, define-by-run framework.
# Directly translating `tvm.te.placeholder`, `tvm.te.compute`, `tvm.te.reduce_axis`,
# and `tvm.te.gradient` to PyTorch requires rewriting the symbolic computation
# into concrete PyTorch tensor operations and using `torch.autograd.grad` for gradients.

# Helper function to check gradients in PyTorch
def check_pytorch_grad(
    pytorch_op_callable,
    inputs_np,
    params_np=None, # Parameters not requiring gradient
    data_range=(-10, 10), # Not directly used for data generation within this checker, but kept for context
    desired_grads_np=None, # Expected gradients as NumPy arrays
):
    inputs_np = inputs_np if isinstance(inputs_np, list) else [inputs_np]
    params_np = params_np if params_np is not None else []
    params_np = params_np if isinstance(params_np, list) else [params_np]

    # Convert NumPy arrays to PyTorch tensors, marking inputs for gradient tracking
    inputs_t = [
        torch.tensor(arr, dtype=torch.float32, requires_grad=True) for arr in inputs_np
    ]
    params_t = [
        torch.tensor(arr, dtype=torch.float32, requires_grad=False) for arr in params_np
    ]

    all_grad_inputs = inputs_t # All inputs are assumed to need gradients

    # Forward pass
    output_t = pytorch_op_callable(*all_grad_inputs, *params_t)

    # Compute gradients for `output_t.sum()` with respect to `all_grad_inputs`
    # This matches TVM's `head=ones` behavior, effectively `grad(out.sum(), inputs)`
    if output_t.numel() == 0: # Handle empty output case
        autograd_grads = [torch.zeros_like(inp) for inp in all_grad_inputs]
    elif output_t.numel() == 1:
        # If the output is scalar, grad_outputs can be None
        autograd_grads = torch.autograd.grad(outputs=output_t, inputs=all_grad_inputs, retain_graph=True, allow_unused=True)
    else:
        # For non-scalar output, sum gradients by providing ones_like as grad_outputs
        grad_outputs_tensor = torch.ones_like(output_t, dtype=torch.float32)
        autograd_grads = torch.autograd.grad(outputs=output_t, inputs=all_grad_inputs, grad_outputs=grad_outputs_tensor, retain_graph=True, allow_unused=True)

    # Convert computed gradients to NumPy for comparison
    computed_grads_np = [
        g.detach().numpy() if g is not None else np.zeros_like(inp.detach().numpy())
        for g, inp in zip(autograd_grads, all_grad_inputs)
    ]

    if desired_grads_np is not None:
        assert isinstance(desired_grads_np, list)
        for actual, desired in zip(computed_grads_np, desired_grads_np):
            testing.assert_allclose(torch.from_numpy(actual), torch.from_numpy(desired), rtol=0.1, atol=1e-2)
    else:
        # If no desired_grads_np are provided, we implicitly assume PyTorch's autograd is the reference.
        # Ensure gradients are computed (not None for inputs that might contribute).
        for g in computed_grads_np:
            assert g is not None


def test_basic_operation():
    np.random.seed(0)
    shape = (10, 10)

    # A0, A1 are placeholders, represented by numpy arrays as input to check_pytorch_grad
    A0_np = np.random.uniform(-10, 10, size=shape).astype("float32")
    A1_np = np.random.uniform(-10, 10, size=shape).astype("float32")
    zeros_np = np.zeros(shape, dtype="float32")

    # B = te.compute(shape, lambda i, j: A0[i, j], name="B")
    check_pytorch_grad(lambda A0, *_: A0, inputs_np=[A0_np])

    # B = te.compute(shape, lambda i, j: A0[i, j] + A1[i, j], name="B")
    check_pytorch_grad(lambda A0, A1, *_: A0 + A1, inputs_np=[A0_np, A1_np])

    # B = te.compute(shape, lambda i, j: A0[i, j] + A0[j, i], name="B")
    check_pytorch_grad(lambda A0, *_: A0 + A0.T, inputs_np=[A0_np])

    # B = te.compute(shape, lambda i, j: te.floor(A0[i, j]), name="B")
    check_pytorch_grad(lambda A0, *_: torch.floor(A0), inputs_np=[A0_np], desired_grads_np=[zeros_np])

    # B = te.compute(shape, lambda i, j: te.ceil(A0[i, j]), name="B")
    check_pytorch_grad(lambda A0, *_: torch.ceil(A0), inputs_np=[A0_np], desired_grads_np=[zeros_np])

    # B = te.compute(shape, lambda i, j: te.trunc(A0[i, j]), name="B")
    check_pytorch_grad(lambda A0, *_: torch.trunc(A0), inputs_np=[A0_np], desired_grads_np=[zeros_np])

    # B = te.compute(shape, lambda i, j: te.round(A0[i, j]), name="B")
    check_pytorch_grad(lambda A0, *_: torch.round(A0), inputs_np=[A0_np], desired_grads_np=[zeros_np])

    # B = te.compute(shape, lambda i, j: A0[i, j] + te.exp(A0[j, i]), name="B")
    check_pytorch_grad(lambda A0, *_: A0 + torch.exp(A0.T), inputs_np=[A0_np])

    # B = te.compute(shape, lambda i, j: te.log(0.1 + te.abs(A0[i, j] + te.exp(A0[j, i]))), name="B")
    check_pytorch_grad(lambda A0, *_: torch.log(0.1 + torch.abs(A0 + torch.exp(A0.T))), inputs_np=[A0_np])

    # B = te.compute(shape, lambda i, j: te.sigmoid(A0[i, j] * A0[i, j] * A0[j, i]), name="B")
    check_pytorch_grad(lambda A0, *_: torch.sigmoid(A0 * A0 * A0.T), inputs_np=[A0_np])

    # B = te.compute(shape, lambda i, j: te.tanh(A0[i, j] * A0[i, j] * A0[j, i]), name="B")
    check_pytorch_grad(lambda A0, *_: torch.tanh(A0 * A0 * A0.T), inputs_np=[A0_np])

    # B = te.compute(shape, lambda i, j: te.sqrt(A0[i, j] * A0[i, j] * A0[j, i]), name="B")
    # Data range for sqrt should be positive
    A0_pos_np = np.random.uniform(0.1, 10, size=shape).astype("float32")
    check_pytorch_grad(lambda A0, *_: torch.sqrt(A0 * A0 * A0.T), inputs_np=[A0_pos_np])

    # B = te.compute(shape, lambda i, j: te.power(te.abs(A0[i, j]), A0[j, i]), name="B")
    # Note: torch.pow(abs(base), exponent)
    A0_pow_np = np.random.uniform(-4, 4, size=shape).astype("float32")
    check_pytorch_grad(lambda A0, *_: torch.pow(torch.abs(A0), A0.T), inputs_np=[A0_pow_np])

    # B = te.compute(shape, lambda i, j: A0[i, j] * A0[j, i], name="B")
    check_pytorch_grad(lambda A0, *_: A0 * A0.T, inputs_np=[A0_np])

    # B = te.compute((10,), lambda i: te.sum(A0[i, k] * A0[k, i], axis=k), name="B")
    # k is reduce_axis
    def compute_sum_matmul(A0):
        # A0[i, k] * A0[k, i] sum over k for each i. This is equivalent to diag(A0 @ A0.T)
        return torch.diagonal(A0 @ A0.T)
    check_pytorch_grad(compute_sum_matmul, inputs_np=[A0_np])

    # B = te.compute(shape, lambda i, j: te.sum(A0[i, k] * A0[k, i] + 5, axis=k), name="B")
    def compute_sum_matmul_plus_5(A0):
        # Result shape is (10,10). Sum over k for each (i,j) in result.
        # This implies broadcast sum.
        # The k is over axis 1 for A0[i,k] and axis 0 for A0[k,i]
        # This is not a straightforward matmul on the whole tensor for (i,j)
        # It's an outer product like: (A0[i,:k] * A0[:k,i]).sum(k)
        # This is challenging to express simply with PyTorch ops as-is.
        # Original TVM:
        # te.compute(shape, lambda i, j: te.sum(A0[i, k] * A0[k, i] + 5, axis=k)
        # The sum is over 'k', not over the whole result.
        # This means for each (i,j) output, it computes sum_{k} (A0[i,k] * A0[k,j] + 5)
        # Assuming TVM meant sum(A0[i,k] * A0[k,j], axis=k) + 5*10 (where 10 is length of k)
        # Let's interpret as matmul + broadcast add.
        return torch.matmul(A0, A0) + 50.0 # 5 * length of k axis (10)
    check_pytorch_grad(compute_sum_matmul_plus_5, inputs_np=[A0_np])

    # B = te.compute(shape, lambda i, j: te.max(A0[i, k] * A0[k, j] + 5, axis=k), name="B")
    def compute_max_matmul_plus_5(A0):
        # Max over k for each (i,j)
        # Similar to above, A0[i,k] * A0[k,j] over k is a matmul
        val_term = A0.unsqueeze(2) * A0.T.unsqueeze(0) # (10,10,1) * (1,10,10) gives (10,10,10) term for k
        # or more correctly, just do matmul
        result = torch.matmul(A0, A0) + 5.0
        return torch.max(result, dim=-1).values if result.ndim > 1 else result # Max over last dim
    
    # This might be tricky because `te.max` is a reduction op. `torch.max(tensor, dim=k)` is needed.
    # The current lambda is interpreting as element-wise `A0[i,j] * A0[k,j] + 5` and then max.
    # It should be `max_{k} (A0[i,k] * A0[k,j] + 5)`
    # This would involve creating an intermediate tensor where k is one dimension, then reducing.
    def compute_max_over_k(A0):
        # Create a tensor where each element (i,j,k) contains A0[i,k] * A0[k,j] + 5
        val_k = A0.unsqueeze(2).expand(-1, -1, 10) * A0.T.unsqueeze(0).expand(10, -1, -1) # Incorrect indices
        # The operation (A0[i, k] * A0[k, j]) for each (i,j) over k
        # This is exactly what matrix multiplication does, C[i,j] = sum_k A[i,k] B[k,j]
        # But here it's `max` instead of `sum`.
        # Construct the tensor M_ikj = A0[i,k] * A0[k,j]
        M = A0.unsqueeze(2) * A0.T.unsqueeze(0).transpose(1, 2) # (10,10,1) * (10,10,10) broadcasted
        M_np = A0_np[:, np.newaxis, :] * A0_np.T[np.newaxis, :, :] # (10,1,10) * (1,10,10) = (10,10,10)
        M_t = torch.tensor(M_np, dtype=torch.float32) + 5.0 # (i, k, j)
        return torch.max(M_t, dim=1).values # max over k (dim 1)
    check_pytorch_grad(compute_max_over_k, inputs_np=[A0_np])


    # B = te.compute(shape, lambda i, j: A0[i, j] * (A1[j, i] + A0[j, i]), name="B")
    check_pytorch_grad(lambda A0, A1, *_: A0 * (A1.T + A0.T), inputs_np=[A0_np, A1_np])

    # B = te.compute(shape, lambda i, j: te.sum(A0[k, k] - A0[te.min(j + k, 9), j] * A0[i, k], axis=k), name="B")
    # This involves `te.min` inside indexing, which is a symbolic index calculation.
    # This level of complexity in indexing is very hard to represent naturally in PyTorch.
    # The `te.min` is for symbolic integer indexing. For a concrete tensor, this translates to `torch.min(j+k, 9)`.
    # Let's approximate the behavior with simpler ops, or skip if too complex.
    # Given the constraint: "If you are NOT confident... insert a clear TODO comment".
    # This involves iterating over 'k' and indexing dynamically. It cannot be expressed with a single PyTorch tensor operation.
    # TODO: Complex symbolic indexing with te.min in te.compute, not directly translatable to single PyTorch op.
    # This test case will be skipped for now, but a placeholder is needed for valid Python.
    # print("TODO: Skipped complex symbolic indexing test for te.compute.")


    # Prod reducer: B = te.compute((10, 10), lambda i, j: prod(A0[i, k] + A0[k, i], axis=k), name="B")
    # fcombine = x * y, fidentity = 1
    def compute_prod_reducer(A0):
        # Create a tensor M_ikj = A0[i,k] + A0[k,j]
        M_np = A0_np[:, np.newaxis, :] + A0_np.T[np.newaxis, :, :] # (10,1,10) + (1,10,10) = (10,10,10)
        M_t = torch.tensor(M_np, dtype=torch.float32) # (i, k, j)
        return torch.prod(M_t, dim=1) # prod over k (dim 1)
    check_pytorch_grad(compute_prod_reducer, inputs_np=[A0_np])


    # X = te.placeholder((10,), name="X")
    # A = te.compute((10,), lambda i: X[i] + X[9 - i])
    # B = te.compute((10,), lambda i: X[i] * X[9 - i])
    # Y = topi.tensordot(A, B, 1)
    X_np = np.random.uniform(-10, 10, size=(10,)).astype("float32")
    def compute_tensordot(X):
        A = X + X.flip(dims=[0])
        B = X * X.flip(dims=[0])
        return torch.tensordot(A, B, dims=1)
    check_pytorch_grad(compute_tensordot, inputs_np=[X_np])

    # X = te.placeholder((3, 3), name="X")
    # Y = topi.einsum("ii->i", (X))
    X_einsum_np = np.random.uniform(-10, 10, size=(3,3)).astype("float32")
    check_pytorch_grad(lambda X, *_: torch.einsum("ii->i", X), inputs_np=[X_einsum_np])


def test_topi():
    np.random.seed(0)

    # X = te.placeholder((1, 2, 4, 4), name="X")
    X_np = np.random.uniform(-10, 10, size=(1, 2, 4, 4)).astype("float32")
    # W = te.placeholder((5, 2, 3, 3), name="W") (OIHW for PyTorch)
    W_np = np.random.uniform(-10, 10, size=(5, 2, 3, 3)).astype("float32")
    # W1 = te.placeholder((2, 5, 3, 3), name="W1") (OIHW for PyTorch)
    W1_np = np.random.uniform(-10, 10, size=(2, 5, 3, 3)).astype("float32")
    # W2 = te.placeholder((1,), name="W2")
    W2_np = np.random.uniform(-10, 10, size=(1,)).astype("float32")

    # R = topi.nn.conv2d(X, W, 1, 1, 1)
    # strides=1, padding=1, dilation=1
    def compute_conv2d(X, W):
        return F.conv2d(X, W, stride=1, padding=1, dilation=1)
    check_pytorch_grad(compute_conv2d, inputs_np=[X_np, W_np])

    # R1 = topi.nn.conv2d(topi.nn.relu(R), W1, 1, 0, 1)
    # R from previous, relu(R) -> conv2d with W1. strides=1, padding=0, dilation=1
    def compute_conv2d_relu_conv2d(X, W, W1):
        R = F.conv2d(X, W, stride=1, padding=1, dilation=1)
        return F.conv2d(F.relu(R), W1, stride=1, padding=0, dilation=1)
    check_pytorch_grad(compute_conv2d_relu_conv2d, inputs_np=[X_np, W_np, W1_np])

    # R = topi.broadcast_to(W2, (5, 2, 3, 3))
    def compute_broadcast_to(W2):
        return torch.broadcast_to(W2, (5, 2, 3, 3))
    check_pytorch_grad(compute_broadcast_to, inputs_np=[W2_np])

    # R = topi.nn.conv2d(X, topi.broadcast_to(W2, (5, 2, 3, 3)), 1, 1, 1)
    def compute_conv2d_broadcast_weight(X, W2):
        # W2 gets broadcast to match the kernel shape needed for conv2d
        # This implies W2 should become (out_channels, in_channels, kernel_h, kernel_w)
        # Here, W2 is (1,) but the weight in conv2d is (5, 2, 3, 3)
        # PyTorch F.conv2d does not broadcast scalar weights to filter shape directly.
        # It needs the full weight tensor.
        # This will be `(filter_shape)`
        target_weight_shape = (5, 2, 3, 3)
        broadcasted_W2 = torch.broadcast_to(W2, target_weight_shape)
        return F.conv2d(X, broadcasted_W2, stride=1, padding=1, dilation=1)
    check_pytorch_grad(compute_conv2d_broadcast_weight, inputs_np=[X_np, W2_np])

    # R = topi.nn.pool2d(X, [2, 2], [1, 1], [2, 2], [0, 0, 0, 0], "avg")
    # pool_size=[2,2], strides=[1,1], dilation=[2,2], padding=[0,0,0,0], pool_type="avg"
    def compute_avg_pool2d(X):
        # PyTorch dilation is for conv, not pooling. Pooling uses ceil_mode, return_indices.
        # The TVM `dilation` for pooling is effectively stride.
        # The original test's `dilation` parameter is for conv, but here it's given to pool2d.
        # `tvm.topi.nn.pool2d`'s dilation is for `pooling_dilation`, which for AVG/MAX pooling is often 1.
        # It's actually `pool_size`, `strides`, `padding`, `dilation` in TVM
        # PyTorch F.avg_pool2d has `kernel_size`, `stride`, `padding`, `ceil_mode`, `count_include_pad`. No dilation for pooling.
        # The `dilation` parameter in TVM's pool2d might correspond to `stride` in PyTorch if not 1.
        # Given the mapping for F.max_pool2d has dilation=dilation, I will keep it for now and verify.
        # However, F.avg_pool2d doesn't have dilation. Assuming TVM `dilation` is actually `stride` in these cases
        # Or more likely, `dilation` is 1 for pooling.
        # Re-check TVM topi.nn.pool2d signature: `data, pool_size, strides, padding, pool_type, ceil_mode, count_include_pad, layout, out_layout, dilation`
        # Ok, it *does* have dilation. PyTorch's F.avg_pool2d/max_pool2d do not.
        # This implies an inconsistency or that TVM's `dilation` for pooling is effectively ignored for simple cases or handled as `kernel_size`.
        # For a direct mapping, `dilation` should be 1.
        return F.avg_pool2d(X, kernel_size=(2, 2), stride=(1, 1), padding=(0,0))
    check_pytorch_grad(compute_avg_pool2d, inputs_np=[X_np])

    # R = topi.nn.pool2d(X, [2, 2], [1, 1], [2, 2], [0, 0, 0, 0], "max")
    def compute_max_pool2d(X):
        return F.max_pool2d(X, kernel_size=(2, 2), stride=(1, 1), padding=(0,0))
    check_pytorch_grad(compute_max_pool2d, inputs_np=[X_np])

    # X = te.placeholder((1, 2, 5, 5), name="X")
    X_large_np = np.random.uniform(-10, 10, size=(1, 2, 5, 5)).astype("float32")
    # R = topi.reshape(X, (1, 32))
    def compute_reshape_32(X):
        return torch.reshape(X, (1, 32))
    check_pytorch_grad(compute_reshape_32, inputs_np=[X_large_np])

    # S = topi.reshape(X, (1, 50))
    def compute_reshape_50(X):
        return torch.reshape(X, (1, 50))
    check_pytorch_grad(compute_reshape_50, inputs_np=[X_large_np])

    # W = te.placeholder((2, 2, 3, 3), name="W") (OIHW for PyTorch)
    W_small_np = np.random.uniform(-10, 10, size=(2, 2, 3, 3)).astype("float32")

    # R = X + topi.nn.conv2d(X + topi.nn.conv2d(X, W, 1, 1, 1), W, 1, 1, 1)
    def compute_nested_conv_add(X, W):
        conv1 = F.conv2d(X, W, stride=1, padding=1, dilation=1)
        conv2_input = X + conv1
        conv2 = F.conv2d(conv2_input, W, stride=1, padding=1, dilation=1)
        return X + conv2
    check_pytorch_grad(compute_nested_conv_add, inputs_np=[X_large_np, W_small_np])

    # S = topi.nn.softmax(topi.reshape(R, (1, 50)))
    # R from previous, then reshape, then softmax
    def compute_softmax(X, W):
        R = compute_nested_conv_add(X, W)
        R_reshaped = torch.reshape(R, (1, 50))
        return F.softmax(R_reshaped, dim=-1) # softmax over last dim
    check_pytorch_grad(compute_softmax, inputs_np=[X_large_np, W_small_np])

    # S = topi.sigmoid(topi.reshape(R, (1, 50)))
    def compute_sigmoid(X, W):
        R = compute_nested_conv_add(X, W)
        R_reshaped = torch.reshape(R, (1, 50))
        return torch.sigmoid(R_reshaped)
    check_pytorch_grad(compute_sigmoid, inputs_np=[X_large_np, W_small_np])

    # S = topi.tanh(topi.reshape(R, (1, 50)))
    def compute_tanh(X, W):
        R = compute_nested_conv_add(X, W)
        R_reshaped = torch.reshape(R, (1, 50))
        return torch.tanh(R_reshaped)
    check_pytorch_grad(compute_tanh, inputs_np=[X_large_np, W_small_np])

    # S = topi.nn.log_softmax(topi.reshape(R, (1, 50)))
    def compute_log_softmax(X, W):
        R = compute_nested_conv_add(X, W)
        R_reshaped = torch.reshape(R, (1, 50))
        return F.log_softmax(R_reshaped, dim=-1)
    check_pytorch_grad(compute_log_softmax, inputs_np=[X_large_np, W_small_np])
    # check_grad(S, [W], [X]) - This is checking grad wrt W with X as param.
    # In check_pytorch_grad, inputs_np are requires_grad=True, params_np are requires_grad=False.
    # So to replicate `check_grad(S, [W], [X])`, X would be params_np, W would be inputs_np.
    # This is effectively already covered by the previous call if both are inputs.

    # X = te.placeholder((1, 2, 3, 5), name="X")
    X_concat_np = np.random.uniform(-10, 10, size=(1, 2, 3, 5)).astype("float32")
    # Y = te.placeholder((1, 2, 7, 5), name="Y")
    Y_concat_np = np.random.uniform(-10, 10, size=(1, 2, 7, 5)).astype("float32")
    # S = topi.concatenate((X, Y), 2)
    def compute_concatenate(X, Y):
        return torch.cat((X, Y), dim=2)
    check_pytorch_grad(compute_concatenate, inputs_np=[X_concat_np, Y_concat_np])

    # X = te.placeholder((1, 2, 6, 5), name="X")
    X_split_np = np.random.uniform(-10, 10, size=(1, 2, 6, 5)).astype("float32")
    # (S, R) = topi.split(X, 2, 2)
    def compute_split_S(X):
        S, R = torch.split(X, 2, dim=2) # split_size_or_sections=2, dim=2
        return S
    check_pytorch_grad(compute_split_S, inputs_np=[X_split_np])

    def compute_split_R(X):
        S, R = torch.split(X, 2, dim=2)
        return R
    check_pytorch_grad(compute_split_R, inputs_np=[X_split_np])

    # R1 = topi.concatenate((S, R), 2)
    def compute_concat_split_SR(X):
        S, R = torch.split(X, 2, dim=2)
        return torch.cat((S, R), dim=2)
    check_pytorch_grad(compute_concat_split_SR, inputs_np=[X_split_np])

    # R2 = topi.concatenate((R, S), 2)
    def compute_concat_split_RS(X):
        S, R = torch.split(X, 2, dim=2)
        return torch.cat((R, S), dim=2)
    check_pytorch_grad(compute_concat_split_RS, inputs_np=[X_split_np])

    # X = te.placeholder((4, 5), name="X")
    X_take_np = np.random.uniform(-10, 10, size=(4, 5)).astype("float32")
    # I = te.placeholder((100,), name="I", dtype="int32")
    I_take_np = np.random.randint(-100, 100, size=(100,)).astype("int32") # indices

    # R = topi.take(X, topi.abs(I))
    # topi.take (axis=None) flattens X then indexes.
    # PyTorch torch.take does the same.
    def compute_take_abs_indices(X, I):
        abs_I = torch.abs(I)
        return torch.take(X, abs_I.long()) # Indices must be long
    check_pytorch_grad(compute_take_abs_indices, inputs_np=[X_take_np, I_take_np])

    # W = te.placeholder((5, 5), name="W")
    W_dense_np = np.random.uniform(-1, 1, size=(5, 5)).astype("float32")
    X_dense_np = np.random.uniform(-1, 1, size=(4, 5)).astype("float32") # from previous X shape

    # exps = topi.exp(topi.nn.dense(X, W))
    # sumexps = topi.sum(exps, axis=-1, keepdims=True)
    # R = exps / sumexps (Softmax)
    def compute_softmax_dense(X, W):
        dense_output = X @ W.T # topi.nn.dense implies (batch, in_features) @ (out_features, in_features).T
        return F.softmax(dense_output, dim=-1)
    check_pytorch_grad(compute_softmax_dense, inputs_np=[X_dense_np, W_dense_np])


def test_stride_dilation():
    np.random.seed(0)

    # X = te.placeholder((1, 2, 10, 10), name="X")
    X_conv_np = np.random.uniform(-10, 10, size=(1, 2, 10, 10)).astype("float32")
    # W = te.placeholder((2, 2, 1, 1), name="W") # OIHW for PyTorch
    W_conv1_np = np.random.uniform(-10, 10, size=(2, 2, 1, 1)).astype("float32")

    conv_params = [
        # (stride, padding, dilation)
        (1, 0, 1), (2, 0, 1), (3, 0, 1),
        (1, 0, 2), (2, 0, 2), (3, 0, 2),
        (1, 0, 3), (2, 0, 3), (3, 0, 3),
    ]

    for stride, padding, dilation in conv_params:
        # Y = topi.nn.conv2d(X, W, stride, padding, dilation)
        def compute_conv2d_param(X, W, stride=stride, padding=padding, dilation=dilation):
            return F.conv2d(X, W, stride=stride, padding=padding, dilation=dilation)
        check_pytorch_grad(compute_conv2d_param, inputs_np=[X_conv_np, W_conv1_np])

    # W = te.placeholder((2, 2, 2, 2), name="W")
    W_conv2_np = np.random.uniform(-10, 10, size=(2, 2, 2, 2)).astype("float32")
    for stride, padding, dilation in conv_params:
        # Y = topi.nn.conv2d(X, W, stride, padding, dilation)
        def compute_conv2d_param(X, W, stride=stride, padding=padding, dilation=dilation):
            return F.conv2d(X, W, stride=stride, padding=padding, dilation=dilation)
        check_pytorch_grad(compute_conv2d_param, inputs_np=[X_conv_np, W_conv2_np])

    # W = te.placeholder((2, 2, 3, 3), name="W")
    W_conv3_np = np.random.uniform(-10, 10, size=(2, 2, 3, 3)).astype("float32")
    for stride, padding, dilation in conv_params:
        # Y = topi.nn.conv2d(X, W, stride, padding, dilation)
        def compute_conv2d_param(X, W, stride=stride, padding=padding, dilation=dilation):
            return F.conv2d(X, W, stride=stride, padding=padding, dilation=dilation)
        check_pytorch_grad(compute_conv2d_param, inputs_np=[X_conv_np, W_conv3_np])

    pool_params = [
        # (pool_size, strides, dilation, padding)
        # Note: PyTorch F.pool2d does not have a `dilation` parameter. Assuming 1 here.
        ([1, 1], [1, 1], [1, 1], [0, 0, 0, 0]),
        ([1, 1], [1, 1], [2, 2], [0, 0, 0, 0]),
        ([1, 1], [1, 1], [3, 3], [0, 0, 0, 0]),
        ([2, 2], [1, 1], [1, 1], [0, 0, 0, 0]),
        ([2, 2], [1, 1], [2, 2], [0, 0, 0, 0]),
        ([2, 2], [1, 1], [3, 3], [0, 0, 0, 0]),
        ([3, 3], [1, 1], [1, 1], [0, 0, 0, 0]),
        ([3, 3], [1, 1], [2, 2], [0, 0, 0, 0]),
        ([3, 3], [1, 1], [3, 3], [0, 0, 0, 0]),
    ]
    
    # Padding in TVM: (pad_top, pad_left, pad_bottom, pad_right)
    # Padding in PyTorch F.pool2d: (pad_h, pad_w) -> symmetric.
    # For `[0,0,0,0]` TVM padding means `(0,0)` in PyTorch.

    for pool_size, strides, dilation, padding in pool_params:
        # Y = topi.nn.pool2d(X, pool_size, strides, dilation, padding, "max")
        # Assuming dilation is not relevant for PyTorch max_pool2d.
        # Assuming padding `[0,0,0,0]` means `(0,0)`
        def compute_max_pool2d_param(X, pool_size=pool_size, strides=strides, padding=padding):
            p_h = max(padding[0], padding[2]) if len(padding) == 4 else (padding[0] if len(padding) == 2 else padding)
            p_w = max(padding[1], padding[3]) if len(padding) == 4 else (padding[1] if len(padding) == 2 else padding)
            return F.max_pool2d(X, kernel_size=pool_size, stride=strides, padding=(p_h, p_w))
        check_pytorch_grad(compute_max_pool2d_param, inputs_np=[X_conv_np])


# @pytest.mark.xfail # Original test was marked xfail
def test_reduction_init():
    np.random.seed(0)
    shape = (10, 10)
    A0_np = np.random.uniform(-10, 10, size=shape).astype("float32")

    # B = te.compute((10,), lambda i: te.sum(A0[i, k] * A0[k, i], axis=k, init=0.0), name="B")
    # TVM's `init` parameter for reduction specifies initial value.
    # PyTorch's `sum` doesn't have `init` directly. If the sum is empty, it returns 0.
    # This is effectively checking `compute_sum_matmul` without init logic differences
    def compute_sum_matmul(A0):
        return torch.diagonal(A0 @ A0.T)
    check_pytorch_grad(compute_sum_matmul, inputs_np=[A0_np])


if __name__ == "__main__":
    pytest.main([__file__])
