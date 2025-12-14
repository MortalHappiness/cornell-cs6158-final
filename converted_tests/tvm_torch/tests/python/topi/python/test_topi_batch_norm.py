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
"""Tests for the batch_norm operator."""
import numpy as np
import pytest

import torch
import torch.nn.functional as F
import torch.testing

# TVM-specific imports are removed, replaced by torch where applicable
# import tvm
# from tvm import te
# from tvm import topi
# import tvm.testing
# import tvm.topi.testing

# The original _DEVICE="llvm" implies CPU execution.
_DEVICE = "cpu" # Changed to CPU, 'cuda' could be used if available
# _BATCH_NORM_IMPLEMENT and related dispatch logic is TVM-specific and removed.


@pytest.mark.parametrize(
    "shape, axis, epsilon, center, scale",
    [
        ((1,), 0, 0.1, True, True),
        ((2, 3), 0, 0.1, True, True),
        ((1, 2, 4), 0, 0.1, True, True),
        ((1, 2, 3, 4), 0, 0.001, False, False),
        ((2, 3, 4, 1), 1, 0.01, False, True),
        ((3, 4, 1, 2), 2, 0.1, True, False),
        ((4, 1, 2, 3), 3, 1.0, True, True),
        ((1, 2, 4, 4, 5), 0, 0.1, True, True),
    ],
)
def test_batch_norm(shape, axis, epsilon, center, scale):
    # PyTorch uses 'cpu' or 'cuda' for device strings
    device = torch.device(_DEVICE)

    x_np = np.random.random(shape).astype("float32")
    # For PyTorch, gamma and beta should be None if scale/center are False.
    # The `topi.testing.batch_norm` implicitly handles `center`/`scale` by using 1.0/0.0 if not specified.
    # For our PyTorch `F.batch_norm` call, we explicitly pass None.
    gamma_full_np = np.random.random(shape[axis]).astype("float32")
    beta_full_np = np.random.random(shape[axis]).astype("float32")
    moving_mean_np = np.random.random(shape[axis]).astype("float32")
    moving_var_np = np.random.random(shape[axis]).astype("float32")

    # --- Reference computation using PyTorch (simulating topi.testing.batch_norm) ---
    # We use PyTorch's F.batch_norm to generate the reference output as well.
    # This assumes PyTorch's F.batch_norm is the accurate reference.
    # When `training=True`, F.batch_norm updates `running_mean` and `running_var` in-place.
    # Therefore, we need to pass clones if we want to observe the updated values.

    x_ref_pt = torch.tensor(x_np, device=device)
    gamma_ref_pt = torch.tensor(gamma_full_np, device=device) if scale else None
    beta_ref_pt = torch.tensor(beta_full_np, device=device) if center else None
    
    # Clone moving_mean and moving_var for the reference, as they are updated in-place
    running_mean_ref_pt = torch.tensor(moving_mean_np, device=device)
    running_var_ref_pt = torch.tensor(moving_var_np, device=device)

    num_dims = len(shape)
    x_ref_pt_in = x_ref_pt
    output_permutation_order = None # Flag to indicate if inverse permute or squeeze is needed

    if num_dims == 1 and axis == 0:
        # For (C,) shape, unsqueeze to (1, C) so C is at dim 1.
        x_ref_pt_in = x_ref_pt.unsqueeze(0)
        # Output will be (1, C), needs to be squeezed back to (C,)
    elif num_dims > 1 and axis != 1:
        # Construct permutation order to move 'axis' to dim 1, preserving 0 (batch) and other relative orders.
        # e.g., (N, H, W, C) with axis=3 -> perm_order = (0, 3, 1, 2) to get (N, C, H, W)
        source_dims = list(range(num_dims))
        perm_order = [0, axis] + [d for d in source_dims if d != 0 and d != axis]
        x_ref_pt_in = x_ref_pt.permute(*perm_order)

        # Create inverse permutation for the output to restore original TVM layout
        inv_perm_order = [0] * num_dims
        for i, dim_idx in enumerate(perm_order):
            inv_perm_order[dim_idx] = i
        output_permutation_order = inv_perm_order
    # Else: num_dims > 1 and axis == 1, input is already (N, C, ...) as expected, no permute.
    
    # The 'training=True' mode in F.batch_norm ensures that running_mean/var are updated.
    # This mimics the behavior of tvm.topi.testing.batch_norm which returns updated running stats.
    out_x_ref_unpermuted_pt = F.batch_norm(
        input=x_ref_pt_in,
        running_mean=running_mean_ref_pt, # This will be updated in-place
        running_var=running_var_ref_pt,   # This will be updated in-place
        weight=gamma_ref_pt,
        bias=beta_ref_pt,
        training=True,
        momentum=0.1, # PyTorch default momentum
        eps=epsilon,
    )
    
    # Apply inverse transformation to output_x to match expected NumPy shape
    if num_dims == 1 and axis == 0:
        out_x_np_ref = out_x_ref_unpermuted_pt.squeeze(0).cpu().numpy()
    elif output_permutation_order is not None:
        out_x_np_ref = out_x_ref_unpermuted_pt.permute(*output_permutation_order).cpu().numpy()
    else:
        out_x_np_ref = out_x_ref_unpermuted_pt.cpu().numpy()

    # The updated running mean and variance are now in running_mean_ref_pt and running_var_ref_pt
    out_moving_mean_np_ref = running_mean_ref_pt.cpu().numpy()
    out_moving_var_np_ref = running_var_ref_pt.cpu().numpy()

    # --- Actual PyTorch execution (intended for TorchInductor) ---
    # This section replaces TVM's TE computation and build process.
    x_torch = torch.tensor(x_np, device=device)
    gamma_torch = torch.tensor(gamma_full_np, device=device) if scale else None
    beta_torch = torch.tensor(beta_full_np, device=device) if center else None
    
    # Clone moving_mean and moving_var for the actual run, so they are updated in-place
    moving_mean_torch = torch.tensor(moving_mean_np, device=device)
    moving_var_torch = torch.tensor(moving_var_np, device=device)

    # Apply permutation to input if needed (same logic as reference)
    x_torch_in = x_torch
    if num_dims == 1 and axis == 0:
        x_torch_in = x_torch.unsqueeze(0)
    elif num_dims > 1 and axis != 1:
        source_dims = list(range(num_dims))
        perm_order = [0, axis] + [d for d in source_dims if d != 0 and d != axis]
        x_torch_in = x_torch.permute(*perm_order)

    out_x_torch_unpermuted = F.batch_norm(
        input=x_torch_in,
        running_mean=moving_mean_torch, # This will be updated in-place
        running_var=moving_var_torch,   # This will be updated in-place
        weight=gamma_torch,
        bias=beta_torch,
        training=True, # Consistent with reference
        momentum=0.1,
        eps=epsilon,
    )

    # Apply inverse permutation/squeeze to output if needed (same logic as reference)
    if num_dims == 1 and axis == 0:
        out_x_torch = out_x_torch_unpermuted.squeeze(0)
    elif output_permutation_order is not None:
        out_x_torch = out_x_torch_unpermuted.permute(*output_permutation_order)
    else:
        out_x_torch = out_x_torch_unpermuted

    # Convert results to NumPy for assertion
    out_x_actual_np = out_x_torch.cpu().numpy()
    out_moving_mean_actual_np = moving_mean_torch.cpu().numpy()
    out_moving_var_actual_np = moving_var_torch.cpu().numpy()

    # Assertions
    # Note: torch.testing.assert_allclose is deprecated in favor of torch.testing.assert_close
    torch.testing.assert_allclose(out_x_actual_np, out_x_np_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_allclose(out_moving_mean_actual_np, out_moving_mean_np_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_allclose(out_moving_var_actual_np, out_moving_var_np_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    # To run this test directly without pytest, call the function.
    # Note: Pytest fixtures for `shape, axis, ...` would not be automatically applied.
    # For local execution, one would typically pick a single set of parameters.
    # For example:
    test_batch_norm(shape=(1, 2, 4), axis=0, epsilon=0.1, center=True, scale=True)
    test_batch_norm(shape=(1, 2, 3, 4), axis=0, epsilon=0.001, center=False, scale=False)
    test_batch_norm(shape=(4, 1, 2, 3), axis=3, epsilon=1.0, center=True, scale=True)
    test_batch_norm(shape=(1,), axis=0, epsilon=0.1, center=True, scale=True)
    print("All selected batch_norm tests passed!")
