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
import pytest
import numpy as np


def test_lstm_cell_inline():
    num_step = 128
    num_input = 256
    num_hidden = 1152
    batch_size = 4
    dtype = torch.float32

    # Global transition matrices (weights)
    # Equivalent to te.placeholder in TVM
    X_val = torch.randn(num_step - 1, batch_size, num_input, dtype=dtype)
    Wi2h_val = torch.randn(4, num_hidden, num_input, dtype=dtype)
    Wh2h_val = torch.randn(4, num_hidden, num_hidden, dtype=dtype)

    # Initial hidden and cell states
    # Equivalent to te.compute((1, ...), lambda *i: 0.0) in TVM
    init_c_val = torch.zeros(batch_size, num_hidden, dtype=dtype)
    init_h_val = torch.zeros(batch_size, num_hidden, dtype=dtype)

    # Define the combine function for torch.scan to simulate LSTM cell
    # This function encapsulates the logic for a single time step,
    # converting TVM's te.compute expressions.
    # The 'inlining' of gates, etc. is naturally handled by direct function calls in Python.
    def lstm_combine_fn(carry, x_step):
        # carry: (h_prev, c_prev) where each is (batch_size, num_hidden)
        # x_step: (batch_size, num_input) for the current time step's input X[t-1]

        h_prev, c_prev = carry

        # Equivalent to s_i2h in TVM TE
        # s_i2h[t, x, i, j] = sum_k(X[t-1, i, k] * Wi2h[x, j, k])
        # x_step (batch_size, num_input)
        # Wi2h_val (4, num_hidden, num_input)
        # Output should be (batch_size, 4, num_hidden)
        # Using einsum for clear mapping of index contraction
        i2h_term = torch.einsum("bk,gjk->bgj", x_step, Wi2h_val)

        # Equivalent to s_h2h in TVM TE
        # s_h2h[t, x, i, j] = sum_k(s_state_h[t-1, i, k] * Wh2h[x, j, k])
        # h_prev (batch_size, num_hidden)
        # Wh2h_val (4, num_hidden, num_hidden)
        # Output should be (batch_size, 4, num_hidden)
        h2h_term = torch.einsum("bh,gjh->bgj", h_prev, Wh2h_val)

        # Equivalent to gates in TVM TE (s_i2h + s_h2h)
        gates = i2h_term + h2h_term # (batch_size, 4, num_hidden)

        # Extract gate components and apply activations
        # These correspond to in_gate, in_transform, forget_gate, out_gate in TVM TE
        in_gate = torch.sigmoid(gates[:, 0, :])
        in_transform = torch.tanh(gates[:, 1, :])
        forget_gate = torch.sigmoid(gates[:, 2, :])
        out_gate = torch.sigmoid(gates[:, 3, :])

        # Compute next cell state
        # Equivalent to next_c in TVM TE
        next_c = forget_gate * c_prev + in_gate * in_transform

        # Compute next hidden state
        # Equivalent to next_h in TVM TE
        next_h = out_gate * torch.tanh(next_c)

        # The new carry state for the next iteration of the scan
        new_carry = (next_h, next_c)
        # The outputs for the current time step, to be stacked by torch.scan
        output_step = (next_h, next_c)
        return new_carry, output_step

    # Perform the scan operation
    # torch.scan will iterate over the first dimension of X_val.
    # The initial carry state for h and c are init_h_val and init_c_val.
    # stacked_outputs will be a tuple of tensors, where each tensor corresponds
    # to a sequence of outputs (h_seq, c_seq).
    final_carry, stacked_outputs = torch.scan(
        lstm_combine_fn,
        (init_h_val, init_c_val), # initial_state (h_0, c_0)
        X_val,                     # input sequence X
    )

    # Unpack the stacked outputs (sequences of h and c over time)
    scan_h_output, scan_c_output = stacked_outputs

    # Verify the shapes of the output sequences
    # The scan runs for num_step - 1 steps, each producing (batch_size, num_hidden)
    assert scan_h_output.shape == (num_step - 1, batch_size, num_hidden)
    assert scan_c_output.shape == (num_step - 1, batch_size, num_hidden)

    # In TVM, the purpose of tvm.lower was to check if the schedule could be lowered.
    # In PyTorch, simply running the code successfully demonstrates its validity and executability.
    # Further checks would involve comparing against a reference PyTorch LSTM if numerical
    # correctness was being tested. For this test, verifying the computation graph construction
    # and output shapes is sufficient.


if __name__ == "__main__":
    test_lstm_cell_inline()
