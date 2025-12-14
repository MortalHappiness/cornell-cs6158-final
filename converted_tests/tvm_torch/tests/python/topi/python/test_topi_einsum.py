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
import torch.testing # For assert_close, replacing tvm.testing.assert_allclose

# Helper to run PyTorch einsum, simulating the TVM test setup structure
def run_einsum_on_torch(einsum_func, *args):
    """Take numpy arrays as args, convert them to PyTorch tensors and call `einsum_func`.
    Result of einsum_func is converted back to numpy array and returned.
    """
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert numpy inputs to PyTorch tensors
    # Ensure dtype consistency with original TVM tests (float32)
    torch_inputs = [torch.tensor(arg, dtype=torch.float32, device=device) for arg in args]

    # Call the PyTorch einsum equivalent through the provided lambda
    # The `einsum_func` here is expected to be a lambda like `lambda *tensors: torch.einsum(subscripts, *tensors)`
    torch_output = einsum_func(*torch_inputs)

    # Convert the output PyTorch tensor back to numpy array
    return torch_output.cpu().numpy()


def verify_einsum(subscripts, shapes):
    ops = []
    for shape in shapes:
        tmp = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
        ops.append(tmp)

    # NumPy reference computation
    c1 = np.einsum(subscripts, *ops)

    # PyTorch computation, wrapped to fit the original test structure
    # The lambda takes *torch_inputs and unpacks them for torch.einsum
    c2 = run_einsum_on_torch(lambda *torch_inputs: torch.einsum(subscripts, *torch_inputs), *ops)

    # Use torch.testing.assert_close for comparison, as assert_allclose is deprecated
    torch.testing.assert_close(c1, c2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "equation,inputs",
    [
        ("ii", [(5, 5)]),
        ("ii->i", [(5, 5)]),
        ("ij->i", [(5, 5)]),
        ("...j->...", [(5, 5)]),
        ("...j, j", [(5, 5), (5,)]),
        ("..., ...", [(), (2, 3)]),
        ("ijk, jil->kl", [(3, 4, 5), (4, 3, 2)]),
        ("ij, ij -> i", [(1, 4), (2, 4)]),
        ("...ij, ...jk -> ...ik", [(1, 4), (4, 2)]),
        ("...ij, ...ik -> ...jk", [(1, 1, 1, 4), (1, 1, 1, 3)]),
        ("...ik, ...jk, ...hk -> i...jh", [(3, 4, 4), (1, 5, 3, 8, 4), (2, 5, 3, 6, 4)]),
        ("ij,jk->ik", [(2, 3), (3, 4)]),
        ("ij,jk,km->im", [(2, 3), (3, 4), (4, 5)]),
    ],
)
def test_einsum(equation, inputs):
    verify_einsum(equation, inputs)


if __name__ == "__main__":
    # In a PyTorch test environment, pytest automatically discovers and runs tests.
    # The equivalent of tvm.testing.main() in this context is running pytest.main()
    # on the current file, allowing it to be executed directly from the command line.
    pytest.main([__file__])
