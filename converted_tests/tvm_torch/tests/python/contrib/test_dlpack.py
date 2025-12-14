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
import numpy as np
import torch.utils.dlpack
import torch.testing


def verify_torch_dlpack():
    a = np.random.randn(1337)
    # Original TVM: tvm_a = tvm.nd.array(a)
    # Original TVM: np.testing.assert_equal(tvm.nd.from_dlpack(tvm_a.to_dlpack()).numpy(), a)
    # This section tested TVM's internal DLPack capabilities and conversion from NumPy.
    # In a PyTorch-only context, this translates to PyTorch's internal DLPack capabilities.
    torch_a = torch.tensor(a)
    reconstructed_a = torch.utils.dlpack.from_dlpack(torch_a.__dlpack__())
    np.testing.assert_equal(reconstructed_a.numpy(), a)
    assert torch_a.data_ptr() == reconstructed_a.data_ptr() # Ensure memory sharing

    try:
        x = torch.rand(56, 56)
        
        # Original TVM: tvm_x = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        # This tested PyTorch -> DLPack -> TVM. In PyTorch-only, it's PyTorch -> DLPack -> PyTorch.
        x_from_dlpack = torch.utils.dlpack.from_dlpack(x.__dlpack__())
        np.testing.assert_equal(x.numpy(), x_from_dlpack.numpy())
        assert x.data_ptr() == x_from_dlpack.data_ptr()

        # Original TVM: y = tvm.nd.from_dlpack(tvm_x)
        # This tested TVM -> DLPack -> TVM. In PyTorch-only, it's PyTorch -> DLPack -> PyTorch.
        y_from_dlpack = torch.utils.dlpack.from_dlpack(x_from_dlpack.__dlpack__())
        np.testing.assert_equal(y_from_dlpack.numpy(), x_from_dlpack.numpy())
        assert y_from_dlpack.data_ptr() == x_from_dlpack.data_ptr()

        # Original TVM: torch.utils.dlpack.from_dlpack(y.to_dlpack()).numpy()
        # This tested TVM -> DLPack -> PyTorch. In PyTorch-only, it's PyTorch -> DLPack -> PyTorch.
        final_dlpack_view = torch.utils.dlpack.from_dlpack(y_from_dlpack.__dlpack__())
        np.testing.assert_equal(final_dlpack_view.numpy(), x_from_dlpack.numpy())
        assert final_dlpack_view.data_ptr() == y_from_dlpack.data_ptr()

        # --- Original section involving TVM Tensor Expression (TE) and compilation ---
        # The original test used TVM's Tensor Expression (TE) to define a matrix
        # multiplication and compiled it using tvm.build. It then demonstrated
        # calling this TVM-compiled function with PyTorch tensors via `to_pytorch_func`,
        # leveraging DLPack for efficient tensor transfer between TVM and PyTorch.
        #
        # Since this conversion aims for PyTorch-only tests, the TVM TE definition
        # (`te.placeholder`, `te.reduce_axis`, `te.compute`, `te.create_schedule`)
        # and TVM compilation (`tvm.build`, `to_pytorch_func`) are inherently
        # TVM-specific and cannot be directly translated to PyTorch.
        #
        # We replace the call to the TVM-compiled function with its direct PyTorch
        # equivalent (`torch.matmul`) to ensure numerical correctness.
        # The aspect of testing the TVM-PyTorch DLPack bridge itself is thus removed.
        n = 137
        xx = torch.rand(n, n)
        yy = torch.rand(n, n)
        
        # PyTorch ground truth for matrix multiplication
        zz_ground_truth = xx.mm(yy)

        # Output tensor for the operation, mimicking TVM's output buffer pattern
        zz2_result = torch.empty(n, n)
        
        # Replaces f_pytorch(xx, yy, zz2) with direct PyTorch computation
        torch.matmul(xx, yy, out=zz2_result)
        
        torch.testing.assert_allclose(zz_ground_truth.numpy(), zz2_result.numpy(), rtol=1e-4, atol=1e-4)

        # TODO: The original test explicitly verified a TVM-compiled operator
        # could be called via DLPack in a PyTorch context. This specific cross-framework
        # interoperability testing is removed in a PyTorch-only conversion.
        # The numerical equivalence is preserved, but the DLPack bridge testing for
        # TVM-compiled functions is inherently lost.

    except ImportError:
        # If torch is not available, skip the torch-specific tests.
        # This block is kept as per the original TVM test structure,
        # although in a PyTorch environment, torch is expected.
        pass


def test_torch_dlpack():
    # Run dlpack interoperability test a few times to make sure it's stable.
    for i in range(5):
        verify_torch_dlpack()


if __name__ == "__main__":
    test_torch_dlpack()
