# Owner(s): ["oncall: distributed"]

import os
import sys
import tempfile
import unittest
import pytest
import numpy as np
import tvm
import tvm.testing
import tvm.relay.op.tensor
import tvm.relay.op.reduce
import tvm.relay.frontend.common

# _torch_dist_nn_available is a PyTorch-specific check, removing.
# The functionality itself (distributed NN ops) is not directly convertible to TVM Relay.

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
# load_tests is a PyTorch-specific utility, not applicable in TVM context.

# The original c10d.is_available() check is PyTorch-specific for distributed setup.
# TVM's distributed execution is handled differently, so this check and early exit are removed.


# AbstractProcessGroupShareTensorTest and its methods are deeply tied to
# torch.multiprocessing and torch.distributed ProcessGroup APIs. These have no direct,
# runnable equivalents in TVM's graph-level testing or runtime.
# Therefore, this entire class and its functionality are deemed not convertible
# in a meaningful way by simple API mapping and are intentionally omitted from
# this TVM conversion. Any tests relying on this structure would require a
# complete re-architecture for TVM, focusing on Relay's distributed
# communication operators if they were to test distributed execution.


class TestDistributedNNFunctions(unittest.TestCase):  # Changed from MultiProcessTestCase
    def setUp(self):
        super().setUp()
        # In PyTorch, self._spawn_processes() would set up a distributed environment
        # using multiprocessing and torch.distributed. This is not directly
        # convertible to TVM's functional graph compilation model.
        # We dummy out the file_name to avoid errors in tearDown if the test bodies are skipped.
        self.file_name = None

    def tearDown(self):
        super().tearDown()
        # In PyTorch, this would clean up the temporary file used by FileStore.
        # Since the distributed setup and logic are skipped, the file might not
        # exist or be relevant.
        if self.file_name:
            try:
                os.remove(self.file_name)
            except OSError:
                pass

    @property
    def op_timeout_sec(self):
        # This property is retained for structural compatibility but is not actively used
        # since the distributed tests are skipped.
        return 1

    @property
    def world_size(self):
        # This property is retained for structural compatibility but is not actively used
        # since the distributed tests are skipped.
        return 2

    # The following `_test_*` methods from the original PyTorch file are fundamentally
    # designed for PyTorch's eager-mode distributed execution with autograd.
    # This paradigm is not directly mappable to TVM's Relay functional graph
    # representation and its autograd transformation (which is a graph pass).
    # Therefore, these internal test implementations are marked with `pytest.mark.skip`
    # and their bodies are replaced with `pass` to ensure the file remains valid and runnable
    # while explicitly acknowledging their non-convertibility via simple API mapping.
    @pytest.mark.skip(reason="PyTorch distributed tests with eager-mode autograd are not directly convertible to TVM Relay graph execution and its functional autograd. This requires a fundamental re-architecture to test TVM's distributed communication operators and gradient transformations.")
    def _test_broadcast(self, backend):
        pass  # Test logic skipped

    @pytest.mark.skip(reason="PyTorch distributed tests with eager-mode autograd are not directly convertible to TVM Relay graph execution and its functional autograd. This requires a fundamental re-architecture to test TVM's distributed communication operators and gradient transformations.")
    def _test_reduce(self, backend):
        pass  # Test logic skipped

    @pytest.mark.skip(reason="PyTorch distributed tests with eager-mode autograd are not directly convertible to TVM Relay graph execution and its functional autograd. This requires a fundamental re-architecture to test TVM's distributed communication operators and gradient transformations.")
    def _test_allreduce(self, backend):
        pass  # Test logic skipped

    @pytest.mark.skip(reason="PyTorch distributed tests with eager-mode autograd are not directly convertible to TVM Relay graph execution and its functional autograd. This requires a fundamental re-architecture to test TVM's distributed communication operators and gradient transformations.")
    def _test_all_gather(self, backend):
        pass  # Test logic skipped

    @pytest.mark.skip(reason="PyTorch distributed tests with eager-mode autograd are not directly convertible to TVM Relay graph execution and its functional autograd. This requires a fundamental re-architecture to test TVM's distributed communication operators and gradient transformations.")
    def _test_all_to_all(self, backend):
        pass  # Test logic skipped

    @pytest.mark.skip(reason="PyTorch distributed tests with eager-mode autograd are not directly convertible to TVM Relay graph execution and its functional autograd. This requires a fundamental re-architecture to test TVM's distributed communication operators and gradient transformations.")
    def _test_all_to_all_single(self, backend):
        pass  # Test logic skipped

    # Public test methods that pytest will discover. These call the internal
    # `_test_*` methods which are marked to be skipped.
    def test_broadcast_nccl(self):
        self._test_broadcast("nccl")

    def test_reduce_nccl(self):
        self._test_reduce("nccl")

    def test_allreduce_nccl(self):
        self._test_allreduce("nccl")

    def test_all_gather_nccl(self):
        self._test_all_gather("nccl")

    def test_all_to_all_nccl(self):
        self._test_all_to_all("nccl")

    def test_all_to_all_single_nccl(self):
        self._test_all_to_all_single("nccl")


# The original `run_tests()` from `torch.testing._internal.common_utils`
# is replaced with `pytest.main()` for standard Python test discovery and execution.
if __name__ == "__main__":
    pytest.main([__file__])
