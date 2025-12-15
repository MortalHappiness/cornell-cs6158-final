# Owner(s): ["oncall: distributed"]

import operator
import os
import sys
import threading
from functools import reduce
from unittest import SkipTest
import unittest
import pytest
import numpy as np

# Removed torch, torch.autograd, torch.distributed, ReduceOp imports as they are PyTorch-specific.
# Removed common_distributed and common_utils imports as their functionality is not directly portable or handled by pytest.

# Removed torch.distributed.is_available() check.
# Removed device_type assignment as torch.accelerator.current_accelerator() is PyTorch-specific.

DEFAULT_WORLD_SIZE = 4


class TestCollectivesWithWrapper(unittest.TestCase):
    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_broadcast_object_list(self):
        # TODO: This test relies on torch.distributed's runtime collective `broadcast_object_list`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_collective_error_on_rank_zero(self):
        # TODO: This test relies on torch.distributed's runtime collective communication
        # and error propagation in a multi-process setup, which is not directly portable to TVM.
        # The inner `_test_method` is also not portable.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_collective_error_on_rank_non_zero(self):
        # TODO: This test relies on torch.distributed's runtime collective communication
        # and error propagation in a multi-process setup, which is not directly portable to TVM.
        # The inner `_test_method` is also not portable.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_collective_error_on_rank_non_zero_all(self):
        # TODO: This test relies on torch.distributed's runtime collective communication
        # and error propagation in a multi-process setup, which is not directly portable to TVM.
        # The inner `_test_method` is also not portable.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests skip exception capture, marked as skipped itself.")
    def test_skip(self):
        # This test checks skip exceptions, effectively it tests the test framework itself.
        # The original @skip decorator is replaced by pytest.mark.skip.
        # The internal _test_method is skipped along with the wrapper.
        # IS_SANDCASTLE is PyTorch-specific and removed.
        raise unittest.SkipTest("check if skip exception can be captured correctly.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_all_to_all_single_tensor(self):
        # TODO: This test relies on torch.distributed's runtime collective `all_to_all_single`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_all_to_all_single_list(self):
        # TODO: This test relies on torch.distributed's runtime collective `all_to_all_single`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_all_to_all_single_none(self):
        # TODO: This test relies on torch.distributed's runtime collective `all_to_all_single`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")


class TestCollectivesWithBaseClass(unittest.TestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        # os.environ["TORCH_DIST_INIT_BARRIER"] is PyTorch-specific and removed.
        super().setUp()
        # self._spawn_threads() is PyTorch-specific and removed.

    def tearDown(self):
        super().tearDown()
        # os.environ["TORCH_DIST_INIT_BARRIER"] is PyTorch-specific and removed.

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_allgather(self):
        # TODO: This test relies on torch.distributed's runtime collective `all_gather`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_broadcast(self):
        # TODO: This test relies on torch.distributed's runtime collective `broadcast`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_scatter(self):
        # TODO: This test relies on torch.distributed's runtime collective `scatter`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_reduce_scatter(self):
        # TODO: This test relies on torch.distributed's runtime collective `reduce_scatter`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_broadcast_object_list(self):
        # TODO: This test relies on torch.distributed's runtime collective `broadcast_object_list`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_all_reduce(self):
        # TODO: This test relies on torch.distributed's runtime collective `all_reduce`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_all_to_all(self):
        # TODO: This test relies on torch.distributed's runtime collective `all_to_all`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_all_reduce_ops(self):
        # TODO: This test relies on torch.distributed's runtime collective `all_reduce` with various ops,
        # including `ReduceOp` enum, which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior and assertions on specific ranks, not directly portable to TVM.")
    def test_assert_equal_on_rank(self):
        # TODO: This test relies on torch.distributed's runtime information and custom assertions
        # like `assertEqualOnRank` and `assertNotEqualOnRank`, which are not directly portable to TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior and process groups, not directly portable to TVM.")
    def test_subpg(self):
        # TODO: This test relies on torch.distributed's process groups (`new_group`) and collectives on subgroups,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior and thread interaction, not directly portable to TVM.")
    def test_using_pg_from_another_thread(self):
        # TODO: This test involves torch.distributed's process group usage from another thread,
        # which is a runtime coordination mechanism not directly portable to TVM.
        raise unittest.S_kipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_gather(self):
        # TODO: This test relies on torch.distributed's runtime collective `gather`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior, not directly portable to TVM.")
    def test_all_reduce_coalesced(self):
        # TODO: This test relies on torch.distributed's runtime collective `all_reduce_coalesced`,
        # which has no direct API equivalent for Python-level distributed execution in TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior, not directly portable to TVM.")

    @pytest.mark.skip(reason="Tests torch.distributed runtime behavior with autograd, not directly portable to TVM.")
    # @skip_if_lt_x_gpu(1) # Removed, as it is PyTorch-specific GPU check and the test is skipped anyway.
    def test_bwd_sees_fwd_pg(self):
        # TODO: This test involves torch.autograd and distributed context within forward/backward,
        # which is a complex runtime interaction not directly portable to TVM.
        raise unittest.SkipTest("Tests torch.distributed runtime behavior with autograd, not directly portable to TVM.")


if __name__ == "__main__":
    unittest.main()
