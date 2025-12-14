import os
import sys
import numpy as np
import pytest

import torch
import torch.testing
import tempfile


# This entire file tests TVM's `AnnotateTarget` pass, which is a TVM-specific compiler
# infrastructure feature for marking regions of the Relay IR graph for external code generation.
# There is no direct functional or conceptual equivalent in PyTorch, which operates on
# Python-level models and computational graphs derived from eager execution or tracing.
# Therefore, all tests in this file are marked as skipped.


def check_result(
    mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device="cpu", params=None
):
    pytest.skip("TODO: TVM-specific runtime checking (VM, Graph Executor) is not convertible to PyTorch.")
    # The original TVM `check_result` function involves:
    # 1. Exporting and loading TVM compiled libraries (`lib.export_library`, `runtime.load_module`).
    # 2. Compiling and running with TVM's Relay Virtual Machine (`relay.vm.compile`, `runtime.vm.VirtualMachine`).
    # 3. Compiling and running with TVM's Graph Executor (`relay.build`, `tvm.contrib.graph_executor.create`).
    # These are low-level TVM compilation and execution details that do not have
    # direct, high-level functional equivalents in PyTorch.
    # The `torch.testing.assert_allclose` could be used for numerical comparison
    # if the computation itself were convertible, but the entire setup is TVM-specific.


def test_extern_dnnl():
    pytest.skip("TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (e.g., `relay.nn.conv2d`, `relay.add`, `relay.annotation.compiler_begin`/`_end`) are not convertible to PyTorch. This test targets external DNNL codegen.")


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation are not convertible to PyTorch. This test specifically targets DNNL codegen on a MobileNet model within TVM's framework.")
def test_extern_dnnl_mobilenet():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass, `tvm.ir.register_op_attr`, and Relay IR manipulation are not convertible to PyTorch.")
def test_multiple_ends():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass, `tvm.ir.register_op_attr`, and Relay IR manipulation are not convertible to PyTorch.")
def test_type_propagation():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (RefCreate, RefRead, RefWrite) are not convertible to PyTorch.")
def test_ref_create_read_write():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (TupleNode, TupleGetItem) are not convertible to PyTorch.")
def test_tuple():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (Composite functions) are not convertible to PyTorch.")
def test_composite_function():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and `tvm.ir.register_op_attr` are not convertible to PyTorch.")
def test_double_target():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and `tvm.ir.register_op_attr` are not convertible to PyTorch.")
def test_different_targets():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and `tvm.ir.register_op_attr` are not convertible to PyTorch.")
def test_multiple_runs():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (Tuple, TupleGetItem, clip) are not convertible to PyTorch.")
def test_ends_with_tuple():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (If-else, equal, tanh, sigmoid, erf) are not convertible to PyTorch.")
def test_if_else():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (While-Let, less, add, zeros_like) are not convertible to PyTorch.")
def test_while_let():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (If-else with free vars) are not convertible to PyTorch.")
def test_if_free_vars():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation are not convertible to PyTorch.")
def test_free_vars_zeros():
    pass


@pytest.mark.skip(reason="TODO: TVM's `AnnotateTarget` pass and Relay IR manipulation (empty tuple) are not convertible to PyTorch.")
def test_empty_tuple():
    pass


if __name__ == "__main__":
    # Original TVM tests were executed here.
    # Since all tests are marked as skipped due to non-convertibility,
    # calling them directly would just result in skipped tests.
    # To avoid confusion or attempting to run non-existent sub-functions,
    # the main execution block is updated.
    print("Skipping all tests in this file as TVM-specific compiler passes are not convertible to PyTorch.")
