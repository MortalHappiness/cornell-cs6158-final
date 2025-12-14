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
# pylint: disable=invalid-name

"""
Testing for the pass that annotates used memory for each primitive
Relay function.
"""

import pytest

# TODO: This entire test file is highly TVM-specific, as it tests a Relay IR analysis pass
# (AnnotateUsedMemory) and its interaction with Relay's internal function attributes,
# composite functions, and memory modeling. There is no direct, equivalent concept
# or API in PyTorch for analyzing memory usage at this level of a symbolic graph
# representation in a user-facing manner. PyTorch's memory management is typically
# handled dynamically during eager execution or implicitly during torch.compile.
# Therefore, these tests are marked as skipped.


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_simple():
    """
    Test simple graph with one primitive function.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_multiple_functions():
    """
    Test a graph with multiple primitive functions.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_mixed_data_types():
    """
    Test a graph with a primitive function that has mixed datatypes.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_parallel_function_call():
    """
    Test a graph when the results of two functions are concatenated
    into a single result. The second function will also have the result
    of the first function alive so will be annotated with a larger
    "used memory" value.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_many_different_parallel_calls():
    """
    Test a graph that calls many different functions in parallel.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_nested_branches():
    """
    Tests a graph with branches that also branch.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_composite_inner_function():
    """
    Tests the typical BYOC use case where a primitive function
    contains a composite function.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_multiple_calls_to_same_function():
    """
    Tests the case when there are multiple calls to the same function.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_parallel_calls_to_same_function():
    """
    Test parallel calls to the same function.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_parallel_calls_with_non_ifm_input():
    """
    Test a graph that calls many different functions in parallel where
    the input is not the input to the function.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_dynamic_io_tensor_not_supported():
    """
    Test to check dynamic IO tensor error.
    """
    pass


@pytest.mark.skip("TVM-specific test for Relay IR analysis pass (AnnotateUsedMemory). No direct PyTorch equivalent.")
def test_dynamic_callsite_tensor_not_supported():
    """
    Test to check dynamic callsite tensor error.
    """
    pass
