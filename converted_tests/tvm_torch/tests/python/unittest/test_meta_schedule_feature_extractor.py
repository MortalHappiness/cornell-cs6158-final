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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import re
from typing import List

import numpy as np
import torch # Added torch

# Dummy classes for TVM-specific concepts
# These replace tvm.meta_schedule.TuneContext and tvm.meta_schedule.search_strategy.MeasureCandidate
# as they represent internal TVM objects with no direct PyTorch functional equivalent for these tests.
class TuneContext:
    def __str__(self):
        # Mimic TVM's specific string format for testing purposes.
        # Use a fixed placeholder address for consistent regex matching in tests.
        return "meta_schedule.TuneContext(0xCAFEBABE)"

class MeasureCandidate:
    pass

# A dummy base class to simulate tvm.meta_schedule.feature_extractor.PyFeatureExtractor.
# Its purpose is to define the interface that derived classes should implement,
# without relying on TVM's FFI or runtime system.
class PyFeatureExtractor:
    def extract_from(
        self,
        context: TuneContext,
        candidates: List[MeasureCandidate],
    ) -> List[torch.Tensor]:  # Changed return type hint from np.ndarray to torch.Tensor
        """Simulates the abstract method to be implemented by derived feature extractors."""
        raise NotImplementedError("Derived classes must implement extract_from")

    def __str__(self):
        # Mimic TVM's specific string format for testing purposes.
        # The original TVM regex pattern included "meta_schedule.", so we explicitly add it here.
        # Use a fixed placeholder address for consistent regex matching in tests.
        return f"meta_schedule.{self.__class__.__name__}(0xCAFEBABE)"

# A no-op decorator to replace tvm.meta_schedule.utils.derived_object.
# This decorator is TVM-specific for FFI binding and has no direct PyTorch equivalent or necessity
# for the functional behavior tested here.
def derived_object(cls):
    return cls


def test_meta_schedule_feature_extractor():
    @derived_object
    class FancyFeatureExtractor(PyFeatureExtractor):  # Inherit from the dummy PyFeatureExtractor
        def extract_from(
            self,
            context: TuneContext,
            candidates: List[MeasureCandidate],
        ) -> List[torch.Tensor]:  # Changed return type hint
            # Convert tvm.runtime.ndarray.array to torch.tensor.
            # np.random.rand typically produces float64, using torch.float32 is a common ML default.
            return [torch.tensor(np.random.rand(4, 5), dtype=torch.float32)]

    extractor = FancyFeatureExtractor()
    features = extractor.extract_from(TuneContext(), [])
    assert len(features) == 1
    # Check shape, which works for both TVM NDArray and PyTorch Tensor.
    assert features[0].shape == (4, 5)
    # Add a specific check that the returned object is a PyTorch Tensor.
    assert isinstance(features[0], torch.Tensor)


def test_meta_schedule_feature_extractor_as_string():
    @derived_object
    class NotSoFancyFeatureExtractor(PyFeatureExtractor):  # Inherit from the dummy PyFeatureExtractor
        def extract_from(
            self,
            context: TuneContext,
            candidates: List[MeasureCandidate],
        ) -> List[torch.Tensor]:  # Changed return type hint
            return []

    feature_extractor = NotSoFancyFeatureExtractor()
    # The original TVM pattern implied a specific `__str__` format.
    # The dummy PyFeatureExtractor.__str__ is crafted to match this pattern using 0xCAFEBABE.
    # We use a regex that expects this specific format including the module prefix and the placeholder.
    pattern = re.compile(r"meta_schedule.NotSoFancyFeatureExtractor\(0x[a-fA-F0-9]+\)")
    # Assert that the string representation matches the pattern.
    assert pattern.match(str(feature_extractor)) is not None


if __name__ == "__main__":
    # A simple check for PyTorch backend availability, for runtime information.
    if torch.cuda.is_available():
        print("PyTorch CUDA is available.")
    else:
        print("PyTorch CUDA is not available, running on CPU.")

    test_meta_schedule_feature_extractor()
    test_meta_schedule_feature_extractor_as_string()
    print("All tests passed!")
