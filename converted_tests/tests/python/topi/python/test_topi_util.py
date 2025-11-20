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
"""Test code for util"""

import torch
import pytest
import numpy as np

# TODO: tvm.topi.utils.get_shape is for inferring output shapes after layout transformations
# in TVM's Tensor Expression (TE) framework. PyTorch operates on concrete tensor shapes
# and has implicit layout handling (e.g., NCHW or NHWC via memory_format or permute).
# There is no direct PyTorch API that takes (source_shape, source_layout, dest_layout)
# to predict the destination shape. This functionality is TVM-specific and cannot be
# directly translated to an equivalent PyTorch test without reimplementing a layout
# inference engine.
#
# The original test functions are commented out as they rely on TVM-specific constructs.

# def verify_get_shape(src_shape, src_layout, dst_layout, expect_shape):
#     dst_shape = topi.utils.get_shape(src_shape, src_layout, dst_layout)
#     assert dst_shape == expect_shape, "Shape mismatch: expecting %s but got %s" % (
#         expect_shape,
#         dst_shape,
#     )


# def test_get_shape():
#     verify_get_shape((1, 3, 224, 224), "NCHW", "NCHW", (1, 3, 224, 224))
#     verify_get_shape((1, 3, 224, 224), "NCHW", "NHWC", (1, 224, 224, 3))
#     verify_get_shape((3, 2, 32, 48, 16), "NCHW16c", "NC16cWH", (3, 2, 16, 48, 32))
#     verify_get_shape((2, 3, 32, 32, 16, 8), "OIHW16i8o", "HWO8oI16i", (32, 32, 2, 8, 3, 16))


# if __name__ == "__main__":
#     test_get_shape()

print("TODO: This file contains TVM-specific layout inference tests that cannot be directly translated to PyTorch.")
print("The original tests are commented out.")
