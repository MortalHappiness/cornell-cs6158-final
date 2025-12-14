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
"""OpenCL ML network tests."""

import pytest
import numpy as np
import torch
import torch.testing

# TODO: This file contains tests for TVM's integration with Keras models and the OpenCL ML backend.
# The core functionalities used, such as `relay.frontend.from_keras` for model import
# and `build_and_run` for TVM-specific compilation and execution (especially targeting
# OpenCL ML with TVM's infrastructure), are deeply intertwined with the TVM compilation
# and runtime ecosystem.
#
# There are no direct, equivalent PyTorch APIs for:
# - Converting Keras models to a PyTorch native graph format (without external tools like ONNX).
# - TVM's `relay` IR and its associated operations.
# - TVM's custom `build_and_run` infrastructure for specific TVM backends like OpenCL ML.
#
# Therefore, these tests are not convertible to equivalent PyTorch/TorchInductor tests
# through direct API mapping. The original test logic has been commented out, and the
# tests are marked to be skipped.

# Original helper functions are not convertible as they rely on TVM's Relay IR and runtime.
# def _build_and_run_network(mod, params, inputs, data, device, atol, rtol, tvm_log=""):
#     """Helper function to build and run a network."""
#
#     outputs = []
#     for clml in [True, False]:
#         outputs.append(
#             build_and_run(mod, data, 1, params, device, enable_clml=clml, tune_log=tvm_log)[0][0]
#         )
#     return outputs
#
#
# def _get_keras_model(keras_model, inputs_dict, data):
#     """Convert Keras graph to relay."""
#     inputs = {}
#     for name, (shape, _) in inputs_dict.items():
#         inputs[keras_model.input_names[0]] = shape
#
#     from tensorflow.keras.layers import Input
#     from tensorflow.keras.models import Model
#
#     def get_bottom_top_model(model, layer_name):
#         layer = model.get_layer(layer_name)
#         bottom_input = model.layers[0].input
#         bottom_output = layer.output
#         bottom_model = Model(bottom_input, bottom_output)
#         return bottom_model
#
#     keras_model = get_bottom_top_model(keras_model, "predictions")
#     ref_output = keras_model.predict(data["input_1"].transpose(0, 2, 3, 1))
#
#     mod, params = relay.frontend.from_keras(keras_model, inputs, layout="NCHW")
#     return mod, params, ref_output


@pytest.mark.skip(reason="Tests TVM-specific Keras->Relay conversion and OpenCL ML backend integration, no direct PyTorch equivalent.")
@pytest.mark.parametrize("dtype", ["float16"])
def test_mobilenet(device, dtype):
    # The original test logic involves TVM-specific components:
    # - Loading Keras models and converting them to TVM Relay IR (`_get_keras_model`).
    # - Running the Relay IR with TVM's build and runtime infrastructure (`_build_and_run_network`),
    #   which targets specific TVM backends like OpenCL ML.
    # - The assertion compares results obtained from different TVM compilation paths (OpenCL vs. CLML).
    #
    # This setup is not directly translatable to PyTorch API calls.
    pass


@pytest.mark.skip(reason="Tests TVM-specific Keras->Relay conversion and OpenCL ML backend integration, no direct PyTorch equivalent.")
@pytest.mark.parametrize("dtype", ["float16"])
def test_inception_v3(device, dtype):
    # Similar to test_mobilenet, this test relies heavily on TVM's Keras frontend
    # and backend-specific execution flow, which cannot be directly mapped to PyTorch.
    pass


@pytest.mark.skip(reason="Tests TVM-specific Keras->Relay conversion and OpenCL ML backend integration, no direct PyTorch equivalent.")
@pytest.mark.parametrize("dtype", ["float16"])
def test_resnet50v2(device, dtype):
    # Similar to test_mobilenet, this test relies heavily on TVM's Keras frontend
    # and backend-specific execution flow, which cannot be directly mapped to PyTorch.
    pass
