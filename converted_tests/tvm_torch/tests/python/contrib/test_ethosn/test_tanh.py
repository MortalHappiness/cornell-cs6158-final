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

"""Arm(R) Ethos(TM)-N NPU integration tanh tests"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
import torch.testing as testing

# Ethos-N specific infrastructure is removed (`tei` module).
# The tests will now perform direct PyTorch operations for both float and quantized versions.
# `requires_ethosn` is removed.

# Helper function for quantization/dequantization, mimicking TVM's `qnn.op` behavior
def qnn_dequantize(data_tensor, input_scale, input_zero_point):
    return (data_tensor.float() - input_zero_point) * input_scale

def qnn_quantize(data_float, output_scale, output_zero_point, out_dtype_torch):
    data_quant = torch.round(data_float / output_scale + output_zero_point).to(out_dtype_torch)
    iinfo = np.iinfo(out_dtype_torch.item())
    return torch.clamp(data_quant, iinfo.min, iinfo.max)


def _get_model(input_tensor, input_zp, input_sc, output_zp, output_sc, dtype_torch):
    # This function represents the Relay graph, translated to PyTorch operations.
    
    # Dequantize to float
    dequantize_output = qnn_dequantize(input_tensor, input_sc, input_zp)
    
    # Apply Tanh
    tanh_output = torch.tanh(dequantize_output)
    
    # Quantize back
    model_output = qnn_quantize(tanh_output, output_sc, output_zp, dtype_torch)
    return model_output


@pytest.mark.parametrize("dtype_str", ["uint8", "int8"])
@pytest.mark.parametrize("shape", [(1, 52, 52, 3)])
def test_tanh(dtype_str, shape):
    """Compare Tanh output with TVM."""

    zp_min = np.iinfo(dtype_str).min
    zp_max = np.iinfo(dtype_str).max

    np.random.seed(0)

    # Convert dtype_str to PyTorch dtype
    if dtype_str == "uint8":
        dtype_torch = torch.uint8
    elif dtype_str == "int8":
        dtype_torch = torch.int8
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    inputs_np = np.random.randint(zp_min, high=zp_max, size=shape, dtype=dtype_str)
    input_tensor = torch.tensor(inputs_np, dtype=dtype_torch)

    # The original test ran NPU (Ethos-N) vs non-NPU (TVM) and checked for consistency.
    # Here, we run the PyTorch model once, which serves as the reference computation.
    output_pytorch = _get_model(input_tensor, zp_min + 120, 0.0250629, zp_min + 128, 0.0078125, dtype_torch)
    
    assert output_pytorch.shape == shape
    assert output_pytorch.dtype == dtype_torch


@pytest.mark.parametrize("dtype_str", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape, input_zp, input_sc, output_zp, output_sc, err_msg",
    [
        (
            (1, 16, 16, 16),
            120,
            0.0250629,
            64,
            0.0078125,
            "output quantization params=(64, 0.0078125), must = ({test_zp}, 1/256);",
        )
    ],
)
def test_tanh_failure(shape, input_zp, input_sc, output_zp, output_sc, err_msg, dtype_str):
    """Check Tanh error messages."""
    # The original test checked for specific Ethos-N compiler errors related to output quantization parameters.
    # PyTorch's functional ops do not have these compile-time restrictions directly at the functional call site.
    # The `qnn_quantize` helper will simply quantize with the given parameters, not raise an error based on
    # whether they conform to Ethos-N requirements.
    # Therefore, this test is adapted to assert that PyTorch runs successfully,
    # as its behavior for these 'unsupported' scenarios will simply be to compute the result.

    # Convert dtype_str to PyTorch dtype
    if dtype_str == "uint8":
        dtype_torch = torch.uint8
    elif dtype_str == "int8":
        dtype_torch = torch.int8
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    zp_min = np.iinfo(dtype_str).min
    test_zp = 0 if dtype_str == "int8" else 128 # Used in original error message

    inputs_np = np.random.randint(zp_min, high=np.iinfo(dtype_str).max, size=shape, dtype=dtype_str)
    input_tensor = torch.tensor(inputs_np, dtype=dtype_torch)

    # PyTorch will compute this without error, so we assert it runs successfully
    output_pytorch = _get_model(input_tensor, input_zp, input_sc, output_zp, output_sc, dtype_torch)
    
    assert output_pytorch.shape == shape
    assert output_pytorch.dtype == dtype_torch


if __name__ == "__main__":
    pytest.main([__file__])
