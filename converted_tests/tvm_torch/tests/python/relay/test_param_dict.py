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
import os
import numpy as np
import torch
import io
import pytest

# Helper to map TVM dtypes to PyTorch dtypes (if needed for string conversions)
TVM_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int8": torch.int8,
    "bool": torch.bool,
}


def test_save_load():
    x_np = np.ones((10, 2)).astype("float32")
    y_np = np.ones((1, 2, 3)).astype("float32")

    # Convert numpy arrays to torch tensors for saving
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    params = {"x": x, "y": y}

    # Simulate runtime.save_param_dict using torch.save to a BytesIO object
    buffer = io.BytesIO()
    torch.save(params, buffer)
    param_bytes = buffer.getvalue()

    # The original TVM test asserts isinstance(param_bytes, bytearray).
    # `getvalue()` returns `bytes`. `bytearray` can be created from `bytes`.
    # For compatibility with a byte-oriented check, `bytes` is fine.
    assert isinstance(param_bytes, bytes)

    # Simulate relay.load_param_dict using torch.load from BytesIO
    buffer_load = io.BytesIO(param_bytes)
    param2 = torch.load(buffer_load)

    assert len(param2) == 2
    torch.testing.assert_allclose(param2["x"], x)
    torch.testing.assert_allclose(param2["y"], y)


def test_ndarray_reflection():
    # Make two `NDArrayWrapper`s that point to the same underlying array.
    np_array = np.random.uniform(size=(10, 2)).astype("float32")

    # In TVM, `tvm.nd.array(np_array)` creates a new TVM NDArray, copying data.
    # When `param_dict = {"x": tvm_array, "y": tvm_array}` is then used,
    # "x" and "y" refer to the *same Python object*. This is checked by `.same_as()`.
    tvm_array_like_obj = torch.tensor(np_array)  # Creates a new tensor, copying data from numpy

    param_dict = {"x": tvm_array_like_obj, "y": tvm_array_like_obj}

    # Check for same Python object reference (equivalent to TVM's .same_as() in this context)
    assert param_dict["x"] is param_dict["y"]

    # Serialize then deserialize `param_dict`.
    buffer = io.BytesIO()
    torch.save(param_dict, buffer)
    param_bytes = buffer.getvalue()

    buffer_load = io.BytesIO(param_bytes)
    deser_param_dict = torch.load(buffer_load)

    # After deserialization with `torch.load`, `deser_param_dict["x"]` and `deser_param_dict["y"]`
    # will generally be *new, distinct Python objects*, even if their content is identical.
    # The original TVM test implicitly checks content equality after deserialization.

    # Make sure the data matches the original data.
    torch.testing.assert_allclose(deser_param_dict["x"], tvm_array_like_obj)
    # Make sure `x` and `y` contain the same data after deserialization (content equality).
    torch.testing.assert_allclose(deser_param_dict["x"], deser_param_dict["y"])


# TODO: The `test_bigendian_rpc_param` function involves TVM-specific RPC
# (remote procedure call) and compilation for specific hardware targets (PowerPC).
# There is no direct functional equivalent in PyTorch/TorchInductor for this kind of
# remote compilation, module loading, and execution via a compiler-specific RPC protocol.
# PyTorch's distributed features focus on distributed tensor computation, not compiler-level
# interaction with remote hardware targets like TVM's RPC.
# This test is therefore not convertible to PyTorch.
# To allow the file to run and be imported, a pytest.skip is used.
def test_bigendian_rpc_param():
    pytest.skip("TVM RPC and remote compilation for specific hardware targets is not convertible to PyTorch.")


if __name__ == "__main__":
    test_save_load()
    test_ndarray_reflection()
    test_bigendian_rpc_param()
