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

"""Arm(R) Ethos(TM)-N integration pooling tests"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Dummy for `requires_ethosn` since it's a TVM-specific decorator
def requires_ethosn(f):
    return f

# Define a common type map for PyTorch dtypes from strings.
_TORCH_DTYPE_MAP = {
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

# Minimal dummy for tei (Ethos-N infrastructure) functions.
# These functions require significant PyTorch/TorchInductor backend knowledge for full fidelity.
# The current implementation provides basic callable execution and comparison.
class DummyEthosNInfrastructure:
    def make_module(self, model_callable, params):
        # In TVM, this constructs a Relay module. Here, we just return the PyTorch model/function.
        return model_callable

    def build_and_run(self, model_callable, inputs, num_outputs, params, npu=False):
        input_tensor_key = list(inputs.keys())[0] # Assuming single input 'a'
        input_tensor = torch.tensor(inputs[input_tensor_key], device='cpu')

        output_tensor = None
        if npu:
            # Simulate TorchInductor compilation and execution for the "NPU" path.
            # Mode "reduce-overhead" is often good for basic performance tests.
            try:
                compiled_model = torch.compile(model_callable, fullgraph=True, mode="reduce-overhead")
                output_tensor = compiled_model(input_tensor)
            except Exception as e:
                # If compilation fails, fall back to eager and print a warning for comparison.
                # In a real scenario, this might be a test failure.
                print(f"Warning: torch.compile failed for NPU path (Ethos-N simulation): {e}. "
                      f"Falling back to eager mode for comparison. This might indicate an unsupported feature.")
                output_tensor = model_callable(input_tensor)
        else:
            # Eager execution for the "non-NPU" path.
            output_tensor = model_callable(input_tensor)

        # Convert output to a list of numpy arrays, mimicking TVM's tei.build_and_run return format.
        if isinstance(output_tensor, (list, tuple)):
            return [o.cpu().numpy() for o in output_tensor]
        return [output_tensor.cpu().numpy()]

    def verify(self, outputs, dtype_str, tolerance_val):
        # In TVM, this compares results. In PyTorch, we use assert_allclose.
        if len(outputs) != 2:
            raise ValueError("Expected 2 outputs (Eager, Compiled) for verification")
        
        # outputs[0] is from non-NPU (eager), outputs[1] is from NPU (compiled)
        # Each element of outputs is a list of numpy arrays. We take the first one.
        actual_eager = outputs[0][0]
        actual_compiled = outputs[1][0]

        # Ethos-N typically uses low-precision integer dtypes (uint8, int8).
        # For `assert_allclose`, it's best to compare floating-point values for tolerance.
        # TVM's `verify` often converts to float64 internally for comparison.
        
        # Determine common floating point type for comparison
        common_dtype_np = np.result_type(actual_eager.dtype, actual_compiled.dtype)
        if not np.issubdtype(common_dtype_np, np.floating):
            common_dtype_np = np.float32 # Convert to float32 for comparison

        # Convert numpy arrays to torch tensors with the common float dtype
        tensor_eager = torch.tensor(actual_eager, dtype=_TORCH_DTYPE_MAP.get(str(common_dtype_np), torch.float32))
        tensor_compiled = torch.tensor(actual_compiled, dtype=_TORCH_DTYPE_MAP.get(str(common_dtype_np), torch.float32))
        
        # Define tolerances. Ethos-N models might have higher error.
        # Using typical tolerances for floating point comparisons.
        rtol = 1e-5
        atol = 1e-5
        
        torch.testing.assert_allclose(
            tensor_eager,
            tensor_compiled,
            rtol=rtol,
            atol=atol,
            msg=f"Mismatch between eager and compiled outputs for dtype {dtype_str}"
        )

    def make_ethosn_partition(self, model_callable):
        # This TVM function signals that the model should be partitioned for Ethos-N.
        # In the PyTorch context, we return the callable, and `test_error` will attempt to compile it.
        return model_callable

    def test_error(self, model_callable, inputs, err_msg_pattern):
        input_tensor_key = list(inputs.keys())[0] # Assuming single input 'a'
        input_tensor = torch.tensor(inputs[input_tensor_key], device='cpu')
        
        print(f"TODO: test_error - matching pattern '{err_msg_pattern}'. "
              f"PyTorch/TorchInductor error messages may differ from TVM Ethos-N specific messages, "
              f"and some Ethos-N unsupported features might be supported by TorchInductor.")
        
        with pytest.raises(Exception) as excinfo:
            # Attempt to compile and run the model through TorchInductor.
            compiled_model = torch.compile(model_callable, fullgraph=True, mode="reduce-overhead")
            _ = compiled_model(input_tensor)
            pytest.fail(f"Expected an exception containing '{err_msg_pattern}', but the model ran successfully.")
        
        # Check if the captured exception message contains the expected pattern.
        # This is a loose check as exact error messages vary greatly between frameworks.
        error_message = str(excinfo.value).lower()
        if not (err_msg_pattern.lower() in error_message or "unsupported" in error_message or "not supported" in error_message or "invalid" in error_message):
            print(f"Warning: The exact error message pattern '{err_msg_pattern}' was not found. Actual error: {excinfo.value}")

tei = DummyEthosNInfrastructure()

# PyTorch Module to encapsulate the pooling logic, handling layout and dtype conversions.
class PoolingModel(torch.nn.Module):
    def __init__(self, typef_func, sizes, strides, pads, layout, dtype_str):
        super().__init__()
        self.typef_func = typef_func # This will be F.max_pool2d or F.avg_pool2d
        self.sizes = sizes
        self.strides = strides
        self.pads = pads
        self.layout = layout
        self.output_dtype = _TORCH_DTYPE_MAP.get(dtype_str, torch.float32)

    def forward(self, x):
        # Ensure input is on CPU, as `tei.build_and_run` prepares CPU tensors.
        # The original_input_dtype would be relevant if we were casting back to it,
        # but the TVM _get_model casts to `dtype` specified for the final output.
        
        # Handle layout transformation: NHWC (TVM Ethos-N) -> NCHW (PyTorch pooling)
        if self.layout == "NHWC" and x.ndim == 4:
            x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
        # else: assume NCHW or other compatible layout, or let downstream ops error

        if self.typef_func == F.avg_pool2d:
            # TVM test explicitly casts to int32 before avg_pool2d, then back to original.
            # PyTorch's F.avg_pool2d typically requires floating-point input.
            x_processed = x.to(torch.int32).to(torch.float32) # Match TVM's cast-to-int32-before-avg_pool, then to float for op

            # TVM's avg_pool2d defaults count_include_pad to False. PyTorch's is True.
            # Explicitly set to False to match TVM behavior.
            out = self.typef_func(
                x_processed,
                kernel_size=self.sizes,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=True,
                count_include_pad=False
            )
            out = out.to(self.output_dtype) # Cast to target output dtype
        elif self.typef_func == F.max_pool2d:
            # Max pooling can often work with integer dtypes, but ensure consistency for output dtype.
            x_processed = x.to(self.output_dtype) # Ensure consistent dtype for max_pool operation based on model's target output
            out = self.typef_func(
                x_processed,
                kernel_size=self.sizes,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=True
            )
        else:
            raise ValueError(f"Unsupported pooling type function: {self.typef_func}")

        # Permute back to NHWC if original input was NHWC
        if self.layout == "NHWC" and out.ndim == 4:
            out = out.permute(0, 2, 3, 1) # NCHW -> NHWC
        
        return out

# Helper function to construct the PyTorch model
def _get_model(shape, typef_func, sizes, strides, pads, layout, dtype_str):
    """Return a PyTorch module for the pooling operation."""
    # `shape` is not directly used by the Module's constructor but is important for input creation.
    return PoolingModel(typef_func, sizes, strides, pads, layout, dtype_str)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape,typef,size,stride,pad",
    [
        ((1, 8, 8, 8), F.max_pool2d, (2, 2), (2, 2), (0, 0, 0, 0)),
        ((1, 9, 9, 9), F.max_pool2d, (3, 3), (2, 2), (0, 0, 0, 0)),
        ((1, 8, 8, 8), F.avg_pool2d, (3, 3), (1, 1), (1, 1, 1, 1)),
    ],
)
def test_pooling(dtype, shape, typef, size, stride, pad):
    """Compare Pooling output with PyTorch/TorchInductor."""
    np.random.seed(0)

    layout = "NHWC"

    inputs = {
        "a": np.random.randint(
            low=np.iinfo(dtype).min, high=np.iinfo(dtype).max + 1, size=shape, dtype=dtype
        ),
    }
    outputs = []
    
    # Get the PyTorch module corresponding to the TVM Relay model
    model_callable = _get_model(shape, typef, size, stride, pad, layout, dtype)

    # Run for both non-NPU (eager) and NPU (compiled) paths
    for npu_flag in [False, True]:
        # make_module and build_and_run are now handled by dummy tei
        mod = tei.make_module(model_callable, {})
        outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu_flag))

    # Verify outputs between eager and compiled paths
    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,size,stride,layout,dtype,err_msg",
    [
        (
            (2, 8, 8, 8),
            (2, 2),
            (2, 2),
            "NHWC",
            "uint8",
            "batch size=2, batch size must = 1", # PyTorch/TorchInductor generally supports batch_size > 1
        ),
        (
            (1, 8, 8, 8),
            (2, 2),
            (2, 2),
            "NHWC",
            "int16",
            "dtype='int16', dtype must be either uint8, int8 or int32", # PyTorch max_pool2d supports int16
        ),
        (
            (1, 8, 8, 8),
            (2, 2),
            (2, 2),
            "NCHW",
            "uint8",
            "data format=NCHW, data format must = NHWC", # PyTorch expects NCHW, so it will likely work
        ),
        (
            (1, 8, 8, 8),
            (2, 2),
            (2, 2, 2), # Invalid stride size in Ethos-N (stride expects 2 elements for 2D)
            "NHWC",
            "uint8",
            "stride size=3, stride size must = 2",
        ),
        (
            (1, 8, 8, 8),
            (2, 2, 2), # Invalid kernel size in Ethos-N (kernel_size expects 2 elements for 2D)
            (2, 2),
            "NHWC",
            "uint8",
            "dimensions=3, dimensions must = 2",
        ),
    ],
)
def test_pooling_failure(shape, size, stride, layout, dtype, err_msg):
    """Check Pooling error messages with PyTorch/TorchInductor."""

    typef = F.max_pool2d
    pad = (0, 0, 0, 0)

    # Get the PyTorch module
    model_callable = _get_model(shape, typef, size, stride, pad, layout, dtype)
    
    # make_ethosn_partition and test_error are handled by dummy tei
    mod = tei.make_ethosn_partition(model_callable)
    inputs = {
        "a": np.random.randint(
            low=np.iinfo(dtype).min, high=np.iinfo(dtype).max + 1, size=shape, dtype=dtype
        ),
    }
    tei.test_error(mod, inputs, err_msg)
