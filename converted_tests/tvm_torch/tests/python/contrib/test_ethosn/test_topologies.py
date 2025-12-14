import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

# Minimal replacement for tei infrastructure
class DummyModule(torch.nn.Module):
    def __init__(self, model_fn, *args, **kwargs):
        super().__init__()
        self.model_fn = model_fn
        self.args = args
        self.kwargs = kwargs

    def forward(self, *inputs_tensors):
        return self.model_fn(*inputs_tensors)

# Map TVM string dtypes to PyTorch dtypes
_torch_dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "int64": torch.int64,
    "qint8": torch.qint8, # For quantized types
    "quint8": torch.quint8,
    "qint32": torch.qint32,
}

def _to_torch_dtype(dtype_str):
    return _torch_dtype_map.get(dtype_str, None)

def _get_abs_tol_from_dtype(dtype_str):
    # Determine appropriate atol based on dtype for verification
    if dtype_str == "float32":
        return 1e-4
    elif dtype_str == "float16":
        return 1e-2
    elif dtype_str == "uint8" or dtype_str == "int8":
        # For quantized integer types, comparison should be exact after dequantization or within 1 LSB.
        # Since we dequantize for verification, float tolerances apply, allowing for potential minor differences.
        return 2.0 # A larger tolerance for quantization paths due to potential dequant-quant errors
    else:
        return 1e-5 # Default for other types

# Helper for composite QNN add
def _qnn_add_composite(lhs_q_tensor, rhs_q_tensor, output_scale, output_zero_point, output_dtype):
    float_lhs = torch.dequantize(lhs_q_tensor)
    float_rhs = torch.dequantize(rhs_q_tensor)
    float_output = torch.add(float_lhs, float_rhs)
    quantized_output = torch.quantize_per_tensor(float_output, scale=output_scale, zero_point=output_zero_point, dtype=output_dtype)
    return quantized_output

# Helper for composite QNN concatenate
def _qnn_concatenate_composite(q_tensors_list, axis, output_scale, output_zero_point, output_dtype):
    float_tensors = [torch.dequantize(t) for t in q_tensors_list]
    float_concat = torch.cat(float_tensors, dim=axis)
    quantized_output = torch.quantize_per_tensor(float_concat, scale=output_scale, zero_point=output_zero_point, dtype=output_dtype)
    return quantized_output

# Helper for composite QNN requantize
def _qnn_requantize_composite(input_q_tensor, output_scale, output_zero_point, output_dtype):
    float_input = torch.dequantize(input_q_tensor)
    quantized_output = torch.quantize_per_tensor(float_input, scale=output_scale, zero_point=output_zero_point, dtype=output_dtype)
    return quantized_output

# Minimal tei (TVM EthosN Infrastructure) replacement
class tei:
    @staticmethod
    def make_module(model_func, params_list):
        return model_func

    @staticmethod
    def make_ethosn_partition(model_func):
        return model_func

    @staticmethod
    def build(model_func, params_dict, npu=False, additional_config_args=None, expected_host_ops=0, npu_partitions=0):
        if npu:
            # For PyTorch, we try to compile the model.
            # `model_func` here is an instance of a `torch.nn.Module`.
            return torch.compile(model_func)
        else:
            return model_func # Return the eager model

    @staticmethod
    def _prepare_inputs(inputs_np):
        # This function prepares torch tensors from numpy inputs.
        # `Model.forward` methods are designed to accept float32, and handle their own quantization.
        torch_inputs = []
        for name, np_array in inputs_np.items():
            # Convert NumPy array to float32 torch.Tensor
            torch_inputs.append(torch.tensor(np_array.astype(np.float32), device='cpu'))
        # If there's only one input, return it directly. Otherwise, return as a tuple.
        return torch_inputs[0] if len(torch_inputs) == 1 else tuple(torch_inputs)

    @staticmethod
    def run(model_or_compiled_model, inputs_np, num_outputs, npu=False):
        prepared_inputs = tei._prepare_inputs(inputs_np)
        with torch.no_grad():
            if isinstance(model_or_compiled_model, (nn.Module, torch._dynamo.OptimizedModule)):
                model_or_compiled_model.eval()
                # If `prepared_inputs` is a tuple, unpack it. Otherwise, pass directly.
                if isinstance(prepared_inputs, tuple):
                    output = model_or_compiled_model(*prepared_inputs)
                else:
                    output = model_or_compiled_model(prepared_inputs)
            else: # Callable function
                if isinstance(prepared_inputs, tuple):
                    output = model_or_compiled_model(*prepared_inputs)
                else:
                    output = model_or_compiled_model(prepared_inputs)
        return output

    @staticmethod
    def build_and_run(
        model_func_factory,
        inputs_np,
        num_outputs,
        params_dict,
        npu=False,
        expected_host_ops=0,
        npu_partitions=0,
        additional_config_args=None,
    ):
        results = []
        prepared_inputs = tei._prepare_inputs(inputs_np)

        eager_model = model_func_factory()
        with torch.no_grad():
            if isinstance(eager_model, (nn.Module, torch._dynamo.OptimizedModule)):
                eager_model.eval()
                if isinstance(prepared_inputs, tuple):
                    output_eager = eager_model(*prepared_inputs)
                else:
                    output_eager = eager_model(prepared_inputs)
            else:
                if isinstance(prepared_inputs, tuple):
                    output_eager = eager_model(*prepared_inputs)
                else:
                    output_eager = eager_model(prepared_inputs)
        results.append(output_eager)

        if npu:
            compiled_model = torch.compile(eager_model)
            with torch.no_grad():
                if isinstance(compiled_model, (nn.Module, torch._dynamo.OptimizedModule)):
                    compiled_model.eval()
                    if isinstance(prepared_inputs, tuple):
                        output_compiled = compiled_model(*prepared_inputs)
                    else:
                        output_compiled = compiled_model(prepared_inputs)
                else:
                    if isinstance(prepared_inputs, tuple):
                        output_compiled = compiled_model(*prepared_inputs)
                    else:
                        output_compiled = compiled_model(prepared_inputs)
            results.append(output_compiled)

        return results

    @staticmethod
    def verify(outputs_list_raw, dtype_str, rtol=1e-5, atol=None):
        if not outputs_list_raw:
            return

        # Dequantize all outputs for comparison as floating-point numbers
        outputs_list = []
        for out in outputs_list_raw:
            if isinstance(out, (tuple, list)):
                # If a tuple/list contains tensors, dequantize them
                outputs_list.append(tuple(torch.dequantize(x) if x.is_quantized else x for x in out))
            else:
                # If a single tensor, dequantize it
                outputs_list.append(torch.dequantize(out) if out.is_quantized else out)

        if atol is None:
            atol = _get_abs_tol_from_dtype(dtype_str)

        # Compare output from compiled vs eager runs
        if len(outputs_list) > 1:
            if isinstance(outputs_list[0], (tuple, list)):
                assert len(outputs_list[0]) == len(outputs_list[1]), "Number of outputs mismatch"
                for i in range(len(outputs_list[0])):
                    torch.testing.assert_close(outputs_list[1][i], outputs_list[0][i], rtol=rtol, atol=atol)
            else:
                torch.testing.assert_close(outputs_list[1], outputs_list[0], rtol=rtol, atol=atol)

# Dummy EthosN specific imports and functions
class Available:
    SW_ONLY = 0 # Placeholder for ethosn_available() return
    HW = 1

def ethosn_available():
    return Available.HW # Assume available for running tests

def requires_ethosn(func):
    return func # Decorator becomes no-op


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_split_add_concat(dtype):
    """Test a model with split, add and contatenate."""

    class Model(nn.Module):
        def __init__(self, input_shape, dtype_str):
            super().__init__()
            self.input_shape = input_shape
            self.dtype_str = dtype_str
            self.output_dtype = _to_torch_dtype(dtype_str)

            # Define constants
            self.split_scale = 0.25
            self.split_zp = 100
            self.add_scale = 0.75
            self.add_zp = 120
            self.axis = 2

        def forward(self, a_float):
            # Input `a_float` is unquantized, so first quantize it
            q_a = torch.quantize_per_tensor(
                a_float,
                scale=self.split_scale,
                zero_point=self.split_zp,
                dtype=self.output_dtype
            )

            # relay.split(a, indices_or_sections=4, axis=axis)
            # 4 means split into 4 equal sections.
            split = torch.split(q_a, split_size_or_sections=self.input_shape[self.axis] // 4, dim=self.axis)

            # relay.qnn.op.add(...)
            q_b = _qnn_add_composite(
                split[0],
                split[1],
                output_scale=self.add_scale,
                output_zero_point=self.add_zp,
                output_dtype=self.output_dtype,
            )
            
            # relay.qnn.op.concatenate(...)
            conc = _qnn_concatenate_composite(
                [q_b, split[2], split[3]],
                axis=self.axis,
                output_scale=self.add_scale,
                output_zero_point=self.add_zp,
                output_dtype=self.output_dtype,
            )
            return conc

    np.random.seed(0)
    input_shape = (1, 16, 16, 4)
    inputs = {
        "a": np.random.randint(
            np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=input_shape, dtype=dtype
        ),
    }

    outputs = []
    # `get_model_wrapper` is a factory function for tei.build_and_run
    def get_model_wrapper():
        return Model(input_shape, dtype)

    results_eager_and_compiled = tei.build_and_run(
        get_model_wrapper,
        inputs,
        1,
        {},
        npu=True, # Always run both paths for comparison
        expected_host_ops=0, # Ignored in PyTorch
        npu_partitions=1,    # Ignored in PyTorch
        additional_config_args={"inline_non_compute_intensive_partitions": False}, # Ignored
    )
    outputs.extend(results_eager_and_compiled)

    if len(outputs) > 0: # Ensure outputs were generated before verifying
        tei.verify(outputs, dtype, rtol=1e-5)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_multiple_command_streams(dtype):
    """Check that multiple Ethos-N partitions are correctly handled.
    """

    class Model(nn.Module):
        def __init__(self, dtype_str):
            super().__init__()
            self.dtype_str = dtype_str
            self.output_dtype = _to_torch_dtype(dtype_str)

        def forward(self, x_float):
            # Input `x_float` (float32 from tei._prepare_inputs).
            # Quantize it with scale=1.0, zero_point=0 to `self.output_dtype` to simulate `relay.var` of `dtype`.
            q_x = torch.quantize_per_tensor(
                x_float,
                scale=1.0,
                zero_point=0,
                dtype=self.output_dtype
            )
            x_dequant = torch.dequantize(q_x) # Dequantize for float ops

            # Layout is "NHWC" in TVM. PyTorch uses NCHW. So permute input.
            x_dequant_nchw = x_dequant.permute(0, 3, 1, 2) # NHWC to NCHW

            # relay.nn.max_pool2d -> F.max_pool2d
            out_float = F.max_pool2d(x_dequant_nchw, kernel_size=(2, 2), stride=(2, 2), padding=0)
            
            # relay.op.abs -> torch.abs
            out_float = torch.abs(out_float)
            
            # relay.nn.max_pool2d -> F.max_pool2d
            out_float = F.max_pool2d(out_float, kernel_size=(2, 2), stride=(2, 2), padding=0)
            
            # Permute output back to NHWC for consistency with TVM output layout
            out_float = out_float.permute(0, 2, 3, 1) # NCHW to NHWC

            # Requantize final float output
            q_out = torch.quantize_per_tensor(out_float, scale=1.0, zero_point=0, dtype=self.output_dtype)
            return q_out

    np.random.seed(0)
    input_shape = (1, 4, 4, 4)
    inputs_np = {
        "x": np.random.randint(
            np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=input_shape, dtype=dtype
        )
    }
    
    def get_model_wrapper():
        return Model(dtype)

    # In TVM, the model has 2 NPU partitions and 1 host op (abs).
    # We will run both eager and compiled and compare outputs.
    # The `npu=True` path simulates the ethosn backend.
    
    # Mock inference is only supported when the whole graph is offloaded to the NPU
    if ethosn_available() == Available.SW_ONLY:
        # In PyTorch, we can't directly replicate the "build only" test.
        # We will proceed with build_and_run and verify outputs.
        # TODO: Add specific TorchInductor graph break verification if applicable.
        pass
    
    outputs = tei.build_and_run(get_model_wrapper, inputs_np, 1, {}, npu=True)
    
    if outputs:
        tei.verify(outputs, dtype, rtol=1e-5)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_output_order(dtype):
    """Test the output order."""

    class Model(nn.Module):
        def __init__(self, input_shape, dtype_str):
            super().__init__()
            self.input_shape = input_shape
            self.dtype_str = dtype_str
            self.output_dtype = _to_torch_dtype(dtype_str)

            self.min_value = np.iinfo(dtype_str).min
            self.max_value = np.iinfo(dtype_str).max

        def forward(self, a_float):
            # Input `a_float` is unquantized. Quantize first to simulate `relay.var` of `dtype`.
            # Assume an identity quantization for `a` with scale=1, zp=0, and `dtype`.
            q_a = torch.quantize_per_tensor(
                a_float,
                scale=1.0, # Identity scale
                zero_point=0, # Identity zero point
                dtype=self.output_dtype
            )
            
            # Dequantize for float operations (as PyTorch clamp operates on float)
            a_dequant = torch.dequantize(q_a)

            # relay.op.clip -> torch.clamp (float ops)
            op_z = torch.clamp(a_dequant, self.min_value, self.max_value)
            op_b = torch.clamp(op_z, self.min_value, self.min_value + 15)
            op_c = torch.clamp(op_z, self.min_value + 16, self.min_value + 31)
            op_d = torch.clamp(op_z, self.min_value + 32, self.min_value + 47)
            op_e = torch.clamp(op_z, self.min_value + 48, self.min_value + 63)
            op_f = torch.clamp(op_z, self.min_value + 64, self.min_value + 79)
            op_g = torch.clamp(op_z, self.min_value + 80, self.min_value + 95)
            op_h = torch.clamp(op_z, self.min_value + 96, self.min_value + 111)
            op_i = torch.clamp(op_z, self.min_value + 112, self.max_value)

            # Re-quantize each output to the same identity quantization parameters as input `q_a`
            q_b = torch.quantize_per_tensor(op_b, scale=1.0, zero_point=0, dtype=self.output_dtype)
            q_c = torch.quantize_per_tensor(op_c, scale=1.0, zero_point=0, dtype=self.output_dtype)
            q_d = torch.quantize_per_tensor(op_d, scale=1.0, zero_point=0, dtype=self.output_dtype)
            q_e = torch.quantize_per_tensor(op_e, scale=1.0, zero_point=0, dtype=self.output_dtype)
            q_f = torch.quantize_per_tensor(op_f, scale=1.0, zero_point=0, dtype=self.output_dtype)
            q_g = torch.quantize_per_tensor(op_g, scale=1.0, zero_point=0, dtype=self.output_dtype)
            q_h = torch.quantize_per_tensor(op_h, scale=1.0, zero_point=0, dtype=self.output_dtype)
            q_i = torch.quantize_per_tensor(op_i, scale=1.0, zero_point=0, dtype=self.output_dtype)

            # Return as a tuple of quantized tensors, matching TVM's Relay.Tuple output order
            return (q_d, q_c, q_e, q_f, q_i, q_b, q_h, q_g)

    np.random.seed(0)
    input_shape = (1, 16, 16, 4)
    inputs = {
        "a": np.random.randint(
            np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=input_shape, dtype=dtype
        ),
    }

    outputs = []
    def get_model_wrapper():
        return Model(input_shape, dtype)

    results_eager_and_compiled = tei.build_and_run(
        get_model_wrapper,
        inputs,
        8, # num_outputs in the tuple
        {},
        npu=True,
        additional_config_args={"inline_non_compute_intensive_partitions": False},
    )
    outputs.extend(results_eager_and_compiled)

    if len(outputs) > 0:
        # For multiple outputs, `tei.verify` compares each tuple element
        tei.verify(outputs, dtype, rtol=1e-5)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_output_order_different_sizes(dtype):
    """
    Test the output order when there are multiple outputs of different sizes.
    """

    class Model(nn.Module):
        def __init__(self, input_shape, dtype_str, dtype_min, dtype_max):
            super().__init__()
            self.input_shape = input_shape
            self.dtype_str = dtype_str
            self.output_dtype = _to_torch_dtype(dtype_str)
            self.dtype_min = dtype_min
            self.dtype_max = dtype_max
            
            # Constants for requantize
            self.input_scale_requant = 0.0784314
            self.input_zero_point_requant = self.dtype_min + 128
            self.output_scale_requant = 0.0784314
            self.output_zero_point_requant = self.dtype_min + 128

        def forward(self, var_float):
            # Input `var_float` is unquantized. Quantize first to simulate `relay.var` of `dtype`.
            # Assume an identity quantization for `var` with scale=1, zp=0, and `dtype`.
            q_var = torch.quantize_per_tensor(
                var_float,
                scale=1.0,
                zero_point=0,
                dtype=self.output_dtype
            )

            # relay.op.clip -> torch.clamp
            # Dequantize for float clamp, then re-quantize with identity params
            clip_float = torch.clamp(torch.dequantize(q_var), self.dtype_min, self.dtype_max)
            q_clip = torch.quantize_per_tensor(clip_float, scale=1.0, zero_point=0, dtype=self.output_dtype)

            # relay.nn.max_pool2d -> F.max_pool2d
            # PyTorch max_pool2d expects NCHW, so permute q_clip
            q_clip_nchw = q_clip.permute(0, 3, 1, 2) # NHWC to NCHW
            max_pool_float = F.max_pool2d(
                torch.dequantize(q_clip_nchw), # Dequantize for float op
                kernel_size=(2, 2),
                stride=(2, 2),
                ceil_mode=True,
                padding=0
            )
            # Requantize max_pool output to the same quantization as q_clip
            q_max_pool = torch.quantize_per_tensor(max_pool_float, scale=1.0, zero_point=0, dtype=self.output_dtype)
            q_max_pool_nhwc = q_max_pool.permute(0, 2, 3, 1) # NCHW to NHWC

            # relay.op.cast(clip, "int32") followed by relay.mean
            # Dequantize q_clip, cast to int32, then to float32 for mean op
            mean_int32_float = torch.dequantize(q_clip).to(torch.int32).to(torch.float32)
            mean_float = torch.mean(mean_int32_float, dim=[1, 2], keepdim=True)

            # relay.qnn.op.requantize
            # Need to create a quantized tensor for `mean_float` with the specified input_scale/zp
            q_mean_input = torch.quantize_per_tensor(
                mean_float, 
                scale=self.input_scale_requant,
                zero_point=self.input_zero_point_requant,
                dtype=_to_torch_dtype("int32") # Intermediate dtype for q_mean_input
            )

            q_mean_output = _qnn_requantize_composite(
                q_mean_input,
                output_scale=self.output_scale_requant,
                output_zero_point=self.output_zero_point_requant,
                output_dtype=self.output_dtype,
            )

            # Return as a tuple of quantized tensors, matching TVM's Relay.Tuple output order
            return (q_mean_output, q_max_pool_nhwc, q_clip)

    np.random.seed(0)
    input_name = "a"
    input_shape = (1, 8, 8, 4)
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    inputs = {
        input_name: np.random.randint(dtype_min, dtype_max + 1, size=input_shape, dtype=dtype)
    }

    outputs = []
    def get_model_wrapper():
        return Model(input_shape, dtype, dtype_min, dtype_max)

    results_eager_and_compiled = tei.build_and_run(
        get_model_wrapper, inputs, 3, {}, npu=True, expected_host_ops=0, npu_partitions=1
    )
    outputs.extend(results_eager_and_compiled)

    if len(outputs) > 0:
        tei.verify(outputs, dtype, rtol=1e-5)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape,splits,axis",
    [
        ((1, 16, 16, 32), (2, 7, 10), 2),
    ],
)
def test_split_with_asym_concats(dtype, shape, splits, axis):
    """Test a model with split and contatenates."""
    np.random.seed(0)

    class Model(nn.Module):
        def __init__(self, shape, dtype_str, splits, axis):
            super().__init__()
            self.shape = shape
            self.dtype_str = dtype_str
            self.output_dtype = _to_torch_dtype(dtype_str)
            self.splits = splits
            self.axis = axis

            self.zeroi = 1 # integer zero_point for input/output scales
            self.zerof = 0.5 # float scale for input/output scales

        def forward(self, a_float):
            # Input `a_float` is unquantized. Quantize `a` with zerof/zeroi.
            q_a = torch.quantize_per_tensor(
                a_float,
                scale=self.zerof,
                zero_point=self.zeroi,
                dtype=self.output_dtype
            )

            # relay.op.split(a, indices_or_sections=splits, axis=axis)
            # `splits` is a tuple of indices where to split.
            current_dim_size = self.shape[self.axis]
            split_points = sorted(self.splits)
            split_sizes_for_torch = []
            last_idx = 0
            for idx in split_points:
                split_sizes_for_torch.append(idx - last_idx)
                last_idx = idx
            split_sizes_for_torch.append(current_dim_size - last_idx)

            split_tensors = torch.split(q_a, split_sizes_for_torch, dim=self.axis)

            # relay.qnn.op.concatenate([split[0], split[1]], ...)
            con1 = _qnn_concatenate_composite(
                [split_tensors[0], split_tensors[1]],
                axis=self.axis,
                output_scale=self.zerof,
                output_zero_point=self.zeroi,
                output_dtype=self.output_dtype,
            )

            # relay.qnn.op.concatenate([split[2], split[3]], ...)
            con2 = _qnn_concatenate_composite(
                [split_tensors[2], split_tensors[3]],
                axis=self.axis,
                output_scale=self.zerof,
                output_zero_point=self.zeroi,
                output_dtype=self.output_dtype,
            )
            return (con2, con1)

    outputs = []
    inputs = {
        "a": np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
    }

    def get_model_wrapper():
        return Model(shape, dtype, splits, axis)

    if ethosn_available() == Available.SW_ONLY:
        pass # No direct mapping for TVM's build-only check, proceed to build_and_run
    
    # Run eager and compiled
    results_eager_and_compiled = tei.build_and_run(
        get_model_wrapper,
        inputs,
        2, # num_outputs for the tuple
        {},
        npu=True,
        expected_host_ops=0, # Ignored in PyTorch
        npu_partitions=1,    # Ignored in PyTorch
        additional_config_args={"inline_non_compute_intensive_partitions": False}, # Ignored
    )
    outputs.extend(results_eager_and_compiled)

    if outputs:
        tei.verify(outputs, dtype, rtol=1e-5)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_output_tuple_propagation(dtype):
    """This tests the case where the output tuple must be inferred
    as having dummy tensor information."""

    class Model(nn.Module):
        def __init__(self, dtype_str):
            super().__init__()
            self.dtype_str = dtype_str
            self.output_dtype = _to_torch_dtype(dtype_str)

        def forward(self, a_float):
            # Input `a_float` is unquantized. Quantize first to simulate `relay.var` of `dtype`.
            # Assume an identity quantization for `a` with scale=1, zp=0, and `dtype`.
            q_a = torch.quantize_per_tensor(
                a_float,
                scale=1.0,
                zero_point=0,
                dtype=self.output_dtype
            )
            
            # relay.op.split(a, indices_or_sections=4, axis=2)
            # `indices_or_sections=4` implies 4 equal sections
            split = torch.split(q_a, split_size_or_sections=q_a.shape[2] // 4, dim=2)
            
            # relay.Tuple(...) -> Python tuple
            return (split[0], split[1], split[2], split[3])

    np.random.seed(0)
    input_shape = (1, 4, 4, 16)
    inputs = {
        "a": np.random.randint(
            np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=input_shape, dtype=dtype
        )
    }
    outputs = []
    def get_model_wrapper():
        return Model(dtype)

    results_eager_and_compiled = tei.build_and_run(
        get_model_wrapper,
        inputs,
        4, # num_outputs
        {},
        npu=True,
        additional_config_args={"inline_non_compute_intensive_partitions": False},
    )
    outputs.extend(results_eager_and_compiled)

    if len(outputs) > 0:
        # For multiple outputs, `tei.verify` compares each tuple element
        tei.verify(outputs, dtype, rtol=1e-5)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_input_tuples(dtype):
    """Test a model with a tuple as input."""

    class Model(nn.Module):
        def __init__(self, shapes, dtype_str, axis):
            super().__init__()
            self.shapes = shapes
            self.dtype_str = dtype_str
            self.output_dtype = _to_torch_dtype(dtype_str)
            self.axis = axis

            # Constants for input/output scales/zps, as used in TVM model
            self.zeroi = 1
            self.zerof = 0.5

        def forward(self, in0_float, in1_float): # Accept float inputs
            # Quantize inputs using self.zerof/self.zeroi as per TVM model
            q_in0 = torch.quantize_per_tensor(
                in0_float,
                scale=self.zerof,
                zero_point=self.zeroi,
                dtype=self.output_dtype
            )
            q_in1 = torch.quantize_per_tensor(
                in1_float,
                scale=self.zerof,
                zero_point=self.zeroi,
                dtype=self.output_dtype
            )

            # relay.qnn.op.concatenate(tup, ...)
            con = _qnn_concatenate_composite(
                [q_in0, q_in1], # Pass quantized tensors
                axis=self.axis,
                output_scale=self.zerof,
                output_zero_point=self.zeroi,
                output_dtype=self.output_dtype,
            )
            return con

    np.random.seed(0)
    input_shapes = [(1, 4), (1, 6)]
    inputs_np = {
        "in0": np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=input_shapes[0], dtype=dtype),
        "in1": np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=input_shapes[1], dtype=dtype),
    }

    outputs = []
    
    # Instantiate the model
    model = Model(input_shapes, dtype, 1)

    # NPU=False path (eager PyTorch)
    mod_eager = tei.make_module(model, [])
    lib_eager = tei.build(mod_eager, {}, npu=False, additional_config_args={"inline_non_compute_intensive_partitions": False})
    outputs.append(tei.run(lib_eager, inputs_np, 1))

    # NPU=True path (compiled PyTorch)
    mod_npu = tei.make_ethosn_partition(model)
    lib_npu = tei.build(mod_npu, {}, npu=True, additional_config_args={"inline_non_compute_intensive_partitions": False})
    outputs.append(tei.run(lib_npu, inputs_np, 1, npu=True))

    tei.verify(outputs, dtype, rtol=1e-5)


@requires_ethosn
def test_inline_non_compute_intensive_operations():
    """Tests the case when a subgraph is unpartitioned."""
    np.random.seed(0)
    dtype = "int8"
    shape = (1, 2, 2, 4)

    class Model(nn.Module):
        def __init__(self, shape, dtype_str):
            super().__init__()
            self.shape = shape
            self.output_dtype = _to_torch_dtype(dtype_str)

        def forward(self, x_float):
            # Input `x_float` is float32 from tei._prepare_inputs.
            # Convert to target dtype (e.g., int8) before reshape.
            # This simulates TVM's `relay.reshape` on an `int8` variable.
            x = x_float.to(self.output_dtype)
            
            # relay.reshape(inp, newshape=(1, 1, 4, 4)) -> torch.reshape
            reshape = torch.reshape(x, newshape=(1, 1, 4, 4))
            return reshape

    inputs = {
        "x": np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
    }
    outputs = []

    def get_model_wrapper():
        return Model(shape, dtype)

    # In TVM, expected_host_ops=1, npu_partitions=0 means the whole graph runs on CPU.
    # In PyTorch, this means that `torch.compile` might not offer speedup for this simple op,
    # or the graph is trivially compiled.
    results_eager_and_compiled = tei.build_and_run(get_model_wrapper, inputs, 1, {}, npu=True, expected_host_ops=1, npu_partitions=0)
    outputs.extend(results_eager_and_compiled)

    if len(outputs) > 0:
        # For reshape on integer types, we expect exact comparison.
        tei.verify(outputs, dtype, rtol=1e-5, atol=0)
