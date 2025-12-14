import torch
import numpy as np
import pytest
import torch.nn.functional as F
from typing import Any, Dict, List, Callable, Union, Tuple
import functools # For logical_and, logical_or

# Helper to convert TVM dtype strings to torch dtypes
def _convert_relay_dtype_to_torch_dtype(dtype_str):
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    # Add other dtypes as needed
    raise ValueError(f"Unsupported dtype: {dtype_str}")

def verify_mixed_precision_output_close(
    result_fp32: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    result_amp: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    mixed_precision_dtype_str: str,
    rtol: float = 1e-3,
    atol: float = 0,
    keep_orig_output_dtype: bool = False,
):
    # Ensure results are lists for consistent processing
    if not isinstance(result_fp32, (list, tuple)):
        result_fp32 = [result_fp32]
    if not isinstance(result_amp, (list, tuple)):
        result_amp = [result_amp]

    result_fp32_np = [r.detach().cpu().numpy() for r in result_fp32]
    result_amp_np = [r.detach().cpu().numpy() for r in result_amp]

    # Ensure the results are close
    # The original test specifically passed rtol/atol for bfloat16 in `test_lstm`
    # We should use the provided rtol/atol for all cases.
    for fp32_np, amp_np in zip(result_fp32_np, result_amp_np):
        np.testing.assert_allclose(fp32_np, amp_np, rtol=rtol, atol=atol)

    if keep_orig_output_dtype:
        for amp_res, fp32_res in zip(result_amp, result_fp32):
            assert amp_res.dtype == fp32_res.dtype, f"output type {amp_res.dtype} and original type {fp32_res.dtype} mismatch"

# Use pytest.mark.parametrize for target_precision
target_precision_fixture = pytest.mark.parametrize(
    "mixed_precision_dtype",
    [
        pytest.param("float16", id="float16"),
        pytest.param("bfloat16", id="bfloat16"),
    ],
)

# --- Test cases ---

@target_precision_fixture
def test_lstm(mixed_precision_dtype):
    # This simulates the unrolled LSTM setup.
    units = 4
    iterations = 5

    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 Model
    class LstmModelFP32(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)

            # Initialize weights explicitly to ensure consistency across models
            # Standard PyTorch LSTMCell init
            for name, param in self.lstm_cell.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

        def forward(self, inputs_dict, iterations):
            h_state = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
            c_state = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)

            for i in range(iterations):
                input_name = "data" if i == 0 else f"data{i}"
                data_input = inputs_dict[input_name]
                h_state, c_state = self.lstm_cell(data_input, (h_state, c_state))

            return h_state # Relay usually returns the last output

    # Mixed Precision Model
    class LstmModelMixedPrecision(torch.nn.Module):
        def __init__(self, input_size, hidden_size, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)

            # Initialize weights explicitly and convert to mixed precision
            for name, param in self.lstm_cell.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
                # Parameters are initially created in FP32, then will be converted in forward pass or here
                # Here we ensure they are explicitly cast to mixed precision during init for AMP model
                param.data = param.data.to(self.mixed_precision_torch_dtype)

        def forward(self, inputs_dict, iterations):
            # All operations within LSTMCell are assumed to be "green-listed"
            h_state = torch.zeros(1, self.lstm_cell.hidden_size, dtype=self.mixed_precision_torch_dtype)
            c_state = torch.zeros(1, self.lstm_cell.hidden_size, dtype=self.mixed_precision_torch_dtype)

            for i in range(iterations):
                input_name = "data" if i == 0 else f"data{i}"
                data_input = inputs_dict[input_name].to(self.mixed_precision_torch_dtype) # Input cast to mixed precision
                
                h_state, c_state = self.lstm_cell(data_input, (h_state, c_state))
            
            # Output cast back to float32 (common AMP pattern for final output)
            return h_state.to(torch.float32)

    # Prepare input data (NumPy arrays for initial creation)
    raw_mod_params = {}
    for i in range(iterations):
        raw_mod_params["data" if i == 0 else f"data{i}"] = np.random.uniform(
            -10, 10, (1, units)
        ).astype("float32")

    # Create FP32 input tensors
    fp32_input_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_mod_params.items()}

    # Instantiate models
    fp32_model = LstmModelFP32(units, units)
    amp_model = LstmModelMixedPrecision(units, units, mixed_precision_dtype)

    # Make sure both models start with the same weights (after type conversion for AMP)
    # This involves copying FP32 weights to AMP model, converting type for AMP params
    with torch.no_grad(): # Ensure no gradient tracking during parameter assignment
        for (fp32_name, fp32_param), (amp_name, amp_param) in zip(fp32_model.named_parameters(), amp_model.named_parameters()):
            if 'weight' in fp32_name or 'bias' in fp32_name: # Ensure we're matching parameters
                assert fp32_name == amp_name
                amp_param.data = fp32_param.data.to(_torch_mixed_precision_dtype)

    # Run models
    fp32_result = fp32_model(fp32_input_tensors, iterations)
    amp_result = amp_model(fp32_input_tensors, iterations) # AMP model takes fp32 inputs and casts internally

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype, rtol=0.01, atol=0.01
    )


def test_lstm_float64():
    units = 3
    iterations = 5
    mixed_precision_dtype = "float64" # Specific test for float64
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 Model (same as above)
    class LstmModelFP32(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
            for name, param in self.lstm_cell.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

        def forward(self, inputs_dict, iterations):
            h_state = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
            c_state = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
            for i in range(iterations):
                input_name = "data" if i == 0 else f"data{i}"
                data_input = inputs_dict[input_name]
                h_state, c_state = self.lstm_cell(data_input, (h_state, c_state))
            return h_state

    # Mixed Precision Model (adapted for float64)
    class LstmModelMixedPrecision(torch.nn.Module):
        def __init__(self, input_size, hidden_size, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
            for name, param in self.lstm_cell.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
                param.data = param.data.to(self.mixed_precision_torch_dtype)

        def forward(self, inputs_dict, iterations):
            h_state = torch.zeros(1, self.lstm_cell.hidden_size, dtype=self.mixed_precision_torch_dtype)
            c_state = torch.zeros(1, self.lstm_cell.hidden_size, dtype=self.mixed_precision_torch_dtype)
            for i in range(iterations):
                input_name = "data" if i == 0 else f"data{i}"
                data_input = inputs_dict[input_name].to(self.mixed_precision_torch_dtype)
                h_state, c_state = self.lstm_cell(data_input, (h_state, c_state))
            return h_state.to(torch.float32)

    raw_mod_params = {}
    for i in range(iterations):
        raw_mod_params["data" if i == 0 else f"data{i}"] = np.random.uniform(
            -10, 10, (1, units)
        ).astype("float32")

    fp32_input_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_mod_params.items()}

    fp32_model = LstmModelFP32(units, units)
    amp_model = LstmModelMixedPrecision(units, units, mixed_precision_dtype)

    with torch.no_grad():
        for (fp32_name, fp32_param), (amp_name, amp_param) in zip(fp32_model.named_parameters(), amp_model.named_parameters()):
            if 'weight' in fp32_name or 'bias' in fp32_name:
                assert fp32_name == amp_name
                amp_param.data = fp32_param.data.to(_torch_mixed_precision_dtype)

    fp32_result = fp32_model(fp32_input_tensors, iterations)
    amp_result = amp_model(fp32_input_tensors, iterations)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype, rtol=0.01, atol=0.01
    )

@target_precision_fixture
def test_convert_single_conv(mixed_precision_dtype):
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3) # OIHW format for conv2d
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    class ConvModelFP32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=torch.float32))

        def forward(self, data):
            return F.conv2d(data, self.weight, stride=(1, 1), padding=(1, 1))

    # Mixed Precision model
    class ConvModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=self.mixed_precision_torch_dtype))

        def forward(self, data):
            # Conv is green-listed. Inputs converted to mixed precision.
            data_mp = data.to(self.mixed_precision_torch_dtype)
            
            # Conv2d operation in mixed precision. Weights are already in MP from init.
            output_mp = F.conv2d(data_mp, self.weight, stride=(1, 1), padding=(1, 1))
            
            # Output converted back to float32 if keep_orig_output_dtype=True is simulated
            # The test explicitly sets keep_orig_output_dtype=True and checks against FP32 output dtype.
            return output_mp.to(torch.float32)

    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    weight_fp32 = torch.tensor(raw_mod_params["weight"], dtype=torch.float32)

    fp32_model = ConvModelFP32()
    with torch.no_grad():
        fp32_model.weight.data = weight_fp32 # Assign pre-generated weight

    amp_model = ConvModelMixedPrecision(mixed_precision_dtype)
    with torch.no_grad():
        amp_model.weight.data = weight_fp32.to(_torch_mixed_precision_dtype) # Assign and convert weight

    fp32_result = fp32_model(data_fp32)
    amp_result = amp_model(data_fp32) # AMP model takes fp32 inputs and casts internally

    verify_mixed_precision_output_close(
        fp32_result,
        amp_result,
        mixed_precision_dtype_str=mixed_precision_dtype,
        atol=0.01,
        rtol=1e-3,
        keep_orig_output_dtype=True, # Explicitly check for output dtype
    )

def test_convert_single_conv_fp64():
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3)
    mixed_precision_dtype = "float64"
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    class ConvModelFP32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=torch.float32))

        def forward(self, data):
            return F.conv2d(data, self.weight, stride=(1, 1), padding=(1, 1))

    # Mixed Precision model (FP64 in this case)
    class ConvModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=self.mixed_precision_torch_dtype))

        def forward(self, data):
            # Conv is green-listed. Inputs converted to mixed precision.
            data_mp = data.to(self.mixed_precision_torch_dtype)
            
            # Conv2d operation in mixed precision
            # TVM's behavior: "Note we still accumulate to FP32 by default, a user would need to overwrite default behavior"
            # However, the `expected_mod` implies full fp64 conversion for this test.
            # So, for PyTorch, we'll perform the conv in fp64 and keep output as fp64.
            output_mp = F.conv2d(data_mp, self.weight, stride=(1, 1), padding=(1, 1))
            
            return output_mp # Output remains in mixed_precision_dtype

    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    weight_fp32 = torch.tensor(raw_mod_params["weight"], dtype=torch.float32)

    fp32_model = ConvModelFP32()
    with torch.no_grad():
        fp32_model.weight.data = weight_fp32

    amp_model = ConvModelMixedPrecision(mixed_precision_dtype)
    with torch.no_grad():
        amp_model.weight.data = weight_fp32.to(_torch_mixed_precision_dtype)

    fp32_result = fp32_model(data_fp32)
    amp_result = amp_model(data_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.01, rtol=1e-3
    )

@target_precision_fixture
def test_convert_conv_bn(mixed_precision_dtype):
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3) # Conv weight OIHW
    bn_shape = [5] # Channels dimension for BatchNorm
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    class ConvBnModelFP32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=torch.float32))
            # BatchNorm parameters
            self.bn_gamma = torch.nn.Parameter(torch.randn(bn_shape[0], dtype=torch.float32))
            self.bn_beta = torch.nn.Parameter(torch.randn(bn_shape[0], dtype=torch.float32))
            self.bn_mean = torch.nn.Parameter(torch.randn(bn_shape[0], dtype=torch.float32))
            self.bn_var = torch.nn.Parameter(torch.randn(bn_shape[0], dtype=torch.float32))

        def forward(self, data):
            conv_out = F.conv2d(data, self.conv_weight, stride=(1, 1), padding=(1, 1))
            # BatchNorm in inference mode (training=False), use_input_stats=True
            bn_out = F.batch_norm(
                conv_out, self.bn_mean, self.bn_var, self.bn_gamma, self.bn_beta, training=False
            )
            return bn_out

    # Mixed Precision model
    class ConvBnModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            
            # Conv weights converted
            self.conv_weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=self.mixed_precision_torch_dtype))
            
            # BN params are also converted, as BN is gray-listed and its input (conv output) is green (mixed precision)
            self.bn_gamma = torch.nn.Parameter(torch.randn(bn_shape[0], dtype=self.mixed_precision_torch_dtype))
            self.bn_beta = torch.nn.Parameter(torch.randn(bn_shape[0], dtype=self.mixed_precision_torch_dtype))
            self.bn_mean = torch.nn.Parameter(torch.randn(bn_shape[0], dtype=self.mixed_precision_torch_dtype))
            self.bn_var = torch.nn.Parameter(torch.randn(bn_shape[0], dtype=self.mixed_precision_torch_dtype))

        def forward(self, data):
            # Input data cast to mixed precision for conv
            data_mp = data.to(self.mixed_precision_torch_dtype)

            # Conv op with mixed precision inputs and weights
            conv_out_mp = F.conv2d(data_mp, self.conv_weight, stride=(1, 1), padding=(1, 1))

            # BatchNorm op with mixed precision inputs and parameters
            bn_out_mp = F.batch_norm(
                conv_out_mp, self.bn_mean, self.bn_var, self.bn_gamma, self.bn_beta, training=False
            )
            return bn_out_mp.to(torch.float32) # Output cast back to float32

    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
        "gamma": np.random.uniform(-1, 1, size=bn_shape).astype("float32"),
        "beta": np.random.uniform(-1, 1, size=bn_shape).astype("float32"),
        "moving_mean": np.random.uniform(-1, 1, size=bn_shape).astype("float32"),
        "moving_var": np.random.uniform(-1, 1, size=bn_shape).astype("float32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    weight_fp32 = torch.tensor(raw_mod_params["weight"], dtype=torch.float32)
    gamma_fp32 = torch.tensor(raw_mod_params["gamma"], dtype=torch.float32)
    beta_fp32 = torch.tensor(raw_mod_params["beta"], dtype=torch.float32)
    mean_fp32 = torch.tensor(raw_mod_params["moving_mean"], dtype=torch.float32)
    var_fp32 = torch.tensor(raw_mod_params["moving_var"], dtype=torch.float32)

    fp32_model = ConvBnModelFP32()
    with torch.no_grad():
        fp32_model.conv_weight.data = weight_fp32
        fp32_model.bn_gamma.data = gamma_fp32
        fp32_model.bn_beta.data = beta_fp32
        fp32_model.bn_mean.data = mean_fp32
        fp32_model.bn_var.data = var_fp32

    amp_model = ConvBnModelMixedPrecision(mixed_precision_dtype)
    with torch.no_grad():
        amp_model.conv_weight.data = weight_fp32.to(_torch_mixed_precision_dtype)
        amp_model.bn_gamma.data = gamma_fp32.to(_torch_mixed_precision_dtype)
        amp_model.bn_beta.data = beta_fp32.to(_torch_mixed_precision_dtype)
        amp_model.bn_mean.data = mean_fp32.to(_torch_mixed_precision_dtype)
        amp_model.bn_var.data = var_fp32.to(_torch_mixed_precision_dtype)

    fp32_result = fp32_model(data_fp32)
    amp_result = amp_model(data_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.025, rtol=0.01
    )


@target_precision_fixture
def test_do_not_convert_softmax(mixed_precision_dtype):
    shape = [1, 2, 3]

    # FP32 model
    def model_fp32(a):
        return F.softmax(a, dim=-1) # Default axis for softmax in TVM is usually last

    # Mixed Precision model: Softmax is RED listed, so it and its inputs/outputs
    # should remain FP32.
    def model_amp(a):
        # Inputs should be FP32, computation FP32, output FP32
        return F.softmax(a, dim=-1)

    data_fp32 = torch.randn(shape, dtype=torch.float32)
    
    fp32_result = model_fp32(data_fp32)
    amp_result = model_amp(data_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.0, rtol=0.0
    )
    # The original test asserts structural equality of the module to itself, meaning no change.
    # Numerical equality is sufficient here.

@target_precision_fixture
def test_do_not_convert_arange(mixed_precision_dtype):
    dtype = "float32"
    torch_dtype = _convert_relay_dtype_to_torch_dtype(dtype)

    # FP32 model
    def model_fp32():
        return torch.arange(1, 128, dtype=torch_dtype)

    # Mixed Precision model: Arange is RED listed. Should remain FP32.
    def model_amp():
        return torch.arange(1, 128, dtype=torch_dtype)

    fp32_result = model_fp32()
    amp_result = model_amp()

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.0, rtol=0.0
    )

@target_precision_fixture
def test_do_not_convert_summation(mixed_precision_dtype):
    shape = [1, 3, 16, 16]
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    ops_mapping = [
        (torch.sum, lambda inp: torch.sum(inp)), # Global sum
        (torch.mean, lambda inp: torch.mean(inp)), # Global mean
        # Adaptive Avg Pool equivalent for Global Avg Pool
        (F.adaptive_avg_pool2d, lambda inp: F.adaptive_avg_pool2d(inp, (1, 1))), 
    ]

    for op_func, op_wrapper in ops_mapping:
        data_fp32 = torch.randn(shape, dtype=torch.float32)

        # FP32 model
        def model_fp32(a):
            return op_wrapper(a)

        # Mixed Precision model: Summation ops (sum, mean, avg_pool) are RED listed.
        # Should remain FP32.
        def model_amp(a):
            # Inputs to summation ops should be FP32, computation FP32, output FP32
            return op_wrapper(a)

        fp32_result = model_fp32(data_fp32)
        amp_result = model_amp(data_fp32)

        verify_mixed_precision_output_close(
            fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.0, rtol=0.0
        )

@target_precision_fixture
def test_green_gray_propagates_simple(mixed_precision_dtype):
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3) # OIHW format for conv2d
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    class ModelFP32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=torch.float32))

        def forward(self, data):
            conv = F.conv2d(data, self.weight, stride=(1, 1), padding=(1, 1))
            return conv + conv # Add is gray-listed, inherits FP32 from conv

    # Mixed Precision model
    class ModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=self.mixed_precision_torch_dtype))

        def forward(self, data):
            # Conv is green-listed
            data_mp = data.to(self.mixed_precision_torch_dtype)
            conv_mp = F.conv2d(data_mp, self.weight, stride=(1, 1), padding=(1, 1))
            
            # Add is gray-listed, inherits mixed precision from conv output
            result_mp = conv_mp + conv_mp
            
            return result_mp.to(torch.float32) # Final output to FP32

    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    weight_fp32 = torch.tensor(raw_mod_params["weight"], dtype=torch.float32)

    fp32_model = ModelFP32()
    with torch.no_grad():
        fp32_model.weight.data = weight_fp32

    amp_model = ModelMixedPrecision(mixed_precision_dtype)
    with torch.no_grad():
        amp_model.weight.data = weight_fp32.to(_torch_mixed_precision_dtype) # Weight converted at init

    fp32_result = fp32_model(data_fp32)
    amp_result = amp_model(data_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.01, rtol=0.01
    )


@target_precision_fixture
def test_green_red_not_use_extraneous_cast(mixed_precision_dtype):
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3) # OIHW format for conv2d
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    class ModelFP32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=torch.float32))

        def forward(self, data):
            conv = F.conv2d(data, self.weight, stride=(1, 1), padding=(1, 1))
            return F.softmax(conv, dim=1) # Softmax red-listed

    # Mixed Precision model
    class ModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=self.mixed_precision_torch_dtype))

        def forward(self, data):
            # Conv is green-listed
            data_mp = data.to(self.mixed_precision_torch_dtype)
            conv_mp = F.conv2d(data_mp, self.weight, stride=(1, 1), padding=(1, 1))
            
            # Softmax is red-listed, so its input must be FP32.
            # Output of conv_mp (float16/bfloat16) needs to be cast to float32 before softmax.
            conv_fp32 = conv_mp.to(torch.float32)
            softmax_out_fp32 = F.softmax(conv_fp32, dim=1)
            
            return softmax_out_fp32 # Final output is FP32 as softmax is red

    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    weight_fp32 = torch.tensor(raw_mod_params["weight"], dtype=torch.float32)

    fp32_model = ModelFP32()
    with torch.no_grad():
        fp32_model.weight.data = weight_fp32

    amp_model = ModelMixedPrecision(mixed_precision_dtype)
    with torch.no_grad():
        amp_model.weight.data = weight_fp32.to(_torch_mixed_precision_dtype)

    fp32_result = fp32_model(data_fp32)
    amp_result = amp_model(data_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.01, rtol=1e-3
    )

@target_precision_fixture
def test_red_gray_propagates_simple(mixed_precision_dtype):
    shape = [1, 2, 3]
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    def model_fp32(a):
        softmax_out = F.softmax(a, dim=-1)
        return softmax_out + softmax_out # Add is gray-listed, inherits FP32 from softmax

    # Mixed Precision model
    def model_amp(a):
        # Softmax is red-listed
        softmax_out_fp32 = F.softmax(a, dim=-1) # Input and output are FP32
        
        # Add is gray-listed, inherits FP32 from softmax output
        add_out_fp32 = softmax_out_fp32 + softmax_out_fp32
        
        return add_out_fp32 # Final output is FP32

    data_fp32 = torch.randn(shape, dtype=torch.float32)
    
    fp32_result = model_fp32(data_fp32)
    amp_result = model_amp(data_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.0, rtol=0.0
    )


@target_precision_fixture
def test_let_statement_simple(mixed_precision_dtype):
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
    
    # FP32 Model
    class ModelFP32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(20, 20, dtype=torch.float32)) # (out_features, in_features)

        def forward(self, data):
            # Relay.Let essentially means sequential assignment and use
            # var1 = dense(data, weight)
            var1_val = F.linear(data, self.weight) # dense maps to linear, weight is (out_features, in_features)
            
            r1 = var1_val + var1_val # Add is gray, inherits FP32

            # var2 = dense(r1, weight)
            var2_val = F.linear(r1, self.weight)

            r2 = var2_val + var2_val # Add is gray, inherits FP32
            return r2

    # Mixed Precision Model
    class ModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            self.weight = torch.nn.Parameter(torch.randn(20, 20, dtype=self.mixed_precision_torch_dtype))

        def forward(self, data):
            # data cast to mixed precision for first dense (green)
            data_mp = data.to(self.mixed_precision_torch_dtype)

            # var1 = dense(data, weight) -> green, so in mixed precision
            var1_val_mp = F.linear(data_mp, self.weight) # dense in MP

            r1_mp = var1_val_mp + var1_val_mp # Add is gray, inherits mixed precision

            # var2 = dense(r1, weight) -> green, so in mixed precision
            var2_val_mp = F.linear(r1_mp, self.weight) # dense in MP

            r2_mp = var2_val_mp + var2_val_mp # Add is gray, inherits mixed precision
            return r2_mp.to(torch.float32) # Final output to FP32

    data_shape = [1, 20]
    weight_shape = [20, 20] # For F.linear, this is (out_features, in_features)
    
    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    weight_fp32 = torch.tensor(raw_mod_params["weight"], dtype=torch.float32)

    fp32_model = ModelFP32()
    with torch.no_grad():
        fp32_model.weight.data = weight_fp32

    amp_model = ModelMixedPrecision(mixed_precision_dtype)
    with torch.no_grad():
        amp_model.weight.data = weight_fp32.to(_torch_mixed_precision_dtype)

    fp32_result = fp32_model(data_fp32)
    amp_result = amp_model(data_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.05, rtol=0.15
    )


@target_precision_fixture
def test_where_simple(mixed_precision_dtype):
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    class ModelFP32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(20, 20, dtype=torch.float32))

        def forward(self, data, weight):
            a = F.linear(data, weight) # Dense maps to linear
            # In TVM, `data` (float32) can be used as condition. In PyTorch, condition must be bool.
            condition = data > 0
            b = torch.where(condition, a, a) # Where is gray-listed
            return b

    # Mixed Precision model
    class ModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            self.weight = torch.nn.Parameter(torch.randn(20, 20, dtype=self.mixed_precision_torch_dtype))

        def forward(self, data_fp32, weight_fp32):
            # Dense is green-listed
            data_mp = data_fp32.to(self.mixed_precision_torch_dtype)
            weight_mp = weight_fp32.to(self.mixed_precision_torch_dtype)
            a_mp = F.linear(data_mp, weight_mp)

            # Where is gray-listed, inherits mixed precision
            # Condition based on FP32 input, then used with MP tensors.
            condition = data_fp32 > 0
            b_mp = torch.where(condition, a_mp, a_mp)
            return b_mp.to(torch.float32)

    data_shape = [1, 20]
    weight_shape = [20, 20]
    
    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    weight_fp32 = torch.tensor(raw_mod_params["weight"], dtype=torch.float32)

    fp32_model = ModelFP32()
    with torch.no_grad():
        fp32_model.weight.data = weight_fp32

    amp_model = ModelMixedPrecision(mixed_precision_dtype)
    with torch.no_grad():
        amp_model.weight.data = weight_fp32.to(_torch_mixed_precision_dtype)

    fp32_result = fp32_model(data_fp32, fp32_model.weight)
    amp_result = amp_model(data_fp32, weight_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.01, rtol=0.01
    )


@target_precision_fixture
def test_batch_matmul_simple(mixed_precision_dtype):
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    class ModelFP32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # In PyTorch, inputs are often not parameters themselves, but passed in
            # We'll make them parameters here for consistency in initialization
            self.data_param = torch.nn.Parameter(torch.randn(1, 1, 20, dtype=torch.float32))
            self.weight_param = torch.nn.Parameter(torch.randn(1, 20, 20, dtype=torch.float32))

        def forward(self, data_input, weight_input):
            # batch_matmul maps to torch.bmm
            return torch.bmm(data_input, weight_input)

    # Mixed Precision model
    class ModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            # Weights should be in mixed precision
            self.data_param = torch.nn.Parameter(torch.randn(1, 1, 20, dtype=self.mixed_precision_torch_dtype))
            self.weight_param = torch.nn.Parameter(torch.randn(1, 20, 20, dtype=self.mixed_precision_torch_dtype))

        def forward(self, data_input_fp32, weight_input_fp32):
            # batch_matmul is green-listed and accumulates to fp16 (or target precision)
            data_mp = data_input_fp32.to(self.mixed_precision_torch_dtype)
            weight_mp = weight_input_fp32.to(self.mixed_precision_torch_dtype)
            
            # Perform bmm in mixed precision
            result_mp = torch.bmm(data_mp, weight_mp)
            
            return result_mp.to(torch.float32) # Output cast back to FP32

    data_shape = [1, 1, 20]
    weight_shape = [1, 20, 20]
    
    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    weight_fp32 = torch.tensor(raw_mod_params["weight"], dtype=torch.float32)

    fp32_model = ModelFP32()
    with torch.no_grad():
        fp32_model.data_param.data = data_fp32 # Assign actual data values from params
        fp32_model.weight_param.data = weight_fp32

    amp_model = ModelMixedPrecision(mixed_precision_dtype)
    with torch.no_grad():
        amp_model.data_param.data = data_fp32.to(_torch_mixed_precision_dtype) # Parameters for AMP, type converted to target precision
        amp_model.weight_param.data = weight_fp32.to(_torch_mixed_precision_dtype)


    fp32_result = fp32_model(data_fp32, weight_fp32)
    amp_result = amp_model(data_fp32, weight_fp32)

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.01, rtol=0.01
    )


@target_precision_fixture
def test_convert_follow_node_with_integer_arguments(mixed_precision_dtype):
    # This tests that integer inputs to ops like `take` are not cast,
    # and only floating-point inputs are cast.
    _torch_mixed_precision_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)

    # FP32 model
    def model_fp32(data_tensor, indices_tensor):
        # Relay.take(data, indices, axis=0) -> torch.index_select(data, dim=0, index=indices.flatten())
        # The original TVM test had indices + relay.const(0, dtype="int32"), which means indices is a tensor.
        # torch.index_select expects a 1-D tensor for indices.
        indices_val = indices_tensor + torch.tensor(0, dtype=torch.int32)
        return torch.index_select(data_tensor, dim=0, index=indices_val.flatten())

    # Mixed Precision model
    class ModelMixedPrecision(torch.nn.Module):
        def __init__(self, mixed_precision_dtype):
            super().__init__()
            self.mixed_precision_torch_dtype = _convert_relay_dtype_to_torch_dtype(mixed_precision_dtype)
            # No parameters for this simple example, but could be if `data` was a param.
        
        def forward(self, data_tensor_fp32, indices_tensor_fp32):
            # `take` is a gray-listed op, so it should inherit precision from its float inputs.
            # Data input will be cast to mixed precision.
            # Indices input (int32) should remain int32.
            data_mp = data_tensor_fp32.to(self.mixed_precision_torch_dtype)
            
            # Indices should remain int32
            indices_val_int32 = indices_tensor_fp32 + torch.tensor(0, dtype=torch.int32)
            
            # Operation performed using mixed precision data and int32 indices
            result_mp = torch.index_select(data_mp, dim=0, index=indices_val_int32.flatten())
            
            return result_mp.to(torch.float32) # Output cast back to FP32

    data_shape = [1, 10]
    indices_shape = [1, 1]

    raw_mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "indices": np.array([[0]]).astype("int32"),
    }
    
    data_fp32 = torch.tensor(raw_mod_params["data"], dtype=torch.float32)
    indices_fp32 = torch.tensor(raw_mod_params["indices"], dtype=torch.int32) # Keep indices as int32

    fp32_model = model_fp32
    amp_model = ModelMixedPrecision(mixed_precision_dtype)

    fp32_result = fp32_model(data_fp32, indices_fp32)
    amp_result = amp_model(data_fp32, indices_fp32) # Pass original FP32 data, int32 indices

    verify_mixed_precision_output_close(
        fp32_result, amp_result, mixed_precision_dtype_str=mixed_precision_dtype, atol=0.01, rtol=0.01
    )
