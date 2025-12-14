import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch.ao.quantization as ao_q

# Custom utility to convert TVM string dtypes to PyTorch dtypes
def to_torch_dtype(dtype_str):
    if dtype_str == "int8":
        return torch.int8
    elif dtype_str == "uint8":
        return torch.uint8
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

# For quantized types in PyTorch, which typically start with 'q'
def get_q_dtype(dtype_str):
    if dtype_str == "int8":
        return torch.qint8
    elif dtype_str == "uint8":
        return torch.quint8
    elif dtype_str == "int32":
        return torch.qint32 # Usually for accumulator, not direct tensor storage
    else:
        raise ValueError(f"Unsupported quantized dtype: {dtype_str}")

# Helper to broadcast scalar scales/zero_points to tensors if needed
def ensure_tensor_params(param, target_shape, axis, dtype):
    if isinstance(param, (float, int)):
        return torch.tensor(param, dtype=dtype)
    if isinstance(param, np.ndarray):
        param = torch.tensor(param, dtype=dtype)
    
    if axis is None or param.numel() == 1:
        return param # Already scalar or single-element tensor
    
    # Per-channel parameter, broadcast to target_shape for element-wise ops
    # The param's shape should be `(target_shape[axis],)`
    # Need to reshape `param` to `(1, ..., 1, param_len, 1, ...)`
    new_shape = [1] * len(target_shape)
    new_shape[axis] = param.numel()
    return param.reshape(new_shape)


# General comparison function for Fake Quantization to Integer pass
def compare_fq_to_int(
    float_model_func, # Callable representing the original fake-quantized float graph
    int_model_func,   # Callable representing the transformed integer-quantized graph
    args_np,          # NumPy arrays as inputs
    allow_rounding_error=False
):
    # Run the float model
    result_float_np = float_model_func(*args_np)

    # Run the integer model
    result_int_np = int_model_func(*args_np)

    # In TVM, it also asserts `not tvm.ir.structural_equal(mod, mod_int)`.
    # This structural check is not directly applicable in PyTorch's eager mode.
    # We focus on numerical equivalence.
    # print("TODO: TVM IR structural equality checks are not directly applicable to PyTorch eager mode.")

    # Convert numpy results to torch tensors for comparison
    res_float_torch = torch.tensor(result_float_np, dtype=torch.float32)
    res_int_torch = torch.tensor(result_int_np, dtype=torch.float32)

    if allow_rounding_error:
        assert torch.all(torch.abs(res_float_torch.to(torch.int32) - res_int_torch.to(torch.int32)) <= 1), \
            f"Results differ by more than 1 (rounding error allowed).\nFloat: {result_float_np}\nInt: {result_int_np}"
    else:
        # Use np.array_equal as TVM tests often expect exact matches unless rounding error is allowed
        assert np.array_equal(result_float_np, result_int_np), \
            f"Results are not array_equal.\nFloat: {result_float_np}\nInt: {result_int_np}"


# The `compare_expected_fq_qat_to_int` function
def compare_expected_fq_qat_to_int(
    expr_float_func,       # Original expression in float
    expected_expr_int_func, # Expected transformed expression (integer)
    args_np,
    allow_rounding_error=False
):
    # Execute the original (fake-quantized float) graph
    result_def_np = expr_float_func(*args_np)

    # Execute the expected (integer quantized) graph
    result_exp_np = expected_expr_int_func(*args_np)

    # In TVM, it also asserts `not tvm.ir.structural_equal(mod, mod_int)`
    # and `tvm.ir.structural_equal(mod_int, mod_exp)`.
    # These structural checks are not directly applicable in PyTorch's eager mode.
    # We focus on numerical equivalence.
    # print("TODO: TVM IR structural equality checks for QAT are not directly applicable to PyTorch eager mode.")

    # For comparison, convert to torch tensors
    res_def_torch = torch.tensor(result_def_np, dtype=torch.float32)
    res_exp_torch = torch.tensor(result_exp_np, dtype=torch.float32)

    if allow_rounding_error:
        assert torch.all(torch.abs(res_def_torch.to(torch.int32) - res_exp_torch.to(torch.int32)) <= 1), \
            f"QAT results differ by more than 1 (rounding error allowed).\nFloat: {result_def_np}\nInt: {result_exp_np}"
    else:
        assert np.array_equal(result_def_np, result_exp_np), \
            f"QAT results are not array_equal.\nFloat: {result_def_np}\nInt: {result_exp_np}"


@pytest.mark.parametrize("out_dtype_str", ["int8", "uint8"])
def test_fake_quantize_conv(out_dtype_str):
    x_shape = [1, 3, 224, 224]
    w_shape = [16, 3, 5, 5]
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 0.5, 0
    output_scale, output_zp = 1.0, 0
    kernel_size = [5, 5]

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    args_np = [x_np, w_np]

    def float_model(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        x_deq = (x_val - x_zp) * x_scale
        w_deq = (w_val - w_zp) * w_scale

        conv_float = F.conv2d(x_deq, w_deq, kernel_size=kernel_size)

        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            conv_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        # Create quantized tensors (inputs to qnn.conv2d)
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        int_conv = ao_q.quantized.functional.conv2d(
            q_x,
            q_w,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_conv.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


@pytest.mark.parametrize("out_dtype_str", ["int8", "uint8"])
def test_fake_quantize_conv_per_channel(out_dtype_str):
    x_shape = [1, 3, 224, 224]
    w_shape = [16, 3, 5, 5]
    x_scale, x_zp = 2.0, 0
    w_scale_np = np.random.random([16]).astype("float32")
    w_zp_np = np.array([np.random.randint(0, 255)] * 16, dtype="int32")
    output_scale, output_zp = 1.0, 0
    kernel_size = [5, 5]
    w_axis = 0 # Axis for per-channel quantization for weights (output channels)

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    args_np = [x_np, w_np]

    def float_model(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        w_scale_tensor = torch.tensor(w_scale_np, dtype=torch.float32)
        w_zp_tensor = torch.tensor(w_zp_np, dtype=torch.int32)
        
        # Broadcast scales and zero points for element-wise operation
        # Weight shape is OIHW. w_axis=0 means scale/zp broadcast along Output channels
        w_scale_bcast = w_scale_tensor.reshape(w_shape[w_axis], 1, 1, 1)
        w_zp_bcast = w_zp_tensor.reshape(w_shape[w_axis], 1, 1, 1)

        x_deq = (x_val - x_zp) * x_scale
        w_deq = (w_val - w_zp_bcast.float()) * w_scale_bcast

        conv_float = F.conv2d(x_deq, w_deq, kernel_size=kernel_size, channels=16) # channels=16 is info from TVM
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            conv_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))
        
        w_scale_tensor = torch.tensor(w_scale_np, dtype=torch.float32)
        w_zp_tensor = torch.tensor(w_zp_np, dtype=torch.int32)

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        # Per-channel quantization for weights
        q_w = ao_q.quantize_per_channel(w_q_val.float(), scales=w_scale_tensor, zero_points=w_zp_tensor, axis=w_axis, dtype=get_q_dtype("int8"))

        int_conv = ao_q.quantized.functional.conv2d(
            q_x,
            q_w,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_conv.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


@pytest.mark.parametrize("out_dtype_str", ["int8", "uint8"])
def test_fake_quantize_transposeconv(out_dtype_str):
    x_shape = [1, 3, 224, 224]
    w_shape = [3, 16, 5, 5] # IOHW format for conv2d_transpose
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 0.5, 0
    output_scale, output_zp = 1.0, 0
    kernel_size = [5, 5]
    # data_layout="NCHW", kernel_layout="IOHW" are default for PyTorch conv_transpose2d
    
    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    args_np = [x_np, w_np]

    def float_model(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        x_deq = (x_val - x_zp) * x_scale
        w_deq = (w_val - w_zp) * w_scale

        conv_t_float = F.conv_transpose2d(x_deq, w_deq, kernel_size=kernel_size)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            conv_t_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        # PyTorch quantized functional does not have a direct conv_transpose2d, use dequantize-float-quantize pattern
        # This reflects a case where TVM's pass might not have a direct QNN op.
        # So the "integer model" here will still be dequantize-float-quantize but derived directly.
        
        # Intermediate float computation
        x_deq = q_x.dequantize()
        w_deq = q_w.dequantize()
        conv_t_float = F.conv_transpose2d(x_deq, w_deq, kernel_size=kernel_size)

        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            conv_t_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


@pytest.mark.parametrize("out_dtype_str", ["int8", "uint8"])
def test_fake_quantize_dense(out_dtype_str):
    x_shape = [128, 64]
    w_shape = [256, 64] # PyTorch F.linear expects (out_features, in_features)
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 0.5, 0
    output_scale, output_zp = 1.0, 0

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    args_np = [x_np, w_np]

    def float_model(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        x_deq = (x_val - x_zp) * x_scale
        w_deq = (w_val - w_zp) * w_scale

        dense_float = F.linear(x_deq, w_deq) # No bias in this test
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            dense_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        int_dense = ao_q.quantized.functional.linear(
            q_x,
            q_w,
            bias=None,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_dense.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


@pytest.mark.parametrize("out_dtype_str", ["int8", "uint8"])
def test_fake_quantize_dense_per_channel(out_dtype_str):
    x_shape = [128, 64]
    w_shape = [256, 64] # PyTorch F.linear expects (out_features, in_features)
    x_scale, x_zp = 2.0, 0
    w_scale_np = np.random.random([256]).astype("float32")
    w_zp_np = np.array([0] * 256, dtype="int32")
    output_scale, output_zp = 1.0, 0
    w_axis = 0 # Axis for per-channel quantization for weights (output channels)

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    args_np = [x_np, w_np]

    def float_model(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        w_scale_tensor = torch.tensor(w_scale_np, dtype=torch.float32)
        w_zp_tensor = torch.tensor(w_zp_np, dtype=torch.int32)

        # Broadcast scales and zero points for element-wise operation
        w_scale_bcast = w_scale_tensor.reshape(w_shape[w_axis], 1)
        w_zp_bcast = w_zp_tensor.reshape(w_shape[w_axis], 1)

        x_deq = (x_val - x_zp) * x_scale
        w_deq = (w_val - w_zp_bcast.float()) * w_scale_bcast

        dense_float = F.linear(x_deq, w_deq)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            dense_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))
        
        w_scale_tensor = torch.tensor(w_scale_np, dtype=torch.float32)
        w_zp_tensor = torch.tensor(w_zp_np, dtype=torch.int32)

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = ao_q.quantize_per_channel(w_q_val.float(), scales=w_scale_tensor, zero_points=w_zp_tensor, axis=w_axis, dtype=get_q_dtype("int8"))

        int_dense = ao_q.quantized.functional.linear(
            q_x,
            q_w,
            bias=None,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_dense.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


def test_fake_quantize_dense_bias():
    out_dtype_str = "int8"
    x_shape = [128, 64]
    w_shape = [256, 64]
    bias_shape = [256]
    x_scale, x_zp = 2.0, 0
    w_zp = 0
    output_scale, output_zp = 1.0, 0
    w_scale_np = np.random.random([256]).astype("float32")
    bias_scale_factor = 2.0 # for bias dequantize scale = x_scale * w_scale

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    bias_np = np.random.randint(-128, 127, size=bias_shape, dtype="int32") # Bias is int32 quantized
    args_np = [x_np, w_np, bias_np]

    def float_model(x_val_np, w_val_np, bias_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()
        bias_val = torch.tensor(bias_val_np, dtype=to_torch_dtype("int32")).float()
        
        w_scale_tensor = torch.tensor(w_scale_np, dtype=torch.float32)

        x_deq = (x_val - x_zp) * x_scale
        w_deq = (w_val - w_zp) * w_scale_tensor.reshape(-1, 1)

        dense_float = F.linear(x_deq, w_deq)

        # Dequantize bias, where bias_scale = x_scale * w_scale
        bias_deq_scale = bias_scale_factor * w_scale_tensor
        bias_deq = (bias_val - output_zp) * bias_deq_scale # Using output_zp=0 for bias here

        op_add = dense_float + bias_deq
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_add, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np, bias_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))
        bias_q_val = torch.tensor(bias_val_np, dtype=to_torch_dtype("int32"))
        
        w_scale_tensor = torch.tensor(w_scale_np, dtype=torch.float32)

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = ao_q.quantize_per_channel(w_q_val.float(), scales=w_scale_tensor, zero_points=w_zp, axis=0, dtype=get_q_dtype("int8"))

        bias_scale = bias_scale_factor * w_scale_tensor # This is the float scale for bias
        q_bias = ao_q.quantize_per_channel(bias_q_val.float(), scales=bias_scale, zero_points=output_zp, axis=0, dtype=get_q_dtype("int32"))

        int_dense_bias = ao_q.quantized.functional.linear(
            q_x,
            q_w,
            bias=q_bias, # PyTorch expects quantized bias tensor here
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_dense_bias.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


@pytest.mark.parametrize("out_dtype_str", ["int8", "uint8"])
def test_fake_quantize_batch_matmul(out_dtype_str):
    x_shape = [1, 128, 64]
    w_shape = [1, 256, 64] # PyTorch matmul aligns last two for non-batch dimensions
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 0.5, 0
    output_scale, output_zp = 1.0, 0

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    args_np = [x_np, w_np]

    def float_model(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        x_deq = (x_val - x_zp) * x_scale
        w_deq = (w_val - w_zp) * w_scale

        # For batch_matmul, PyTorch's torch.matmul automatically handles batch dimensions
        # It performs A @ B.T for the last two dimensions typically in Relay
        # Relay `batch_matmul` implies `(M, K) @ (K, N)` without explicit transpose.
        # So for PyTorch, we would need to transpose W.
        op_float = torch.matmul(x_deq, w_deq.transpose(-1, -2))
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        # PyTorch quantized functional does not have a direct `batch_matmul` or `matmul`
        # for `qint8` inputs that produce `qint8` output.
        # It typically converts to float, does the op, then quantizes.
        # This reflects a scenario where TVM might also dequantize, compute, then quantize internally.
        
        # Dequantize, perform float batch matmul, then quantize
        x_deq = q_x.dequantize()
        w_deq = q_w.dequantize()
        float_result = torch.matmul(x_deq, w_deq.transpose(-1, -2))
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            float_result, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_transpose_quantize_conv():
    x_shape = [1, 224, 224, 3] # NHWC
    w_shape = [16, 3, 5, 5] # OIHW
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 0.5, 0
    output_scale, output_zp = 1.0, 0
    kernel_size = [5, 5]
    out_dtype_str = "int8" # Default for TVM qnn.quantize if not specified

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    args_np = [x_np, w_np]

    def float_model(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        x_deq = (x_val - x_zp) * x_scale
        # Transpose from NHWC to NCHW for PyTorch conv2d
        x_transposed = x_deq.permute(0, 3, 1, 2)
        
        w_deq = (w_val - w_zp) * w_scale

        conv_float = F.conv2d(x_transposed, w_deq, kernel_size=kernel_size)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            conv_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        # Dequantize, transpose, conv, quantize as the `FakeQuantizationToInteger` would apply to this structure
        x_deq = q_x.dequantize()
        x_transposed = x_deq.permute(0, 3, 1, 2) # NHWC -> NCHW

        # PyTorch quantized functional expects quantized inputs
        # The transpose op itself doesn't have a qnn.op
        # so this is effectively float op within an int quantized graph simulation
        
        # NOTE: For PyTorch `quantized.functional.conv2d`, input `x` must be quantized.
        # This implies that `transpose` should ideally operate on quantized tensors,
        # or there is an implicit dequantize-transpose-quantize.
        # TVM's pass might introduce temporary float stages for ops not having qnn counterparts.
        # For simplicity, if an op is not a qnn.op, we dequantize, apply float op, then requantize if needed for next qnn op.
        
        # Here, x_transposed is float. It needs to be quantized before passing to qnn.conv2d
        # If the input to qnn.conv2d (in the TVM transformed graph) is not directly a quantized tensor,
        # this path would represent a dequantize -> float_op -> quantize -> qnn.conv2d
        # Assuming the TVM pass makes it dequantize -> transpose -> quantize (temp) -> qnn.conv2d
        # However, it's more likely `dequantize -> transpose -> conv2d(float) -> quantize`.
        # Let's align with the `dequantize -> float_op -> quantize` pattern for the whole chain if no specific qnn op exists.
        
        conv_float = F.conv2d(x_transposed, q_w.dequantize(), kernel_size=kernel_size)

        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            conv_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


@pytest.mark.parametrize("const_bias", [False, True])
def test_fake_transpose_quantize_conv_bias_add(const_bias):
    x_shape = [1, 224, 224, 3]
    w_shape = [16, 3, 5, 5]
    bias_shape = [16]
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 0.5, 0
    output_scale, output_zp = 1.0, 0
    kernel_size = [5, 5]
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=bias_shape, dtype="int32")
    args_np = [x_np, w_np]
    
    if not const_bias:
        args_np.append(bias_np)
    
    # Bias scale and zero point for float_model and int_model
    # This comes from the `relay.qnn.op.dequantize(bias, one, zero)` or `relay.const(bias_np)`
    # If const_bias, then it's a direct float tensor
    bias_deq_scale_val = 1.0 # From `one = relay.const(1.0)`
    bias_deq_zp_val = 0 # From `zero = relay.const(0)`

    def float_model(*input_nps):
        x_val_np, w_val_np = input_nps[0], input_nps[1]
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        x_deq = (x_val - x_zp) * x_scale
        x_transposed = x_deq.permute(0, 3, 1, 2)
        
        w_deq = (w_val - w_zp) * w_scale

        conv_float = F.conv2d(x_transposed, w_deq, kernel_size=kernel_size)
        
        if const_bias:
            bias_deq = torch.tensor(bias_np.astype("float32")) # Direct float
        else:
            bias_val_np = input_nps[2]
            bias_val = torch.tensor(bias_val_np, dtype=to_torch_dtype("int32")).float()
            bias_deq = (bias_val - bias_deq_zp_val) * bias_deq_scale_val

        op_add_float = conv_float + bias_deq.reshape(1, -1, 1, 1) # Reshape bias for broadcasting
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_add_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(*input_nps):
        x_val_np, w_val_np = input_nps[0], input_nps[1]
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        # Dequantize for transpose
        x_deq = q_x.dequantize()
        x_transposed = x_deq.permute(0, 3, 1, 2)

        # PyTorch quantized functional.conv2d expects quantized bias
        bias_for_qconv = None
        if const_bias:
            # If bias is constant, it's a float32 tensor
            bias_for_qconv = ao_q.quantize_per_tensor(
                torch.tensor(bias_np, dtype=torch.float32), 
                scale=bias_deq_scale_val, zero_point=bias_deq_zp_val, dtype=get_q_dtype("int32")
            )
        else:
            bias_val_np = input_nps[2]
            bias_q_val = torch.tensor(bias_val_np, dtype=to_torch_dtype("int32"))
            bias_for_qconv = ao_q.quantize_per_tensor(
                bias_q_val.float(), scale=bias_deq_scale_val, zero_point=bias_deq_zp_val, dtype=get_q_dtype("int32")
            )

        # qnn.conv2d takes quantized inputs, but transposed `x` is currently float
        # This implies `dequantize -> transpose -> float conv2d + bias -> quantize`
        # OR `dequantize -> transpose -> quantize (temp) -> qnn.conv2d + quantized bias`
        # Based on TVM's pass logic, typically `dequantize -> float_op -> quantize_to_float_op`
        # If conv2d+bias_add does not have a native QNN op, the whole block stays float computation
        # followed by final quantization.

        # So `conv_float` needs to be calculated in float
        conv_float_res = F.conv2d(x_transposed, q_w.dequantize(), kernel_size=kernel_size)
        op_add_float_res = conv_float_res + bias_for_qconv.dequantize().reshape(1, -1, 1, 1) # Reshape bias for broadcasting
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_add_float_res, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_transpose_quantize_conv_bias_add_per_channel():
    x_shape = [1, 224, 224, 3]
    w_shape = [16, 3, 5, 5]
    bias_shape = [16]
    x_scale, x_zp = 2.0, 0
    w_zp = 0
    output_scale, output_zp = 1.0, 0
    kernel_size = [5, 5]
    out_dtype_str = "int8"

    w_scale_np = (np.random.random([16]).astype("float32") - 0.5) / 10 + 0.5
    noise = (np.random.random([16]).astype("float32") - 0.5) * 1e-15
    w_zp_np = np.array([0] * 16, dtype="int32")

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=bias_shape, dtype="int32")
    args_np = [x_np, w_np, bias_np]
    
    bias_deq_scale_val_np = 2.0 * w_scale_np + noise # Per-channel bias scale
    bias_deq_zp_val_np = w_zp_np # Per-channel bias zero_point

    def float_model(x_val_np, w_val_np, bias_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()
        bias_val = torch.tensor(bias_val_np, dtype=to_torch_dtype("int32")).float()

        w_scale_tensor = torch.tensor(w_scale_np, dtype=torch.float32)
        w_zp_tensor = torch.tensor(w_zp_np, dtype=torch.int32)
        bias_deq_scale_tensor = torch.tensor(bias_deq_scale_val_np, dtype=torch.float32)
        bias_deq_zp_tensor = torch.tensor(bias_deq_zp_val_np, dtype=torch.int32)

        # Broadcast per-channel parameters
        w_scale_bcast = w_scale_tensor.reshape(16, 1, 1, 1)
        w_zp_bcast = w_zp_tensor.reshape(16, 1, 1, 1)
        bias_scale_bcast = bias_deq_scale_tensor.reshape(1, -1, 1, 1)
        bias_zp_bcast = bias_deq_zp_tensor.reshape(1, -1, 1, 1)


        x_deq = (x_val - x_zp) * x_scale
        x_transposed = x_deq.permute(0, 3, 1, 2) # NHWC -> NCHW
        
        w_deq = (w_val - w_zp_bcast.float()) * w_scale_bcast

        conv_float = F.conv2d(x_transposed, w_deq, kernel_size=kernel_size)
        
        bias_deq = (bias_val - bias_zp_bcast.float()) * bias_scale_bcast
        op_add_float = conv_float + bias_deq
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_add_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np, bias_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))
        bias_q_val = torch.tensor(bias_val_np, dtype=to_torch_dtype("int32"))
        
        w_scale_tensor = torch.tensor(w_scale_np, dtype=torch.float32)
        w_zp_tensor = torch.tensor(w_zp_np, dtype=torch.int32)
        bias_deq_scale_tensor = torch.tensor(bias_deq_scale_val_np, dtype=torch.float32)
        bias_deq_zp_tensor = torch.tensor(bias_deq_zp_val_np, dtype=torch.int32)

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        # Per-channel quantization for weights
        q_w = ao_q.quantize_per_channel(w_q_val.float(), scales=w_scale_tensor, zero_points=w_zp_tensor, axis=0, dtype=get_q_dtype("int8"))

        # Dequantize for transpose
        x_deq = q_x.dequantize()
        x_transposed = x_deq.permute(0, 3, 1, 2) # NHWC -> NCHW

        # Per-channel quantized bias
        q_bias = ao_q.quantize_per_channel(bias_q_val.float(), scales=bias_deq_scale_tensor, zero_points=bias_deq_zp_tensor, axis=0, dtype=get_q_dtype("int32"))

        # Simulate `qnn.conv2d` which has no specific PyTorch functional with float input
        conv_float_res = F.conv2d(x_transposed, q_w.dequantize(), kernel_size=kernel_size)
        op_add_float_res = conv_float_res + q_bias.dequantize().reshape(1, -1, 1, 1)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_add_float_res, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


def test_fake_transpose_quantize_conv_bias_add_mismatch():
    x_shape = [1, 224, 224, 3]
    w_shape = [16, 3, 5, 5]
    bias_shape = [16]
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 0.5, 0
    bias_scale_val, bias_zp_val = 2.0, 0 # Explicitly set for bias dequantize
    output_scale, output_zp = 1.0, 0
    kernel_size = [5, 5]
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    w_np = np.random.randint(-128, 127, size=w_shape, dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=bias_shape, dtype="int32")
    args_np = [x_np, w_np, bias_np]

    def float_model(x_val_np, w_val_np, bias_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()
        bias_val = torch.tensor(bias_val_np, dtype=to_torch_dtype("int32")).float()

        x_deq = (x_val - x_zp) * x_scale
        x_transposed = x_deq.permute(0, 3, 1, 2)
        
        w_deq = (w_val - w_zp) * w_scale

        conv_float = F.conv2d(x_transposed, w_deq, kernel_size=kernel_size)
        
        bias_deq = (bias_val - bias_zp_val) * bias_scale_val
        op_add_float = conv_float + bias_deq.reshape(1, -1, 1, 1) # Reshape bias for broadcasting
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_add_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np, bias_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))
        bias_q_val = torch.tensor(bias_val_np, dtype=to_torch_dtype("int32"))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))
        
        # Dequantize for transpose
        x_deq = q_x.dequantize()
        x_transposed = x_deq.permute(0, 3, 1, 2) # NHWC -> NCHW

        # PyTorch quantized functional.conv2d expects quantized bias
        q_bias = ao_q.quantize_per_tensor(bias_q_val.float(), scale=bias_scale_val, zero_point=bias_zp_val, dtype=get_q_dtype("int32"))

        conv_float_res = F.conv2d(x_transposed, q_w.dequantize(), kernel_size=kernel_size)
        op_add_float_res = conv_float_res + q_bias.dequantize().reshape(1, -1, 1, 1)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_add_float_res, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_maxpool():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    pool_size = [3, 3]
    out_dtype_str = "int8" # Default for Relay var

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        maxpool_float = F.max_pool2d(x_deq, kernel_size=pool_size)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            maxpool_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        int_maxpool = ao_q.quantized.functional.max_pool2d(
            q_x,
            kernel_size=pool_size,
            stride=pool_size, # Default stride is kernel_size in TVM unless specified
            padding=0,
            dilation=1,
            ceil_mode=False,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_maxpool.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


@pytest.mark.parametrize("output_size_val", [None, 1])
def test_fake_quantize_adaptive_avgpool1d(output_size_val):
    x_shape = [1, 128, 768]
    x_scale, x_zp = 2.0, -12
    output_scale, output_zp = 0.5, 10
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    # For PyTorch, output_size=None needs to be handled.
    # If output_size is 1, it implies (1,) for 1D.
    if output_size_val is None:
        pytorch_output_size = x_shape[-1] # Adaptive pool to original size if None
    else:
        pytorch_output_size = output_size_val

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        
        # PyTorch F.adaptive_avg_pool1d expects tuple for output_size
        output_size_tuple = (pytorch_output_size,) if isinstance(pytorch_output_size, int) else pytorch_output_size
        if output_size_tuple is None: # for the case when output_size_val is None and x_shape[-1] is used
            output_size_tuple = (x_shape[-1],)

        adaptive_avgpool_float = F.adaptive_avg_pool1d(x_deq, output_size=output_size_tuple)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            adaptive_avgpool_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # PyTorch F.adaptive_avg_pool1d expects tuple for output_size
        output_size_tuple = (pytorch_output_size,) if isinstance(pytorch_output_size, int) else pytorch_output_size
        if output_size_tuple is None:
            output_size_tuple = (x_shape[-1],)

        int_adaptive_avgpool = ao_q.quantized.functional.adaptive_avg_pool1d(
            q_x,
            output_size=output_size_tuple,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_adaptive_avgpool.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


def test_fake_quantize_avgpool():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, -12
    output_scale, output_zp = 0.5, 10
    pool_size = [3, 3]
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        avgpool_float = F.avg_pool2d(x_deq, kernel_size=pool_size)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            avgpool_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        int_avgpool = ao_q.quantized.functional.avg_pool2d(
            q_x,
            kernel_size=pool_size,
            stride=pool_size, # Default stride is kernel_size in TVM unless specified
            padding=0,
            ceil_mode=False,
            count_include_pad=True, # PyTorch default, matches TVM default
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_avgpool.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


def test_fake_quantize_global_avg_pool():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, -12
    output_scale, output_zp = 0.5, 10
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        global_avgpool_float = F.adaptive_avg_pool2d(x_deq, output_size=1)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            global_avgpool_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        int_global_avgpool = ao_q.quantized.functional.adaptive_avg_pool2d(
            q_x,
            output_size=1,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_global_avgpool.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


class TestUnaryQNNOp:
    def helper_test_fake_quantize_unary_op(self, fp32_op_func, pos_values=False):
        for dtype_str in ["int8", "uint8"]:
            x_shape = [1, 3, 3, 3]
            
            zero_val = -128 if dtype_str == "int8" else 0
            if pos_values:
                input_zp_val = zero_val
                output_zp_val = zero_val
            else:
                input_zp_val = np.random.randint(0, 255) + zero_val
                output_zp_val = np.random.randint(0, 255) + zero_val

            input_scale_val = np.random.rand()
            output_scale_val = np.random.rand()

            x_np = np.random.randint(0 + zero_val, 255 + zero_val, size=x_shape, dtype=dtype_str)
            args_np = [x_np]

            def float_model(x_val_np):
                x_val = torch.tensor(x_val_np, dtype=to_torch_dtype(dtype_str)).float()
                x_deq = (x_val - input_zp_val) * input_scale_val
                op_float = fp32_op_func(x_deq)
                
                output_q_dtype = get_q_dtype(dtype_str)
                return ao_q.quantize_per_tensor(
                    op_float, scale=output_scale_val, zero_point=output_zp_val, dtype=output_q_dtype
                ).dequantize().numpy()

            def int_model(x_val_np):
                x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype(dtype_str))
                q_x = torch.quantize_per_tensor(x_q_val.float(), scale=input_scale_val, zero_point=input_zp_val, dtype=get_q_dtype(dtype_str))

                # For these unary ops, PyTorch quantized functional ops are often dequantize-float_op-quantize
                op_float = fp32_op_func(q_x.dequantize())
                
                output_q_dtype = get_q_dtype(dtype_str)
                return ao_q.quantize_per_tensor(
                    op_float, scale=output_scale_val, zero_point=output_zp_val, dtype=output_q_dtype
                ).dequantize().numpy()

            compare_fq_to_int(float_model, int_model, args_np, True)

    def test_sqrt(self):
        self.helper_test_fake_quantize_unary_op(fp32_op_func=torch.sqrt, pos_values=True)

    def test_rsqrt(self):
        self.helper_test_fake_quantize_unary_op(fp32_op_func=torch.rsqrt, pos_values=True)

    def test_exp(self):
        self.helper_test_fake_quantize_unary_op(fp32_op_func=torch.exp)

    def test_erf(self):
        # torch.erf is in torch.special in newer versions, but also directly in torch
        self.helper_test_fake_quantize_unary_op(fp32_op_func=torch.erf)

    def test_sigmoid(self):
        self.helper_test_fake_quantize_unary_op(fp32_op_func=torch.sigmoid)

    def test_tanh(self):
        self.helper_test_fake_quantize_unary_op(fp32_op_func=torch.tanh)

    def test_log(self):
        self.helper_test_fake_quantize_unary_op(fp32_op_func=torch.log, pos_values=True)


def test_fake_quantize_reshape():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    new_shape = [1, 3, -1]
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        reshape_float = torch.reshape(x_deq, new_shape)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            reshape_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # Reshape for quantized tensors
        int_reshape = q_x.reshape(new_shape) # PyTorch quantized tensors support .reshape()
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            int_reshape.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype # PyTorch reshape on q-tensor keeps qparams
        ).dequantize().numpy() # This re-quantization is a bit redundant if q-params are the same, but matches TVM

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_image_resize_bilinear():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    size = [4, 4]
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        resize_float = F.interpolate(x_deq, size=size, mode="bilinear", align_corners=False) # TVM "linear" for resize2d usually maps to bilinear for 2D
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            resize_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # For interpolate, PyTorch quantized functional ops are often dequantize-float_op-quantize
        # This implies TVM might also handle it this way if no specific QNN op
        resize_float = F.interpolate(q_x.dequantize(), size=size, mode="bilinear", align_corners=False)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            resize_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


def test_fake_quantize_abs():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        abs_float = torch.abs(x_deq)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            abs_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        abs_float = torch.abs(q_x.dequantize())
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            abs_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_expand_dims():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    axis_val = 1
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        expand_dims_float = torch.unsqueeze(x_deq, dim=axis_val)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            expand_dims_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        int_expand_dims = torch.unsqueeze(q_x, dim=axis_val)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            int_expand_dims.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_squeeze():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    axis_val = [0] # TVM uses list, PyTorch expects int for dim
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        squeeze_float = torch.squeeze(x_deq, dim=axis_val[0])
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            squeeze_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        int_squeeze = torch.squeeze(q_x, dim=axis_val[0])
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            int_squeeze.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_strided_slice():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    begin = [0, 0, 0, 0]
    end = [1, 1, 112, 112]
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        
        # PyTorch slicing
        slice_float = x_deq[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2], begin[3]:end[3]]
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            slice_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # Slicing on quantized tensors
        int_slice = q_x[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2], begin[3]:end[3]]
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            int_slice.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_split():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    axis_val = 3
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    # Test case 1: split into 2 sections (equal parts)
    def float_model_sections_2(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        
        splits = torch.split(x_deq, x_shape[axis_val] // 2, dim=axis_val) # Assuming even split if integer
        op = splits[0]
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model_sections_2(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        splits = torch.split(q_x, x_shape[axis_val] // 2, dim=axis_val)
        op = splits[0]
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model_sections_2, int_model_sections_2, args_np)

    # Test case 2: split with specific indices
    indices_or_sections = [56, 112, 168] # These are split POINTS, not sizes.
                                         # PyTorch split uses split_sizes_or_sections
                                         # which means lengths of each split.
                                         # So lengths would be [56, 112-56, 168-112, 224-168]
    split_sizes = [indices_or_sections[0]]
    for i in range(1, len(indices_or_sections)):
        split_sizes.append(indices_or_sections[i] - indices_or_sections[i-1])
    split_sizes.append(x_shape[axis_val] - indices_or_sections[-1])
    
    def float_model_indices(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        
        splits = torch.split(x_deq, split_sizes, dim=axis_val)
        op = splits[1] # Taking the second split
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model_indices(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        splits = torch.split(q_x, split_sizes, dim=axis_val)
        op = splits[1]
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model_indices, int_model_indices, args_np)


def test_fake_quantize_batch_flatten():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        batch_flatten_float = x_deq.flatten(start_dim=1)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            batch_flatten_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # Flatten on quantized tensors
        int_batch_flatten = q_x.flatten(start_dim=1)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            int_batch_flatten.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_transpose_reshape():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    transpose_axes = [1, 0, 2, 3]
    reshape_new_shape = [3, -1]
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        
        transposed_float = x_deq.permute(transpose_axes)
        reshape_float = torch.reshape(transposed_float, reshape_new_shape)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            reshape_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # These ops directly apply to quantized tensors in PyTorch
        transposed_int = q_x.permute(transpose_axes)
        int_reshape = transposed_int.reshape(reshape_new_shape)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            int_reshape.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_concat():
    input_shapes = [[1, 4]] * 4
    input_scales = [0.5, 1.5, 2.5, 3.5] # i + 0.5
    input_zps = [0] * 4
    output_scale, output_zp = 3.5, 0 # From concat relay.const(3.5)
    axis_val = 1
    out_dtype_str = "int8"

    inputs_np = [np.random.randint(-128, 127, size=shape, dtype="int8") for shape in input_shapes]
    args_np = inputs_np

    def float_model(*input_nps):
        dequantized_inputs = []
        for i, x_val_np in enumerate(input_nps):
            x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
            x_deq = (x_val - input_zps[i]) * input_scales[i]
            dequantized_inputs.append(x_deq)
        
        concat_float = torch.cat(dequantized_inputs, dim=axis_val)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            concat_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(*input_nps):
        quantized_inputs = []
        for i, x_val_np in enumerate(input_nps):
            x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
            q_x = torch.quantize_per_tensor(x_q_val.float(), scale=input_scales[i], zero_point=input_zps[i], dtype=get_q_dtype("int8"))
            quantized_inputs.append(q_x)
        
        # In TVM's pass, qnn.concatenate expects all inputs to have the same scale and zero_point.
        # If they don't, it will either perform dequantize-float_cat-quantize, or some form of requantization.
        # The mapping for `tvm.relay.qnn.op.qnn.concatenate` specifically uses dequantize-float_op-quantize.
        
        dequantized_to_cat = [q_in.dequantize() for q_in in quantized_inputs]
        float_concat = torch.cat(dequantized_to_cat, dim=axis_val)

        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            float_concat, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


@pytest.mark.parametrize("k", [0, 1, 5])
@pytest.mark.parametrize("axis_val", [0, -1, 1])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("dtype_str", ["int8", "uint8"])
def test_fake_quantize_topk(k, axis_val, is_ascend, dtype_str):
    x_shape = [20, 100]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0 # TVM quantize ops output with same params
    ret_type = "values" # TVM topk specifies "values" for this test

    # For k=0, PyTorch topk requires k > 0.
    # TVM allows k=0, which returns empty tensor.
    # We will adjust PyTorch's k to 1 if k=0 to prevent error, but actual check needs to handle empty
    if k == 0:
        pytest.skip("PyTorch torch.topk requires k > 0. TVM allows k=0 for empty tensor.")

    x_np = np.random.randint(0, 127, size=x_shape, dtype=dtype_str)
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype(dtype_str)).float()
        x_deq = (x_val - x_zp) * x_scale
        
        # PyTorch topk uses `largest` instead of `is_ascend`
        largest_val = not is_ascend
        topk_float = torch.topk(x_deq, k=k, dim=axis_val, largest=largest_val, sorted=True).values
        
        output_q_dtype = get_q_dtype(dtype_str)
        return ao_q.quantize_per_tensor(
            topk_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype(dtype_str))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype(dtype_str))

        # PyTorch quantized functional ops do not directly have topk on quantized inputs
        # Dequantize-float_op-quantize
        largest_val = not is_ascend
        topk_float = torch.topk(q_x.dequantize(), k=k, dim=axis_val, largest=largest_val, sorted=True).values
        
        output_q_dtype = get_q_dtype(dtype_str)
        return ao_q.quantize_per_tensor(
            topk_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_clip():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 114
    output_scale, output_zp = 2.0, 114
    clip_min, clip_max = 0, 6
    out_dtype_str = "uint8"

    x_np = np.random.randint(0, 255, size=x_shape, dtype="uint8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8")).float()
        x_deq = (x_val - x_zp) * x_scale
        clip_float = torch.clamp(x_deq, min=clip_min, max=clip_max)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            clip_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("uint8"))

        int_clip = ao_q.quantized.functional.clamp( # torch.ao.quantized.functional has clamp
            q_x,
            min=q_x.quantize_per_tensor(torch.tensor(clip_min, dtype=torch.float32), q_x.q_scale(), q_x.q_zero_point(), q_x.dtype()),
            max=q_x.quantize_per_tensor(torch.tensor(clip_max, dtype=torch.float32), q_x.q_scale(), q_x.q_zero_point(), q_x.dtype()),
        )
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            int_clip.dequantize(), scale=output_scale, zero_point=output_zp, dtype=output_q_dtype # PyTorch clamp on q-tensor keeps qparams
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_clip_per_channel():
    x_shape = [1, 3, 224, 224]
    x_scale_np = np.array([1.0, 2.0, 3.0], dtype="float32")
    x_zp_np = np.array([96, 114, 128], dtype="int32")
    output_scale_np = np.array([1.0, 2.0, 3.0], dtype="float32")
    output_zp_np = np.array([96, 114, 128], dtype="int32")
    clip_min, clip_max = 0, 6
    x_axis = 1 # Channel axis
    out_dtype_str = "uint8"

    x_np = np.random.randint(0, 255, size=x_shape, dtype="uint8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8")).float()
        
        x_scale_tensor = torch.tensor(x_scale_np, dtype=torch.float32)
        x_zp_tensor = torch.tensor(x_zp_np, dtype=torch.int32)
        
        x_scale_bcast = x_scale_tensor.reshape(1, -1, 1, 1)
        x_zp_bcast = x_zp_tensor.reshape(1, -1, 1, 1)

        x_deq = (x_val - x_zp_bcast.float()) * x_scale_bcast
        clip_float = torch.clamp(x_deq, min=clip_min, max=clip_max)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_channel(
            clip_float, scales=torch.tensor(output_scale_np), zero_points=torch.tensor(output_zp_np), axis=x_axis, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8"))
        
        x_scale_tensor = torch.tensor(x_scale_np, dtype=torch.float32)
        x_zp_tensor = torch.tensor(x_zp_np, dtype=torch.int32)

        q_x = ao_q.quantize_per_channel(x_q_val.float(), scales=x_scale_tensor, zero_points=x_zp_tensor, axis=x_axis, dtype=get_q_dtype("uint8"))

        # For per-channel quantization, `q_x.quantize_per_tensor` needs to align scales/zps
        # This is where PyTorch's quantized.functional.clamp gets complicated for per-channel inputs.
        # It's often safer to dequantize -> float_op -> quantize for cases like this.
        
        clip_float = torch.clamp(q_x.dequantize(), min=clip_min, max=clip_max)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_channel(
            clip_float, scales=torch.tensor(output_scale_np), zero_points=torch.tensor(output_zp_np), axis=x_axis, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_relu():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 114
    output_scale, output_zp = 2.0, 114
    out_dtype_str = "uint8"

    x_np = np.random.randint(0, 255, size=x_shape, dtype="uint8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8")).float()
        x_deq = (x_val - x_zp) * x_scale
        relu_float = F.relu(x_deq)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            relu_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("uint8"))

        # PyTorch quantized functional relu
        int_relu = ao_q.quantized.functional.relu(
            q_x,
            scale=output_scale, # Output scale for the quantized relu
            zero_point=output_zp, # Output zero_point for the quantized relu
            dtype=get_q_dtype(out_dtype_str)
        )
        return int_relu.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_mean():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 114
    output_scale, output_zp = 2.0, 114
    out_dtype_str = "uint8"

    x_np = np.random.randint(0, 255, size=x_shape, dtype="uint8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8")).float()
        x_deq = (x_val - x_zp) * x_scale
        mean_float = torch.mean(x_deq)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            mean_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("uint8"))

        # PyTorch quantized functional ops do not directly have mean on quantized inputs
        # Dequantize-float_op-quantize
        mean_float = torch.mean(q_x.dequantize())
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            mean_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


def test_fake_quantize_relu_per_channel():
    x_shape = [1, 3, 224, 224]
    x_scale_np = np.array([1.0, 2.0, 3.0], dtype="float32")
    x_zp_np = np.array([96, 114, 128], dtype="int32")
    output_scale_np = np.array([1.0, 2.0, 3.0], dtype="float32")
    output_zp_np = np.array([96, 114, 128], dtype="int32")
    x_axis = 1 # Channel axis
    out_dtype_str = "uint8"

    x_np = np.random.randint(0, 255, size=x_shape, dtype="uint8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8")).float()
        
        x_scale_tensor = torch.tensor(x_scale_np, dtype=torch.float32)
        x_zp_tensor = torch.tensor(x_zp_np, dtype=torch.int32)
        
        x_scale_bcast = x_scale_tensor.reshape(1, -1, 1, 1)
        x_zp_bcast = x_zp_tensor.reshape(1, -1, 1, 1)

        x_deq = (x_val - x_zp_bcast.float()) * x_scale_bcast
        relu_float = F.relu(x_deq)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_channel(
            relu_float, scales=torch.tensor(output_scale_np), zero_points=torch.tensor(output_zp_np), axis=x_axis, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8"))
        
        x_scale_tensor = torch.tensor(x_scale_np, dtype=torch.float32)
        x_zp_tensor = torch.tensor(x_zp_np, dtype=torch.int32)

        q_x = ao_q.quantize_per_channel(x_q_val.float(), scales=x_scale_tensor, zero_points=x_zp_tensor, axis=x_axis, dtype=get_q_dtype("uint8"))

        # PyTorch quantized functional relu
        int_relu = ao_q.quantized.functional.relu_relu( # relu_relu for per_channel
            q_x,
            scale=torch.tensor(output_scale_np),
            zero_point=torch.tensor(output_zp_np),
            axis=x_axis,
            dtype=get_q_dtype(out_dtype_str)
        )
        return int_relu.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_leaky_relu():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 114
    output_scale, output_zp = 2.0, 114
    alpha = 0.1
    out_dtype_str = "uint8"

    x_np = np.random.randint(0, 255, size=x_shape, dtype="uint8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8")).float()
        x_deq = (x_val - x_zp) * x_scale
        leaky_relu_float = F.leaky_relu(x_deq, negative_slope=alpha)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            leaky_relu_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("uint8"))

        # PyTorch quantized functional leaky_relu is not available, dequantize-float_op-quantize
        leaky_relu_float = F.leaky_relu(q_x.dequantize(), negative_slope=alpha)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            leaky_relu_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)


@pytest.mark.parametrize(
    "operator_func",
    [torch.add, torch.mul, torch.sub, torch.minimum, torch.maximum],
    ids=["add", "multiply", "subtract", "minimum", "maximum"],
)
def test_fake_quantize_binary(operator_func):
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 0.1, 0
    y_scale, y_zp = 0.2, 0
    out_dtype_str = "int8"
    
    # TVM specific logic for multiply output scale
    if operator_func == torch.mul:
        output_scale = 20.0
    else:
        output_scale = 0.1
    output_zp = 0

    x_np = np.random.randint(-25, 25, size=x_shape, dtype="int8")
    y_np = np.random.randint(-25, 25, size=x_shape, dtype="int8")
    args_np = [x_np, y_np]

    def float_model(x_val_np, y_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        y_val = torch.tensor(y_val_np, dtype=to_torch_dtype("int8")).float()

        x_deq = (x_val - x_zp) * x_scale
        y_deq = (y_val - y_zp) * y_scale

        op_float = operator_func(x_deq, y_deq)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, y_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        y_q_val = torch.tensor(y_val_np, dtype=to_torch_dtype("int8"))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_y = torch.quantize_per_tensor(y_q_val.float(), scale=y_scale, zero_point=y_zp, dtype=get_q_dtype("int8"))

        # For binary ops, the qnn.op.* version in TVM (e.g. qnn.add) needs explicit input/output scales/zps.
        # The mapping table suggests dequantize-float_op-quantize.
        op_float = operator_func(q_x.dequantize(), q_y.dequantize())
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


@pytest.mark.parametrize(
    "operator_func",
    [torch.add, torch.mul, torch.sub, torch.minimum, torch.maximum],
    ids=["add", "multiply", "subtract", "minimum", "maximum"],
)
def test_fake_quantize_binary_per_channel(operator_func):
    x_shape = [1, 3, 224, 224]
    out_dtype_str = "int8"

    # Helper function from original TVM test
    def verify_binary_per_channel(lhs_scale_val, rhs_scale_val, lhs_zp_val, rhs_zp_val, out_zp_val, lhs_axis_val, rhs_axis_val):
        if operator_func == torch.mul:
            output_scale_val = 2.0
            rhs_axis_val = lhs_axis_val # TODO: Support different axes for per-channel quantized multiply
        else:
            output_scale_val = 0.1

        x_np = np.random.randint(-25, 25, size=x_shape, dtype="int8")
        y_np = np.random.randint(-25, 25, size=x_shape, dtype="int8")
        args_np = [x_np, y_np]

        def float_model(x_val_np, y_val_np):
            x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
            y_val = torch.tensor(y_val_np, dtype=to_torch_dtype("int8")).float()

            # Ensure scales/zps are tensors and broadcastable
            lhs_scale_tensor = ensure_tensor_params(lhs_scale_val, x_shape, lhs_axis_val, torch.float32)
            lhs_zp_tensor = ensure_tensor_params(lhs_zp_val, x_shape, lhs_axis_val, torch.int32)
            rhs_scale_tensor = ensure_tensor_params(rhs_scale_val, y_shape, rhs_axis_val, torch.float32)
            rhs_zp_tensor = ensure_tensor_params(rhs_zp_val, y_shape, rhs_axis_val, torch.int32)

            x_deq = (x_val - lhs_zp_tensor.float()) * lhs_scale_tensor
            y_deq = (y_val - rhs_zp_tensor.float()) * rhs_scale_tensor

            op_float = operator_func(x_deq, y_deq)
            
            output_q_dtype = get_q_dtype(out_dtype_str)
            return ao_q.quantize_per_tensor( # This assume per_tensor output, not per_channel
                op_float, scale=output_scale_val, zero_point=out_zp_val, dtype=output_q_dtype
            ).dequantize().numpy()

        def int_model(x_val_np, y_val_np):
            x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
            y_q_val = torch.tensor(y_val_np, dtype=to_torch_dtype("int8"))

            lhs_scale_tensor = ensure_tensor_params(lhs_scale_val, x_shape, lhs_axis_val, torch.float32)
            lhs_zp_tensor = ensure_tensor_params(lhs_zp_val, x_shape, lhs_axis_val, torch.int32)
            rhs_scale_tensor = ensure_tensor_params(rhs_scale_val, y_shape, rhs_axis_val, torch.float32)
            rhs_zp_tensor = ensure_tensor_params(rhs_zp_val, y_shape, rhs_axis_val, torch.int32)

            # Quantize_per_channel if the scale/zp is a tensor, otherwise quantize_per_tensor
            q_x = (ao_q.quantize_per_channel(x_q_val.float(), scales=lhs_scale_tensor, zero_points=lhs_zp_tensor, axis=lhs_axis_val, dtype=get_q_dtype("int8"))
                   if lhs_scale_tensor.numel() > 1 else
                   ao_q.quantize_per_tensor(x_q_val.float(), scale=lhs_scale_tensor.item(), zero_point=lhs_zp_tensor.item(), dtype=get_q_dtype("int8")))
            q_y = (ao_q.quantize_per_channel(y_q_val.float(), scales=rhs_scale_tensor, zero_points=rhs_zp_tensor, axis=rhs_axis_val, dtype=get_q_dtype("int8"))
                   if rhs_scale_tensor.numel() > 1 else
                   ao_q.quantize_per_tensor(y_q_val.float(), scale=rhs_scale_tensor.item(), zero_point=rhs_zp_tensor.item(), dtype=get_q_dtype("int8")))

            # PyTorch `torch.ao.nn.quantized.functional` does not have all element-wise binary ops.
            # Use dequantize-float_op-quantize for now, as TVM also might not have optimized QNN ops for all cases.
            op_float = operator_func(q_x.dequantize(), q_y.dequantize())
            
            output_q_dtype = get_q_dtype(out_dtype_str)
            return ao_q.quantize_per_tensor(
                op_float, scale=output_scale_val, zero_point=out_zp_val, dtype=output_q_dtype
            ).dequantize().numpy()

        compare_fq_to_int(float_model, int_model, args_np, allow_rounding_error=True)

    # Same axis
    verify_binary_per_channel(
        lhs_scale_val=np.random.uniform(1.0, 5.0, 3),
        rhs_scale_val=np.random.uniform(1.0, 5.0, 3),
        lhs_zp_val=0,
        rhs_zp_val=0,
        out_zp_val=0,
        lhs_axis_val=1,
        rhs_axis_val=1,
    )
    verify_binary_per_channel(
        lhs_scale_val=np.random.uniform(1.0, 5.0, 3),
        rhs_scale_val=np.random.uniform(1.0, 5.0, 3),
        lhs_zp_val=np.random.randint(1, 3),
        rhs_zp_val=np.random.randint(1, 3),
        out_zp_val=0,
        lhs_axis_val=1,
        rhs_axis_val=1,
    )
    verify_binary_per_channel(
        lhs_scale_val=np.random.uniform(1.0, 5.0, 3),
        rhs_scale_val=np.random.uniform(1.0, 5.0, 3),
        lhs_zp_val=np.random.randint(1, 3),
        rhs_zp_val=np.random.randint(1, 3),
        out_zp_val=np.random.randint(1, 3),
        lhs_axis_val=1,
        rhs_axis_val=1,
    )
    verify_binary_per_channel(
        lhs_scale_val=np.random.uniform(1.0, 5.0, 224),
        rhs_scale_val=np.random.uniform(1.0, 5.0, 224),
        lhs_zp_val=np.random.randint(1, 3),
        rhs_zp_val=np.random.randint(1, 3),
        out_zp_val=np.random.randint(1, 3),
        lhs_axis_val=-1,
        rhs_axis_val=-1,
    )

    # Different axes - these will require more complex broadcasting logic for dequantization
    # PyTorch's `quantize_per_channel` requires a specific axis, if dimensions are different,
    # it becomes a mix of `per_channel` and `per_tensor` equivalent.
    # The `ensure_tensor_params` helper now creates broadcastable tensors for `float_model`.
    # For `int_model`, `quantize_per_channel` should correctly handle the axis.
    verify_binary_per_channel(
        lhs_scale_val=np.random.uniform(1.0, 5.0, 224),
        rhs_scale_val=np.random.uniform(1.0, 5.0, 224),
        lhs_zp_val=0,
        rhs_zp_val=0,
        out_zp_val=0,
        lhs_axis_val=2,
        rhs_axis_val=3,
    )
    verify_binary_per_channel(
        lhs_scale_val=np.random.uniform(1.0, 5.0, 224),
        rhs_scale_val=np.random.uniform(1.0, 5.0, 224),
        lhs_zp_val=np.random.randint(1, 3),
        rhs_zp_val=np.random.randint(1, 3),
        out_zp_val=0,
        lhs_axis_val=2,
        rhs_axis_val=3,
    )
    verify_binary_per_channel(
        lhs_scale_val=np.random.uniform(1.0, 5.0, 224),
        rhs_scale_val=np.random.uniform(1.0, 5.0, 224),
        lhs_zp_val=np.random.randint(1, 3),
        rhs_zp_val=np.random.randint(1, 3),
        out_zp_val=np.random.randint(1, 3),
        lhs_axis_val=2,
        rhs_axis_val=3,
    )


@pytest.mark.parametrize(
    "operator_func",
    [
        torch.add,
        torch.mul,
        torch.sub,
        torch.minimum,
        torch.maximum,
    ],
    ids=["add", "multiply", "subtract", "minimum", "maximum"],
)
def test_fake_quantize_binary_const(operator_func):
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 0.1, 10
    y_const_val = 1.0 # This is a float constant
    output_scale, output_zp = 0.1, 10
    out_dtype_str = "int8"

    x_np = np.random.randint(-25, 25, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        y_const = torch.tensor(y_const_val, dtype=torch.float32)

        op_float = operator_func(x_deq, y_const)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # In TVM, constants are often dequantized to float, the operation occurs, then result quantized.
        # This implies dequantize-float_op-quantize.
        y_const = torch.tensor(y_const_val, dtype=torch.float32)
        op_float = operator_func(q_x.dequantize(), y_const)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            op_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_subtract_different_output_zp():
    x_shape = [1, 128, 128, 3] # NHWC
    x_scale, x_zp = 0.1, 0
    y_const_val = 0.5
    output_scale, output_zp = 0.2, 128
    out_dtype_str = "uint8"
    x_axis = 1 # Channel axis for x_dequantize

    x_np = np.random.randint(0, 255, size=x_shape, dtype="uint8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8")).float()
        
        x_deq = (x_val - x_zp) * x_scale
        y_const = torch.tensor(y_const_val, dtype=torch.float32)

        sub_float = torch.sub(x_deq, y_const)
        transposed_float = sub_float.permute(0, 3, 1, 2) # NHWC to NCHW after sub
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_channel( # output axis for this test implies per_channel
            transposed_float, scales=output_scale, zero_points=output_zp, axis=x_axis, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("uint8"))
        q_x = ao_q.quantize_per_channel(x_q_val.float(), scales=torch.tensor(x_scale), zero_points=torch.tensor(x_zp), axis=x_axis, dtype=get_q_dtype("uint8"))

        y_const = torch.tensor(y_const_val, dtype=torch.float32)

        # Dequantize for sub
        sub_float = torch.sub(q_x.dequantize(), y_const)
        transposed_float = sub_float.permute(0, 3, 1, 2)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_channel(
            transposed_float, scales=torch.tensor(output_scale), zero_points=torch.tensor(output_zp), axis=x_axis, dtype=output_q_dtype
        ).dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np, True)


def test_fake_quantize_pad():
    x_shape = [1, 383, 128]
    x_scale, x_zp = 1.0, 10
    output_scale, output_zp = 1.0, 10
    pad_before = [0, 0, 0] # Matches TVM [0,0] for dim0, [0,1] for dim1, [0,0] for dim2
    pad_after = [0, 1, 0]
    pad_value = 0.0
    out_dtype_str = "int8"

    x_np = np.random.randint(-25, 25, size=x_shape, dtype="int8")
    args_np = [x_np]

    # Convert TVM pad_before, pad_after into PyTorch pad format (last dim first)
    # TVM: ((0,0), (0,1), (0,0)) for 3 dims. For torch.pad, it's (d2_l, d2_r, d1_l, d1_r, d0_l, d0_r)
    # So for `[[0, 0], [0, 1], [0, 0]]` from TVM means `(0,0, 1,0, 0,0)` in PyTorch's format
    pytorch_pad = []
    for before, after in zip(reversed(pad_before), reversed(pad_after)):
        pytorch_pad.extend([before, after])
    
    # Correction: pad_before and pad_after are lists that specify the padding for EACH dimension.
    # TVM: relay.op.nn.pad(x, [[0, 0], [0, 1], [0, 0]], 0.0)
    # This means dimension 0: [0,0], dimension 1: [0,1], dimension 2: [0,0].
    # PyTorch: (d_N_left, d_N_right, ..., d_0_left, d_0_right)
    # So for 3 dims: (d2_l, d2_r, d1_l, d1_r, d0_l, d0_r)
    # In this case: (0,0, 0,1, 0,0) - this is [pad_left, pad_right] for each dimension.
    # The example TVM is pad_width=((0, 0), (0, 1), (0, 0)) -> `pad_width` is a list of pairs.
    # The order is: (0,0) for dim 0, (0,1) for dim 1, (0,0) for dim 2.
    # So for PyTorch it will be: (0,0, 1,0, 0,0)
    
    pad_width_tvm_style = [[0, 0], [0, 1], [0, 0]]
    # PyTorch expects padding in reverse dimension order, flattened
    pytorch_pad_tuple = []
    for dim_pad in reversed(pad_width_tvm_style):
        pytorch_pad_tuple.extend(dim_pad)

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        pad_float = F.pad(x_deq, pad=pytorch_pad_tuple, mode='constant', value=pad_value)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            pad_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # PyTorch quantized functional.pad
        int_pad = ao_q.quantized.functional.pad(
            q_x,
            pad=pytorch_pad_tuple,
            mode='constant',
            value=pad_value,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_pad.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_depth_to_space():
    x_shape = [1, 3, 224, 224]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    block_size = 4
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def float_model(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        x_deq = (x_val - x_zp) * x_scale
        
        # PyTorch F.pixel_shuffle is equivalent to depth_to_space
        depth_to_space_float = F.pixel_shuffle(x_deq, upscale_factor=block_size)
        
        output_q_dtype = get_q_dtype(out_dtype_str)
        return ao_q.quantize_per_tensor(
            depth_to_space_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

        # PyTorch quantized functional.pixel_shuffle is available
        int_depth_to_space = ao_q.quantized.functional.pixel_shuffle(
            q_x,
            upscale_factor=block_size,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(out_dtype_str),
        )
        return int_depth_to_space.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fake_quantize_max_min():
    x_shape = [1, 3, 10, 10]
    x_scale, x_zp = 2.0, 0
    output_scale, output_zp = 2.0, 0
    block_size = 4 # For depth_to_space pre-op
    out_dtype_str = "int8"

    x_np = np.random.randint(-128, 127, size=x_shape, dtype="int8")
    args_np = [x_np]

    def run_test_case(partial_op_func):
        def float_model(x_val_np):
            x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
            x_deq = (x_val - x_zp) * x_scale
            # Pre-op: depth_to_space
            pre_op_res = F.pixel_shuffle(x_deq, upscale_factor=block_size)
            
            main_op_res = partial_op_func(pre_op_res)
            
            output_q_dtype = get_q_dtype(out_dtype_str)
            return ao_q.quantize_per_tensor(
                main_op_res, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
            ).dequantize().numpy()

        def int_model(x_val_np):
            x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
            q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))

            # Pre-op: depth_to_space
            q_pre_op_res = ao_q.quantized.functional.pixel_shuffle(
                q_x,
                upscale_factor=block_size,
                scale=x_scale,
                zero_point=x_zp,
                dtype=get_q_dtype(out_dtype_str)
            )
            
            # For max/min, PyTorch's quantized functional version is not direct
            # so dequantize -> float_op -> quantize
            main_op_res = partial_op_func(q_pre_op_res.dequantize())
            
            output_q_dtype = get_q_dtype(out_dtype_str)
            return ao_q.quantize_per_tensor(
                main_op_res, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
            ).dequantize().numpy()

        compare_fq_to_int(float_model, int_model, args_np)

    # Note: torch.max/min return (values, indices) when dim is specified, so select .values
    run_test_case(lambda x: torch.max(x).values if x.ndim == 0 else torch.max(x)) # For global max/min
    run_test_case(lambda x: torch.min(x).values if x.ndim == 0 else torch.min(x)) # For global min/min

    # Test forwarding kwargs works
    run_test_case(lambda x: torch.max(x, dim=1).values)
    run_test_case(lambda x: torch.min(x, dim=1).values)


def test_fq_avg_pool_conv2d():
    dtype_str = "uint8"
    x_shape = [1, 4, 24, 24]
    w_shape = [8, 4, 1, 1]
    
    x_scale, x_zp = 0.64, 2
    w_scale, w_zp = 0.5, 10
    output_scale, output_zp = 1.0, 0

    x_np = np.random.randint(0, 255, size=x_shape, dtype=dtype_str)
    w_np = np.random.randint(0, 255, size=w_shape, dtype=dtype_str)
    args_np = [x_np, w_np]

    def float_model(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype(dtype_str)).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype(dtype_str)).float()

        # op0
        x_deq = (x_val - x_zp) * x_scale
        # op1
        avg_pool_float = F.avg_pool2d(x_deq, kernel_size=[3, 3])
        # op2
        w_deq = (w_val - w_zp) * w_scale
        # op3
        conv_float = F.conv2d(avg_pool_float, w_deq, kernel_size=[1, 1])
        
        output_q_dtype = get_q_dtype(dtype_str)
        return ao_q.quantize_per_tensor(
            conv_float, scale=output_scale, zero_point=output_zp, dtype=output_q_dtype
        ).dequantize().numpy()

    def int_model(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype(dtype_str))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype(dtype_str))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype(dtype_str))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype(dtype_str))

        # op1: qnn.avg_pool2d
        # PyTorch quantized functional.avg_pool2d
        q_avg_pool = ao_q.quantized.functional.avg_pool2d(
            q_x,
            kernel_size=[3, 3],
            stride=[3,3], # Default stride is kernel_size in TVM unless specified
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
            scale=q_x.q_scale(), # AvgPool usually preserves qparams if not followed by requant
            zero_point=q_x.q_zero_point(),
            dtype=q_x.dtype(),
        )

        # op3: qnn.conv2d
        # PyTorch quantized functional.conv2d
        q_conv = ao_q.quantized.functional.conv2d(
            q_avg_pool,
            q_w,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            scale=output_scale,
            zero_point=output_zp,
            dtype=get_q_dtype(dtype_str),
        )
        return q_conv.dequantize().numpy()

    compare_fq_to_int(float_model, int_model, args_np)


def test_fq_hard_fail():
    pytest.skip("TVM specific `ir.register_op_attr` and exception handling is not convertible.")
    # TODO: This test case relies on TVM's internal IR registration and exception
    # mechanism for graph transformation passes. There is no direct PyTorch equivalent
    # for registering a custom behavior for an operator within a quantization pass
    # at this level of abstraction. The test also asserts on `tvm.ir.structural_equal`
    # which is not applicable.


def test_fq_qat_op_positive_part():
    # Only the first operation is converted, since the next operation("add") is not enabled.
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]
    
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 6.0, 0
    op2_output_scale, op2_output_zp = 12.0, 0 # From x_scale * w_scale
    
    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    args_np = [x_np, w_np]

    # Original expr: dequantize -> dequantize -> batch_matmul -> add(const) -> erf
    def expr_float_func(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        op0 = (x_val - x_zp) * x_scale
        op1 = (w_val - w_zp) * w_scale
        op2 = torch.matmul(op0, op1.transpose(-1, -2)) # Assuming Relay batch_matmul transposes 2nd arg
        op3 = op2 + 1.0
        expr = torch.erf(op3)
        return expr.numpy()

    # Expected expr: qnn.batch_matmul -> qnn.dequantize -> add(const) -> erf
    def expected_expr_int_func(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))
        
        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        # PyTorch quantized functional does not have a direct `batch_matmul`.
        # This implies TVM QAT also might dequantize for batch_matmul for now.
        # However, the `expected_expr` in TVM *explicitly* shows `qnn.op.qnn.batch_matmul`.
        # So we simulate the outcome if `qnn.batch_matmul` existed and produced quantized output.
        # This means intermediate dequantization, then matmul, then requantizing for the `qnn.dequantize` node output.
        
        # Simulating qnn.batch_matmul -> qnn.dequantize as a single float computation:
        op0_float = torch.matmul(q_x.dequantize(), q_w.dequantize().transpose(-1, -2))
        
        # This corresponds to `relay.qnn.op.qnn.dequantize(op0, relay.const(12.0), relay.const(0))`
        # So op0_float is already dequantized value.
        op1 = op0_float # It's already float, so no further dequantize needed here.
        
        op2 = op1 + 1.0
        expected_expr = torch.erf(op2)
        return expected_expr.numpy()

    compare_expected_fq_qat_to_int(expr_float_func, expected_expr_int_func, args_np)


def test_fq_qat_negative_all():
    # None of the operations are converted, since the first operation("add") is not enabled.
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]

    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 6.0, 0
    
    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    args_np = [x_np, w_np]

    # Original expr, same as expected_expr (no conversion expected by pass)
    def expr_float_func(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        op0 = (x_val - x_zp) * x_scale
        op1 = (w_val - w_zp) * w_scale
        op2 = op1 + 1.0 # This "add" is the reason for no conversion in TVM
        op3 = torch.matmul(op0, op2.transpose(-1, -2))
        expr = torch.erf(op3)
        return expr.numpy()

    expected_expr_int_func = expr_float_func # Expected no change

    compare_expected_fq_qat_to_int(expr_float_func, expected_expr_int_func, args_np)


def test_fq_qat_positive_single():
    # The single operation is converted.
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]

    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 6.0, 0
    op2_output_scale, op2_output_zp = 12.0, 0 # From x_scale * w_scale

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    args_np = [x_np, w_np]

    # Original expr: dequantize -> dequantize -> batch_matmul
    def expr_float_func(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        op0 = (x_val - x_zp) * x_scale
        op1 = (w_val - w_zp) * w_scale
        expr = torch.matmul(op0, op1.transpose(-1, -2))
        return expr.numpy()

    # Expected expr: qnn.batch_matmul -> qnn.dequantize
    def expected_expr_int_func(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        # Simulating qnn.batch_matmul -> qnn.dequantize as a single float computation:
        op0_float = torch.matmul(
            torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8")).dequantize(),
            torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8")).dequantize().transpose(-1, -2)
        )
        expected_expr = op0_float # This is the dequantized result
        return expected_expr.numpy()

    compare_expected_fq_qat_to_int(expr_float_func, expected_expr_int_func, args_np)


def test_fq_qat_positive_nothing_to_do():
    # All operations are converted by the non-QAT pass.
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]
    
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 6.0, 0
    op2_output_scale_initial, op2_output_zp_initial = 12.0, 0 # Intermediate (float * float)
    op3_output_scale, op3_output_zp = 1.0, 0 # Final output quantization

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    args_np = [x_np, w_np]

    # Original expr: dequantize -> dequantize -> batch_matmul -> add(const) -> quantize
    def expr_float_func(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        op0 = (x_val - x_zp) * x_scale
        op1 = (w_val - w_zp) * w_scale
        op2_float = torch.matmul(op0, op1.transpose(-1, -2))
        op3_float = op2_float + 1.0
        
        output_q_dtype = get_q_dtype("int8")
        expr = ao_q.quantize_per_tensor(
            op3_float, scale=op3_output_scale, zero_point=op3_output_zp, dtype=output_q_dtype
        ).dequantize().numpy()
        return expr

    # Expected expr: qnn.batch_matmul -> qnn.quantize(const) -> add -> requantize
    def expected_expr_int_func(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        q_x = torch.quantize_per_tensor(x_q_val.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_w = torch.quantize_per_tensor(w_q_val.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        # Simulating qnn.batch_matmul with specific output scale/zp
        # This will be dequantize-float_op-quantize for the batch_matmul
        op0_float = torch.matmul(q_x.dequantize(), q_w.dequantize().transpose(-1, -2))
        op0_quantized = ao_q.quantize_per_tensor(op0_float, scale=op2_output_scale_initial, zero_point=op2_output_zp_initial, dtype=get_q_dtype("int32"))

        # Simulating qnn.quantize(const_1.0) with q_output_scale = 1.0 and q_output_zp = 0
        # This constant needs to be quantified as int32
        const_val = 1.0
        const_q_val = ao_q.quantize_per_tensor(torch.tensor(const_val, dtype=torch.float32), scale=op2_output_scale_initial, zero_point=op2_output_zp_initial, dtype=get_q_dtype("int32"))

        # qnn.add: dequantize inputs, add, quantize result
        op2_add_float = op0_quantized.dequantize() + const_q_val.dequantize()
        op2_add_quantized = ao_q.quantize_per_tensor(op2_add_float, scale=op2_output_scale_initial, zero_point=op2_output_zp_initial, dtype=get_q_dtype("int32"))
        
        # qnn.requantize: changes quantization parameters
        final_float_result = op2_add_quantized.dequantize()
        expected_expr = ao_q.quantize_per_tensor(final_float_result, scale=op3_output_scale, zero_point=op3_output_zp, dtype=get_q_dtype("int8")).dequantize().numpy()
        return expected_expr

    compare_expected_fq_qat_to_int(expr_float_func, expected_expr_int_func, args_np)


def test_fq_qat_positive_couple():
    # Several consecutive operations are converted.
    shape_x = [1, 2, 4]
    shape_w = [2]
    
    x_scale, x_zp = 2.0, 0
    w_scale, w_zp = 6.0, 0
    final_output_scale, final_output_zp = 12.0, 0 # From batch_matmul intermediate product (x_scale * w_scale)

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    args_np = [x_np, w_np]

    # Original expr: dequantize(a) -> dequantize(b) -> reshape(a_deq) -> broadcast_to(b_deq) -> batch_matmul -> erf
    def expr_float_func(x_val_np, w_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()
        w_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8")).float()

        op0 = (x_val - x_zp) * x_scale
        op1 = (w_val - w_zp) * w_scale
        op2 = op0.reshape((1, 4, 2))
        op3 = torch.broadcast_to(op1, (2, 2, 2))
        op4 = torch.matmul(op2, op3.transpose(-1, -2)) # Matmul with implicit transpose for B
        expr = torch.erf(op4)
        return expr.numpy()

    # Expected expr: reshape(a) -> broadcast_to(b) -> qnn.batch_matmul -> qnn.dequantize -> erf
    def expected_expr_int_func(x_val_np, w_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        w_q_val = torch.tensor(w_val_np, dtype=to_torch_dtype("int8"))

        # The initial reshape/broadcast ops in TVM are on the *quantized* tensors
        op0_reshaped_q = x_q_val.reshape((1, 4, 2))
        op1_bcast_q = torch.broadcast_to(w_q_val, (2, 2, 2))

        # qnn.batch_matmul from TVM needs quantized inputs
        q_op0 = torch.quantize_per_tensor(op0_reshaped_q.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8"))
        q_op1 = torch.quantize_per_tensor(op1_bcast_q.float(), scale=w_scale, zero_point=w_zp, dtype=get_q_dtype("int8"))

        # Simulating qnn.batch_matmul -> qnn.dequantize as dequantize-float_op
        op3_float = torch.matmul(q_op0.dequantize(), q_op1.dequantize().transpose(-1, -2))
        
        op4_float = torch.erf(op3_float)
        return op4_float.numpy()

    compare_expected_fq_qat_to_int(expr_float_func, expected_expr_int_func, args_np)


def test_fq_positive_single_arg_part():
    # The single-argument operation is converted.
    shape_x = [1, 2, 4]
    x_scale, x_zp = 2.0, 0

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    args_np = [x_np]

    # Original expr: dequantize -> reshape -> erf
    def expr_float_func(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8")).float()

        op0 = (x_val - x_zp) * x_scale
        op1 = op0.reshape((1, 4, 2))
        expr = torch.erf(op1)
        return expr.numpy()

    # Expected expr: reshape -> qnn.dequantize -> erf
    def expected_expr_int_func(x_val_np):
        x_q_val = torch.tensor(x_val_np, dtype=to_torch_dtype("int8"))
        
        op0_reshaped_q = x_q_val.reshape((1, 4, 2))
        
        # qnn.dequantize: dequantizes the tensor
        op1_float = torch.quantize_per_tensor(op0_reshaped_q.float(), scale=x_scale, zero_point=x_zp, dtype=get_q_dtype("int8")).dequantize()

        expected_expr = torch.erf(op1_float)
        return expected_expr.numpy()

    compare_expected_fq_qat_to_int(expr_float_func, expected_expr_int_func, args_np)


def test_fq_qat_intermediate_infertype():
    # Complex conversion of non-QAT and QAT passes that form FakeQuantizationToInteger.
    shape_x = [1, 2, 4]
    const_0_shape = [1, 4, 2]
    
    # Quantize/dequantize for x
    x_quant_scale, x_quant_zp = 17.0, 0
    x_dequant_scale, x_dequant_zp = 17.0, 0
    x_reshape_quant_scale, x_reshape_quant_zp = 10.0, 0 # After reshape

    # Quantize/dequantize for const_0
    const_0_quant_scale, const_0_quant_zp = 1.0, 8
    const_0_dequant_scale, const_0_dequant_zp = 4.0, 9

    # Output of batch_matmul will be (10.0 * 4.0 = 40.0) scale and (0) zp
    batch_matmul_out_scale, batch_matmul_out_zp = 40.0, 0


    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int32").astype("float32")
    const_0_np = np.random.uniform(size=const_0_shape).astype("float32")
    args_np = [x_np]

    # Original expr (Float ops with fake quantization)
    def expr_float_func(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=torch.float32)
        const_0 = torch.tensor(const_0_np, dtype=torch.float32)

        op0 = ao_q.quantize_per_tensor(x_val, scale=x_quant_scale, zero_point=x_quant_zp, dtype=get_q_dtype("int8"))
        op1 = op0.dequantize()
        op2 = op1.reshape((1, 4, 2))
        op3 = ao_q.quantize_per_tensor(op2, scale=x_reshape_quant_scale, zero_point=x_reshape_quant_zp, dtype=get_q_dtype("int8"))
        op4 = ao_q.quantize_per_tensor(const_0, scale=const_0_quant_scale, zero_point=const_0_quant_zp, dtype=get_q_dtype("int8"))
        op5 = op3.dequantize()
        op6 = op4.dequantize() # The scale/zp here should be const_0_dequant_scale, const_0_dequant_zp
        
        op7 = torch.matmul(op5, op6.transpose(-1,-2)) # Matmul with implicit transpose for B
        expr = op7 + 5.0
        return expr.numpy()

    # Expected expr (Integer quantized ops)
    def expected_expr_int_func(x_val_np):
        x_val = torch.tensor(x_val_np, dtype=torch.float32)
        const_0 = torch.tensor(const_0_np, dtype=torch.float32)

        # op0: qnn.quantize(x)
        op0_q = ao_q.quantize_per_tensor(x_val, scale=x_quant_scale, zero_point=x_quant_zp, dtype=get_q_dtype("int8"))
        # op1: reshape(op0)
        op1_reshaped_q = op0_q.reshape((1, 4, 2))
        # op2: qnn.requantize(op1, x_dequant_scale, x_dequant_zp, x_reshape_quant_scale, x_reshape_quant_zp)
        # Requantize `op1_reshaped_q` (which has scale x_quant_scale, zp x_quant_zp)
        # to target scale x_reshape_quant_scale, zp x_reshape_quant_zp
        op2_requantized_q = ao_q.quantize_per_tensor(
            op1_reshaped_q.dequantize(), 
            scale=x_reshape_quant_scale, zero_point=x_reshape_quant_zp, dtype=get_q_dtype("int8")
        )
        # op3: qnn.quantize(const_0)
        op3_const_q = ao_q.quantize_per_tensor(
            const_0, scale=const_0_quant_scale, zero_point=const_0_quant_zp, dtype=get_q_dtype("int8")
        )
        # op4: qnn.batch_matmul(op2, op3)
        # Simulating qnn.batch_matmul from TVM (dequantize inputs, matmul, then quantize)
        op4_float = torch.matmul(op2_requantized_q.dequantize(), op3_const_q.dequantize().transpose(-1,-2))
        op4_quantized = ao_q.quantize_per_tensor(op4_float, scale=batch_matmul_out_scale, zero_point=batch_matmul_out_zp, dtype=get_q_dtype("int32"))

        # op5: qnn.dequantize(op4)
        op5_dequantized = op4_quantized.dequantize()
        
        expected_expr = op5_dequantized + 5.0
        return expected_expr.numpy()

    compare_expected_fq_qat_to_int(expr_float_func, expected_expr_int_func, args_np)


# No direct equivalent for tvm.testing.main() in PyTorch.
# Tests run automatically by pytest.
