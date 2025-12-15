import pytest
import numpy as np
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay.op import nn
from tvm.relay.op import tensor
from tvm.relay.op import transform
from tvm.relay.op import algorithm
from tvm.relay.op import image
import itertools

# --- Helper functions for TVM graph execution and assertion ---

def _get_tvm_outputs_internal(mod, *args_tvm):
    """
    Builds and executes a TVM Relay module, returning its outputs as NumPy arrays.
    """
    target = tvm.target.Target("llvm", host="llvm") # Default target
    dev = tvm.cpu(0) # Default device

    params = {} # No external parameters needed for these tests

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    for i, arg in enumerate(args_tvm):
        rt_mod.set_input(i, arg)
    rt_mod.run()
    
    num_outputs = rt_mod.get_num_outputs()
    if num_outputs > 1:
        return [rt_mod.get_output(i).numpy() for i in range(num_outputs)]
    else:
        return rt_mod.get_output(0).numpy()

def _get_tvm_output_shape(mod, *args_tvm):
    """
    Helper to get the shape of the first output without running a full numerical check.
    """
    target = tvm.target.Target("llvm", host="llvm")
    dev = tvm.cpu(0)
    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    for i, arg in enumerate(args_tvm):
        rt_mod.set_input(i, arg)
    rt_mod.run()
    return rt_mod.get_output(0).shape


def _check_tvm_operator_internal(
    tvm_model_func,  # A function that takes relay.Var(s) and returns a Relay expression
    inputs_np,       # List of NumPy arrays as inputs
    expected_outputs_np_or_list, # Single NumPy array or list of NumPy arrays as expected outputs
    rtol=1e-5,
    atol=1e-8,
):
    """
    Compares the numerical output of a TVM Relay graph with expected NumPy outputs.
    """
    # Construct the Relay graph
    relay_inputs = []
    tvm_inputs_nd = []
    for i, inp_np in enumerate(inputs_np):
        shape = inp_np.shape
        dtype = inp_np.dtype.name
        var = relay.var(f"p{i}", shape=shape, dtype=dtype)
        relay_inputs.append(var)
        tvm_inputs_nd.append(tvm.nd.array(inp_np, device=tvm.cpu(0)))

    relay_output = tvm_model_func(*relay_inputs)
    func = relay.Function(relay_inputs, relay_output)
    mod = tvm.IRModule.from_expr(func)

    tvm_actual_outputs = _get_tvm_outputs_internal(mod, *tvm_inputs_nd)

    if isinstance(expected_outputs_np_or_list, list):
        assert isinstance(tvm_actual_outputs, list), f"Expected list of outputs, got {type(tvm_actual_outputs)}"
        assert len(tvm_actual_outputs) == len(expected_outputs_np_or_list), f"Expected {len(expected_outputs_np_or_list)} outputs, got {len(tvm_actual_outputs)}"
        for actual, expected in zip(tvm_actual_outputs, expected_outputs_np_or_list):
            tvm.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    else:
        tvm.testing.assert_allclose(tvm_actual_outputs, expected_outputs_np_or_list, rtol=rtol, atol=atol)

class TestONNXOpset:
    def test_opset_fallback(self):
        def my_module_tvm(x_relay):
            return tensor.isnan(x_relay)

        x_np = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        expected_output_np = np.array([False, True, False], dtype=np.bool_)
        _check_tvm_operator_internal(my_module_tvm, [x_np], expected_output_np)

    def test_topk(self):
        # Static k=3 case (PyTorch `torch.topk(x, 3)` means k=3, largest=True, sorted=True)
        def my_module_static_k_tvm(x_relay):
            # TVM topk needs ret_type='both' to return values and indices.
            # is_ascend=False for largest=True (descending order).
            # Default output dtype for indices is int64 in TVM.
            return algorithm.topk(x_relay, k=3, axis=-1, ret_type='both', is_ascend=False, dtype='int64')

        x_np = np.arange(1.0, 6.0, dtype=np.float32) # [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_values_np = np.array([5.0, 4.0, 3.0], dtype=np.float32)
        expected_indices_np = np.array([4, 3, 2], dtype=np.int64) 

        _check_tvm_operator_internal(my_module_static_k_tvm, [x_np], [expected_values_np, expected_indices_np])

        # Test with dynamic k (PyTorch input `k` is a tensor, but `topk` expects static `k` in TVM)
        # The PyTorch ONNX exporter treats `k` from `torch.tensor(3)` as a constant.
        x_np_dynamic = np.arange(1.0, 6.0, dtype=np.float32)
        k_val_for_tvm = 3 # `k` is a Python integer literal for TVM's topk operator

        def my_module_dynamic_k_tvm_graph(input_relay):
            # The 'k' parameter to topk is a Python integer or None, not a Relay expression.
            return algorithm.topk(input_relay, k=k_val_for_tvm, axis=-1, ret_type='both', is_ascend=False, dtype='int64')
            
        expected_values_np_dynamic = np.array([5.0, 4.0, 3.0], dtype=np.float32)
        expected_indices_np_dynamic = np.array([4, 3, 2], dtype=np.int64)
        _check_tvm_operator_internal(
            my_module_dynamic_k_tvm_graph,
            [x_np_dynamic],
            [expected_values_np_dynamic, expected_indices_np_dynamic]
        )

    def test_maxpool(self):
        # Case 1: MaxPool1d(kernel_size=2, stride=1)
        x_np = np.random.randn(20, 16, 50).astype(np.float32) # N, C, W input layout (typical for MaxPool1d)

        def my_maxpool1d_tvm(data_relay):
            # MaxPool1d expects N, C, L. TVM's `max_pool1d` takes `layout="NCW"`.
            return nn.max_pool1d(data_relay, pool_size=(2,), strides=(1,), layout="NCW")

        data_var = relay.var("data", shape=x_np.shape, dtype=x_np.dtype.name)
        relay_output = my_maxpool1d_tvm(data_var)
        func = relay.Function([data_var], relay_output)
        mod = tvm.IRModule.from_expr(func)
        
        # Calculate expected output shape
        # L_out = (L_in - kernel_size) / stride + 1 = (50 - 2) / 1 + 1 = 49
        expected_shape_c1 = (20, 16, 49)
        actual_shape = _get_tvm_output_shape(mod, tvm.nd.array(x_np, device=tvm.cpu(0)))
        assert actual_shape == expected_shape_c1
        # TODO: Add numerical assertion for max_pool1d once a reliable PyTorch-free ground truth can be established.

        # Case 2: MaxPool1d(kernel_size=2, stride=1, dilation=2)
        def my_maxpool1d_dilation_tvm(data_relay):
            return nn.max_pool1d(data_relay, pool_size=(2,), strides=(1,), dilations=(2,), layout="NCW")

        data_var_d = relay.var("data", shape=x_np.shape, dtype=x_np.dtype.name)
        relay_output_d = my_maxpool1d_dilation_tvm(data_var_d)
        func_d = relay.Function([data_var_d], relay_output_d)
        mod_d = tvm.IRModule.from_expr(func_d)

        # Calculate expected output shape with dilation
        # L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        # L_out = (50 + 2*0 - 2 * (2 - 1) - 1) / 1 + 1 = (50 - 2 - 1) + 1 = 48
        expected_shape_c2 = (20, 16, 48)
        actual_shape_d = _get_tvm_output_shape(mod_d, tvm.nd.array(x_np, device=tvm.cpu(0)))
        assert actual_shape_d == expected_shape_c2
        # TODO: Add numerical assertion for max_pool1d with dilation.

    def test_upsample(self):
        # Using a small input for deterministic output and ease of manual verification
        x_small_np = np.array([[[[1., 2.], [3., 4.]]]], dtype=np.float32) # (1, 1, 2, 2)
        
        # PyTorch calculation from `x.size()[2:]` would be `(2, 2)`. Multiplied by 2 gives `(4, 4)`.
        target_size_hw = (4, 4)
        
        def my_upsample_tvm(x_relay):
            # tvm.relay.op.image.resize2d takes `size` as (height, width).
            # PyTorch `mode="nearest"` maps to TVM `method="nearest"`.
            return image.resize2d(x_relay, size=target_size_hw, layout="NCHW", method="nearest")

        # Manually calculated nearest neighbor upsample output
        expected_small_output_np = np.array([[[[1., 1., 2., 2.],
                                                [1., 1., 2., 2.],
                                                [3., 3., 4., 4.],
                                                [3., 3., 4., 4.]]]], dtype=np.float32)

        _check_tvm_operator_internal(
            my_upsample_tvm,
            [x_small_np],
            expected_small_output_np
        )

    def test_cast_constant(self):
        def my_module_tvm(x_relay):
            # torch.long maps to 'int64'. Constant 1 needs to be of the same dtype for subtraction.
            return tensor.subtract(x_relay, relay.const(1, dtype=x_relay.dtype))

        x_np = np.ones((5, 6), dtype=np.int64) # torch.ones(..., dtype=torch.long)
        expected_output_np = x_np - 1 # Expecting all zeros
        _check_tvm_operator_internal(my_module_tvm, [x_np], expected_output_np)

    def test_slice(self):
        # Case 1: Static slice x[0:1]
        def my_module_static_slice_tvm(x_relay):
            # Slice 0:1 on axis 0
            return transform.strided_slice(x_relay, begin=[0], end=[1], strides=[1], axes=[0])

        x_np = np.arange(3, dtype=np.float32) # [0., 1., 2.]
        expected_output_np = np.array([0.], dtype=np.float32)
        _check_tvm_operator_internal(my_module_static_slice_tvm, [x_np], expected_output_np)

        # Case 2: Dynamic slice x[1 : x.size(0)]
        x_dynamic_np = np.random.rand(1, 2).astype(np.float32) # PyTorch input: torch.rand(1, 2)

        def my_module_dynamic_slice_tvm(x_relay):
            # x.size(0) -> get 0-th dimension from shape. `shape_of` returns a 1D tensor [dim0, dim1, ...].
            # `strided_slice` extracts the first element.
            shape_of_x = transform.shape_of(x_relay)
            dim0_size = transform.strided_slice(shape_of_x, begin=relay.const([0], "int64"), end=relay.const([1], "int64"), strides=relay.const([1], "int64"), axes=relay.const([0], "int64"))
            
            # strided_slice expects begin, end, strides, axes to be 1D tensors/lists of ints.
            return transform.strided_slice(
                x_relay,
                begin=relay.const([1], dtype="int64"),
                end=dim0_size, # This is a Relay Expression representing `[1]`
                strides=relay.const([1], dtype="int64"),
                axes=relay.const([0], dtype="int64")
            )
        
        # x[1 : x.size(0)] on a (1, 2) tensor. x.size(0) is 1.
        # Slice from index 1 to 1 (exclusive end). This results in an empty slice.
        expected_output_dynamic_np = np.empty((0, 2), dtype=np.float32) # Shape (0, 2)

        _check_tvm_operator_internal(
            my_module_dynamic_slice_tvm,
            [x_dynamic_np],
            expected_output_dynamic_np
        )

    def test_flip(self):
        def my_module_tvm(x_relay):
            # torch.flip(x, dims=[0]) maps to tvm.relay.op.transform.reverse with axis=0
            return transform.reverse(x_relay, axis=0)

        x_np = np.arange(6.0, dtype=np.float32).reshape(2, 3) # [[0., 1., 2.], [3., 4., 5.]]
        expected_output_np = np.array([[3., 4., 5.], [0., 1., 2.]], dtype=np.float32)
        _check_tvm_operator_internal(my_module_tvm, [x_np], expected_output_np)

    def test_dropout(self):
        x_np = np.random.randn(1, 2, 3).astype(np.float32)
        rate_val = 0.5

        # Test TRAINING mode (Dropout is active)
        def my_dropout_training_tvm(x_relay):
            # nn.dropout returns a tuple (output, mask). We test the first element (output).
            dropout_output, _ = nn.dropout(x_relay, rate=rate_val)
            return dropout_output

        tvm_inputs_nd = [tvm.nd.array(x_np, device=tvm.cpu(0))]
        relay_output_training = my_dropout_training_tvm(relay.var("x", shape=x_np.shape, dtype=x_np.dtype.name))
        func_training = relay.Function([relay.var("x", shape=x_np.shape, dtype=x_np.dtype.name)], relay_output_training)
        mod_training = tvm.IRModule.from_expr(func_training)
        
        tvm_output_training_np = _get_tvm_outputs_internal(mod_training, *tvm_inputs_nd)
        assert tvm_output_training_np.shape == x_np.shape
        assert tvm_output_training_np.dtype == x_np.dtype
        # TODO: Add statistical test for dropout, e.g., verify that values are scaled and some are zeroed,
        # if a deterministic RNG is available and exposed for TVM. Current check is only for shape/dtype.

        # Test EVAL mode (Dropout is a no-op, acts as Identity)
        def my_dropout_eval_tvm(x_relay):
            # In eval mode, PyTorch dropout is an identity op.
            return x_relay

        expected_output_eval_np = x_np # Identity operation
        _check_tvm_operator_internal(my_dropout_eval_tvm, [x_np], expected_output_eval_np)

    def test_full(self):
        fill_value_np = np.array(12.0, dtype=np.float32) # torch.tensor(12.0) is float32 by default
        target_shape = (3, 4)

        def my_full_tvm(x_scalar_relay):
            # x_scalar_relay represents the fill_value (12.0)
            return transform.full(x_scalar_relay, shape=target_shape, dtype=x_scalar_relay.dtype)

        expected_output_np = np.full(target_shape, fill_value_np, dtype=np.float32)
        _check_tvm_operator_internal(my_full_tvm, [fill_value_np], expected_output_np)

    def test_interpolate(self):
        # Using a small input for deterministic output and ease of manual verification
        x_small_np = np.array([[[[1., 2.], [3., 4.]]]], dtype=np.float32) # (1, 1, 2, 2)
        
        # PyTorch calculation: `size = [v * 2 for v in x.size()[2:]]`
        # For x_small_np (1,1,2,2), x.shape[2:] is (2, 2). Multiplied by 2 gives `(4, 4)`.
        target_size_hw = (4, 4)
        
        def my_interpolate_tvm(x_relay):
            # tvm.relay.op.image.resize2d takes `size` as (height, width).
            # PyTorch `mode="nearest"` maps to TVM `method="nearest"`.
            return image.resize2d(x_relay, size=target_size_hw, layout="NCHW", method="nearest")

        # Manually calculated nearest neighbor upsample output
        expected_small_output_np = np.array([[[[1., 1., 2., 2.],
                                                [1., 1., 2., 2.],
                                                [3., 3., 4., 4.],
                                                [3., 3., 4., 4.]]]], dtype=np.float32)

        _check_tvm_operator_internal(
            my_interpolate_tvm,
            [x_small_np],
            expected_small_output_np
        )

    def test_affine_grid(self):
        # Helper to extract target_shape from PyTorch's `size` tuple (N, C, H, W) or (N, C, D, H, W)
        def extract_target_shape(size_tuple):
            if len(size_tuple) == 4: # 2D case: N, C, H, W -> (H, W)
                return (int(size_tuple[2]), int(size_tuple[3]))
            elif len(size_tuple) == 5: # 3D case: N, C, D, H, W -> (D, H, W)
                return (int(size_tuple[2]), int(size_tuple[3]), int(size_tuple[4]))
            else:
                raise ValueError("Unsupported size dimension for affine_grid")

        # 2D affine example from PyTorch test
        theta_2d_np = np.random.rand(1, 2, 3).astype(np.float64) # N, 2, 3
        size_2d_torch = (1, 1, 2, 2) # N, C, H, W
        target_shape_2d = extract_target_shape(size_2d_torch) # (2, 2)

        # Expected output shape for 2D `affine_grid`: (N, H_out, W_out, 2)
        expected_output_shape_2d = (theta_2d_np.shape[0], target_shape_2d[0], target_shape_2d[1], 2)
        
        def my_affine_grid_2d_tvm(theta_relay):
            # PyTorch's `align_corners` is not a direct parameter in TVM's `affine_grid`
            # Assume TVM's default behavior is compatible or note mismatch.
            return image.affine_grid(theta_relay, target_shape=target_shape_2d)

        theta_2d_var = relay.var("theta", shape=theta_2d_np.shape, dtype=theta_2d_np.dtype.name)
        relay_output_2d = my_affine_grid_2d_tvm(theta_2d_var)
        func_2d = relay.Function([theta_2d_var], relay_output_2d)
        mod_2d = tvm.IRModule.from_expr(func_2d)
        actual_output_shape_2d = _get_tvm_output_shape(mod_2d, tvm.nd.array(theta_2d_np, device=tvm.cpu(0)))
        assert actual_output_shape_2d == expected_output_shape_2d
        # TODO: Add numerical assertion for affine_grid. This currently only verifies shape.

        # 3D affine example from PyTorch test
        theta_3d_np = np.random.rand(1, 3, 4).astype(np.float64) # N, 3, 4
        size_3d_torch = (1, 1, 2, 2, 2) # N, C, D, H, W
        target_shape_3d = extract_target_shape(size_3d_torch) # (2, 2, 2)

        # Expected output shape for 3D `affine_grid`: (N, D_out, H_out, W_out, 3)
        expected_output_shape_3d = (theta_3d_np.shape[0], target_shape_3d[0], target_shape_3d[1], target_shape_3d[2], 3)

        def my_affine_grid_3d_tvm(theta_relay):
            # PyTorch's `align_corners` is not a direct parameter in TVM's `affine_grid`
            return image.affine_grid(theta_relay, target_shape=target_shape_3d)

        theta_3d_var = relay.var("theta", shape=theta_3d_np.shape, dtype=theta_3d_np.dtype.name)
        relay_output_3d = my_affine_grid_3d_tvm(theta_3d_var)
        func_3d = relay.Function([theta_3d_var], relay_output_3d)
        mod_3d = tvm.IRModule.from_expr(func_3d)
        actual_output_shape_3d = _get_tvm_output_shape(mod_3d, tvm.nd.array(theta_3d_np, device=tvm.cpu(0)))
        assert actual_output_shape_3d == expected_output_shape_3d
        # TODO: Add numerical assertion for affine_grid. This currently only verifies shape.

    def test_grid_sample(self):
        # Pick one representative permutation for 2D (bilinear, zeros, align_corners=True)
        mode_2d = "bilinear"
        padding_mode_2d = "zeros"
        align_corners_2d = True
        x_shape_2d = (1, 1, 4, 4) # N, C, H_in, W_in
        grid_shape_2d = (1, 2, 2, 2) # N, H_out, W_out, 2 (for (x,y) coords)

        # Using a simple input for deterministic output and shape check
        x_np_2d = np.arange(1, 17, dtype=np.float32).reshape(x_shape_2d)
        grid_np_2d = np.array([[[
            [-0.5, -0.5], [0.5, -0.5]],
            [[-0.5, 0.5], [0.5, 0.5]]
        ]]], dtype=np.float32) # Normalized grid coordinates

        # Expected output shape for 2D `grid_sample`: (N, C, H_out, W_out)
        expected_output_shape_2d = (x_shape_2d[0], x_shape_2d[1], grid_shape_2d[1], grid_shape_2d[2]) # (1, 1, 2, 2)

        def my_grid_sample_2d_tvm(x_relay, grid_relay):
            return image.grid_sample(
                x_relay,
                grid_relay,
                method=mode_2d,
                layout="NCHW",
                padding_mode=padding_mode_2d,
                align_corners=align_corners_2d,
            )
        
        x_var_2d = relay.var("x", shape=x_shape_2d, dtype=x_np_2d.dtype.name)
        grid_var_2d = relay.var("grid", shape=grid_shape_2d, dtype=grid_np_2d.dtype.name)
        
        relay_output_2d = my_grid_sample_2d_tvm(x_var_2d, grid_var_2d)
        func_2d = relay.Function([x_var_2d, grid_var_2d], relay_output_2d)
        mod_2d = tvm.IRModule.from_expr(func_2d)
        actual_output_shape_2d = _get_tvm_output_shape(mod_2d, tvm.nd.array(x_np_2d), tvm.nd.array(grid_np_2d))
        assert actual_output_shape_2d == expected_output_shape_2d
        # TODO: Add numerical assertion for grid_sample. Computing grid_sample output manually
        # for arbitrary grid/modes is complex, usually requires golden output from PyTorch.

        # Pick one representative permutation for 3D (nearest, border, align_corners=False)
        # Note: PyTorch's `bicubic` mode for 3D `grid_sample` is opset >= 20.
        mode_3d = "nearest" 
        padding_mode_3d = "border"
        align_corners_3d = False
        x_shape_3d = (1, 1, 2, 3, 2) # N, C, D_in, H_in, W_in
        grid_shape_3d = (1, 3, 2, 4, 3) # N, D_out, H_out, W_out, 3 (for (z,y,x) coords)
        
        x_np_3d = np.random.randn(*x_shape_3d).astype(np.float32)
        grid_np_3d = np.random.randn(*grid_shape_3d).astype(np.float32)

        # Expected output shape for 3D `grid_sample`: (N, C, D_out, H_out, W_out)
        expected_output_shape_3d = (x_shape_3d[0], x_shape_3d[1], grid_shape_3d[1], grid_shape_3d[2], grid_shape_3d[3])

        def my_grid_sample_3d_tvm(x_relay, grid_relay):
            return image.grid_sample(
                x_relay,
                grid_relay,
                method=mode_3d,
                layout="NCDHW",
                padding_mode=padding_mode_3d,
                align_corners=align_corners_3d,
            )

        x_var_3d = relay.var("x", shape=x_shape_3d, dtype=x_np_3d.dtype.name)
        grid_var_3d = relay.var("grid", shape=grid_shape_3d, dtype=grid_np_3d.dtype.name)
        
        relay_output_3d = my_grid_sample_3d_tvm(x_var_3d, grid_var_3d)
        func_3d = relay.Function([x_var_3d, grid_var_3d], relay_output_3d)
        mod_3d = tvm.IRModule.from_expr(func_3d)
        actual_output_shape_3d = _get_tvm_output_shape(mod_3d, tvm.nd.array(x_np_3d), tvm.nd.array(grid_np_3d))
        assert actual_output_shape_3d == expected_output_shape_3d
        # TODO: Add numerical assertion for 3D grid_sample.

    def test_flatten(self):
        # Case 1: 0D tensor (scalar)
        x_0d_np = np.array(42.0, dtype=np.float32) 
        # torch.flatten on 0D tensor returns 1D tensor of size 1 (e.g., [42.0])
        expected_0d_output_np = np.array([42.0], dtype=np.float32)
        
        def my_flatten_0d_tvm(x_relay):
            return transform.reshape(x_relay, newshape=(1,))
        
        _check_tvm_operator_internal(my_flatten_0d_tvm, [x_0d_np], expected_0d_output_np)

        # Case 2: 1D tensor
        x_1d_np = np.random.randn(3).astype(np.float32)
        # torch.flatten on 1D tensor returns itself (shape remains (3,))
        expected_1d_output_np = x_1d_np # Identity
        
        def my_flatten_1d_tvm(x_relay):
            # Reshaping a 1D tensor to (-1,) preserves its shape (e.g., (3,) -> (3,))
            return transform.reshape(x_relay, newshape=(-1,))
        
        _check_tvm_operator_internal(my_flatten_1d_tvm, [x_1d_np], expected_1d_output_np)

        # Case 3: Higher-dim tensor (e.g., 2D) flattens to 1D
        x_2d_np = np.random.randn(2, 3).astype(np.float32)
        expected_2d_output_np = x_2d_np.flatten() # Flattens to (6,)
        
        def my_flatten_2d_tvm(x_relay):
            return transform.reshape(x_relay, newshape=(-1,))
            
        _check_tvm_operator_internal(my_flatten_2d_tvm, [x_2d_np], expected_2d_output_np)
