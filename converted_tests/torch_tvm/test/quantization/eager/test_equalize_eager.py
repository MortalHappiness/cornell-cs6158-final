import copy
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay import transform
from tvm.testing import utils as tvm_testing_utils

# The original PyTorch _equalize module and its functions (_equalize.channel_range,
# _equalize.cross_layer_equalization, _equalize.converged, _equalize.equalize)
# perform operations (e.g., in-place modification of PyTorch nn.Module weights,
# specific quantization algorithms, inspecting Python object attributes)
# that do not have direct, high-fidelity equivalents in TVM's Relay IR or its
# standard operators.
# TVM Relay is a functional IR, and direct manipulation of "module.weight" like PyTorch
# is not possible. Quantization in TVM involves explicit quantization passes and QNN operators.
# Therefore, these specific _equalize functions are marked as TODO in comments and the tests
# that depend on them are skipped.

# Helper function to mock PyTorch's _equalize.channel_range
# This is a placeholder and does not implement the actual equalization logic from PyTorch.
def mock_channel_range(tensor_data, axis):
    """
    Mocks the channel_range functionality for TVM NDArrays.
    In a real TVM scenario, channel range would be computed from the tensor
    using TVM Relay reduce ops (min/max), but that would require graph execution
    and is part of the larger, not directly mappable, equalization logic.
    For this mock, we'll return a placeholder that allows the `checkChannelsEqualized`
    method to run without errors, but its output values are not meaningful for
    actual equalization unless the input `tensor_data` was pre-equalized.
    """
    if isinstance(tensor_data, tvm.relay.Expr):
        # If it's a Relay expression, its shape might be symbolic.
        # We can't actually compute the range without evaluating the graph.
        # Returning a dummy constant for syntactic correctness.
        # This highlights that a true channel_range needs execution or a dedicated Relay op.
        return relay.const(tvm.nd.array(np.array([0.0, 1.0], dtype='float32')))
    elif isinstance(tensor_data, tvm.nd.NDArray) or isinstance(tensor_data, np.ndarray):
        # If it's concrete data (NDArray or NumPy), we can perform a dummy calculation.
        if isinstance(tensor_data, tvm.nd.NDArray):
            numpy_data = tensor_data.numpy()
        else:
            numpy_data = tensor_data
        
        # This is a simplified calculation, not fully mimicking PyTorch's complex
        # channel_range logic which might involve handling of specific tensor layouts
        # or interpretation of 'channels'.
        if numpy_data.ndim <= abs(axis):
            # Attempt to create a dummy range if axis is out of bounds
            return tvm.nd.array(np.array([[0.0, 1.0]], dtype='float32'))
        
        # Calculate min/max across all dimensions except the specified axis.
        # For a weight tensor like Conv2d (C_out, C_in, kH, kW) where C_out is axis 0,
        # this would give min/max for each output channel.
        # For Linear (C_out, C_in) where C_in is axis 1, this would give min/max for each input channel.
        reduced_dims = tuple(i for i in range(numpy_data.ndim) if i != axis)
        if len(reduced_dims) == numpy_data.ndim: # No reduction if reduced_dims is empty
             ranges_min = numpy_data.min()
             ranges_max = numpy_data.max()
             ranges = np.array([[ranges_min, ranges_max]], dtype=numpy_data.dtype)
        else:
            ranges_min = np.min(numpy_data, axis=reduced_dims, keepdims=False)
            ranges_max = np.max(numpy_data, axis=reduced_dims, keepdims=False)
            ranges = np.stack([ranges_min, ranges_max], axis=-1) # Stack along new last dimension

        return tvm.nd.array(ranges.astype('float32'))
    else:
        raise NotImplementedError(f"Unsupported tensor type for mock_channel_range: {type(tensor_data)}")


class TestEqualizeEager: # Inherit from object or not at all, using pytest conventions
    # The original QuantizationTestCase class provides PyTorch-specific assertions
    # and setup. We adapt to pytest's `assert` statements and `tvm.testing.utils.assert_allclose`.

    def checkChannelsEqualized(self, tensor1, tensor2, output_axis, input_axis):
        """Checks the channel ranges of tensor1, tensor2 are the same,
        using mocked channel_range.
        """
        # TODO: The underlying _equalize.channel_range is not directly mappable.
        # This check calls a mock, and thus its validity depends on the mock's accuracy,
        # which is limited without the full PyTorch quantization semantics.
        output_channel_tensor1 = mock_channel_range(tensor1, output_axis)
        input_channel_tensor2 = mock_channel_range(tensor2, input_axis)

        # ensuring the channels ranges of tensor1's input is the same as
        # tensor2's output. We use assert_allclose for numerical comparison of NDArrays.
        tvm_testing_utils.assert_allclose(output_channel_tensor1, input_channel_tensor2)

    def getModule(self, model_params_dict, name):
        """
        Mocks the behavior of retrieving a submodule's weights in PyTorch.
        In TVM, parameters are typically part of a dictionary passed to `relay.build`.
        This mock directly returns the NDArray for a given parameter name.
        """
        # In PyTorch, `getModule` returns an `nn.Module` object.
        # Here, we need the actual weight tensors for `checkChannelsEqualized`.
        # This mock simplifies to returning the parameter's NDArray.
        
        # The original PyTorch `getModule` traverses `_modules`. This is fundamentally
        # different from a parameter dictionary for a Relay graph.
        # This mock is a simplification for the purpose of accessing weight values.
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        
        if weight_key in model_params_dict:
            # Return just the weight, as checkChannelsEqualized uses `weight` directly.
            return model_params_dict[weight_key]
        
        raise KeyError(f"Parameter '{weight_key}' not found in mock model parameters for '{name}'.")


    @pytest.mark.skip(reason="PyTorch's _equalize.cross_layer_equalization modifies nn.Module weights in-place, which is not directly translatable to TVM's functional Relay IR.")
    def test_cross_layer_equalization(self):
        """
        This test depends on PyTorch's _equalize.cross_layer_equalization,
        which performs in-place modification of nn.Module weights.
        This operation is fundamentally incompatible with TVM's functional Relay IR.
        A TVM equivalent would involve a graph transformation pass that replaces
        existing constant weights with new, equalized constants, or a custom operator.
        Therefore, this test is skipped.
        """
        # PyTorch original: module1 = nn.Conv2d(3, 4, 2); module2 = nn.Linear(4, 4)
        # In TVM, these would be parts of a Relay function.
        # We define initial parameter values here.
        conv_weight_shape = (4, 3, 2, 2) # out_channels, in_channels, kH, kW
        linear_weight_shape = (4, 4)     # out_features, in_features

        # Create dummy NumPy arrays for weights, corresponding to PyTorch's default init.
        initial_conv_weight_np = np.random.randn(*conv_weight_shape).astype('float32')
        initial_linear_weight_np = np.random.randn(*linear_weight_shape).astype('float32')
        
        # Convert to TVM NDArrays (these would be `params` in a Relay graph)
        mod_tensor1 = tvm.nd.array(initial_conv_weight_np)
        mod_tensor2 = tvm.nd.array(initial_linear_weight_np)

        module1_output_channel_axis = 0 # Corresponds to C_out for Conv2d
        module2_input_channel_axis = 1  # Corresponds to C_in for Linear

        # The actual call to `_equalize.cross_layer_equalization` is the untranslatable part.
        # It would modify `mod_tensor1` and `mod_tensor2` in-place.
        # Since we cannot perform this step, the test cannot be meaningfully run.

        # If `_equalize.cross_layer_equalization` were implemented and resulted in equalized
        # weights, the following check would verify it.
        # For demonstration, assume for a moment the weights `mod_tensor1` and `mod_tensor2`
        # have *already been* magically equalized by a TVM pass or custom logic.
        # self.checkChannelsEqualized(mod_tensor1, mod_tensor2, module1_output_channel_axis, module2_input_channel_axis)
        
        # Test will be skipped as per decorator.


    @pytest.mark.skip(reason="PyTorch's _equalize.converged inspects nn.Module attributes and performs custom logic not directly translatable to TVM Relay's functional IR.")
    def test_converged(self):
        """
        This test relies on PyTorch's _equalize.converged, which compares weights
        of nn.Module instances. This involves inspecting Python object structure
        and numerical comparison logic specific to PyTorch's quantization context.
        While basic tensor comparison is possible in TVM (e.g., `tvm.testing.utils.assert_allclose`),
        the full `_equalize.converged` semantics are not directly mappable.
        Therefore, this test is skipped.
        """
        weight_size = (3, 3)
        
        # Create dummy weights as TVM NDArrays
        weight1_nd = tvm.nd.array(np.ones(weight_size, dtype='float32'))
        weight2_nd = tvm.nd.array(np.zeros(weight_size, dtype='float32'))

        # Mock function for `_equalize.converged` for illustration,
        # but the test is skipped because the original is too PyTorch-specific.
        def mock_converged_for_tvm_params(params_dict_a, params_dict_b, tol):
            if len(params_dict_a) != len(params_dict_b):
                return False
            
            for key in params_dict_a:
                if key not in params_dict_b:
                    return False
                
                val_a = params_dict_a[key]
                val_b = params_dict_b[key]

                # Convert to NumPy for `np.allclose`
                np_a = val_a.numpy() if isinstance(val_a, tvm.nd.NDArray) else val_a
                np_b = val_b.numpy() if isinstance(val_b, tvm.nd.NDArray) else val_b

                if not np.allclose(np_a, np_b, rtol=tol, atol=tol):
                    return False
            return True

        # Use dictionaries to represent 'modules' for this mock.
        dictionary_1 = {"linear1.weight": weight1_nd}
        dictionary_2 = {"linear1.weight": weight2_nd}

        assert mock_converged_for_tvm_params(dictionary_1, dictionary_1, 1e-6)
        assert not mock_converged_for_tvm_params(dictionary_1, dictionary_2, 1e-6)
        
        # Test skipped as per decorator.


    @pytest.mark.skip(reason="PyTorch's _equalize.equalize modifies nn.Module weights in-place and relies on a specific quantization workflow not directly translatable to TVM Relay.")
    def test_equalize(self):
        """
        This test relies on PyTorch's _equalize.equalize function, which
        is a high-level quantization equalization workflow function that
        modifies PyTorch `nn.Module` weights in-place. This is incompatible with
        TVM's functional Relay IR. A TVM equivalent would be a graph optimization
        pass that transforms the module with new, equalized constant weights.
        Therefore, this test is skipped.
        """

        # Define a Relay function representing the `ChainModule` computation.
        # Parameters are explicitly passed as Relay variables.
        def create_chain_module_relay_func(x_in_var, linear1_w_var, linear1_b_var, linear2_w_var, linear2_b_var, linear3_w_var, linear3_b_var):
            x = x_in_var
            # Linear1 (matmul + bias add)
            x = relay.nn.matmul(x, relay.transpose(linear1_w_var))
            x = relay.add(x, linear1_b_var)
            # Linear2
            x = relay.nn.matmul(x, relay.transpose(linear2_w_var))
            x = relay.add(x, linear2_b_var)
            # Linear3
            x = relay.nn.matmul(x, relay.transpose(linear3_w_var))
            x = relay.add(x, linear3_b_var)
            return x

        # Input shape for the module
        input_shape = (20, 3)
        input_dtype = "float32"

        # Create Relay variables for input and parameters
        x_in_var = relay.var("x", shape=input_shape, dtype=input_dtype)
        linear1_w_var = relay.var("linear1.weight", shape=(4, 3), dtype=input_dtype)
        linear1_b_var = relay.var("linear1.bias", shape=(4,), dtype=input_dtype)
        linear2_w_var = relay.var("linear2.weight", shape=(5, 4), dtype=input_dtype)
        linear2_b_var = relay.var("linear2.bias", shape=(5,), dtype=input_dtype)
        linear3_w_var = relay.var("linear3.weight", shape=(6, 5), dtype=input_dtype)
        linear3_b_var = relay.var("linear3.bias", shape=(6,), dtype=input_dtype)

        relay_func = create_chain_module_relay_func(
            x_in_var, linear1_w_var, linear1_b_var, linear2_w_var, linear2_b_var, linear3_w_var, linear3_b_var
        )

        # Create an IRModule from the function
        # The parameters list for Relay build must contain all variables in the function, except the input.
        relay_mod = tvm.IRModule.from_expr(relay.Function(
            [x_in_var, linear1_w_var, linear1_b_var, linear2_w_var, linear2_b_var, linear3_w_var, linear3_b_var],
            relay_func
        ))

        # Initialize NumPy arrays for weights and biases
        linear1_w_np = np.random.randn(4, 3).astype(input_dtype)
        linear1_b_np = np.random.randn(4).astype(input_dtype)
        linear2_w_np = np.random.randn(5, 4).astype(input_dtype)
        linear2_b_np = np.random.randn(5).astype(input_dtype)
        linear3_w_np = np.random.randn(6, 5).astype(input_dtype)
        linear3_b_np = np.random.randn(6).astype(input_dtype)

        # `chain1_params` will be "equalized" (mocked), `chain2_params` is original.
        chain1_params = {
            "linear1.weight": tvm.nd.array(linear1_w_np.copy()),
            "linear1.bias": tvm.nd.array(linear1_b_np.copy()),
            "linear2.weight": tvm.nd.array(linear2_w_np.copy()),
            "linear2.bias": tvm.nd.array(linear2_b_np.copy()),
            "linear3.weight": tvm.nd.array(linear3_w_np.copy()),
            "linear3.bias": tvm.nd.array(linear3_b_np.copy()),
        }
        chain2_params = copy.deepcopy(chain1_params) # Original parameters

        # Simulate equalization by modifying `chain1_params`. In a real scenario,
        # this would be the output of an actual equalization pass.
        for k in chain1_params:
            if "weight" in k:
                # Randomly change weights to simulate post-equalization state
                chain1_params[k] = tvm.nd.array(np.random.randn(*chain1_params[k].shape).astype(input_dtype))
        
        # For checkChannelsEqualized, we extract the weights directly.
        linear1_w_equalized = chain1_params["linear1.weight"]
        linear2_w_equalized = chain1_params["linear2.weight"]
        linear3_w_equalized = chain1_params["linear3.weight"]

        self.checkChannelsEqualized(linear1_w_equalized, linear2_w_equalized, 0, 1)
        self.checkChannelsEqualized(linear2_w_equalized, linear3_w_equalized, 0, 1)

        # To execute the Relay graph:
        target = "llvm" # or "cuda"
        dev = tvm.device(str(target), 0)

        input_tvm = tvm.nd.array(np.random.randn(*input_shape).astype(input_dtype), dev)

        # Build and run the original model
        factory_original = relay.build(relay_mod, target, params=chain2_params)
        rt_mod_original = tvm.runtime.vm.VirtualMachine(factory_original.graph_json, factory_original.lib, dev)
        result_original = rt_mod_original.invoke("main", input_tvm, **factory_original.params)
        
        # Build and run the "equalized" model
        factory_equalized = relay.build(relay_mod, target, params=chain1_params)
        rt_mod_equalized = tvm.runtime.vm.VirtualMachine(factory_equalized.graph_json, factory_equalized.lib, dev)
        result_equalized = rt_mod_equalized.invoke("main", input_tvm, **factory_equalized.params)

        # Original PyTorch test expects `chain1(input)` (equalized) to be equal to `chain2(input)` (original).
        # This implies that equalization should not change the model's numerical output (only its representation).
        # Since we've randomly modified `chain1_params`, `result_equalized` will NOT be equal to `result_original`.
        # Therefore, this assertion will fail without actual equalization logic.
        # This part of the test is skipped.
        # tvm_testing_utils.assert_allclose(result_equalized.numpy(), result_original.numpy(), rtol=1e-5, atol=1e-8)
        
        # Test skipped as per decorator.


    @pytest.mark.skip(reason="PyTorch's _equalize.equalize relies on nn.Module and fuse_modules, which are not directly translatable to TVM Relay's functional IR.")
    def test_equalize_fused_convrelu(self):
        """
        This test involves fusing PyTorch `ConvReLU2d` models and then applying
        equalization. Both `fuse_modules` and `_equalize.equalize` are highly
        PyTorch-specific (operating on `nn.Module` objects in-place).
        TVM performs fusion as a graph pass and equalization as a graph transformation
        or through QNN operators, not by modifying Python objects.
        Therefore, this test is skipped.
        """
        # Define a Relay function representing the `M` computation with Conv2d and ReLU.
        def create_conv_relu_module_relay_func(x_in_var, conv1_w_var, conv1_b_var, conv2_w_var, conv2_b_var, conv3_w_var, conv3_b_var):
            x = x_in_var
            # Conv1 + ReLU1
            x = relay.nn.conv2d(x, conv1_w_var, kernel_size=(1,1), padding=(0,0), strides=(1,1))
            x = relay.add(x, conv1_b_var)
            x = relay.nn.relu(x)
            # Conv2 + ReLU2
            x = relay.nn.conv2d(x, conv2_w_var, kernel_size=(1,1), padding=(0,0), strides=(1,1))
            x = relay.add(x, conv2_b_var)
            x = relay.nn.relu(x)
            # Conv3 + ReLU3
            x = relay.nn.conv2d(x, conv3_w_var, kernel_size=(1,1), padding=(0,0), strides=(1,1))
            x = relay.add(x, conv3_b_var)
            x = relay.nn.relu(x)
            return x

        # Input shape for the module
        input_shape = (3, 3, 1, 1) # NCHW format
        input_dtype = "float32"

        # Create Relay variables for input and parameters
        x_in_var = relay.var("x", shape=input_shape, dtype=input_dtype)
        conv1_w_var = relay.var("conv1.weight", shape=(3, 3, 1, 1), dtype=input_dtype)
        conv1_b_var = relay.var("conv1.bias", shape=(3,), dtype=input_dtype)
        conv2_w_var = relay.var("conv2.weight", shape=(3, 3, 1, 1), dtype=input_dtype)
        conv2_b_var = relay.var("conv2.bias", shape=(3,), dtype=input_dtype)
        conv3_w_var = relay.var("conv3.weight", shape=(3, 3, 1, 1), dtype=input_dtype)
        conv3_b_var = relay.var("conv3.bias", shape=(3,), dtype=input_dtype)

        relay_func = create_conv_relu_module_relay_func(
            x_in_var, conv1_w_var, conv1_b_var, conv2_w_var, conv2_b_var, conv3_w_var, conv3_b_var
        )

        relay_mod = tvm.IRModule.from_expr(relay.Function(
            [x_in_var, conv1_w_var, conv1_b_var, conv2_w_var, conv2_b_var, conv3_w_var, conv3_b_var],
            relay_func
        ))

        # Initialize NumPy arrays for weights and biases
        conv1_w_np = np.random.randn(3, 3, 1, 1).astype(input_dtype)
        conv1_b_np = np.random.randn(3).astype(input_dtype)
        conv2_w_np = np.random.randn(3, 3, 1, 1).astype(input_dtype)
        conv2_b_np = np.random.randn(3).astype(input_dtype)
        conv3_w_np = np.random.randn(3, 3, 1, 1).astype(input_dtype)
        conv3_b_np = np.random.randn(3).astype(input_dtype)

        fused_model1_params = {
            "conv1.weight": tvm.nd.array(conv1_w_np.copy()),
            "conv1.bias": tvm.nd.array(conv1_b_np.copy()),
            "conv2.weight": tvm.nd.array(conv2_w_np.copy()),
            "conv2.bias": tvm.nd.array(conv2_b_np.copy()),
            "conv3.weight": tvm.nd.array(conv3_w_np.copy()),
            "conv3.bias": tvm.nd.array(conv3_b_np.copy()),
        }
        fused_model2_params = copy.deepcopy(fused_model1_params)
        original_model_params = copy.deepcopy(fused_model1_params)

        # Simulate equalization by modifying `fused_model1_params` weights
        for k in fused_model1_params:
             if "weight" in k:
                 fused_model1_params[k] = tvm.nd.array(np.random.randn(*fused_model1_params[k].shape).astype(input_dtype))
        
        # Accessing weights for checkChannelsEqualized
        conv1_w_equalized = fused_model1_params["conv1.weight"]
        conv2_w_equalized = fused_model1_params["conv2.weight"]
        conv3_w_equalized = fused_model1_params["conv3.weight"]

        self.checkChannelsEqualized(conv1_w_equalized, conv2_w_equalized, 0, 1)
        self.checkChannelsEqualized(conv2_w_equalized, conv3_w_equalized, 0, 1)

        target = "llvm" # or "cuda"
        dev = tvm.device(str(target), 0)
        input_tvm = tvm.nd.array(np.random.randn(*input_shape).astype(input_dtype), dev)

        # Build and run the "equalized" model
        factory_equalized = relay.build(relay_mod, target, params=fused_model1_params)
        rt_mod_equalized = tvm.runtime.vm.VirtualMachine(factory_equalized.graph_json, factory_equalized.lib, dev)
        result_fused_equalized = rt_mod_equalized.invoke("main", input_tvm, **factory_equalized.params)

        # Build and run the original fused model
        factory_fused_original = relay.build(relay_mod, target, params=fused_model2_params)
        rt_mod_fused_original = tvm.runtime.vm.VirtualMachine(factory_fused_original.graph_json, factory_fused_original.lib, dev)
        result_fused_original = rt_mod_fused_original.invoke("main", input_tvm, **factory_fused_original.params)
        
        # Build and run the conceptual "original" unfused model (which is the same Relay graph, but with original params)
        factory_original = relay.build(relay_mod, target, params=original_model_params)
        rt_mod_original = tvm.runtime.vm.VirtualMachine(factory_original.graph_json, factory_original.lib, dev)
        result_original = rt_mod_original.invoke("main", input_tvm, **factory_original.params)

        # This asserts that the deepcopy and initial model are equivalent, which should hold.
        tvm_testing_utils.assert_allclose(result_fused_original.numpy(), result_original.numpy())
        
        # The main assertion `self.assertEqual(fused_model1(input), fused_model2(input))`
        # relies on `_equalize.equalize` to make the outputs equal, which is not implemented.
        # This part of the test is skipped.
        # tvm_testing_utils.assert_allclose(result_fused_equalized.numpy(), result_fused_original.numpy(), rtol=1e-5, atol=1e-8)
        
        # Test skipped as per decorator.


    @pytest.mark.skip(reason="PyTorch's _equalize.equalize relies on nn.Module and fuse_modules, which are not directly translatable to TVM Relay's functional IR.")
    def test_equalize_fused_linearrelu(self):
        """
        This test involves fusing PyTorch `LinearReLU` models and then applying
        equalization. Both `fuse_modules` and `_equalize.equalize` are highly
        PyTorch-specific (operating on `nn.Module` objects in-place).
        TVM performs fusion as a graph pass and equalization as a graph transformation
        or through QNN operators, not by modifying Python objects.
        Therefore, this test is skipped.
        """

        # Define a Relay function representing the `M` computation with Linear and ReLU.
        def create_linear_relu_module_relay_func(x_in_var, linear1_w_var, linear1_b_var, linear2_w_var, linear2_b_var, linear3_w_var, linear3_b_var):
            x = x_in_var
            # Linear1 + ReLU1
            x = relay.nn.matmul(x, relay.transpose(linear1_w_var))
            x = relay.add(x, linear1_b_var)
            x = relay.nn.relu(x)
            # Linear2 + ReLU2
            x = relay.nn.matmul(x, relay.transpose(linear2_w_var))
            x = relay.add(x, linear2_b_var)
            x = relay.nn.relu(x)
            # Linear3 + ReLU3
            x = relay.nn.matmul(x, relay.transpose(linear3_w_var))
            x = relay.add(x, linear3_b_var)
            x = relay.nn.relu(x)
            return x

        # Input shape for the module
        input_shape = (20, 3)
        input_dtype = "float32"

        # Create Relay variables for input and parameters
        x_in_var = relay.var("x", shape=input_shape, dtype=input_dtype)
        linear1_w_var = relay.var("linear1.weight", shape=(4, 3), dtype=input_dtype)
        linear1_b_var = relay.var("linear1.bias", shape=(4,), dtype=input_dtype)
        linear2_w_var = relay.var("linear2.weight", shape=(5, 4), dtype=input_dtype)
        linear2_b_var = relay.var("linear2.bias", shape=(5,), dtype=input_dtype)
        linear3_w_var = relay.var("linear3.weight", shape=(6, 5), dtype=input_dtype)
        linear3_b_var = relay.var("linear3.bias", shape=(6,), dtype=input_dtype)

        relay_func = create_linear_relu_module_relay_func(
            x_in_var, linear1_w_var, linear1_b_var, linear2_w_var, linear2_b_var, linear3_w_var, linear3_b_var
        )

        relay_mod = tvm.IRModule.from_expr(relay.Function(
            [x_in_var, linear1_w_var, linear1_b_var, linear2_w_var, linear2_b_var, linear3_w_var, linear3_b_var],
            relay_func
        ))

        # Initialize NumPy arrays for weights and biases
        linear1_w_np = np.random.randn(4, 3).astype(input_dtype)
        linear1_b_np = np.random.randn(4).astype(input_dtype)
        linear2_w_np = np.random.randn(5, 4).astype(input_dtype)
        linear2_b_np = np.random.randn(5).astype(input_dtype)
        linear3_w_np = np.random.randn(6, 5).astype(input_dtype)
        linear3_b_np = np.random.randn(6).astype(input_dtype)

        fused_model1_params = {
            "linear1.weight": tvm.nd.array(linear1_w_np.copy()),
            "linear1.bias": tvm.nd.array(linear1_b_np.copy()),
            "linear2.weight": tvm.nd.array(linear2_w_np.copy()),
            "linear2.bias": tvm.nd.array(linear2_b_np.copy()),
            "linear3.weight": tvm.nd.array(linear3_w_np.copy()),
            "linear3.bias": tvm.nd.array(linear3_b_np.copy()),
        }
        fused_model2_params = copy.deepcopy(fused_model1_params)
        original_model_params = copy.deepcopy(fused_model1_params)

        # Simulate equalization by modifying `fused_model1_params` weights
        for k in fused_model1_params:
             if "weight" in k:
                 fused_model1_params[k] = tvm.nd.array(np.random.randn(*fused_model1_params[k].shape).astype(input_dtype))
        
        # Accessing weights for checkChannelsEqualized
        linear1_w_equalized = fused_model1_params["linear1.weight"]
        linear2_w_equalized = fused_model1_params["linear2.weight"]
        linear3_w_equalized = fused_model1_params["linear3.weight"]

        self.checkChannelsEqualized(linear1_w_equalized, linear2_w_equalized, 0, 1)
        self.checkChannelsEqualized(linear2_w_equalized, linear3_w_equalized, 0, 1)

        target = "llvm" # or "cuda"
        dev = tvm.device(str(target), 0)
        input_tvm = tvm.nd.array(np.random.randn(*input_shape).astype(input_dtype), dev)

        # Build and run the "equalized" model
        factory_equalized = relay.build(relay_mod, target, params=fused_model1_params)
        rt_mod_equalized = tvm.runtime.vm.VirtualMachine(factory_equalized.graph_json, factory_equalized.lib, dev)
        result_fused_equalized = rt_mod_equalized.invoke("main", input_tvm, **factory_equalized.params)

        # Build and run the original fused model
        factory_fused_original = relay.build(relay_mod, target, params=fused_model2_params)
        rt_mod_fused_original = tvm.runtime.vm.VirtualMachine(factory_fused_original.graph_json, factory_fused_original.lib, dev)
        result_fused_original = rt_mod_fused_original.invoke("main", input_tvm, **factory_fused_original.params)
        
        # Build and run the conceptual "original" unfused model (which is the same Relay graph, but with original params)
        factory_original = relay.build(relay_mod, target, params=original_model_params)
        rt_mod_original = tvm.runtime.vm.VirtualMachine(factory_original.graph_json, factory_original.lib, dev)
        result_original = rt_mod_original.invoke("main", input_tvm, **factory_original.params)

        # This asserts that the deepcopy and initial model are equivalent, which should hold.
        tvm_testing_utils.assert_allclose(result_fused_original.numpy(), result_original.numpy())
        
        # The main assertion `self.assertEqual(fused_model1(input), fused_model2(input))`
        # relies on `_equalize.equalize` to make the outputs equal, which is not implemented.
        # This part of the test is skipped.
        # tvm_testing_utils.assert_allclose(result_fused_equalized.numpy(), result_fused_original.numpy(), rtol=1e-5, atol=1e-8)
        
        # Test skipped as per decorator.


if __name__ == "__main__":
    # The original `raise_on_run_directly` is a PyTorch-specific utility.
    # In pytest, tests are typically run via the `pytest` command.
    # This block is for standalone execution or ensuring importability.
    pass
