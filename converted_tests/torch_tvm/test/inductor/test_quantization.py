import logging
import numpy as np
import unittest
import tvm
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm.runtime import container


log = logging.getLogger(__name__)

# Helper to map PyTorch dtypes to TVM dtypes
def map_torch_dtype_to_tvm(torch_dtype_str):
    dtype_map = {
        "float32": "float32",
        "bfloat16": "bfloat16",
        "float64": "float64",
        "int32": "int32",
        "int64": "int64",
        "bool": "bool",
    }
    # For float8, there's no direct native Relay support. Map to a closest float type and add a TODO.
    if "float8_e5m2" in torch_dtype_str:
        # TODO: TVM does not natively support 'float8_e5m2'. This mapping defaults to 'float16'
        # which might not accurately represent the precision or range characteristics of float8_e5m2.
        # Custom type definitions and lowering might be required for full fidelity.
        return "float16" 
    return dtype_map.get(torch_dtype_str.replace("torch.", ""), "float32") # Default to float32


# Helper to define a TVM Relay function for PyTorch's TargetCPModule
def define_target_cp_module_relay(shape1, shape2, dtype):
    x1_relay = relay.var("x1", relay.Tensor(shape1, dtype))
    x2_relay = relay.var("x2", relay.Tensor(shape2, dtype))
    
    # relued = torch.relu(x1)
    relued_relay = relay.nn.relu(x1_relay)
    
    # tanhed = torch.tanh(relued)
    tanhed_relay = relay.tanh(relued_relay)
    
    # tensor = torch.matmul(tanhed, x2)
    tensor_relay = relay.nn.matmul(tanhed_relay, x2_relay)
    
    return relay.Function([x1_relay, x2_relay], tensor_relay)

# Helper to define a TVM Relay function for PyTorch's FeedforwardNN
def define_feedforward_nn_relay(input_shape, hidden_dim, output_dim, dtype):
    input_dim = input_shape[-1]

    # Parameters for Linear layers (weight and bias)
    # These will be passed as part of the Relay Function's arguments
    fc1_w = relay.var("fc1_weight", relay.Tensor((hidden_dim, input_dim), dtype))
    fc1_b = relay.var("fc1_bias", relay.Tensor((hidden_dim,), dtype))
    fc2_w = relay.var("fc2_weight", relay.Tensor((hidden_dim, hidden_dim), dtype))
    fc2_b = relay.var("fc2_bias", relay.Tensor((hidden_dim,), dtype))
    fc3_w = relay.var("fc3_weight", relay.Tensor((hidden_dim, hidden_dim), dtype))
    fc3_b = relay.var("fc3_bias", relay.Tensor((hidden_dim,), dtype))
    fc4_w = relay.var("fc4_weight", relay.Tensor((output_dim, hidden_dim), dtype))
    fc4_b = relay.var("fc4_bias", relay.Tensor((output_dim,), dtype))

    x_0 = relay.var("x", relay.Tensor(input_shape, dtype))

    # x = torch.relu(self.fc1(x))
    fc1_out = relay.nn.dense(x_0, fc1_w)
    fc1_out_biased = relay.nn.bias_add(fc1_out, fc1_b)
    x_1 = relay.nn.relu(fc1_out_biased) 

    # tanh_x = torch.tanh(x)
    tanh_x_val = relay.tanh(x_1) 

    # x = torch.relu(self.fc2(x))
    fc2_out = relay.nn.dense(x_1, fc2_w)
    fc2_out_biased = relay.nn.bias_add(fc2_out, fc2_b)
    x_2 = relay.nn.relu(fc2_out_biased) 

    # x = torch.relu(self.fc3(tanh_x))
    fc3_out = relay.nn.dense(tanh_x_val, fc3_w)
    fc3_out_biased = relay.nn.bias_add(fc3_out, fc3_b)
    x_3 = relay.nn.relu(fc3_out_biased) 

    # x = self.fc4(x)
    fc4_out = relay.nn.dense(x_3, fc4_w) 
    final_output = relay.nn.bias_add(fc4_out, fc4_b)
    
    # All inputs to the function (first is the actual input, rest are parameters)
    params = [x_0, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, fc4_w, fc4_b]
    return relay.Function(params, final_output)

# Helper to determine layer_norm axis from normalized_shape
def get_layernorm_axis(input_ndim, normalized_shape_len):
    if normalized_shape_len == 0:
        return None # Normalize across all axes
    # PyTorch normalized_shape specifies the *last* dimensions
    return [i for i in range(input_ndim - normalized_shape_len, input_ndim)]

# Helper to define a TVM Relay function for PyTorch's LayernormNN
def define_layernorm_nn_relay(input_shape, normalized_shape_tuple, dtype):
    ln_input_relay = relay.var("input", relay.Tensor(input_shape, dtype))
    
    input_ndim = len(input_shape)
    normalized_shape_len = len(normalized_shape_tuple)
    ln_axis = get_layernorm_axis(input_ndim, normalized_shape_len)
    
    ln_weight_relay = relay.var("weight", relay.Tensor(normalized_shape_tuple, dtype))
    ln_bias_relay = relay.var("bias", relay.Tensor(normalized_shape_tuple, dtype))

    ln_output_relay = relay.nn.layer_norm(
        data=ln_input_relay,
        gamma=ln_weight_relay,
        beta=ln_bias_relay,
        axis=ln_axis,
        epsilon=1e-5,
        center=True, # PyTorch LayerNorm applies bias
        scale=True   # PyTorch LayerNorm applies weight
    )
    
    return relay.Function([ln_input_relay, ln_weight_relay, ln_bias_relay], ln_output_relay)


# A simplified `tvm_compile_model` that directly builds/compiles a Relay Function
def tvm_compile_model(relay_func, target):
    mod = tvm.IRModule.from_expr(relay_func)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    
    # Extract target device from target string (e.g., "cuda -keys=..." -> "cuda")
    dev_type = str(target).split(" ")[0]
    dev = tvm.device(dev_type, 0)
    vm = tvm.runtime.vm.VirtualMachine(lib, dev)
    return vm


class TestQuantization(unittest.TestCase):
    def setUp(self):
        # Store numpy random state to make tests repeatable even with random inputs
        self.rng_state = np.random.get_state()

    def tearDown(self):
        np.random.set_state(self.rng_state)

    def compare_pred(self, tvm_executor, input_tvm_ndarrays, ref_output_np, rtol=1e-3, atol=1e-3):
        res_tvm_nd = tvm_executor(*input_tvm_ndarrays)
        # Ensure result is a numpy array for comparison
        res_np = res_tvm_nd.numpy() if isinstance(res_tvm_nd, tvm.runtime.ndarray.NDArray) else res_tvm_nd 

        tvm.testing.assert_allclose(actual=res_np, desired=ref_output_np, rtol=rtol, atol=atol)

    def compare_parameters(self, *args, **kwargs):
        # TODO: Parameter comparison for TVM would involve extracting parameters from the compiled module
        # and comparing with reference values. This requires a mapping from PyTorch parameter names
        # to TVM Relay variables/constants, which is non-trivial for general cases.
        # For this exercise, it's explicitly marked as a placeholder.
        pass

    def compare_gradients(self, *args, **kwargs):
        # TODO: Gradient comparison for TVM requires auto-differentiation passes and is a complex topic
        # for a basic API mapping exercise. This is a significant functional difference from PyTorch.
        # For this exercise, it's explicitly marked as a placeholder.
        pass

    @tvm.testing.requires_cuda
    def test_activation_quantization_aten_with_scaling(self):
        # counters.clear() # PyTorch specific for inductor, removed.
        log.info("Running test_activation_quantization_aten_with_scaling - TargetCPModule")

        shape1 = (16, 10)
        shape2 = (10, 16)
        # Original test uses bfloat16. TVM support for bfloat16 depends on target capabilities.
        # Mapping to bfloat16 directly, assuming target supports it.
        dtype = map_torch_dtype_to_tvm("bfloat16") 
        device = tvm.cuda(0)
        target_tvm = tvm.target.Target("cuda")

        # 1. Define the Relay function for the model
        relay_func_cp = define_target_cp_module_relay(shape1, shape2, dtype)

        # 2. Generate random numpy inputs
        input1_np = np.random.rand(*shape1).astype(dtype)
        input2_np = np.random.rand(*shape2).astype(dtype)

        # 3. Get reference output from an unquantized TVM Relay model on CPU (acts as "eager" mode result)
        ref_relay_func_cp = define_target_cp_module_relay(shape1, shape2, dtype)
        ref_vm_cp = tvm_compile_model(ref_relay_func_cp, "llvm") # Compile for CPU for reference
        ref_cpu_device = tvm.cpu(0)
        ref_input_tvm_nd_cp = [
            tvm.nd.array(input1_np, device=ref_cpu_device),
            tvm.nd.array(input2_np, device=ref_cpu_device),
        ]
        ref_output_np_cp = ref_vm_cp(*ref_input_tvm_nd_cp).numpy()

        # 4. Compile the target TVM model (on GPU)
        # TODO: The 'activation_quantization_aten_pass' with 'quant_type: torch.float8_e5m2'
        # is a PyTorch Inductor specific pass. TVM does not have a direct equivalent
        # for this pass or native 'float8_e5m2' dtype support in Relay/TIR compilation.
        # The compiled TVM model here represents the *float* equivalent of the original
        # computation graph, *without* explicit float8 quantization.
        compiled_vm_cp = tvm_compile_model(relay_func_cp, target_tvm)
        target_input_tvm_nd_cp = [
            tvm.nd.array(input1_np, device=device),
            tvm.nd.array(input2_np, device=device),
        ]

        # 5. Compare prediction
        self.compare_pred(compiled_vm_cp, target_input_tvm_nd_cp, ref_output_np_cp, rtol=1e-3, atol=1e-3)
        
        # PyTorch-specific assertions removed
        self.compare_parameters(None, None) 
        self.compare_gradients(None, None) 

        # --- Second part of the same test case (FeedforwardNN) ---
        log.info("Running test_activation_quantization_aten_with_scaling - FeedforwardNN")

        input_shape_ffnn = (100, 1) # Matches np.linspace(-10, 10, 100).reshape(-1, 1)
        hidden_dim_ffnn = 64
        output_dim_ffnn = 1
        dtype_ffnn = map_torch_dtype_to_tvm("float32") # Original code uses float32 after numpy conversion

        # 1. Define the Relay function for FeedforwardNN
        relay_func_ffnn = define_feedforward_nn_relay(input_shape_ffnn, hidden_dim_ffnn, output_dim_ffnn, dtype_ffnn)

        # 2. Generate random numpy inputs and parameters
        x_np = np.random.rand(*input_shape_ffnn).astype(dtype_ffnn)
        fc1_w_np = np.random.rand(hidden_dim_ffnn, input_shape_ffnn[-1]).astype(dtype_ffnn)
        fc1_b_np = np.random.rand(hidden_dim_ffnn,).astype(dtype_ffnn)
        fc2_w_np = np.random.rand(hidden_dim_ffnn, hidden_dim_ffnn).astype(dtype_ffnn)
        fc2_b_np = np.random.rand(hidden_dim_ffnn,).astype(dtype_ffnn)
        fc3_w_np = np.random.rand(hidden_dim_ffnn, hidden_dim_ffnn).astype(dtype_ffnn)
        fc3_b_np = np.random.rand(hidden_dim_ffnn,).astype(dtype_ffnn)
        fc4_w_np = np.random.rand(output_dim_ffnn, hidden_dim_ffnn).astype(dtype_ffnn)
        fc4_b_np = np.random.rand(output_dim_ffnn,).astype(dtype_ffnn)

        # Combine input data and parameters for the Relay function call
        ordered_inputs_np_ffnn = [
            x_np,
            fc1_w_np, fc1_b_np,
            fc2_w_np, fc2_b_np,
            fc3_w_np, fc3_b_np,
            fc4_w_np, fc4_b_np,
        ]

        # 3. Get reference output from an unquantized TVM Relay model on CPU
        ref_relay_func_ffnn = define_feedforward_nn_relay(input_shape_ffnn, hidden_dim_ffnn, output_dim_ffnn, dtype_ffnn)
        ref_vm_ffnn = tvm_compile_model(ref_relay_func_ffnn, "llvm") 
        ref_cpu_device_ffnn = tvm.cpu(0)
        ref_input_tvm_nd_ffnn = [tvm.nd.array(arr, device=ref_cpu_device_ffnn) for arr in ordered_inputs_np_ffnn]
        ref_output_ffnn_np = ref_vm_ffnn(*ref_input_tvm_nd_ffnn).numpy()

        # 4. Compile the target TVM model (on GPU)
        # TODO: Quantization pass for FeedforwardNN (same as above).
        compiled_vm_ffnn = tvm_compile_model(relay_func_ffnn, target_tvm)
        target_input_tvm_nd_ffnn = [tvm.nd.array(arr, device=device) for arr in ordered_inputs_np_ffnn]
        
        # 5. Compare prediction
        self.compare_pred(compiled_vm_ffnn, target_input_tvm_nd_ffnn, ref_output_ffnn_np, rtol=1e-3, atol=1e-3)
        
        # PyTorch-specific assertions removed
        self.compare_parameters(None, None)
        self.compare_gradients(None, None)


    @tvm.testing.requires_cuda
    def test_activation_quantization_aten_without_scaling(self):
        # counters.clear() # PyTorch specific for inductor, removed.
        log.info("Running test_activation_quantization_aten_without_scaling - LayernormNN")

        input_shape = (1, 3, 256)
        normalized_shape_py = [256] 
        normalized_shape_tuple = tuple(normalized_shape_py) # For TVM Relay tensor shape
        # Original test uses bfloat16. Mapping to bfloat16 directly, assuming target supports it.
        dtype = map_torch_dtype_to_tvm("bfloat16") 
        device = tvm.cuda(0)
        target_tvm = tvm.target.Target("cuda")

        # 1. Define the Relay function for LayernormNN
        relay_func_layernorm = define_layernorm_nn_relay(input_shape, normalized_shape_tuple, dtype)

        # 2. Generate random numpy inputs and parameters
        ln_input_np = np.random.rand(*input_shape).astype(dtype)
        ln_weight_np = np.random.rand(*normalized_shape_tuple).astype(dtype) 
        ln_bias_np = np.random.rand(*normalized_shape_tuple).astype(dtype)   
        
        ordered_inputs_np_layernorm = [
            ln_input_np,
            ln_weight_np,
            ln_bias_np,
        ]

        # 3. Get reference output from an unquantized TVM Relay model on CPU
        ref_relay_func_layernorm = define_layernorm_nn_relay(input_shape, normalized_shape_tuple, dtype)
        ref_vm_layernorm = tvm_compile_model(ref_relay_func_layernorm, "llvm")
        ref_cpu_device_layernorm = tvm.cpu(0)

        ref_input_tvm_nd_layernorm = [tvm.nd.array(arr, device=ref_cpu_device_layernorm) for arr in ordered_inputs_np_layernorm]
        ref_output_layernorm_np = ref_vm_layernorm(*ref_input_tvm_nd_layernorm).numpy()

        # 4. Compile the target TVM model (on GPU)
        # TODO: Quantization pass for LayernormNN (same as above).
        compiled_vm_layernorm = tvm_compile_model(relay_func_layernorm, target_tvm)
        target_input_tvm_nd_layernorm = [tvm.nd.array(arr, device=device) for arr in ordered_inputs_np_layernorm]

        # 5. Compare prediction
        self.compare_pred(compiled_vm_layernorm, target_input_tvm_nd_layernorm, ref_output_layernorm_np, rtol=1e-3, atol=1e-3)

        # PyTorch-specific assertions removed
        self.compare_parameters(None, None) 
        self.compare_gradients(None, None) 


if __name__ == "__main__":
    # The original PyTorch test used `IS_LINUX and HAS_GPU` to guard `run_tests()`.
    # `@tvm.testing.requires_cuda` decorator handles the GPU requirement for individual tests.
    # `unittest.main()` is the standard way to run unittest test cases.
    unittest.main()
