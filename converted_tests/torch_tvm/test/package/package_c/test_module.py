import tvm
from tvm import relay
from tvm.ir.module import IRModule
from tvm.relay import op
import numpy as np
import tvm.runtime.ndarray
import tvm.transform


# The original PyTorch code defines a class TorchVisionTest
# whose forward method contains the logic we are converting.
# The `resnet18` model instantiation (`self.tvmod = resnet18()`)
# is not used in the forward pass, so it's omitted in the TVM equivalent.

# Original PyTorch helper function:
# def a_non_torch_leaf(a, b):
#     return a + b

# Equivalent TVM helper function operating on Relay expressions
def a_non_tvm_leaf(a_relay, b_relay):
    """Equivalent to PyTorch's a_non_torch_leaf for Relay expressions."""
    return op.add(a_relay, b_relay)

# This function creates a TVM Relay IRModule that represents the
# computation performed by TorchVisionTest's forward method.
def create_tvm_test_module():
    # Define an input variable for the Relay function.
    # The shape and dtype are chosen to be compatible with typical ResNet inputs.
    x_shape = (1, 3, 224, 224)
    x_dtype = "float32"
    x_relay = relay.var("x", shape=x_shape, dtype=x_dtype)

    # 1. Replicate `x = a_non_torch_leaf(x, x)`
    # The helper function `a_non_tvm_leaf` performs element-wise addition.
    temp_x = a_non_tvm_leaf(x_relay, x_relay)

    # 2. Replicate `return torch.relu(x + 3.0)`
    # For `x + 3.0`: The scalar 3.0 needs to be a Relay constant with the correct dtype.
    const_3_0 = relay.const(3.0, dtype=x_dtype)
    add_result = op.add(temp_x, const_3_0)

    # For `torch.relu(...)`: Map to `tvm.relay.op.nn.relu`.
    final_result = op.nn.relu(add_result)

    # Define the main Relay function with the input and final result.
    main_fn = relay.Function([x_relay], final_result)
    
    # Create the IRModule from the main function.
    mod = IRModule.from_expr(main_fn)
    return mod

# This block ensures the file is runnable and demonstrates the generated TVM module.
if __name__ == "__main__":
    # Create the TVM Relay module
    tvm_module = create_tvm_test_module()
    print("--- Generated TVM Relay module ---")
    print(tvm_module.functions["main"].astext())
    print("--- End of TVM Relay module ---")

    # Example of how one might compile and run this module
    # This demonstrates that the generated Relay graph is valid and executable.
    
    # Choose a target and device (e.g., 'llvm' for CPU, 'cuda' for GPU)
    target = "llvm"  # Default to CPU for broad compatibility
    dev = tvm.cpu(0) 

    # Compile the Relay module
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(tvm_module, target=target)

    # Create a TVM runtime module
    module = tvm.runtime.GraphModule(lib["default"](dev))

    # Prepare example input data using NumPy
    input_shape = (1, 3, 224, 224)
    input_dtype = "float32"
    input_data = np.random.rand(*input_shape).astype(input_dtype)

    # Set input and run the module
    module.set_input("x", tvm.nd.array(input_data, dev))
    module.run()

    # Get the output and print its shape and dtype
    output_data = module.get_output(0).numpy()

    print("\n--- Example runtime execution results ---")
    print("Input shape:", input_data.shape)
    print("Input dtype:", input_data.dtype)
    print("Output shape:", output_data.shape)
    print("Output dtype:", output_data.dtype)
    print("--- End of example runtime execution ---")
