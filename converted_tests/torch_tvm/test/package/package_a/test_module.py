# Owner(s): ["oncall: package/deploy"]
# This file is part of the TVM equivalent test suite.

import tvm
from tvm import relay
from tvm.relay import op
import numpy as np
import pytest

# Original PyTorch:
# from torch.fx import wrap
# wrap("a_non_torch_leaf")

# In TVM, external Python functions are typically mapped to Relay operators
# if they are part of the computation graph, or handled outside the graph.
# 'a_non_torch_leaf' performs simple addition, which maps directly to relay.op.tensor.add.
# The 'wrap' mechanism is PyTorch-FX specific and has no direct TVM equivalent.
def a_non_torch_leaf_logic(a, b):
    # This is the Python logic, not a Relay operation directly.
    # The Relay graph will use `op.tensor.add` where this logic is applied.
    return a + b


# Equivalent for PyTorch's ModWithSubmod(torch.nn.Module)
# In TVM, a "module" is often represented as an IRModule containing Relay functions.
# A "submodule" would be another Relay function passed as an argument.
def get_mod_with_submod_relay_module(sub_mod_relay_func) -> tvm.IRModule:
    """
    Creates a Relay IRModule equivalent to PyTorch's ModWithSubmod.
    sub_mod_relay_func: A relay.Function representing the submodule's computation.
    """
    x = relay.var("x", shape=(1, 10), dtype="float32") # Example input tensor
    # The submodule's forward method is called here
    out = sub_mod_relay_func(x)
    main_func = relay.Function([x], out)
    return tvm.IRModule.from_expr(main_func)


# Equivalent for PyTorch's ModWithTensor(torch.nn.Module)
# self.tensor becomes a Relay parameter.
def get_mod_with_tensor_relay_module() -> tvm.IRModule:
    """
    Creates a Relay IRModule equivalent to PyTorch's ModWithTensor.
    `self.tensor` is modeled as a parameter to the Relay function.
    """
    x = relay.var("x", shape=(1, 10), dtype="float32")
    tensor_param = relay.var("tensor", shape=(1, 10), dtype="float32")
    out = op.tensor.multiply(tensor_param, x)
    main_func = relay.Function([x, tensor_param], out)
    return tvm.IRModule.from_expr(main_func)


# Equivalent for PyTorch's ModWithSubmodAndTensor(torch.nn.Module)
def get_mod_with_submod_and_tensor_relay_module(sub_mod_relay_func) -> tvm.IRModule:
    """
    Creates a Relay IRModule equivalent to PyTorch's ModWithSubmodAndTensor.
    sub_mod_relay_func: A relay.Function representing the submodule's computation.
    `self.tensor` is modeled as a parameter.
    """
    x = relay.var("x", shape=(1, 10), dtype="float32")
    tensor_param = relay.var("tensor", shape=(1, 10), dtype="float32")
    sub_mod_output = sub_mod_relay_func(x)
    out = op.tensor.add(sub_mod_output, tensor_param)
    main_func = relay.Function([x, tensor_param], out)
    return tvm.IRModule.from_expr(main_func)


# Equivalent for PyTorch's ModWithTwoSubmodsAndTensor(torch.nn.Module)
def get_mod_with_two_submods_and_tensor_relay_module(
    sub_mod_0_relay_func, sub_mod_1_relay_func
) -> tvm.IRModule:
    """
    Creates a Relay IRModule equivalent to PyTorch's ModWithTwoSubmodsAndTensor.
    `sub_mod_0_relay_func`, `sub_mod_1_relay_func`: Relay Functions for submodules.
    `self.tensor` is modeled as a parameter.
    """
    x = relay.var("x", shape=(1, 10), dtype="float32")
    tensor_param = relay.var("tensor", shape=(1, 10), dtype="float32")
    sub_mod_0_output = sub_mod_0_relay_func(x)
    sub_mod_1_output = sub_mod_1_relay_func(x)
    intermediate_sum = op.tensor.add(sub_mod_0_output, sub_mod_1_output)
    out = op.tensor.add(intermediate_sum, tensor_param)
    main_func = relay.Function([x, tensor_param], out)
    return tvm.IRModule.from_expr(main_func)


# Equivalent for PyTorch's ModWithMultipleSubmods(torch.nn.Module)
def get_mod_with_multiple_submods_relay_module(
    mod1_relay_func, mod2_relay_func
) -> tvm.IRModule:
    """
    Creates a Relay IRModule equivalent to PyTorch's ModWithMultipleSubmods.
    `mod1_relay_func`, `mod2_relay_func`: Relay Functions for submodules.
    """
    x = relay.var("x", shape=(1, 10), dtype="float32")
    mod1_output = mod1_relay_func(x)
    mod2_output = mod2_relay_func(x)
    out = op.tensor.add(mod1_output, mod2_output)
    main_func = relay.Function([x], out)
    return tvm.IRModule.from_expr(main_func)


# Equivalent for PyTorch's SimpleTest(torch.nn.Module)
def get_simple_test_relay_module() -> tvm.IRModule:
    """
    Creates a Relay IRModule equivalent to PyTorch's SimpleTest.
    """
    x = relay.var("x", shape=(1, 10), dtype="float32")
    
    # Original: x = a_non_torch_leaf(x, x)
    # a_non_torch_leaf_logic is 'a + b', so this becomes x + x in Relay.
    sum_x = op.tensor.add(x, x)
    
    # Original: return torch.relu(x + 3.0)
    sum_x_plus_3 = op.tensor.add(sum_x, relay.const(3.0, "float32"))
    out = op.nn.relu(sum_x_plus_3)
    
    main_func = relay.Function([x], out)
    return tvm.IRModule.from_expr(main_func)
