import contextlib
import itertools
import unittest

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.relay import op as relay_op
from tvm.relay import loops
from tvm.relay.op import nn as nn_op
from tvm.relay.op import transform as transform_op
from tvm.relay.op import reduce as reduce_op
from tvm.relay.op import algorithm as alg_op
from tvm.runtime import vm as vm_rt

# Mocking PyTorch-specific common_utils and device types
# For TVM context, we'll use `tvm.device("cuda", 0)` or `tvm.device("cpu", 0)`
# And `pytest.mark.skipif` for conditional tests.

# Dummy for PyTorch's GPU_TYPE, HAS_CPU, HAS_GPU
GPU_TYPE = "cuda"
HAS_CPU = tvm.runtime.cpu().exist
HAS_GPU = tvm.runtime.cuda().exist

# Helper to determine TVM device based on string
def _get_tvm_device(device_str):
    if device_str == "cpu":
        return tvm.cpu(0)
    elif device_str == "cuda":
        return tvm.cuda(0)
    raise ValueError(f"Unsupported device: {device_str}")

# Replicate _prepend_product_of_values logic in numpy for test inputs
def _prepend_product_of_values_np(inputs_np, possible_values, num_to_prepend=1):
    result = []
    for values in itertools.product(*([possible_values] * num_to_prepend)):
        prepended = [np.array(v, dtype=np.bool_) if isinstance(v, bool) else np.array(v) for v in values]
        result.append(tuple(prepended + list(inputs_np)))
    return result

def prepend_predicates_np(inputs_np, num_predicates=1):
    return _prepend_product_of_values_np(inputs_np, [False, True], num_predicates)

def prepend_counters_np(inputs_np, num_counters=1, counter_values=(0, 1, 5)):
    return _prepend_product_of_values_np(inputs_np, counter_values, num_counters)

# A testing loss_fn for NumPy (for reference calculation in autograd-disabled tests)
def loss_fn_np(result_flat_np):
    total_loss = np.array(0.0, dtype=np.float32)
    for res_np in result_flat_np:
        # Convert to float if integer tensor to avoid numerical issues
        if not np.issubdtype(res_np.dtype, np.floating):
            res_np = res_np.astype(np.float32)

        # Simple robust loss: abs values + small constant to avoid inf/nan
        total_loss = total_loss + (np.abs(res_np) / (1.0 + np.abs(res_np))).sum()
    return total_loss

# Helper for flattening possibly nested tuples/lists (like pytree.tree_flatten)
def _flatten_nested_output(obj):
    if isinstance(obj, (tuple, list, tvm.runtime.container.ADT)):
        flat = []
        for item in obj:
            flat.extend(_flatten_nested_output(item))
        return flat
    return [obj]

# Helper to build Relay.Function for torch.cond and torch.while_loop branches
# These are essentially `lambda x, y: ...` which directly translate to Relay.Function
def _relay_if_op(pred, true_branch_func, false_branch_func, operands):
    true_expr = true_branch_func(*operands)
    false_expr = false_branch_func(*operands)
    return relay.If(pred, true_expr, false_expr)

# Relay model builders and NumPy reference calculators for each original PyTorch model

# CondModels.Simple
def _build_simple_relay(relay_input_vars):
    p, a, b = relay_input_vars[0], relay_input_vars[1], relay_input_vars[2]

    x_true = relay.var("x_true", type_annotation=a.checked_type)
    y_true = relay.var("y_true", type_annotation=b.checked_type)
    true_fn_expr = relay.Function([x_true, y_true], relay_op.add(x_true, y_true))

    x_false = relay.var("x_false", type_annotation=a.checked_type)
    y_false = relay.var("y_false", type_annotation=b.checked_type)
    false_fn_expr = relay.Function([x_false, y_false], relay_op.subtract(x_false, y_false))

    return relay.Function(relay_input_vars, _relay_if_op(p, true_fn_expr, false_fn_expr, (a, b)))

def _ref_calc_simple(inputs_np_tuple):
    p_np, a_np, b_np = inputs_np_tuple
    if p_np.item():
        return a_np + b_np
    else:
        return a_np - b_np

# CondModels.SimpleWithIntClosure
def _build_simplewithintclosure_relay(relay_input_vars):
    p, a, b = relay_input_vars[0], relay_input_vars[1], relay_input_vars[2]
    num_const = relay.const(3, "int64") # self.num = 3

    x_true = relay.var("x_true", type_annotation=a.checked_type)
    y_true = relay.var("y_true", type_annotation=b.checked_type)
    true_fn_expr = relay.Function(
        [x_true, y_true],
        relay.Tuple([relay_op.add(relay_op.add(x_true, y_true), num_const)])
    )

    x_false = relay.var("x_false", type_annotation=a.checked_type)
    y_false = relay.var("y_false", type_annotation=b.checked_type)
    false_fn_expr = relay.Function(
        [x_false, y_false],
        relay.Tuple([relay_op.subtract(relay_op.subtract(x_false, y_false), num_const)])
    )
    return relay.Function(relay_input_vars, _relay_if_op(p, true_fn_expr, false_fn_expr, (a, b)))

def _ref_calc_simplewithintclosure(inputs_np_tuple):
    p_np, a_np, b_np = inputs_np_tuple
    num_val = 3
    if p_np.item():
        return (a_np + b_np + num_val,)
    else:
        return (a_np - b_np - num_val,)

# CondModels.Nested
def _build_nested_relay(relay_input_vars):
    p0, p1, p2, a, b, c = relay_input_vars[0], relay_input_vars[1], relay_input_vars[2], relay_input_vars[3], relay_input_vars[4], relay_input_vars[5]
    
    x_type = a.checked_type # Assuming all operands have the same type
    x_shape = x_type.shape
    x_dtype = x_type.dtype

    # Inner-most functions
    x_in_tft = relay.var("x_in", type_annotation=x_type)
    y_in_tft = relay.var("y_in", type_annotation=x_type)
    z_in_tft = relay.var("z_in", type_annotation=x_type)
    true_false_true_fn_expr = relay.Function([x_in_tft, y_in_tft, z_in_tft],
                                             relay_op.divide(relay_op.multiply(relay_op.multiply(x_in_tft, y_in_tft), z_in_tft), relay.const(2.71, x_dtype)))

    x_in_tff = relay.var("x_in", type_annotation=x_type)
    y_in_tff = relay.var("y_in", type_annotation=x_type)
    z_in_tff = relay.var("z_in", type_annotation=x_type)
    true_false_false_fn_expr = relay.Function([x_in_tff, y_in_tff, z_in_tff],
                                              relay_op.multiply(relay_op.add(relay_op.add(x_in_tff, y_in_tff), z_in_tff), relay.const(1.23, x_dtype)))

    x_in_ftt = relay.var("x_in", type_annotation=x_type)
    y_in_ftt = relay.var("y_in", type_annotation=x_type)
    z_in_ftt = relay.var("z_in", type_annotation=x_type)
    false_true_true_fn_expr = relay.Function([x_in_ftt, y_in_ftt, z_in_ftt],
                                             relay_op.add(relay_op.subtract(relay_op.subtract(x_in_ftt, y_in_ftt), z_in_ftt), relay.const(1.23, x_dtype)))

    x_in_ftf = relay.var("x_in", type_annotation=x_type)
    y_in_ftf = relay.var("y_in", type_annotation=x_type)
    z_in_ftf = relay.var("z_in", type_annotation=x_type)
    false_true_false_fn_expr = relay.Function([x_in_ftf, y_in_ftf, z_in_ftf],
                                              relay_op.subtract(relay_op.divide(relay_op.divide(x_in_ftf, y_in_ftf), z_in_ftf), relay.const(3.14, x_dtype)))
    
    # Intermediate functions
    x_in_tt = relay.var("x_in", type_annotation=x_type)
    y_in_tt = relay.var("y_in", type_annotation=x_type)
    z_in_tt = relay.var("z_in", type_annotation=x_type)
    true_true_fn_expr = relay.Function([x_in_tt, y_in_tt, z_in_tt],
                                       relay_op.multiply(
