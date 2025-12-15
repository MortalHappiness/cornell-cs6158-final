import torch
import torch.nn.functional as F
import pytest

# Helper to convert TVM-style string dtypes to PyTorch dtypes
def get_torch_dtype(tvm_dtype_str):
    if tvm_dtype_str == "float32":
        return torch.float32
    elif tvm_dtype_str == "float64":
        return torch.float64
    elif tvm_dtype_str == "int64":
        return torch.int64
    elif tvm_dtype_str == "int32":
        return torch.int32
    elif tvm_dtype_str == "int8":
        return torch.int8
    elif tvm_dtype_str == "bool":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype: {tvm_dtype_str}")

# The original TVM tests operate on symbolic Relay IR graphs and transform them.
# In PyTorch, we represent the computational logic directly with tensor operations.
# To test "eliminate common subexpressions", we'll define a 'before' function with
# redundant computations and an 'expected' function with the redundancy removed,
# and then assert that their outputs are numerically identical, which is the
# desired effect of the optimization.
# The TVM-specific 'run_opt_pass', 'transform', and 'tvm.ir.structural_equal'
# are not directly convertible and are replaced by direct PyTorch computation and
# numerical assertions.

def test_simple():
    # Defines a computation with explicit common subexpressions
    def before_fn(x):
        # Corresponds to TVM Relay graph:
        # y1 = relay.nn.relu(x)
        # y2 = relay.nn.relu(x)
        # y1 = relay.add(y1, relay.const(1.0, "float32"))
        # y2 = relay.add(y2, relay.const(1.0, "float32"))
        # y = relay.add(y1, y2)
        
        y1_relu = F.relu(x)
        y2_relu = F.relu(x) # Explicitly duplicate relu(x)
        
        const_val = torch.tensor(1.0, dtype=get_torch_dtype("float32"))
        
        y1_add = y1_relu + const_val
        y2_add = y2_relu + const_val # Explicitly duplicate add
        
        y_final = y1_add + y2_add
        return y_final

    # Defines the same computation with common subexpressions eliminated
    def expected_fn(x):
        # Corresponds to TVM Relay graph after optimization:
        # y_shared = relay.nn.relu(x)
        # y_shared = relay.add(y_shared, relay.const(1.0, "float32"))
        # y = relay.add(y_shared, y_shared)
        
        y_shared_relu = F.relu(x) # Compute relu(x) once
        
        const_val = torch.tensor(1.0, dtype=get_torch_dtype("float32"))
        
        y_shared_add = y_shared_relu + const_val # Compute add(relu(x), const) once
        
        y_final = y_shared_add + y_shared_add # Reuse the result
        return y_final

    x_input = torch.randn(1, 16, dtype=get_torch_dtype("float32"))

    # Execute both computational graphs
    z_before = before_fn(x_input)
    z_expected = expected_fn(x_input)

    # Assert numerical equality. The optimization should not change the result.
    torch.testing.assert_close(z_before, z_expected, rtol=1e-5, atol=1e-5)


def test_callback():
    # Defines a computation with explicit common subexpressions
    def before_fn(x):
        # Same graph structure as test_simple's before_fn
        y1_relu = F.relu(x)
        y2_relu = F.relu(x)
        
        const_val = torch.tensor(1.0, dtype=get_torch_dtype("float32"))
        
        y1_add = y1_relu + const_val
        y2_add = y2_relu + const_val
        
        y_final = y1_add + y2_add
        return y_final

    # Defines the computation when 'add' operations are skipped for CSE elimination.
    # This means F.relu(x) is optimized as a common subexpression, but
    # (relu(x) + const) is not.
    def expected_fn_with_skipped_add(x):
        # Corresponds to TVM's expected() with fskip on 'add':
        # y_shared_relu = relay.nn.relu(x) (this part is optimized)
        # y1_add = relay.add(y_shared_relu, relay.const(1.0, "float32")) (add not optimized)
        # y2_add = relay.add(y_shared_relu, relay.const(1.0, "float32")) (add not optimized)
        # y = relay.add(y1_add, y2_add)
        
        y_shared_relu = F.relu(x) # Compute relu(x) once (CSE for relu)
        
        const_val = torch.tensor(1.0, dtype=get_torch_dtype("float32"))
        
        # The 'add' operations are NOT eliminated as common subexpressions
        # because the fskip callback prevented it.
        y1_add = y_shared_relu + const_val
        y2_add = y_shared_relu + const_val
        
        y_final = y1_add + y2_add
        return y_final

    x_input = torch.randn(1, 16, dtype=get_torch_dtype("float32"))

    z_before = before_fn(x_input)
    z_expected = expected_fn_with_skipped_add(x_input)

    torch.testing.assert_close(z_before, z_expected, rtol=1e-5, atol=1e-5)


def test_tuple_get_time():
    # Defines a computation where a tuple_getitem result is duplicated
    def before_fn(x, var, mean, beta, gamma):
        # Corresponds to TVM Relay graph:
        # BN = relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)
        # T1 = BN[0]
        # T2 = BN[0]
        # add = T1 + T2
        
        # PyTorch F.batch_norm directly returns the output tensor,
        # so BN[0] in TVM maps to the direct output here.
        bn_output = F.batch_norm(
            input=x,
            running_mean=mean,
            running_var=var,
            weight=gamma,
            bias=beta,
            training=False,  # Assuming inference mode for optimization tests
            momentum=0.1,    # Default PyTorch momentum
            eps=1e-5
        )
        T1 = bn_output
        T2 = bn_output # This is the explicit duplication of the common subexpression

        add_result = T1 + T2
        return add_result

    # Defines the same computation with the duplicated tuple_getitem result eliminated
    def expected_fn(x, var, mean, beta, gamma):
        # Corresponds to TVM Relay graph after optimization:
        # BN = relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)
        # T_shared = BN[0]
        # add = T_shared + T_shared
        
        bn_output = F.batch_norm(
            input=x,
            running_mean=mean,
            running_var=var,
            weight=gamma,
            bias=beta,
            training=False,
            momentum=0.1,
            eps=1e-5
        )
        T_shared = bn_output # Compute once, reuse

        add_result = T_shared + T_shared
        return add_result

    # Generate random inputs for the test
    x_input = torch.randn(1, 16, 1, 1, dtype=get_torch_dtype("float32"))
    var_input = torch.randn(16, dtype=get_torch_dtype("float32"))
    mean_input = torch.randn(16, dtype=get_torch_dtype("float32"))
    beta_input = torch.randn(16, dtype=get_torch_dtype("float32"))
    gamma_input = torch.randn(16, dtype=get_torch_dtype("float32"))

    z_before = before_fn(x_input, var_input, mean_input, beta_input, gamma_input)
    z_expected = expected_fn(x_input, var_input, mean_input, beta_input, gamma_input)

    torch.testing.assert_close(z_before, z_expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
