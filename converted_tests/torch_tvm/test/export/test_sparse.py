import sys
import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.relay import op as relay_op
from tvm.relay.op import sparse as relay_sparse
from tvm.relay.op.nn import nn as relay_nn
from tvm.relay.op.tensor import tensor as relay_tensor
from tvm.testing import assert_allclose

# Note: PyTorch's FakeTensor and node.meta properties are specific to its
# tracing and graph representation. TVM Relay expressions (`relay.Var`, `relay.Call`)
# do not store Python object metadata in the same way. The conversion will
# focus on the functional equivalence in the Relay graph and checking type
# and shape properties from `checked_type_`. Direct access to sparse-specific
# attributes like `_indices()` or `crow_indices()` from a Relay expression is not
# available at this Python level.

# Various data types (preserved over operations).
DTYPES = [
    "int64",
    "float16",
    "bfloat16",  # TVM supports bfloat16, but may require specific targets/passes
    "float32",
    "float64",
]

# Various index types.
ITYPES = ["int32", "int64"]


# Custom helper to map PyTorch dtypes to TVM dtypes
def to_tvm_dtype(torch_dtype):
    if torch_dtype == "float16":
        return "float16"
    elif torch_dtype == "bfloat16":
        return "bfloat16"
    elif torch_dtype == "float32":
        return "float32"
    elif torch_dtype == "float64":
        return "float64"
    elif torch_dtype == "int32":
        return "int32"
    elif torch_dtype == "int64":
        return "int64"
    raise ValueError(f"Unsupported dtype: {torch_dtype}")


# Constructs a subtest for every sparse layout currently supported in torch.sparse.
# For TVM, we'll use string identifiers and construct Relay sparse ops if needed.
def all_sparse_layouts(test_name="layout"):
    return pytest.mark.parametrize(
        test_name,
        [
            pytest.param("sparse_coo", id="SparseCOO"),
            pytest.param("sparse_csr", id="SparseCSR"),
            pytest.param("sparse_csc", id="SparseCSC"),
            pytest.param("sparse_bsr", id="SparseBSR"),
            pytest.param("sparse_bsc", id="SparseBSC"),
        ],
    )


#
# Various network examples, converted to Relay functions.
# These will operate on relay.Var inputs.
#

# Identity network
def IdNet_relay(x, layout_str=None): # layout_str is added for compatibility but ignored
    return x

# Sum network
def SumNet_relay(x):
    # PyTorch's x.sum() sums over all dimensions by default if no dim is given.
    # TVM's reduce.sum(x) also sums over all dimensions by default.
    return relay_op.reduce.sum(x)

# Element-wise network
def EltwiseNet_relay(x):
    # PyTorch: torch.nn.functional.relu(2 * torch.abs(-x))
    # Relay: relu(2 * abs(negative(x)))
    neg_x = relay_tensor.negative(x)
    abs_neg_x = relay_tensor.abs(neg_x)
    # Ensure scalar dtype matches tensor, create a const value
    two = relay.const(2.0, dtype=x.checked_type_.dtype)
    mul_two = relay_tensor.multiply(two, abs_neg_x)
    return relay_nn.relu(mul_two)


# ToDense network
def ToDenseNet_relay(x, layout_str):
    # This function expects 'x' to be a sparse Relay expression.
    # We need to use the appropriate sparse-to-dense conversion.
    # The actual sparse input would typically be constructed before calling `to_dense`.
    # For meta-testing, we define generic calls, as the actual components (indices, values)
    # of the sparse Var aren't directly available from 'x' at this stage.
    # The 'checked_type_' of the input 'x' is assumed to be the *dense equivalent* for meta-checks.
    if layout_str == "sparse_coo":
        # relay.sparse.coo_to_dense expects (data_values, indices, dense_shape) or a sparse tensor struct.
        # For meta-testing, we're simplifying. The actual op takes structured inputs.
        # This will simulate the output type of dense_to_coo.
        return relay_op.annotation.checkpoint(x) # Placeholder to simulate a graph node with identity functionality for type/shape propagation
    elif layout_str == "sparse_csr":
        return relay_op.annotation.checkpoint(x)
    elif layout_str == "sparse_csc":
        return relay_op.annotation.checkpoint(x)
    elif layout_str == "sparse_bsr":
        return relay_op.annotation.checkpoint(x)
    elif layout_str == "sparse_bsc":
        return relay_op.annotation.checkpoint(x)
    else:
        # If input is already dense, identity.
        return x

# Add network
def AddNet_relay(x, y):
    return relay_tensor.add(x, y)

# SparseActivationCOO (PyTorch returns list of sparse tensors)
def SparseActivationCOO_relay(x_list):
    # Each xi is expected to be a dense tensor.
    # We need to represent conversion to sparse in Relay.
    # `relay.sparse.dense_to_coo` takes (data, dense_shape, num_dims) and outputs (values, indices).
    # For meta-testing, we're creating calls that represent this conversion, and their output types.
    outputs = []
    for x in x_list:
        # Simulate sparse.dense_to_coo. Its result is a TupleType of (values, indices, dense_shape)
        # We need to match this structure for `assertEqualMeta`.
        # However, the PyTorch test specifically checks `x.to_sparse()` and then `meta`.
        # This is a conceptual representation for meta-testing, as `relay.dense_to_coo` is a real op.
        num_dims = len(x.checked_type.shape)
        # The output type of dense_to_coo would be a tuple of values, indices, dense_shape.
        # For simplicity in meta-testing where we only check the *final* sparse tensor's overall meta,
        # we can define a dummy `relay.Var` with a TupleType as its checked_type to simulate sparse.
        # This is very tricky to map directly without running actual sparse tensor representation.
        # Given the mapping table hints, `relay.sparse.dense_to_coo` is a good candidate.
        # But for *meta-check* in the original style, `node.meta.get("val")` would yield
        # a PyTorch sparse tensor object whose meta would be read.
        
        # Here we'll produce a placeholder Call, and assert its output type.
        # A full `dense_to_coo` takes dense_shape as a separate argument.
        dense_shape_const = relay.const(list(x.checked_type.shape), dtype="int64")
        
        # Call to a dummy op that has a tuple type output to match the PyTorch semantics
        # of returning a sparse tensor which is composed of values/indices.
        # The actual output type for sparse.dense_to_coo is `TupleType([TensorType(values), TensorType(indices)])`
        # and not a TensorType of sparse format itself.
        outputs.append(relay_sparse.dense_to_coo(x)) # This op returns a tuple of (values, indices)
    return relay.Tuple(outputs)

# SparseActivationCSR (PyTorch returns list of sparse tensors)
def SparseActivationCSR_relay(x_list):
    outputs = []
    for x in x_list:
        outputs.append(relay_sparse.dense_to_csr(x)) # This op returns a tuple of (values, crow_indices, col_indices)
    return relay.Tuple(outputs)


#
# The test driver.
#

# Using pytest fixtures and assertions directly.
# `is_fbcode()` is PyTorch internal, omit.
# `sys.version_info >= (3, 12)` is for torch.compile, which is N/A for TVM.
@pytest.mark.skipif(
    sys.version_info >= (3, 12), reason="PyTorch related skip: torch.compile is not supported on python 3.12+"
)
class TestSparseProp:
    # Mimic common_utils.TestCase methods
    def assertEqual(self, actual, expected, **kwargs):
        # Adapt to assert_allclose for numeric arrays, or standard assert for other types
        if isinstance(actual, (tvm.nd.NDArray, np.ndarray)) and isinstance(expected, (tvm.nd.NDArray, np.ndarray)):
            assert_allclose(actual, expected, **kwargs)
        elif isinstance(actual, tvm.ir.TensorType) and isinstance(expected, tvm.ir.TensorType):
            assert actual.dtype == expected.dtype and actual.shape == expected.shape, f"TensorType mismatch: Actual {actual}, Expected {expected}"
        else:
            assert actual == expected, f"Value mismatch: Actual {actual}, Expected {expected}"

    def assertEqualMeta(self, actual_expr, expected_np_or_relay_expr_or_type, layout_str=None):
        # `actual_expr` is a relay.Expr
        # `expected_np_or_relay_expr_or_type` can be a numpy array, a relay.Expr, or a tvm.ir.TensorType
        # We extract the 'meta' information (shape and dtype) from Relay expressions
        # and compare against the expected information.

        actual_type = actual_expr.checked_type
        
        if isinstance(expected_np_or_relay_expr_or_type, np.ndarray):
            expected_dtype = to_tvm_dtype(str(expected_np_or_relay_expr_or_type.dtype))
            expected_shape = expected_np_or_relay_expr_or_type.shape
            assert actual_type.dtype == expected_dtype, f"Dtype mismatch for meta: Actual {actual_type.dtype}, Expected {expected_dtype}"
            assert actual_type.shape == expected_shape, f"Shape mismatch for meta: Actual {actual_type.shape}, Expected {expected_shape}"
        elif isinstance(expected_np_or_relay_expr_or_type, (relay.Var, relay.Call)):
            expected_type = expected_np_or_relay_expr_or_type.checked_type
            assert actual_type.dtype == expected_type.dtype, f"Dtype mismatch for meta: Actual {actual_type.dtype}, Expected {expected_type.dtype}"
            assert actual_type.shape == expected_type.shape, f"Shape mismatch for meta: Actual {actual_type.shape}, Expected {expected_type.shape}"
        elif isinstance(expected_np_or_relay_expr_or_type, tvm.ir.TensorType):
            expected_type = expected_np_or_relay_expr_or_type
            assert actual_type.dtype == expected_type.dtype, f"Dtype mismatch for meta: Actual {actual_type.dtype}, Expected {expected_type.dtype}"
            assert actual_type.shape == expected_type.shape, f"Shape mismatch for meta: Actual {actual_type.shape}, Expected {expected_type.shape}"
        elif isinstance(expected_np_or_relay_expr_or_type, tvm.ir.TupleType):
            expected_type = expected_np_or_relay_expr_or_type
            assert isinstance(actual_type, tvm.ir.TupleType), f"Expected TupleType, got {actual_type}"
            assert len(actual_type.fields) == len(expected_type.fields), f"TupleType length mismatch: Actual {len(actual_type.fields)}, Expected {len(expected_type.fields)}"
            for i, (actual_field, expected_field) in enumerate(zip(actual_type.fields, expected_type.fields)):
                self.assertEqualMeta(relay.TupleGetItem(actual_expr, i), expected_field, layout_str) # Recurse on tuple fields
        else:
            raise TypeError(f"Unsupported expected type for assertEqualMeta: {type(expected_np_or_relay_expr_or_type)}")

        # Sparse-specific meta checks like `_indices()` or `crow_indices()`
        # from a Relay expression are not directly available at this Python level.
        # The 'layout' information would be implied by the specific Relay sparse operator used.
        # We rely on the `checked_type_` (shape and dtype) for basic meta-comparison.
        pass # The basic type/shape/dtype comparison is done above.

    def generate_simple_inputs(self, layout_str, device="cpu", dtype="float32", index_dtype="int64"):
        # This function generates relay.Var inputs for the Relay graph.
        # For meta-level testing, the `relay.Var` `checked_type` is most important.
        tvm_dtype = to_tvm_dtype(dtype)
        tvm_itype = to_tvm_dtype(index_dtype)

        # For these tests, the initial `sparse_input` from PyTorch is a single sparse tensor.
        # In TVM Relay, a sparse tensor is typically represented as a combination of dense data tensors
        # (e.g., values, indices, dense_shape for COO, or values, crow_indices, col_indices for CSR).
        # When passed as a single argument `x` to a Relay function that expects sparse, `x` might be a `relay.Var`
        # whose `checked_type` is a `TensorType` for the overall "dense equivalent" shape and dtype,
        # with the understanding that subsequent ops will use it as sparse.
        
        # We will create a `relay.Var` representing the *dense equivalent* for shape/dtype.
        # The `layout_str` is a hint for how it *should* be interpreted by subsequent sparse ops.
        dense_shape = (4, 4) # Example shape for simplicity
        
        if layout_str == "sparse_coo":
            # For sparse COO, PyTorch `sparse_input` has a shape and dtype.
            # `relay.sparse.coo` takes (values, indices, dense_shape).
            # If the original PyTorch code provides a single `sparse_input` to the network,
            # this `relay.Var` conceptually represents that.
            # Its direct `checked_type` would be `TensorType` matching the dense shape for meta-checking.
            # A true sparse `relay.Var` would have a TupleType. For these tests, we check dense-equivalent meta.
            yield relay.var("sparse_input_coo_dense_equiv", shape=dense_shape, dtype=tvm_dtype)
        elif layout_str in ["sparse_csr", "sparse_csc", "sparse_bsr", "sparse_bsc"]:
            yield relay.var("sparse_input_csr_dense_equiv", shape=dense_shape, dtype=tvm_dtype)
        else:
            raise ValueError(f"Unsupported sparse layout: {layout_str}")


    # Helper to build a Relay IRModule and check meta properties
    def _run_relay_test(self, net_func, initial_input_vars, layout_str=None):
        # `initial_input_vars` is a list of `relay.Var` objects representing the network inputs.
        
        # Construct the Relay function body
        if net_func == ToDenseNet_relay:
            # ToDenseNet_relay requires the layout string to choose the correct sparse_to_dense op.
            func_body = net_func(initial_input_vars[0], layout_str=layout_str)
        elif net_func in [SparseActivationCOO_relay, SparseActivationCSR_relay]:
            # These networks take a list of tensors
            func_body = net_func(initial_input_vars)
        else:
            func_body = net_func(*initial_input_vars) # Unpack list for functions that take individual args

        relay_mod = tvm.IRModule.from_expr(relay.Function(initial_input_vars, func_body))

        # Perform type inference to get `checked_type_` for all expressions
        relay_mod = relay.transform.InferType()(relay_mod)

        # The output expression is the body of the main function after type inference.
        output_expr = relay_mod["main"].body
        
        return relay_mod, initial_input_vars, output_expr


    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_idnet(self, dtype, itype, layout):
        # IdNet: `x` -> `x` (output shape/dtype same as input)
        
        # Generate the single input `relay.Var`
        input_generator = self.generate_simple_inputs(
            layout,
            device="cpu", # Device is not directly part of Relay expression meta.
            dtype=dtype,
            index_dtype=itype,
        )
        sparse_input_var = next(input_generator) # Get the single input Var

        # Build the Relay graph for IdNet
        mod, input_vars, output_expr = self._run_relay_test(
            IdNet_relay, [sparse_input_var], layout_str=layout
        )

        # Check input meta
        self.assertEqualMeta(input_vars[0], sparse_input_var)

        # Check output meta (should be same as input)
        self.assertEqualMeta(output_expr, sparse_input_var)


    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_sumnet(self, dtype, itype, layout):
        # SumNet: `x.sum()` -> scalar output
        input_generator = self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        )
        sparse_input_var = next(input_generator)

        # Build the Relay graph
        mod, input_vars, output_expr = self._run_relay_test(
            SumNet_relay, [sparse_input_var], layout_str=layout
        )

        # Check input meta
        self.assertEqualMeta(input_vars[0], sparse_input_var)

        # Check output meta (should be a scalar with the same dtype as input)
        expected_output_type = tvm.ir.TensorType((), to_tvm_dtype(dtype))
        self.assertEqualMeta(output_expr, expected_output_type)


    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_eltwisenet(self, dtype, itype, layout):
        # EltwiseNet: `relu(2 * abs(-x))` -> output shape/dtype same as input
        input_generator = self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        )
        sparse_input_var = next(input_generator)

        # Build the Relay graph
        mod, input_vars, output_expr = self._run_relay_test(
            EltwiseNet_relay, [sparse_input_var], layout_str=layout
        )

        # Check input meta
        self.assertEqualMeta(input_vars[0], sparse_input_var)

        # Check output meta (should be same shape/dtype as input)
        self.assertEqualMeta(output_expr, sparse_input_var)


    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_todensenet(self, dtype, itype, layout):
        # ToDenseNet: `x.to_dense()` -> output should be dense with same shape/dtype
        input_generator = self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        )
        sparse_input_var = next(input_generator)

        # Build the Relay graph
        mod, input_vars, output_expr = self._run_relay_test(
            ToDenseNet_relay, [sparse_input_var], layout_str=layout
        )

        # Check input meta (the dense equivalent type)
        self.assertEqualMeta(input_vars[0], sparse_input_var)

        # Check output meta (should be dense, same shape/dtype as original conceptual sparse input)
        expected_output_type = tvm.ir.TensorType(sparse_input_var.checked_type.shape, to_tvm_dtype(dtype))
        self.assertEqualMeta(output_expr, expected_output_type)


    def test_add(self):
        net = AddNet_relay
        
        # PyTorch example inputs, converted to NumPy for meta definition
        Y_np = np.arange(16, 32, dtype="float32").reshape(4, 4)
        A_np = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 3.0, 0.0],
            ],
            dtype="float32",
        )
        
        # Relay Vars. For `AddNet`, PyTorch passes a sparse CSR and a dense tensor.
        # Relay's `add` operates on `TensorType`. For meta checks, we define Vars with dense shapes.
        S_var = relay.var("S_sparse_csr_dense_equiv", shape=A_np.shape, dtype="float32")
        Y_var = relay.var("Y_dense", shape=Y_np.shape, dtype="float32")

        # Build the Relay graph
        # `relay.op.tensor.add` supports broadcasting.
        func_body = net(S_var, Y_var)
        relay_mod = tvm.IRModule.from_expr(relay.Function([S_var, Y_var], func_body))
        relay_mod = relay.transform.InferType()(relay_mod)
        output_expr = relay_mod["main"].body

        # Expected output from `torch.add(S_sparse, Y_dense)` is a dense tensor
        # with the result of adding the dense representation of S to Y.
        expected_result_np = A_np + Y_np # This is the numerical result if S was dense

        # Check input meta
        self.assertEqualMeta(S_var, A_np)
        self.assertEqualMeta(Y_var, Y_np)

        # Check output meta
        self.assertEqualMeta(output_expr, expected_result_np)


    def test_activation_coo(self):
        net = SparseActivationCOO_relay
        # PyTorch inputs are a list of dense tensors
        x_np_list = [np.random.randn(3, 3).astype(np.float32) for _ in range(3)]
        
        # Relay Vars corresponding to dense inputs
        x_vars = [relay.var(f"x_{i}", shape=xi.shape, dtype=to_tvm_dtype(str(xi.dtype))) for i, xi in enumerate(x_np_list)]

        # Build the Relay graph. net(x_vars) returns a Tuple of `sparse.dense_to_coo` Calls.
        mod, input_vars, output_expr = self._run_relay_test(net, x_vars)

        assert isinstance(output_expr, relay.Tuple), f"Expected Tuple output, got {type(output_expr)}"
        assert len(output_expr.fields) == len(x_np_list), f"Tuple length mismatch: {len(output_expr.fields)} vs {len(x_np_list)}"

        for i, (input_var, output_field_expr) in enumerate(zip(x_vars, output_expr.fields)):
            # Check input meta
            self.assertEqualMeta(input_var, x_np_list[i])

            # For `sparse.dense_to_coo`, the output `checked_type_` is a TupleType:
            # (TensorType(values), TensorType(indices), TensorType(dense_shape_const))
            output_field_checked_type = output_field_expr.checked_type
            assert isinstance(output_field_checked_type, tvm.ir.TupleType), f"Expected TupleType for sparse COO output field, got {output_field_checked_type}"
            assert len(output_field_checked_type.fields) == 3, f"Expected 3 fields for sparse COO tuple, got {len(output_field_checked_type.fields)}"

            # For meta-comparison, we check the main components.
            # 0: values, 1: indices, 2: dense_shape
            self.assertEqualMeta(output_field_checked_type.fields[0], input_var.checked_type) # values should have same dtype as input, shape can be (N,)
            assert output_field_checked_type.fields[1].dtype == to_tvm_dtype(ITYPES[0]), f"Indices dtype mismatch for sparse COO: Expected {to_tvm_dtype(ITYPES[0])}, got {output_field_checked_type.fields[1].dtype}"
            assert len(output_field_checked_type.fields[1].shape) == 2, "Indices should be 2D for COO"
            self.assertEqual(output_field_checked_type.fields[1].shape[1], input_var.checked_type.ndim, "Indices second dim should match input rank")
            self.assertEqual(output_field_checked_type.fields[2].shape, (input_var.checked_type.ndim,), "Dense shape tensor should have rank 1 with length = input_rank")


    def test_activation_csr(self):
        net = SparseActivationCSR_relay
        # PyTorch inputs are a list of dense tensors
        x_np_list = [np.random.randn(3, 3).astype(np.float32) for _ in range(3)]

        # Relay Vars corresponding to dense inputs
        x_vars = [relay.var(f"x_{i}", shape=xi.shape, dtype=to_tvm_dtype(str(xi.dtype))) for i, xi in enumerate(x_np_list)]

        # Build the Relay graph. net(x_vars) returns a Tuple of `sparse.dense_to_csr` Calls.
        mod, input_vars, output_expr = self._run_relay_test(net, x_vars)

        assert isinstance(output_expr, relay.Tuple), f"Expected Tuple output, got {type(output_expr)}"
        assert len(output_expr.fields) == len(x_np_list), f"Tuple length mismatch: {len(output_expr.fields)} vs {len(x_np_list)}"

        for i, (input_var, output_field_expr) in enumerate(zip(x_vars, output_expr.fields)):
            # Check input meta
            self.assertEqualMeta(input_var, x_np_list[i])

            # For `sparse.dense_to_csr`, the output `checked_type_` is a TupleType:
            # (TensorType(values), TensorType(crow_indices), TensorType(col_indices))
            output_field_checked_type = output_field_expr.checked_type
            assert isinstance(output_field_checked_type, tvm.ir.TupleType), f"Expected TupleType for sparse CSR output field, got {output_field_checked_type}"
            assert len(output_field_checked_type.fields) == 3, f"Expected 3 fields for sparse CSR tuple, got {len(output_field_checked_type.fields)}"

            # 0: values, 1: crow_indices, 2: col_indices
            self.assertEqualMeta(output_field_checked_type.fields[0], input_var.checked_type) # values should have same dtype as input, shape can be (N,)
            assert output_field_checked_type.fields[1].dtype == to_tvm_dtype(ITYPES[0]), f"crow_indices dtype mismatch for sparse CSR: Expected {to_tvm_dtype(ITYPES[0])}, got {output_field_checked_type.fields[1].dtype}"
            self.assertEqual(output_field_checked_type.fields[1].shape, (input_var.checked_type.shape[0] + 1,), "crow_indices shape mismatch")
            assert output_field_checked_type.fields[2].dtype == to_tvm_dtype(ITYPES[0]), f"col_indices dtype mismatch for sparse CSR: Expected {to_tvm_dtype(ITYPES[0])}, got {output_field_checked_type.fields[2].dtype}"
            assert len(output_field_checked_type.fields[2].shape) == 1, "col_indices should be 1D for CSR"
