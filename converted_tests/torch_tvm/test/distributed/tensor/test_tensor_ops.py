import tvm
from tvm import relay
import numpy as np
import pytest

# Dummy torch types for dtype mapping, as torch cannot be imported
class DummyTorch:
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    int64 = "int64"
    int32 = "int32"
    bool = "bool"
    # Add other dtypes as necessary. For this test, we only see these.

    # Mock `torch.tensor` to return numpy arrays for test setup if needed,
    # but actual ops should be mapped to TVM.
    # The original tests use `torch.tensor` to define values, but then DTensor wraps them.
    # We will use np.array directly.
    def tensor(self, data, dtype=None):
        if dtype:
            return np.array(data, dtype=getattr(np, self.get_numpy_dtype_name(dtype)))
        return np.array(data)
    
    def get_numpy_dtype_name(self, dtype_obj):
        # Maps DummyTorch dtypes to numpy dtype names
        if dtype_obj == self.float32: return "float32"
        if dtype_obj == self.float64: return "float64"
        if dtype_obj == self.bfloat16: return "float16" # numpy uses float16 for bfloat16 sometimes, or needs manual conversion
        if dtype_obj == self.int64: return "int64"
        if dtype_obj == self.int32: return "int32"
        if dtype_obj == self.bool: return "bool"
        return str(dtype_obj) # Fallback

torch = DummyTorch()

# --- Common TVM Test Utilities ---
def get_tvm_target():
    if tvm.cuda().exist:
        return "cuda", tvm.cuda(0)
    return "llvm", tvm.cpu(0)

_TARGET, _DEVICE = get_tvm_target()

def get_tvm_dtype_str(torch_dtype_str):
    # Our DummyTorch dtypes are already TVM-compatible strings
    return torch_dtype_str

def run_tvm_graph(relay_expr, input_dict):
    mod = tvm.IRModule.from_expr(relay_expr)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=_TARGET)
    vm = tvm.runtime.vm.VirtualMachine(lib, _DEVICE)
    tvm_inputs = {name: tvm.nd.array(arr, device=_DEVICE) for name, arr in input_dict.items()}
    tvm_output_nd = vm.invoke_stateful("main", **tvm_inputs)
    
    if isinstance(tvm_output_nd, tvm.runtime.ndarray.NDArray):
        return tvm_output_nd.numpy()
    elif isinstance(tvm_output_nd, tvm.runtime.container.ADT): # Handle tuple outputs
        return tuple(o.numpy() for o in tvm_output_nd)
    else:
        return tvm_output_nd

class TVMTestBase:
    # Mimic common_utils.TestCase assertions
    def assertEqual(self, actual, expected, rtol=1e-5, atol=1e-8, msg=""):
        if isinstance(actual, (np.ndarray, np.generic)) and isinstance(expected, (np.ndarray, np.generic)):
            tvm.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=msg)
        else:
            assert actual == expected, msg

    def assertTrue(self, condition, msg=""):
        assert condition, msg

    def assertFalse(self, condition, msg=""):
        assert not condition, msg

    def assertIsInstance(self, obj, class_or_tuple, msg=""):
        assert isinstance(obj, class_or_tuple), msg

    def assertNotEqual(self, actual, expected, rtol=1e-5, atol=1e-8, msg=""):
        try:
            if isinstance(actual, (np.ndarray, np.generic)) and isinstance(expected, (np.ndarray, np.generic)):
                tvm.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=msg)
            else:
                assert actual == expected, msg
            assert False, f"assertEqual succeeded, but expected not equal. {msg}"
        except AssertionError:
            pass

    def skipTest(self, msg=""):
        pytest.skip(msg) # Using pytest's skip for test functions

    # Helper for running TVM graph and comparing with NumPy reference
    def _run_tvm_and_compare_op(self, np_ref_func, tvm_op_func, *args_raw, rtol=1e-5, atol=1e-8):
        # Convert raw inputs (could be numpy, scalars, or other objects) to standard numpy/scalars for processing
        np_args = []
        tvm_input_data = {} # For actual numpy data passed to the VM
        relay_vars = []     # For relay.var in the graph expression
        
        var_counter = 0
        for arg in args_raw:
            if isinstance(arg, np.ndarray):
                np_args.append(arg)
                var_name = f"input_{var_counter}"
                tvm_input_data[var_name] = arg
                relay_vars.append(relay.var(var_name, shape=arg.shape, dtype=str(arg.dtype)))
                var_counter += 1
            elif isinstance(arg, (int, float, bool, np.number)):
                np_args.append(arg)
                relay_vars.append(relay.const(arg, dtype=str(np.dtype(type(arg))))) # Use numpy dtype for consistency
            else:
                np_args.append(arg) # Pass through other types directly for the `np_ref_func` or `tvm_op_func`
                relay_vars.append(arg) # Same for relay_vars

        # Compute reference output using NumPy/Python equivalent
        out_np_ref = np_ref_func(*np_args)
        if isinstance(out_np_ref, tuple):
            out_np_ref = tuple(o.numpy() if hasattr(o, 'numpy') else o for o in out_np_ref)
        elif hasattr(out_np_ref, 'numpy'):
            out_np_ref = out_np_ref.numpy()

        # Construct the Relay expression
        relay_expr = tvm_op_func(*relay_vars)

        # Run TVM graph
        tvm_output = run_tvm_graph(relay_expr, tvm_input_data)

        # Compare
        self.assertEqual(tvm_output, out_np_ref, rtol=rtol, atol=atol)


# Dummy classes/functions for PyTorch distributed concepts, as they cannot be mapped
class DeviceMesh:
    def __init__(self, device_type, mesh):
        self.device_type = device_type
        self.mesh = mesh
        self.ndim = mesh.ndim
        # Assuming mesh is 1D for world_size
        self.world_size = mesh.size if isinstance(mesh, np.ndarray) else len(mesh)

class Shard:
    def __init__(self, dim):
        self.dim = dim
    def __eq__(self, other):
        return isinstance(other, Shard) and self.dim == other.dim
    def __repr__(self):
        return f"Shard({self.dim})"

class Replicate:
    def __init__(self):
        pass
    def __eq__(self, other):
        return isinstance(other, Replicate)
    def __repr__(self):
        return "Replicate()"

class Partial:
    def __init__(self, reduce_op="sum"):
        self.reduce_op = reduce_op
    def __eq__(self, other):
        return isinstance(other, Partial) and self.reduce_op == other.reduce_op
    def __repr__(self):
        return f"Partial(reduce_op='{self.reduce_op}')"

# Dummy CommDebugMode (distributed debugging, not applicable to local TVM test)
class CommDebugMode:
    def __init__(self):
        self.counts = 0
    def __enter__(self):
        self.counts = 0
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def get_total_counts(self):
        return self.counts

# Dummy skip decorators, just pass through (pytest handles actual skips)
def skip_if_lt_x_gpu(num_gpus):
    def decorator(func):
        return func
    return decorator

def skipIfRocm(func):
    return func

# Dummy with_comms decorator, just pass through (no distributed comms in local TVM test)
def with_comms(func):
    return func

# DTensorConverter and other distributed utilities are not mapped as they are PyTorch-specific.


# --- Converted Tests ---
class DistTensorOpsTest(TVMTestBase):
    device_type = "cpu" # Default to CPU for TVM tests
    world_size = 1 # Single process for local TVM execution by default

    def build_device_mesh(self):
        # Simplified for local testing
        return DeviceMesh(self.device_type, np.arange(self.world_size))

    # The original `_test_op` wrapper from PyTorch test is not directly convertible.
    # Individual test cases will be rewritten to use `_run_tvm_and_compare_op`.
    # For complex indexing, specific helper functions or direct lambda will be used.

    def test_aten_contiguous(self):
        input_np = np.random.randn(16, 32).astype(np.float32)
        
        # NumPy equivalent: `np.ascontiguousarray` ensures a contiguous copy.
        np_ref_func = lambda x: np.ascontiguousarray(x)
        # TVM equivalent: `relay.op.transform.copy` generates a contiguous copy.
        tvm_op_func = lambda x_var: relay.op.transform.copy(x_var)
        
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np)
        
        # Original test also checks DTensor properties like is_contiguous and stride.
        # These are properties of the specific PyTorch DTensor object/local tensor.
        # A TVM NDArray output from `relay.copy` is always contiguous.
        # TODO: PyTorch DTensor-specific properties (is_contiguous, stride) not directly mappable.

    def test_detach(self):
        input_np = np.random.randn(12, 8).astype(np.float32)
        
        # NumPy equivalent: detach creates a new tensor, so a copy is functionally similar.
        np_ref_func = lambda x: x.copy()
        # TVM equivalent: `relay.op.transform.copy` creates a new tensor.
        tvm_op_func = lambda x_var: relay.op.transform.copy(x_var)
        
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np)
        # TODO: PyTorch `requires_grad` and object identity (`is`) checks for `detach` are not directly mappable.

    def test_clone(self):
        input_np = np.random.randn(12, 8).astype(np.float32)
        
        # NumPy equivalent: clone creates a copy.
        np_ref_func = lambda x: x.copy()
        # TVM equivalent: `relay.op.transform.copy` creates a new tensor.
        tvm_op_func = lambda x_var: relay.op.transform.copy(x_var)
        
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np)
        # TODO: PyTorch object identity (`is`) check for `clone` is not directly mappable.

    def test_copy_(self):
        # DTensor.copy_() is an in-place copy. In functional TVM, this is a new tensor.
        # We test the value, ignoring in-place behavior.
        # The test cases use different sharding in PyTorch, which is ignored here.

        # Case 1: basic copy
        src_np = np.random.randn(12, 12).astype(np.float32)
        dst_np_initial = np.zeros((12, 12), dtype=np.float32) # Not directly used in TVM op, but for ref

        np_ref_func_1 = lambda src, dst_initial: src.copy()
        tvm_op_func_1 = lambda src_var, dst_initial_var: relay.op.tensor.copy(src_var) # dst_initial_var is ignored functionally
        self._run_tvm_and_compare_op(np_ref_func_1, tvm_op_func_1, src_np, dst_np_initial)
        
        # Case 2: simple broadcasting copy
        src_np_bcast = np.random.randn(128).astype(np.float32)
        dst_np_initial_bcast = np.zeros((128, 128), dtype=np.float32)
        
        np_ref_func_2 = lambda src, dst_initial: np.broadcast_to(src, dst_initial.shape)
        tvm_op_func_2 = lambda src_var, dst_initial_var: relay.op.transform.broadcast_to(src_var, shape=dst_np_initial_bcast.shape)
        self._run_tvm_and_compare_op(np_ref_func_2, tvm_op_func_2, src_np_bcast, dst_np_initial_bcast)

        # Case 3: complex broadcasting. `src` shape is (64,1), `dst` shape is (16,32,64,128)
        src_np_complex = np.random.randn(64, 1).astype(np.float32)
        dst_np_initial_complex = np.zeros((16, 32, 64, 128), dtype=np.float32)
        
        np_ref_func_3 = lambda src, dst_initial: np.broadcast_to(src, dst_initial.shape)
        tvm_op_func_3 = lambda src_var, dst_initial_var: relay.op.transform.broadcast_to(src_var, shape=dst_np_initial_complex.shape)
        self._run_tvm_and_compare_op(np_ref_func_3, tvm_op_func_3, src_np_complex, dst_np_initial_complex)
        
        # TODO: The complex sharding and redistribution logic of DTensor.copy_() is not directly mappable.
        # Object identity and `full_tensor()` checks are also not directly mappable.

    def test_contiguous(self):
        # This test checks contiguous property after transpose and explicit contiguous call.
        input_np = np.random.rand(3, 5, 6).astype(np.float32)
        
        # 1. Transpose: `dist_tensor.transpose(0, 2)`
        transposed_np = np.transpose(input_np, (0, 2, 1))
        # 2. Contiguous: `new_dt.contiguous()` (after transpose)
        ref_contiguous_np = np.ascontiguousarray(transposed_np)

        # TVM graph: transpose then copy (contiguous)
        tvm_op_func = lambda data_var: relay.op.transform.copy(
            relay.op.transform.transpose(data_var, axes=(0, 2, 1))
        )
        
        self._run_tvm_and_compare_op(lambda x: ref_contiguous_np, tvm_op_func, input_np)
        
        # TODO: PyTorch DTensor-specific checks (is_contiguous, stride, backward propagation) are not directly mappable.

    def test_inplace_op(self):
        # In-place ops (add_, mul_) in PyTorch DTensor. TVM Relay is functional.
        input_np = np.random.randn(12, 3).astype(np.float32)
        scalar_add = 3.0
        scalar_mul = 3.0

        # Reference for add_ (dt_to_add.add_(3))
        np_ref_func_add = lambda x: x + scalar_add
        tvm_op_func_add = lambda x_var: relay.op.tensor.add(x_var, relay.const(scalar_add, dtype=str(x_var.dtype)))
        self._run_tvm_and_compare_op(np_ref_func_add, tvm_op_func_add, input_np)

        # Reference for mul_ (dt_to_mul.mul_(3))
        np_ref_func_mul = lambda x: x * scalar_mul
        tvm_op_func_mul = lambda x_var: relay.op.tensor.multiply(x_var, relay.const(scalar_mul, dtype=str(x_var.dtype)))
        self._run_tvm_and_compare_op(np_ref_func_mul, tvm_op_func_mul, input_np)

        # TODO: Object identity and DTensor placement checks are not mappable.
        # The `partial_grad` part implies distributed gradient, not mappable.

    def test_op_out_variant(self):
        # PyTorch `out` variant `torch.add(..., out=result_tensor)`. TVM Relay is functional.
        input_np = np.random.randn(12, 3).astype(np.float32)
        scalar_add = 3.0

        # Reference:
        np_ref_func = lambda x: x + scalar_add
        tvm_op_func = lambda x_var: relay.op.tensor.add(x_var, relay.const(scalar_add, dtype=str(x_var.dtype)))
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np)

        # TODO: Object identity and DTensor placement checks are not mappable.

    def test_empty_like(self):
        # `torch.empty_like(dist_tensor)`. Maps to `zeros_like` for functional graph.
        input_np = np.random.randn(4, 8).astype(np.float32)
        
        np_ref_func = lambda x: np.zeros_like(x) # Functional equivalent for testing value
        tvm_op_func = lambda x_var: relay.op.tensor.zeros_like(x_var)
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np)
        # TODO: DTensor specific placement assertion is not mappable.

    def test_fill_inplace(self):
        # `torch.fill_(dist_tensor, 42.0)`. In-place fill. Maps to `full_like`.
        input_np = np.random.randn(4, 8).astype(np.float32)
        fill_value = 42.0

        np_ref_func = lambda x, val: np.full_like(x, val)
        tvm_op_func = lambda x_var, val: relay.op.transform.full_like(x_var, relay.const(val, dtype=str(x_var.dtype)))
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np, fill_value)
        # TODO: In-place nature and DTensor specific checks are not mappable.

    def test_full_like(self):
        # `torch.full_like(dist_tensor, 42.0)`
        input_np = np.random.randn(4, 8).astype(np.float32)
        fill_value = 42.0

        np_ref_func = lambda x, val: np.full_like(x, val)
        tvm_op_func = lambda x_var, val: relay.op.transform.full_like(x_var, relay.const(val, dtype=str(x_var.dtype)))
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np, fill_value)

    def test_ones_like(self):
        # `torch.ones_like(dist_tensor)`
        input_np = np.random.randn(4, 8).astype(np.float32)

        np_ref_func = lambda x: np.ones_like(x)
        tvm_op_func = lambda x_var: relay.op.tensor.ones_like(x_var)
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np)

    def test_ones_like_partial_sum(self):
        # `torch.ones_like` then `DTensor.full_tensor()` on `Partial` placement.
        # `full_tensor()` on Partial sums over world_size.
        
        self.world_size = 4 
        input_np = np.random.randn(4, 8).astype(np.float32)

        # NumPy ref: local ones_like, then scale by world_size to simulate full_tensor on Partial.
        np_ref_func = lambda x, world_size: np.ones_like(x) * world_size
        
        # The _run_tvm_and_compare_op helper runs the tvm_op_func. We then manually apply the aggregation.
        # This requires overriding the standard helper slightly.
        data_var = relay.var("data", shape=input_np.shape, dtype=str(input_np.dtype))
        relay_expr_local_ones = relay.op.tensor.ones_like(data_var)
        tvm_local_ones_np = run_tvm_graph(relay_expr_local_ones, {"data": input_np})
        
        ref_expected_aggregated = np_ref_func(input_np, self.world_size)
        tvm_output_aggregated = tvm_local_ones_np * self.world_size # Simulate full_tensor for Partial
        
        self.assertEqual(tvm_output_aggregated, ref_expected_aggregated)
        # TODO: The `Partial()` placement concept is not directly mappable.

    def test_fill_inplace_partial_sum(self):
        # `torch.fill_()` then `DTensor.full_tensor()` on `Partial` placement.
        self.world_size = 4
        input_np = np.random.randn(4, 8).astype(np.float32)
        fill_value = 8.0

        # NumPy ref: local fill, then scale by world_size.
        np_ref_func = lambda x, val, world_size: np.full_like(x, val) * world_size
        
        data_var = relay.var("data", shape=input_np.shape, dtype=str(input_np.dtype))
        relay_expr_local_fill = relay.op.transform.full_like(data_var, relay.const(fill_value, dtype=str(input_np.dtype)))
        tvm_local_fill_np = run_tvm_graph(relay_expr_local_fill, {"data": input_np})
        
        ref_expected_aggregated = np_ref_func(input_np, fill_value, self.world_size)
        tvm_output_aggregated = tvm_local_fill_np * self.world_size
        
        self.assertEqual(tvm_output_aggregated, ref_expected_aggregated)
        # TODO: The `Partial()` placement and in-place semantics are not directly mappable.

    def test_zeros_like_partial_sum(self):
        # `torch.zeros_like()` then `DTensor.full_tensor()` on `Partial` placement.
        self.world_size = 4
        input_np = np.random.randn(4, 8).astype(np.float32)

        # NumPy ref: local zeros_like, then scale by world_size. Sum of zeros is still zero.
        np_ref_func = lambda x, world_size: np.zeros_like(x)
        
        data_var = relay.var("data", shape=input_np.shape, dtype=str(input_np.dtype))
        relay_expr_local_zeros = relay.op.tensor.zeros_like(data_var)
        tvm_local_zeros_np = run_tvm_graph(relay_expr_local_zeros, {"data": input_np})
        
        ref_expected_aggregated = np_ref_func(input_np, self.world_size)
        tvm_output_aggregated = tvm_local_zeros_np * self.world_size # Simulate full_tensor for Partial
        
        self.assertEqual(tvm_output_aggregated, ref_expected_aggregated)
        # TODO: The `Partial()` placement concept is not directly mappable.

    def test_zero_inplace(self):
        # `torch.zero_(dist_tensor)`. In-place fill with zero. Maps to `zeros_like`.
        input_np = np.random.randn(4, 8).astype(np.float32)
        
        np_ref_func = lambda x: np.zeros_like(x)
        tvm_op_func = lambda x_var: relay.op.tensor.zeros_like(x_var)
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np)
        # TODO: In-place semantics and DTensor specific checks are not mappable.

    def test_zeros_like(self):
        # `torch.zeros_like(dist_tensor, dtype=torch.bfloat16)`
        input_np = np.random.randn(4, 8).astype(np.float32)
        target_dtype_str = "bfloat16" # PyTorch.bfloat16 maps to "bfloat16"
        target_np_dtype = tvm.runtime.DataType(target_dtype_str).numpy_dtype # Get numpy dtype for comparison
        
        np_ref_func = lambda x: np.zeros(x.shape, dtype=target_np_dtype)
        tvm_op_func = lambda x_var: relay.op.tensor.zeros_like(x_var, dtype=target_dtype_str)
        self._run_tvm_and_compare_op(np_ref_func, tvm_op_func, input_np)
        # TODO: DTensor specific dtype and side effect checks not mappable.

    @skip_if_lt_x_gpu(4)
    def test_stack(self):
        # `torch.stack([dt1, dt2])`
        # Simplified to local NumPy arrays and `relay.op.tensor.stack`.
        
        # Test 1: stack before/after shard dim (dim 0)
        global_input_np = np.random.randn(8, 8).astype(np.float32)
        
        np_ref_func_0 = lambda i1, i2: np.stack([i1, i2], axis=0)
        tvm_op_func_0 = lambda i1_var, i2_var: relay.op.tensor.stack(relay.expr.Tuple([i1_var, i2_var]), axis=0)
        self._run_tvm_and_compare_op(np_ref_func_0, tvm_op_func_0, global_input_np, global_input_np)

        # Test 2: stack before/after shard dim (dim 1)
        np_ref_func_1 = lambda i1, i2: np.stack([i1, i2], axis=1)
        tvm_op_func_1 = lambda i1_var, i2_var: relay.op.tensor.stack(relay.expr.Tuple([i1_var, i2_var]), axis=1)
        self._run_tvm_and_compare_op(np_ref_func_1, tvm_op_func_1, global_input_np, global_input_np)

        # TODO: 2D mesh, Partial placements, and DTensor-specific assertions are not mappable.

    def test_equal(self):
        # `dt1.equal(dt2)` involves all_reduce and comparison.
        # We test element-wise equality followed by global reduction.
        
        input1_np = np.ones((4, 4), dtype=np.float32)
        input2_np_equal = np.ones((4, 4), dtype=np.float32)
        input2_np_diff_local = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.float32)

        # Case 1: tensors are equal
        np_ref_func_eq_true = lambda x, y: np.array_equal(x, y)
        tvm_op_func_eq = lambda x_var, y_var: relay.op.reduce.all(relay.op.tensor.equal(x_var, y_var))
        self._run_tvm_and_compare_op(np_ref_func_eq_true, tvm_op_func_eq, input1_np, input2_np_equal)

        # Case 2: tensors are different
        np_ref_func_eq_false = lambda x, y: np.array_equal(x, y) # This should evaluate to False for the inputs
        self._run_tvm_and_compare_op(np_ref_func_eq_false, tvm_op_func_eq, input1_np, input2_np_diff_local)
        
        # is_same_size check (local shape comparison) - not a graph op
        self.assertTrue(input1_np.shape == input2_np_equal.shape)
        self.assertTrue(input1_np.shape == input2_np_diff_local.shape) # Same shapes
        # TODO: The complex DTensor `is_same_size` and placement checks are not directly mappable.

    def test_new_full(self):
        # `input_dt.new_full((4, 8), 42.0)`
        input_np = np.random.randn(12, 8).astype(np.float32)
        fill_value = 42.0
        
        # Case 1: different shape
        target_shape_diff = (4, 8)
        np_ref_func_diff = lambda data, shape, val: np.full(shape, val, dtype=str(data.dtype))
        tvm_op_func_diff = lambda data_var, shape, val: relay.op.transform.full(relay.const(val, dtype=str(data_var.dtype)), shape=shape)
        self._run_tvm_and_compare_op(np_ref_func_diff, tvm_op_func_diff, input_np, target_shape_diff, fill_value)

        # Case 2: same shape
        target_shape_same = (12, 8)
        np_ref_func_same = lambda data, shape, val: np.full(shape, val, dtype=str(data.dtype))
        tvm_op_func_same = lambda data_var, shape, val: relay.op.transform.full(relay.const(val, dtype=str(data_var.dtype)), shape=shape)
        self._run_tvm_and_compare_op(np_ref_func_same, tvm_op_func_same, input_np, target_shape_same, fill_value)
        
        # TODO: CommDebugMode checks and DTensor placement assertions are not mappable.

    def test_new_empty_strided(self):
        # `input_dt.new_empty_strided((12, 8), (8, 1))`
        # This is about creating a tensor with specific strides, which is a low-level memory layout detail.
        # TVM Relay operates on logical shapes. The `is_contiguous()` and `stride()` checks relate to this.
        # We will test creation of a new tensor with the specified shape. Stride information is lost in Relay graph.
        
        input_np_base = np.random.randn(12, 8).astype(np.float32)
        
        # Case 1: output shape same as input shape, evenly sharded -> output same sharding (Replicated)
        # For simplicity, we just test the resulting shape and type of a zero-filled array.
        output_shape_1 = (12, 8)
        # Original test ensures it is contiguous. `zeros` or `empty` in numpy is typically contiguous.
        np_ref_func_1 = lambda data, shape: np.zeros(shape, dtype=str(data.dtype))
        tvm_op_func_1 = lambda data_var, shape: relay.op.tensor.zeros(shape=shape, dtype=str(data_var.dtype))
        self._run_tvm_and_compare_op(np_ref_func_1, tvm_op_func_1, input_np_base, output_shape_1)

        # Case 2: output shape same as input shape, unevenly sharded -> output replicated (Replicated)
        # Just testing shape and type.
        output_shape_2 = (12, 7) # different size in dim 1
        input_np_uneven = np.random.randn(12, 7).astype(np.float32)
        np_ref_func_2 = lambda data, shape: np.zeros(shape, dtype=str(data.dtype))
        tvm_op_func_2 = lambda data_var, shape: relay.op.tensor.zeros(shape=shape, dtype=str(data_var.dtype))
        self._run_tvm_and_compare_op(np_ref_func_2, tvm_op_func_2, input_np_uneven, output_shape_2)

        # Case 3: output shape different from input shape -> output replicated (Replicated)
        output_shape_3 = (12, 4)
        np_ref_func_3 = lambda data, shape: np.zeros(shape, dtype=str(data.dtype))
        tvm_op_func_3 = lambda data_var, shape: relay.op.tensor.zeros(shape=shape, dtype=str(data_var.dtype))
        self._run_tvm_and_compare_op(np_ref_func_3, tvm_op_func_3, input_np_base, output_shape_3)

        # TODO: CommDebugMode checks and DTensor placement/stride assertions are not mappable.
        self.skipTest("`new_empty_strided` tests low-level stride/contiguous properties not directly in Relay graphs. Only basic shape/dtype checks are possible.")


    def test_scatter(self):
        # `torch.scatter(input_dt, scatter_dim, index_dt, src_dt)`
        # `dt.full_tensor()` verifies result.
        
        # Simple scatter_elements example to mimic torch.scatter (update mode)
        # PyTorch `scatter(input, dim, index, src)` is roughly `input.select(dim, index) = src`.
        # TVM's `scatter_elements(data, indices, updates, axis, reduction="update")`
        # `indices` and `updates` must have the same rank and be broadcastable to `data` along non-axis dims.
        
        data_np_scatter_base = np.zeros((3, 5), dtype=np.int64)

        # Test with scatter_dim = 0
        indices_np_0 = np.array([[0, 1, 2, 0, 1]], dtype=np.int64) # (1,5)
        updates_np_0 = np.arange(1, 6).reshape(1,5).astype(np.int64) # (1,5)
        
        ref_output_0_np = data_np_scatter_base.copy()
        # manual scatter equivalent for numpy
        # For torch.scatter(input, dim=0, index, src), it's input[index[i,j], j] = src[i,j] (for 2D index/src)
        # For 1D index, it's input[index[j], j] = src[j] if index has same number of columns as src
        # Based on PyTorch doc for input.scatter_(dim, index, src):
        # If dim = 0: for each col j, input[index[i,j], j] = src[i,j] for all rows i
        # But here index is (1,5), src is (1,5).
        # So for each column j from 0 to 4: input[index[0,j], j] = src[0,j]
        for j in range(updates_np_0.shape[1]):
            ref_output_0_np[indices_np_0[0, j], j] = updates_np_0[0, j]

        tvm_op_func_0 = lambda d_var, i_var, u_var: relay.op.transform.scatter_elements(
            d_var, i_var, u_var, axis=0, reduction="update"
        )
        self._run_tvm_and_compare_op(
            lambda d, i, u: ref_output_0_np, 
            tvm_op_func_0,
            data_np_scatter_base, indices_np_0, updates_np_0
        )
            
        # Test with scatter_dim = 1
        indices_np_1 = np.array([[0, 1, 2], [0, 1, 4]], dtype=np.int64) # (2,3)
        updates_np_1 = np.arange(10, 16).reshape(2,3).astype(np.int64) # (2,3)

        ref_output_1_np = data_np_scatter_base.copy()
        # For torch.scatter(input, dim=1, index, src):
        # For each row i, input[i, index[i,j]] = src[i,j] for all cols j
        for i in range(updates_np_1.shape[0]):
            for j in range(updates_np_1.shape[1]):
                ref_output_1_np[i, indices_np_1[i, j]] = updates_np_1[i, j]

        tvm_op_func_1 = lambda d_var, i_var, u_var: relay.op.transform.scatter_elements(
            d_var, i_var, u_var, axis=1, reduction="update"
        )
        self._run_tvm_and_compare_op(
            lambda d, i, u: ref_output_1_np,
            tvm_op_func_1,
            data_np_scatter_base, indices_np_1, updates_np_1
        )
        # TODO: The CommDebugMode and DTensor placements are not mappable.
        self.skipTest("PyTorch's scatter with original test shapes leads to complex broadcast rules not easily mapped to TVM scatter_elements. Simplified cases manually tested.")

    def test_gather(self):
        # `torch.gather(input_dt, gather_dim, index_dt)`
        
        global_input_np = np.random.randn(12, 8, 16).astype(np.float32)
        global_index_np = np.random.randint(8, size=(4, 4, 8), dtype=np.int64) # Index into dim 1 (8)
        
        # NumPy reference for gather on dim 0
        ref_output_np_0 = np.take_along_axis(global_input_np, global_index_np, axis=0)
        tvm_op_func_0 = lambda data_var, indices_var: relay.op.transform.gather(data_var, axis=0, indices=indices_var)
        self._run_tvm_and_compare_op(lambda d,i: ref_output_np_0, tvm_op_func_0, global_input_np, global_index_np)

        # NumPy reference for gather on dim 1
        ref_output_np_1 = np.take_along_axis(global_input_np, global_index_np, axis=1)
        tvm_op_func_1 = lambda data_var, indices_var: relay.op.transform.gather(data_var, axis=1, indices=indices_var)
        self._run_tvm_and_compare_op(lambda d,i: ref_output_np_1, tvm_op_func_1, global_input_np, global_index_np)

        # NumPy reference for gather on dim 2
        ref_output_np_2 = np.take_along_axis(global_input_np, global_index_np, axis=2)
        tvm_op_func_2 = lambda data_var, indices_var: relay.op.transform.gather(data_var, axis=2, indices=indices_var)
        self._run_tvm_and_compare_op(lambda d,i: ref_output_np_2, tvm_op_func_2, global_input_np, global_index_np)

        # Case 2: input sharding + index replicated (original used _MaskPartial, which is DTensor-specific)
        # Simplified to local tensors. `global_index_np_2` has size 1 on gather_dim (1).
        gather_dim_case2 = 1
        global_input_np_2 = np.random.randn(12, 8, 16).astype(np.float32)
        global_index_np_2 = np.random.randint(8, size=(4, 1, 8), dtype=np.int64)
        
        ref_output_np_case2 = np.take_along_axis(global_input_np_2, global_index_np_2, axis=gather_dim_case2)
        
        tvm_op_func_case2 = lambda data_var, indices_var: relay.op.transform.gather(data_var, axis=gather_dim_case2, indices=indices_var)
        self._run_tvm_and_compare_op(lambda d,i: ref_output_np_case2, tvm_op_func_case2, global_input_np_2, global_index_np_2)
        # TODO: The `_MaskPartial` placement is not mappable.

        # Case 3: index sharding (simplified to local tensors)
        # Original: input replicated, index sharded. output sharded.
        gather_dim_case3 = 0
        global_input_np_3 = np.random.randn(12, 8, 16).astype(np.float32)
        global_index_np_3 = np.random.randint(8, size=(4, 4, 8), dtype=np.int64)
        
        ref_output_np_case3 = np.take_along_axis(global_input_np_3, global_index_np_3, axis=gather_dim_case3)
        
        tvm_op_func_case3 = lambda data_var, indices_var: relay.op.transform.gather(data_var, axis=gather_dim_case3, indices=indices_var)
        self._run_tvm_and_compare_op(lambda d,i: ref_output_np_case3, tvm_op_func_case3, global_input_np_3, global_index_np_3)

        # TODO: CommDebugMode checks and DTensor placements are not mappable.


    @skipIfRocm
    def test_index(self):
        # Many advanced indexing patterns `x[y]`, `x[:,y]`, `x[...,y]`, `x[z,y]`, etc.
        # These need to be mapped to `relay.op.transform.take` or `relay.op.transform.gather_nd`.

        # Input tensor (3D)
        input_data_np_3d = np.random.randn(16, 32, 16).astype(np.float32)
        # Input tensor (4D)
        input_data_np_4d = np.random.randn(16, 32, 16, 12).astype(np.float32)

        # Case 1: `x[y]` (single integer tensor index for first dim)
        indices_1_np = np.random.randint(input_data_np_3d.shape[0], size=(4, 8), dtype=np.int64)
        np_ref_func_1 = lambda x, y_idx: x[y_idx]
        tvm_op_func_1 = lambda x_var, y_idx_var: relay.op.transform.gather_nd(x_var, y_idx_var)
        self._run_tvm_and_compare_op(np_ref_func_1, tvm_op_func_1, input_data_np_3d, indices_1_np)
        
        # Case 2: `x.index_select(1, y)`
        indices_2_np = np.random.randint(input_data_np_3d.shape[1], size=(4,), dtype=np.int64)
        np_ref_func_2 = lambda x, dim, y_idx: np.take(x, y_idx, axis=dim)
        tvm_op_func_2 = lambda x_var, dim, y_idx_var: relay.op.transform.take(x_var, y_idx_var, axis=dim)
        self._run_tvm_and_compare_op(np_ref_func_2, tvm_op_func_2, input_data_np_3d, 1, indices_2_np)

        # Case 3: `x.index_select(0, y)`
        indices_3_np = np.random.randint(input_data_np_3d.shape[0], size=(4,), dtype=np.int64)
        np_ref_func_3 = lambda x, dim, y_idx: np.take(x, y_idx, axis=dim)
        tvm_op_func_3 = lambda x_var, dim, y_idx_var: relay.op.transform.take(x_var, y_idx_var, axis=dim)
        self._run_tvm_and_compare_op(np_ref_func_3, tvm_op_func_3, input_data_np_3d, 0, indices_3_np)
        
        # Case 4: `x[y]` where y is a 1D index tensor (same as index_select(0,y))
        indices_4_np = np.random.randint(input_data_np_3d.shape[0], size=(12,), dtype=np.int64)
        np_ref_func_4 = lambda x, y_idx: x[y_idx]
        tvm_op_func_4 = lambda x_var, y_idx_var: relay.op.transform.take(x_var, y_idx_var, axis=0)
        self._run_tvm_and_compare_op(np_ref_func_4, tvm_op_func_4, input_data_np_3d, indices_4_np)
        
        # Case 5: `x[:, y]` - slicing and advanced indexing on second dimension.
        # PyTorch advanced indexing `x[:, y]` for `x(D0,D1,D2)` and `y(I0,I1)` produces `(D0, I0, I1, D2)`.
        # This is `gather_nd` with `batch_dims=1`.
        indices_5_np = np.random.randint(input_data_np_3d.shape[1], size=(4, 8), dtype=np.int64)
        np_ref_func_5 = lambda x, y_idx: x[:, y_idx]
        tvm_op_func_5 = lambda x_var, y_idx_var: relay.op.transform.gather_nd(x_var, y_idx_var, batch_dims=1)
        self._run_tvm_and_compare_op(np_ref_func_5, tvm_op_func_5, input_data_np_3d, indices_5_np)
        
        # Case 6: `x[..., y]` - slicing and advanced indexing on last dimension.
        # PyTorch advanced indexing `x[..., y]` for `x(D0,D1,D2)` and `y(I0,I1)` produces `(D0,D1,I0,I1)`.
        # This is `gather_nd` with `batch_dims = x.ndim - y.ndim`.
        indices_6_np = np.random.randint(input_data_np_3d.shape[2], size=(4, 12), dtype=np.int64)
        np_ref_func_6 = lambda x, y_idx: x[..., y_idx]
        tvm_op_func_6 = lambda x_var, y_idx_var: relay.op.transform.gather_nd(x_var, y_idx_var, batch_dims=x_var.shape.ndim - y_idx_var.shape.ndim)
        self._run_tvm_and_compare_op(np_ref_func_6, tvm_op_func_6, input_data_np_3d, indices_6_np)

        # Case 7: `x[..., y]` with more index dimensions than data dim (e.g. (4,8,16) and (4,8,16) for index) -> full broadcast
        indices_7_np = np.random.randint(input_data_np_3d.shape[2], size=(4, 8, 16), dtype=np.int64)
        np_ref_func_7 = lambda x, y_idx: x[..., y_idx]
        # This also works with gather_nd, batch_dims = x.ndim - y.ndim = 3 - 3 = 0.
        tvm_op_func_7 = lambda x_var, y_idx_var: relay.op.transform.gather_nd(x_var, y_idx_var, batch_dims=x_var.shape.ndim - y_idx_var.shape.ndim)
        self._run_tvm_and_compare_op(np_ref_func_7, tvm_op_func_7, input_data_np_3d, indices_7_np)

        # Case 8: `x[z, y]` (two integer tensor indices)
        y_indices_8_np = np.random.randint(input_data_np_3d.shape[1], size=(12, 8, 12), dtype=np.int64)
        z_indices_8_np = np.random.randint(input_data_np_3d.shape[0], size=(12, 8, 12), dtype=np.int64)
        np_ref_func_8 = lambda x, z_idx, y_idx: x[z_idx, y_idx]
        # For gather_nd, indices must be (..., K) where K is number of indexed dims.
        tvm_op_func_8 = lambda x_var, z_idx_var, y_idx_var: relay.op.transform.gather_nd(
            x_var, relay.op.tensor.stack(relay.expr.Tuple([z_idx_var, y_idx_var]), axis=-1)
        )
        self._run_tvm_and_compare_op(np_ref_func_8, tvm_op_func_8, input_data_np_3d, z_indices_8_np, y_indices_8_np)

        # Case 9: `x[z, :, y]` (two integer tensor indices with slicing)
        y_indices_9_np = np.random.randint(input_data_np_3d.shape[2], size=(12, 8, 12), dtype=np.int64)
        z_indices_9_np = np.random.randint(input_data_np_3d.shape[0], size=(12, 8, 12), dtype=np.int64)
        np_ref_func_9 = lambda x, z_idx, y_idx: x[z_idx, :, y_idx]
        # For this pattern, gather_nd indices should cover axes 0 and 2.
        tvm_op_func_9 = lambda x_var, z_idx_var, y_idx_var: relay.op.transform.gather_nd(
            x_var, relay.op.tensor.stack(relay.expr.Tuple([z_idx_var, y_idx_var]), axis=-1), batch_dims=1
        )
        self._run_tvm_and_compare_op(np_ref_func_9, tvm_op_func_9, input_data_np_3d, z_indices_9_np, y_indices_9_np)

        # Case 10: `x[:, z, :, y]` (4D input, two index tensors, two slices)
        y_indices_10_np = np.random.randint(input_data_np_4d.shape[3], size=(12, 8, 12), dtype=np.int64)
        z_indices_10_np = np.random.randint(input_data_np_4d.shape[1], size=(12, 8, 12), dtype=np.int64)
        np_ref_func_10 = lambda x, z_idx, y_idx: x[:, z_idx, :, y_idx]
        # For this, indices cover axes 1 and 3. batch_dims=2
        tvm_op_func_10 = lambda x_var, z_idx_var, y_idx_var: relay.op.transform.gather_nd(
            x_var, relay.op.tensor.stack(relay.expr.Tuple([z_idx_var, y_idx_var]), axis=-1), batch_dims=2
        )
        self._run_tvm_and_compare_op(np_ref_func_10, tvm_op_func_10, input_data_np_4d, z_indices_10_np, y_indices_10_np)
        
        # Case 11: broadcast in inner dimensions `x[:, z, :, y]`
        # This is handled by gather_nd's implicit broadcasting rules.
        # `z_indices_11_np` has shape (12,1,12), which broadcasts with (12,8,12) for `y`.
        input_11_np = input_data_np_4d # Use 4D input
        y_indices_11_np = np.random.randint(input_11_np.shape[3], size=(12, 8, 12), dtype=np.int64)
        z_indices_11_np = np.random.randint(input_11_np.shape[1], size=(12, 1, 12), dtype=np.int64) # Note: shape is (12,1,12)
        np_ref_func_11 = lambda x, z_idx, y_idx: x[:, z_idx, :, y_idx]
        tvm_op_func_11 = lambda x_var, z_idx_var, y_idx_var: relay.op.transform.gather_nd(
            x_var, relay.op.tensor.stack(relay.expr.Tuple([z_idx_var, y_idx_var]), axis=-1), batch_dims=2
        )
        self._run_tvm_and_compare_op(np_ref_func_11, tvm_op_func_11, input_11_np, z_indices_11_np, y_indices_11_np)
        
        # Case 12: implicit (left-padded) broadcast `x[:, z, :, y]`
        input_12_np = input_data_np_4d # Use 4D input
        y_indices_12_np = np.random.randint(input_12_np.shape[3], size=(12, 8, 12), dtype=np.int64)
        z_indices_12_np = np.random.randint(input_12_np.shape[1], size=(8, 12), dtype=np.int64) # Note: shape is (8,12)
        np_ref_func_12 = lambda x, z_idx, y_idx: x[:, z_idx, :, y_idx]
        tvm_op_func_12 = lambda x_var, z_idx_var, y_idx_var: relay.op.transform.gather_nd(
            x_var, relay.op.tensor.stack(relay.expr.Tuple([z_idx_var, y_idx_var]), axis=-1), batch_dims=2
        )
        self._run_tvm_and_compare_op(np_ref_func_12, tvm_op_func_12, input_12_np, z_indices_12_np, y_indices_12_np)

        # Cases using multiple slices `x[z, y, :, :]`, `x[z, :, y, :]`, `x[z, :, :, y]`
        # These also map to gather_nd with appropriate batch_dims.
        # Case 13: `x[z, y, :, :]` (4D input)
        input_13_np = input_data_np_4d # Use 4D input
        y_indices_13_np = np.random.randint(input_13_np.shape[1], size=(8, 12), dtype=np.int64)
        z_indices_13_np = np.random.randint(input_13_np.shape[0], size=(12, 8, 12), dtype=np.int64)
        np_ref_func_13 = lambda x, z_idx, y_idx: x[z_idx, y_idx, :, :]
        tvm_op_func_13 = lambda x_var, z_idx_var, y_idx_var: relay.op.transform.gather_nd(
            x_var, relay.op.tensor.stack(relay.expr.Tuple([z_idx_var, y_idx_var]), axis=-1), batch_dims=0
        )
        self._run_tvm_and_compare_op(np_ref_func_13, tvm_op_func_13, input_13_np, z_indices_13_np, y_indices_13_np)
        
        # Case 14: `x[z, :, y, :]` (4D input)
        input_14_np = input_data_np_4d # Use 4D input
        y_indices_14_np = np.random.randint(input_14_np.shape[2], size=(8, 12), dtype=np.int64)
        z_indices_14_np = np.random.randint(input_14_np.shape[0], size=(12, 8, 12), dtype=np.int64)
        np_ref_func_14 = lambda x, z_idx, y_idx: x[z_idx, :, y_idx, :]
        tvm_op_func_14 = lambda x_var, z_idx_var, y_idx_var: relay.op.transform.gather_nd(
            x_var, relay.op.tensor.stack(relay.expr.Tuple([z_idx_var, y_idx_var]), axis=-1), batch_dims=1
        )
        self._run_tvm_and_compare_op(np_ref_func_14, tvm_op_func_14, input_14_np, z_indices_14_np, y_indices_14_np)

        # Case 15: `x[z, :, :, y]` (4D input)
        input_15_np = input_data_np_4d # Use 4D input
        y_indices_15_np = np.random.randint(input_15_np.shape[3], size=(8, 1), dtype=np.int64)
        z_indices_15_np = np.random.randint(input_15_np.shape[0], size=(12, 8, 12), dtype=np.int64)
        np_ref_func_15 = lambda x, z_idx, y_idx: x[z_idx, :, :, y_idx]
        tvm_op_func_15 = lambda x_var, z_idx_var, y_idx_var: relay.op.transform.gather_nd(
            x_var, relay.op.tensor.stack(relay.expr.Tuple([z_idx_var, y_idx_var]), axis=-1), batch_dims=2
        )
        self._run_tvm_and_compare_op(np_ref_func_15, tvm_op_func_15, input_15_np, z_indices_15_np, y_indices_15_np)


    def test_index_put_scalar(self):
        # `torch.index_put(global_input, global_index, global_value)` where global_index is list of scalar tensors
        # Maps to `scatter_nd` for a single element update.
        self.world_size = 2 # Dummy value
        
        global_input_np = np.random.randn(2, 4, 8).astype(np
