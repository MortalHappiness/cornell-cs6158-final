import tvm
import tvm.relay as relay
import numpy as np
import pytest
from tvm.relay.testing import run_infer_type
from tvm.ir.module import IRModule

# No direct mapping for torch._dynamo.utils.counters, etc.
# These PyTorch-specific imports and features are removed or replaced.

# Helper for creating Relay functions and executing them
def _run_tvm_model(relay_fn, inputs_np, target="llvm", device=tvm.cpu(0)):
    if not isinstance(inputs_np, (list, tuple)):
        inputs_np = [inputs_np]

    input_vars = []
    for i, arr_np in enumerate(inputs_np):
        shape = arr_np.shape
        dtype = str(arr_np.dtype)
        input_vars.append(relay.var(f"p{i}", relay.TensorType(shape, dtype)))

    outputs_expr = relay_fn(*input_vars)
    
    # If the function returns multiple outputs, wrap them in a Relay tuple
    if isinstance(outputs_expr, (list, tuple)):
        # Relay.expr.TupleWrapper is for representing tuples in the IR,
        # but relay.Function directly accepts a Python tuple of expressions as its body.
        func_body = relay.Tuple(list(outputs_expr))
    else:
        func_body = outputs_expr

    func = relay.Function(input_vars, func_body)
    mod = IRModule.from_expr(func)
    mod = run_infer_type(mod)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    runtime_inputs = []
    for arr_np in inputs_np:
        runtime_inputs.append(tvm.nd.array(arr_np, device=device))

    vm = tvm.runtime.vm.VirtualMachine(lib, device)
    result = vm.run(*runtime_inputs)

    if isinstance(result, tvm.runtime.container.ADT): # Check for Relay tuple output
        ret = []
        for r in result:
            ret.append(r.asnumpy())
        return tuple(ret)
    else:
        return result.asnumpy()

# Replace torch ops with TVM Relay ops
relu_op = relay.op.nn.relu
sin_op = relay.op.tensor.sin
tanh_op = relay.op.tensor.tanh
multiply_op = relay.op.tensor.multiply
copy_op = relay.op.tensor.copy
concatenate_op = relay.op.tensor.concatenate
stack_op = relay.op.tensor.stack
squeeze_op = relay.op.transform.squeeze
unbind_op = relay.frontend.common.unbind
reshape_op = relay.op.transform.reshape
ones_op = relay.op.tensor.ones
reduce_max_op = relay.op.reduce.max
reduce_argmax_op = relay.op.reduce.argmax
dropout_op = relay.op.nn.dropout

FLOAT32 = "float32"
INT64 = "int64"

# Use pytest to mark tests that require GPU
requires_gpu = pytest.mark.skipif(
    not tvm.runtime.enabled("cuda"), reason="Requires GPU"
)
GPU_TARGET = "cuda" if tvm.runtime.enabled("cuda") else "llvm"
GPU_DEVICE = tvm.cuda(0) if tvm.runtime.enabled("cuda") else tvm.cpu(0)

# The 'patch' decorator and 'counters' mechanism from PyTorch Inductor are not applicable to TVM,
# and thus are removed. Tests now focus on functional equivalence.

class TestSplitCatFxPasses:
    def test_split_normalization(self):
        # Helper to mimic torch.split on numpy for expected values
        def numpy_split_by_size(arr, split_size, axis):
            dim_len = arr.shape[axis]
            sections_indices = []
            current_idx = 0
            while current_idx < dim_len:
                sections_indices.append(min(current_idx + split_size, dim_len))
                current_idx += split_size
            
            if sections_indices and sections_indices[-1] == dim_len:
                sections_indices = sections_indices[:-1]
            
            return np.split(arr, sections_indices, axis=axis)

        def numpy_split_by_sections(arr, sections, axis):
            cumulative_indices = np.cumsum(sections)[:-1]
            return np.split(arr, cumulative_indices, axis=axis)

        def numpy_relu(x_np):
            return np.maximum(x_np, 0)
        
        # Helper for PyTorch's `squeeze` behavior (no-op if dim not 1)
        def numpy_conditional_squeeze(arr, axis=None):
            if axis is None:
                return np.squeeze(arr, axis=None)
            
            # For a single axis, if its size is not 1, PyTorch's squeeze is a no-op.
            if arr.shape[axis] != 1:
                return arr
            return np.squeeze(arr, axis=axis)

        def arg_only(x):
            splits = relay.op.transform.split(x, 2, axis=1) # x (2,32) -> 16x (2,2)
            return tuple([relu_op(s) for s in splits])

        def arg_only_dim0(x):
            splits = relay.op.transform.split(x, 2, axis=0) # x (2,32) -> 1x (2,32) if split_size=2
            return tuple([relu_op(s) for s in splits])

        def kwarg1(x):
            splits = relay.op.transform.split(x, 2, axis=1)
            return tuple([relu_op(s) for s in splits])

        def kwarg2(x):
            splits = relay.op.transform.split(x, 2, axis=1)
            return tuple([relu_op(s) for s in splits])

        def kwarg3(x):
            splits = relay.op.transform.split(x, 2, axis=-1)
            return tuple([relu_op(s) for s in splits])

        def list_replace(x):
            splits = relay.op.transform.split(x, sections=(16, 16), axis=1) # TVM expects tuple for sections
            return tuple([relu_op(s) for s in splits])

        def multi_split(x):
            outer_splits = relay.op.transform.split(x, 2, axis=1) # 16 outputs of (2,2)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 2, axis=1) # Each (2,2) splits into 1 output (2,2)
                results.append(inner_splits)
            return tuple(results)

        def unequal_split(x):
            # x (2,32), split by 3 in dim 1. 10 splits of size 3, 1 split of size 2. Total 11 splits.
            splits = relay.op.transform.split(x, 3, axis=1)
            return tuple([relu_op(s) for s in splits])

        # Call method variants
        def arg_only_cm(x):
            splits = relay.op.transform.split(x, 2, axis=1)
            return tuple([relu_op(s) for s in splits])

        def kwarg1_cm(x):
            splits = relay.op.transform.split(x, 2, axis=1)
            return tuple([relu_op(s) for s in splits])

        def kwarg2_cm(x):
            splits = relay.op.transform.split(x, 2, axis=1)
            return tuple([relu_op(s) for s in splits])

        def multi_split_cm(x):
            outer_splits = relay.op.transform.split(x, 2, axis=1)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 2, axis=1)
                results.append(inner_splits)
            return tuple(results)

        def unequal_split_cm(x):
            splits = relay.op.transform.split(x, 3, axis=1)
            return tuple([relu_op(s) for s in splits])

        def cm_with_list(x):
            splits = relay.op.transform.split(x, sections=(16, 16), axis=-1)
            return tuple([relu_op(s) for s in splits])

        def normalize_reshape_with_dynamic_shape(x):
            return reshape_op(x, newshape=(4, 16))

        args_np = [
            np.random.randn(2, 32).astype(FLOAT32),
        ]
        
        # NumPy implementations to calculate expected values
        def numpy_arg_only(x_np):
            splits_np = numpy_split_by_size(x_np, 2, 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_arg_only_dim0(x_np):
            splits_np = numpy_split_by_size(x_np, 2, 0)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_kwarg1(x_np):
            splits_np = numpy_split_by_size(x_np, 2, 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_kwarg2(x_np):
            splits_np = numpy_split_by_size(x_np, 2, 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_kwarg3(x_np):
            splits_np = numpy_split_by_size(x_np, 2, -1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_list_replace(x_np):
            splits_np = numpy_split_by_sections(x_np, (16, 16), 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_multi_split(x_np):
            outer_splits_np = numpy_split_by_size(x_np, 2, 1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 2, 1)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)
        def numpy_unequal_split(x_np):
            splits_np = numpy_split_by_size(x_np, 3, 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_arg_only_cm(x_np):
            splits_np = numpy_split_by_size(x_np, 2, 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_kwarg1_cm(x_np):
            splits_np = numpy_split_by_size(x_np, 2, 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_kwarg2_cm(x_np):
            splits_np = numpy_split_by_size(x_np, 2, 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_multi_split_cm(x_np):
            outer_splits_np = numpy_split_by_size(x_np, 2, 1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 2, 1)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)
        def numpy_unequal_split_cm(x_np):
            splits_np = numpy_split_by_size(x_np, 3, 1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_cm_with_list(x_np):
            splits_np = numpy_split_by_sections(x_np, (16, 16), -1)
            return tuple([numpy_relu(s) for s in splits_np])
        def numpy_normalize_reshape_with_dynamic_shape(x_np):
            return x_np.reshape(4, 16)

        test_cases = [
            (arg_only, numpy_arg_only),
            (arg_only_dim0, numpy_arg_only_dim0),
            (kwarg1, numpy_kwarg1),
            (kwarg2, numpy_kwarg2),
            (kwarg3, numpy_kwarg3),
            (list_replace, numpy_list_replace),
            (multi_split, numpy_multi_split),
            (unequal_split, numpy_unequal_split),
            (arg_only_cm, numpy_arg_only_cm),
            (kwarg1_cm, numpy_kwarg1_cm),
            (kwarg2_cm, numpy_kwarg2_cm),
            (multi_split_cm, numpy_multi_split_cm),
            (unequal_split_cm, numpy_unequal_split_cm),
            (cm_with_list, numpy_cm_with_list),
            (normalize_reshape_with_dynamic_shape, numpy_normalize_reshape_with_dynamic_shape),
        ]
        
        for fn_relay, fn_numpy in test_cases:
            actual = _run_tvm_model(fn_relay, args_np[0])
            expected_np = fn_numpy(args_np[0])
            tvm.testing.assert_allclose(actual, expected_np)


    def test_cat_normalization(self):
        def caoncat_only(x):
            splits = relay.op.transform.split(x, 2, axis=1) # x (2,32) -> 16x (2,2)
            return concatenate_op(splits, axis=1)

        args_np = [
            np.random.randn(2, 32).astype(FLOAT32),
        ]
        
        splits_np = numpy_split_by_size(args_np[0], 2, axis=1) # 16 equal splits
        expected_np = np.concatenate(splits_np, axis=1)

        actual = _run_tvm_model(caoncat_only, args_np[0])
        tvm.testing.assert_allclose(actual, expected_np)


    def test_consecutive_split_merge(self):
        # x is (2, 32)
        def multi_split(x):
            outer_splits = relay.op.transform.split(x, 2, axis=1)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 2, axis=1)
                results.append(inner_splits)
            return tuple(results)

        def multi_split_2(x):
            outer_splits = relay.op.transform.split(x, 2, axis=1) # 16x (2,2)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 1, axis=1) # Each (2,2) splits into 2x (2,1)
                results.append(inner_splits)
            return tuple(results)

        def multi_split_2_neg_dim(x):
            outer_splits = relay.op.transform.split(x, 2, axis=-1)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 1, axis=1)
                results.append(inner_splits)
            return tuple(results)

        def multi_split_with_sizes(x):
            outer_splits = relay.op.transform.split(x, sections=(16, 16), axis=1) # 2x (2,16)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 2, axis=1) # Each (2,16) splits into 8x (2,2)
                results.append(inner_splits)
            return tuple(results)

        def multi_split_kwarg1(x):
            outer_splits = relay.op.transform.split(x, 2, axis=1)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 2, axis=1)
                results.append(inner_splits)
            return tuple(results)

        def multi_split_kwarg2(x):
            outer_splits = relay.op.transform.split(x, 2, axis=1)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 2, axis=1)
                results.append(inner_splits)
            return tuple(results)

        def unequal_multi_split(x):
            fs = relay.op.transform.split(x, sections=(10, 10, 12), axis=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(relay.op.transform.split(item0, sections=(4, 6), axis=1))
            final_items.extend(relay.op.transform.split(item1, sections=(6, 4), axis=1))
            final_items.extend(relay.op.transform.split(item2, sections=(4, 4, 4), axis=1))
            return tuple([relu_op(s) for s in final_items])

        def unequal_multi_split_neg_index(x):
            fs = relay.op.transform.split(x, sections=(10, 10, 12), axis=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]
            
            final_items = []
            final_items.extend(relay.op.transform.split(item0, sections=(4, 6), axis=1))
            final_items.extend(relay.op.transform.split(item1, sections=(6, 4), axis=1))
            final_items.extend(relay.op.transform.split(item2, sections=(4, 4, 4), axis=1))
            return tuple([relu_op(s) for s in final_items])

        def diff_dims(x):
            outer_splits = relay.op.transform.split(x, 2, axis=1)
            results = []
            for s_outer in outer_splits:
                inner_splits = relay.op.transform.split(s_outer, 2, axis=0) # split along dim 0
                results.append(inner_splits)
            return tuple(results)

        def some_users_not_splits(x):
            fs = relay.op.transform.split(x, sections=(10, 10, 12), axis=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(relay.op.transform.split(item0, sections=(4, 6), axis=1))
            final_items.extend(relay.op.transform.split(item1, sections=(6, 4), axis=1))
            final_items.append(sin_op(item2))

            return tuple([relu_op(s) for s in final_items])

        def split_with_cat(x):
            fs = relay.op.transform.split(x, sections=(4, 4, 24), axis=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = [item0, item1]
            final_items.extend(relay.op.transform.split(item2, sections=(4, 4, 4, 4, 4, 4), axis=1))

            return concatenate_op(final_items, axis=1)

        def duplicate_getitems(x):
            fs = relay.op.transform.split(x, sections=(10, 10, 12), axis=1)
            item0 = fs[0]
            item1_1 = fs[1]
            item1_2 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(relay.op.transform.split(item0, sections=(4, 6), axis=1))
            final_items.extend(relay.op.transform.split(item1_1, sections=(6, 4), axis=1))
            final_items.append(item1_2) # Assume append semantic for single tensor
            final_items.append(sin_op(item2))

            return tuple([relu_op(s) for s in final_items])

        def duplicate_getitems_neg_index(x):
            fs = relay.op.transform.split(x, sections=(10, 10, 12), axis=1)
            item0 = fs[0]
            item1_1 = fs[1]
            item1_2 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(relay.op.transform.split(item0, sections=(4, 6), axis=1))
            final_items.extend(relay.op.transform.split(item1_1, sections=(6, 4), axis=1))
            final_items.append(item1_2)
            final_items.append(sin_op(item2))

            return tuple([relu_op(s) for s in final_items])

        def split_getitem_gap(x):
            fs = relay.op.transform.split(x, sections=(4, 4, 24), axis=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [item0]
            final_items.extend(relay.op.transform.split(item2, sections=(4, 4, 4, 4, 4, 4), axis=1))

            return concatenate_op(final_items, axis=1)

        def split_getitem_out_of_order(x):
            fs = relay.op.transform.split(x, sections=(4, 4, 4, 20), axis=1)
            item0 = fs[0]
            item2 = fs[2]
            item1 = fs[1]
            item3 = fs[3]

            final_items = [item0, item2, item1]
            final_items.extend(relay.op.transform.split(item3, sections=(4, 4, 4, 4, 4), axis=1))

            return concatenate_op(final_items, axis=1)

        def split_partial_getitem_cat(x):
            fs = relay.op.transform.split(x, sections=(4, 4, 24), axis=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [item0]
            final_items.extend(relay.op.transform.split(item2, sections=(4, 4, 4, 4, 4, 4), axis=1))

            return concatenate_op(final_items, axis=1)

        def next_split_getitem_partial_used(x):
            fs = relay.op.transform.split(x, sections=(4, 4, 24), axis=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [item0]
            ns = relay.op.transform.split(item2, sections=(4, 4, 4, 4, 4, 4), axis=1)
            final_items.extend(ns[0:1])
            final_items.extend(ns[3:4])

            return concatenate_op(final_items, axis=1)

        args_np = [
            np.random.randn(2, 32).astype(FLOAT32),
        ]
        
        # NumPy implementations for these functions
        def numpy_multi_split_expected(x_np):
            outer_splits_np = numpy_split_by_size(x_np, 2, 1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 2, 1)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)

        def numpy_multi_split_2_expected(x_np):
            outer_splits_np = numpy_split_by_size(x_np, 2, 1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 1, 1)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)
        
        def numpy_multi_split_2_neg_dim_expected(x_np):
            outer_splits_np = numpy_split_by_size(x_np, 2, -1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 1, 1)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)

        def numpy_multi_split_with_sizes_expected(x_np):
            outer_splits_np = numpy_split_by_sections(x_np, (16, 16), 1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 2, 1)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)

        def numpy_multi_split_kwarg1_expected(x_np):
            outer_splits_np = numpy_split_by_size(x_np, 2, 1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 2, 1)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)
        
        def numpy_multi_split_kwarg2_expected(x_np):
            outer_splits_np = numpy_split_by_size(x_np, 2, 1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 2, 1)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)

        def numpy_unequal_multi_split_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (10, 10, 12), 1)
            item0_np, item1_np, item2_np = fs_np
            final_items_np = []
            final_items_np.extend(numpy_split_by_sections(item0_np, (4, 6), 1))
            final_items_np.extend(numpy_split_by_sections(item1_np, (6, 4), 1))
            final_items_np.extend(numpy_split_by_sections(item2_np, (4, 4, 4), 1))
            return tuple([numpy_relu(s) for s in final_items_np])

        def numpy_unequal_multi_split_neg_index_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (10, 10, 12), 1)
            item0_np, item1_np, item2_np = fs_np
            final_items_np = []
            final_items_np.extend(numpy_split_by_sections(item0_np, (4, 6), 1))
            final_items_np.extend(numpy_split_by_sections(item1_np, (6, 4), 1))
            final_items_np.extend(numpy_split_by_sections(item2_np, (4, 4, 4), 1))
            return tuple([numpy_relu(s) for s in final_items_np])

        def numpy_diff_dims_expected(x_np):
            outer_splits_np = numpy_split_by_size(x_np, 2, 1)
            results_np = []
            for s_outer_np in outer_splits_np:
                inner_splits_np = numpy_split_by_size(s_outer_np, 2, 0)
                results_np.append(tuple(inner_splits_np))
            return tuple(results_np)
        
        def numpy_some_users_not_splits_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (10, 10, 12), 1)
            item0_np, item1_np, item2_np = fs_np
            final_items_np = []
            final_items_np.extend(numpy_split_by_sections(item0_np, (4, 6), 1))
            final_items_np.extend(numpy_split_by_sections(item1_np, (6, 4), 1))
            final_items_np.append(np.sin(item2_np))
            return tuple([numpy_relu(s) for s in final_items_np])
        
        def numpy_split_with_cat_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (4, 4, 24), 1)
            item0_np, item1_np, item2_np = fs_np
            final_items_np = [item0_np, item1_np]
            final_items_np.extend(numpy_split_by_sections(item2_np, (4, 4, 4, 4, 4, 4), 1))
            return np.concatenate(final_items_np, axis=1)
        
        def numpy_duplicate_getitems_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (10, 10, 12), 1)
            item0_np, item1_1_np, item2_np = fs_np
            item1_2_np = item1_1_np # same tensor
            final_items_np = []
            final_items_np.extend(numpy_split_by_sections(item0_np, (4, 6), 1))
            final_items_np.extend(numpy_split_by_sections(item1_1_np, (6, 4), 1))
            final_items_np.append(item1_2_np) # Assumed append
            final_items_np.append(np.sin(item2_np))
            return tuple([numpy_relu(s) for s in final_items_np])

        def numpy_duplicate_getitems_neg_index_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (10, 10, 12), 1)
            item0_np, item1_1_np, item2_np = fs_np
            item1_2_np = item1_1_np # same tensor (fs[-2] is fs[1])
            final_items_np = []
            final_items_np.extend(numpy_split_by_sections(item0_np, (4, 6), 1))
            final_items_np.extend(numpy_split_by_sections(item1_1_np, (6, 4), 1))
            final_items_np.append(item1_2_np) # Assumed append
            final_items_np.append(np.sin(item2_np))
            return tuple([numpy_relu(s) for s in final_items_np])

        def numpy_split_getitem_gap_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (4, 4, 24), 1)
            item0_np, _, item2_np = fs_np # item1 is skipped
            final_items_np = [item0_np]
            final_items_np.extend(numpy_split_by_sections(item2_np, (4, 4, 4, 4, 4, 4), 1))
            return np.concatenate(final_items_np, axis=1)

        def numpy_split_getitem_out_of_order_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (4, 4, 4, 20), 1)
            item0_np, item1_np, item2_np, item3_np = fs_np
            final_items_np = [item0_np, item2_np, item1_np] # Specific order
            final_items_np.extend(numpy_split_by_sections(item3_np, (4, 4, 4, 4, 4), 1))
            return np.concatenate(final_items_np, axis=1)

        def numpy_split_partial_getitem_cat_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (4, 4, 24), 1)
            item0_np, _, item2_np = fs_np
            final_items_np = [item0_np]
            final_items_np.extend(numpy_split_by_sections(item2_np, (4, 4, 4, 4, 4, 4), 1))
            return np.concatenate(final_items_np, axis=1)

        def numpy_next_split_getitem_partial_used_expected(x_np):
            fs_np = numpy_split_by_sections(x_np, (4, 4, 24), 1)
            item0_np, _, item2_np = fs_np
            final_items_np = [item0_np]
            ns_np = numpy_split_by_sections(item2_np, (4, 4, 4, 4, 4, 4), 1)
            final_items_np.extend(ns_np[0:1])
            final_items_np.extend(ns_np[3:4])
            return np.concatenate(final_items_np, axis=1)

        test_cases = [
            (multi_split, numpy_multi_split_expected),
            (multi_split_2, numpy_multi_split_2_expected),
            (multi_split_2_neg_dim, numpy_multi_split_2_neg_dim_expected),
            (multi_split_with_sizes, numpy_multi_split_with_sizes_expected),
            (multi_split_kwarg1, numpy_multi_split_kwarg1_expected),
            (multi_split_kwarg2, numpy_multi_split_kwarg2_expected),
            (unequal_multi_split, numpy_unequal_multi_split_expected),
            (unequal_multi_split_neg_index, numpy_unequal_multi_split_neg_index_expected),
            (diff_dims, numpy_diff_dims_expected),
            (some_users_not_splits, numpy_some_users_not_splits_expected),
            (split_with_cat, numpy_split_with_cat_expected),
            (duplicate_getitems, numpy_duplicate_getitems_expected),
            (duplicate_getitems_neg_index, numpy_duplicate_getitems_neg_index_expected),
            (split_getitem_gap, numpy_split_getitem_gap_expected),
            (split_getitem_out_of_order, numpy_split_getitem_out_of_order_expected),
            (split_partial_getitem_cat, numpy_split_partial_getitem_cat_expected),
            (next_split_getitem_partial_used, numpy_next_split_getitem_partial_used_expected),
        ]
        for fn_relay, fn_numpy in test_cases:
            actual = _run_tvm_model(fn_relay, args_np[0])
            expected_np = fn_numpy(args_np[0])
            tvm.testing.assert_allclose(actual, expected_np)


    def test_split_cat_merge(self):
        # x is (2, 32, 32, 16)
        def simple_split_cat(x):
            splits = relay.op.transform.split(x, 4, axis=1)
            return concatenate_op(splits, axis=1)

        def simple_split_cat_argspec1(x):
            splits = relay.op.transform.split(x, 4, axis=1)
            return concatenate_op(splits, axis=1)

        def simple_split_cat_argspec2(x):
            splits = relay.op.transform.split(x, 4, axis=1)
            return concatenate_op(splits, axis=1)

        def simple_split_cat_argspec3(x):
            splits = relay.op.transform.split(x, 4, axis=1)
            return concatenate_op(splits, axis=-2)

        def simple_split_cat_argspec4(x):
            splits = relay.op.transform.split(x, 4, axis=1)
            return concatenate_op(splits, axis=-2)

        def simple_split_stack(x):
            splits = relay.op.transform.split(x, 4, axis=1)
            return stack_op(splits, axis=1)

        def simple_split_stack_argspec1(x):
            splits = relay.op.transform.split(x, 4, axis=1)
            return stack_op(splits, axis=1)

        def simple_split_stack_argspec2(x):
            splits = relay.op.transform.split(x, 4, axis=1)
            return stack_op(splits, axis=1)

        def split_cat_addn_args(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_5_32_16 = ones_op(shape=(2, 5, 32, 16), dtype=FLOAT32)
            one_const_2_6_32_16 = ones_op(shape=(2, 6, 32, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_5_32_16] + split_output + [one_const_2_6_32_16]
            return concatenate_op(all_tensors, axis=1)

        def split_stack_addn_args(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_4_32_16] + split_output + [one_const_2_4_32_16, one_const_2_4_32_16]
            return stack_op(all_tensors, axis=1)

        def split_cat_addn_args_dim2(x):
            split_output = list(relay.op.transform.split(x, 4, axis=2))
            one_const_2_32_5_16 = ones_op(shape=(2, 32, 5, 16), dtype=FLOAT32)
            one_const_2_32_6_16 = ones_op(shape=(2, 32, 6, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_32_5_16] + split_output + [one_const_2_32_6_16]
            return concatenate_op(all_tensors, axis=2)

        def split_cat_dim_mismatch(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_4_32_16] + split_output + [one_const_2_4_32_16]
            return concatenate_op(all_tensors, axis=2)

        def split_stack_dim_mismatch(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_4_32_16] + split_output + [one_const_2_4_32_16]
            return stack_op(all_tensors, axis=2)

        def split_cat_dim_mismatch2(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_4_32_16] + split_output + [one_const_2_4_32_16]
            return concatenate_op(all_tensors, axis=3)

        def split_stack_dim_mismatch2(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_4_32_16] + split_output + [one_const_2_4_32_16]
            return stack_op(all_tensors, axis=3)

        def split_cat_dim_mismatch3(x):
            split_output = list(relay.op.transform.split(x, 4, axis=2))
            one_const_2_32_4_16 = ones_op(shape=(2, 32, 4, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_32_4_16] + split_output + [one_const_2_32_4_16]
            return concatenate_op(all_tensors, axis=0)

        def split_stack_dim_mismatch3(x):
            split_output = list(relay.op.transform.split(x, 4, axis=2))
            one_const_2_32_4_16 = ones_op(shape=(2, 32, 4, 16), dtype=FLOAT32)
            all_tensors = [one_const_2_32_4_16] + split_output + [one_const_2_32_4_16]
            return stack_op(all_tensors, axis=0)

        def input_shuffling(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = (
                [one_const_2_4_32_16]
                + [split_output[1], split_output[2], split_output[3]]
                + [one_const_2_4_32_16]
                + [split_output[5], split_output[6], split_output[7]]
                + [one_const_2_4_32_16]
            )
            return concatenate_op(all_tensors, axis=1)

        def input_shuffling_stack(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = (
                [one_const_2_4_32_16]
                + [split_output[1], split_output[2], split_output[3]]
                + [one_const_2_4_32_16]
                + [split_output[5], split_output[6], split_output[7]]
                + [one_const_2_4_32_16]
            )
            return stack_op(all_tensors, axis=1)

        def input_shuffling_dim_mismatch(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = (
                [one_const_2_4_32_16]
                + [split_output[1], split_output[2], split_output[3]]
                + [one_const_2_4_32_16]
                + [split_output[5], split_output[6], split_output[7]]
                + [one_const_2_4_32_16]
            )
            return concatenate_op(all_tensors, axis=2)

        def input_shuffling_dim_mismatch_stack(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)
            all_tensors = (
                [one_const_2_4_32_16]
                + [split_output[1], split_output[2], split_output[3]]
                + [one_const_2_4_32_16]
                + [split_output[5], split_output[6], split_output[7]]
                + [one_const_2_4_32_16]
            )
            return stack_op(all_tensors, axis=2)

        def input_shuffling_multiple_output(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)

            cat1_tensors = (
                [one_const_2_4_32_16]
                + [split_output[1], split_output[2], split_output[3]]
                + [one_const_2_4_32_16]
            )
            cat1 = concatenate_op(cat1_tensors, axis=2)

            stack1_tensors = [
                one_const_2_4_32_16,
                split_output[4],
                split_output[5],
                one_const_2_4_32_16,
            ]
            stack1 = stack_op(stack1_tensors, axis=1)

            relu1 = relu_op(split_output[6])

            return cat1, stack1, relu1

        def input_shuffling_direct_output(x):
            split_output = list(relay.op.transform.split(x, 4, axis=1))
            one_const_2_4_32_16 = ones_op(shape=(2, 4, 32, 16), dtype=FLOAT32)

            cat1_tensors = (
                [one_const_2_4_32_16]
                + [split_output[1], split_output[2], split_output[3]]
                + [one_const_2_4_32_16]
            )
            cat1 = concatenate_op(cat1_tensors, axis=2)
            stack1_tensors = [
                one_const_2_4_32_16,
                split_output[4],
                split_output[5],
                one_const_2_4_32_16,
            ]
            stack1 = stack_op(stack1_tensors, axis=1)
