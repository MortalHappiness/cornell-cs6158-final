import gc
import re
import textwrap
import unittest
import weakref
from typing import Any, Dict, List, Tuple
import pytest
import numpy as np

# TVM specific imports
import tvm
from tvm import relay
from tvm.relay import op
# from tvm.relay.frontend.common import infer_shape # infer_shape is not for generic expressions, but for frontends.
from tvm.runtime import nd as tvm_nd

# For simplicity, and since the original test uses unittest.TestCase,
# we'll stick to it.
# We will define placeholder classes for PyTorch internal profiling structures,
# as they have no direct TVM equivalent for this kind of detailed runtime introspection.

# Placeholder classes for PyTorch internal profiling types.
# These will NOT capture the actual profiling data like PyTorch,
# but allow the code structure to remain valid.
class _TensorMetadata:
    def __init__(self, impl_ptr=0, storage_data_ptr=0, id=0, sizes=None, strides=None, layout=None, device=None, dtype=None, allocation_id=0):
        self.impl_ptr = impl_ptr
        self.storage_data_ptr = storage_data_ptr
        self.id = id
        self.sizes = sizes if sizes is not None else []
        self.strides = strides if strides is not None else []
        self.layout = layout
        self.device = device
        self.dtype = dtype
        self.allocation_id = allocation_id

class _ExtraFields_TorchOp:
    def __init__(self, inputs=None, sequence_number=0, scope=None):
        self.inputs = inputs if inputs is not None else []
        self.sequence_number = sequence_number
        self.scope = scope

class _ExtraFields_PyCCall:
    def __init__(self):
        pass

class _ExtraFields_Allocation:
    def __init__(self, ptr=0, alloc_size=0, device=None, total_allocated=0, total_reserved=0, id=0, allocation_id=0):
        self.ptr = ptr
        self.alloc_size = alloc_size
        self.device = device
        self.total_allocated = total_allocated
        self.total_reserved = total_reserved
        self.id = id
        self.allocation_id = allocation_id

class _EventType:
    # Dummy value, as TVM doesn't have a direct equivalent for PyTorch's _EventType enum in this context
    Allocation = 1

class _Node:
    def __init__(self, name, extra_fields=None, children=None, start_time_ns=0, tag=None):
        self.name = name
        self.extra_fields = extra_fields
        self.children = children if children is not None else []
        self.start_time_ns = start_time_ns
        self.tag = tag

# Dummy profiler context manager, as direct PyTorch profiler doesn't have TVM equivalent.
# This will not actually profile TVM execution in the same way.
class _DummyProfiler:
    def __init__(self):
        self.kineto_results = self # Mock the structure
        self._event_tree = [] # Stores dummy nodes if needed

    def experimental_event_tree(self):
        # In a real TVM profiling scenario, this would be replaced with actual TVM profiling data.
        # For this conversion, we'll return an empty list or a pre-defined dummy tree.
        return self._event_tree

class _DummyRecordScope:
    # Dummy value, as TVM doesn't have a direct equivalent for PyTorch's RecordScope enum
    FUNCTION = 0

class profile:
    def __init__(self, with_stack=False, profile_memory=False, record_shapes=False):
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.profiler = _DummyProfiler()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Dummy _utils for profiling, as actual traversal relies on PyTorch's internal objects
class _DummyUtils:
    def traverse_dfs(self, nodes):
        yield from nodes # Simple iteration for dummy nodes

_utils = _DummyUtils()

# Dummy class for nn.Module and related objects
# TVM does not have a direct equivalent for Python object introspection like this
class DummyModule:
    def __init__(self, name):
        self.name = name
        self.parameters = [] # List of (name, _TensorMetadata, _TensorMetadata)

class DummyOptimizer:
    def __init__(self, name, parameters):
        self.name = name
        self.self_ptr = id(self)
        self.parameters = parameters # List of (_TensorMetadata, _TensorMetadata, List[Tuple[str, _TensorMetadata]])
        self.param_groups = [{"params": []}] # Simplified representation

# Replace torch.nn.Module with a dummy placeholder.
# A full TVM conversion would involve building a Relay graph for the neural network.
# Since this test focuses on runtime details, we'll use a mock.
class SimpleNet:
    def __init__(self) -> None:
        self.fc1_weight = tvm_nd.array(np.random.randn(5, 10).astype("float32"))
        self.fc1_bias = tvm_nd.array(np.random.randn(5).astype("float32"))
        self.fc2_weight = tvm_nd.array(np.random.randn(2, 5).astype("float32"))
        self.fc2_bias = tvm_nd.array(np.random.randn(2).astype("float32"))

    def parameters(self):
        # Return a simplified structure for the test that expects PyTorch-like parameter objects
        # For the purpose of profiling metadata tracking, these are mock _TensorMetadata
        return [
            (
                "fc1.weight",
                _TensorMetadata(id=101, storage_data_ptr=self.fc1_weight.data.data),
                _TensorMetadata(id=102, storage_data_ptr=0) # Mock grad
            ),
            (
                "fc1.bias",
                _TensorMetadata(id=103, storage_data_ptr=self.fc1_bias.data.data),
                _TensorMetadata(id=104, storage_data_ptr=0) # Mock grad
            ),
            (
                "fc2.weight",
                _TensorMetadata(id=105, storage_data_ptr=self.fc2_weight.data.data),
                _TensorMetadata(id=106, storage_data_ptr=0) # Mock grad
            ),
            (
                "fc2.bias",
                _TensorMetadata(id=107, storage_data_ptr=self.fc2_bias.data.data),
                _TensorMetadata(id=108, storage_data_ptr=0) # Mock grad
            ),
        ]

    def __call__(self, x_in):
        # This part of the model is computational, would typically be a Relay graph.
        # For a profiler test, we mock its effect or return a placeholder.
        # Original test calls model(x).backward(), which implies differentiable ops.
        # This requires a Relay graph with AD, which is too complex to mock simply.
        # For now, we return a dummy output.
        return tvm_nd.array(np.random.randn(2).astype("float32"))

    # Original _get_relay_graph for SimpleNet, currently not used by any non-skipped test parts
    # def _get_relay_graph(self, input_shape, input_dtype):
    #     data = relay.var("data", shape=input_shape, dtype=input_dtype)
    #     w1 = relay.var("fc1_weight", shape=(5, 10), dtype="float32")
    #     b1 = relay.var("fc1_bias", shape=(5,), dtype="float32")
    #     w2 = relay.var("fc2_weight", shape=(2, 5), dtype="float32")
    #     b2 = relay.var("fc2_bias", shape=(2,), dtype="float32")

    #     dense1 = relay.op.nn.dense(data, w1)
    #     add1 = relay.op.add(dense1, b1)
    #     dense2 = relay.op.nn.dense(add1, w2)
    #     out = relay.op.add(dense2, b2)
    #     return relay.Function([data, w1, b1, w2, b2], out)


class TestTorchTidyProfiler(unittest.TestCase):
    def _get_tensor_fields(self, node, index):
        self.assertIsNotNone(node)
        # In TVM, we don't have direct _ExtraFields_TorchOp with inputs as _TensorMetadata
        # This part of the test is inherently PyTorch-specific for runtime introspection.
        # We will mock the return to avoid errors, but it won't reflect actual TVM internals.
        if isinstance(node.extra_fields, _ExtraFields_TorchOp) and node.extra_fields.inputs:
            tensor_info = node.extra_fields.inputs[index]
            self.assertIsInstance(tensor_info, _TensorMetadata)
            # These assertions would rely on mocked data, not real TVM pointers/ids
            # self.assertIsNotNone(tensor_info.impl_ptr)
            # self.assertIsNotNone(tensor_info.storage_data_ptr)
            # self.assertIsNotNone(tensor_info.id)
            return tensor_info.impl_ptr, tensor_info.storage_data_ptr, tensor_info.id
        else:
            # Fallback for nodes that don't have this structure or for mocked behavior
            # Return dummy values
            return 0, 0, 0

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl/Storage behavior, not directly translatable to TVM.")
    def test_pointers_and_ids(self):
        self.skipTest("This test is highly PyTorch-specific due to runtime tensor/storage details.")
        # Original logic:
        # a = torch.randn(4, 3)
        # a_initial_storage_data = a.storage().data_ptr()
        # b = a.view((1, 12))
        # c = torch.randn(4, 1)
        # c_initial_storage_data = c.storage().data_ptr()
        # d = torch.randn(4, 3)

        # with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
        #     _ = a + c
        #     _ = b * c
        #     f = a.resize_(128, 129)
        #     _ = torch.relu(f)
        #     _ = d.sin()
        #     c.set_(d.storage())
        #     _ = c.cos()

        # nodes = p.profiler.kineto_results.experimental_event_tree()

        # def get_fields(op_name, index):
        #     return self._get_tensor_fields(find_node_with_name(nodes, op_name), index)

        # a_impl, a_storage_data, a_id = get_fields("aten::add", 0)
        # b_impl, b_storage_data, _ = get_fields("aten::mul", 0)

        # self.assertEqual(a_storage_data, a_initial_storage_data)
        # self.assertEqual(a_storage_data, b_storage_data)
        # self.assertNotEqual(a_impl, b_impl)

        # c_impl, c_storage_data, c_id = get_fields("aten::add", 1)
        # self.assertEqual((c_impl, c_storage_data, c_id), get_fields("aten::mul", 1))
        # self.assertEqual(c_storage_data, c_initial_storage_data)

        # f_impl, f_storage_data, f_id = get_fields("aten::relu", 0)
        # self.assertEqual(a_impl, f_impl)
        # self.assertNotEqual(a_storage_data, f_storage_data)
        # self.assertEqual(a_id, f_id)

        # d_impl, d_storage_data, d_id = get_fields("aten::sin", 0)
        # c_impl_new, c_storage_data_new, c_id_new = get_fields("aten::cos", 0)
        # self.assertNotEqual(d_impl, c_impl_new)
        # self.assertEqual(d_storage_data, c_storage_data_new)
        # self.assertEqual(c_id, c_id_new)
        # self.assertEqual(d_id, c_id_new)

    @staticmethod
    def _format_allocations(profiled_code):
        # This function and the tests using it (`test_tensorimpl_invalidation_*`, `test_allocation_ids`)
        # are deeply tied to PyTorch's memory profiling features, which monitor individual tensor
        # allocations and deallocations via internal pointers and IDs.
        # There is no direct, equivalent high-level TVM Python API for this kind of memory profiling.
        # TVM's memory model is different, often involving static memory planning or a runtime
        # that doesn't expose object-level allocation IDs in the same way.
        return "TODO: Implement TVM equivalent for _format_allocations or skip test"

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl memory invalidation/lifecycle.")
    def test_tensorimpl_invalidation_set(self) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal memory management.")
        # Original logic:
        # def profiled_code(add_empty_set: bool):
        #     x = torch.ones((1,))
        #     if add_empty_set:
        #         x.set_()
        #     x.set_(torch.ones((1,)).storage())
        #     x.view_as(x)
        # self.assertExpectedInline(
        #     self._format_allocations(lambda: profiled_code(add_empty_set=False)),
        #     """\
        #         0          1      Allocation
        #         0          2      Allocation
        #         0          1      Free
        #         0          2      Free""",
        # )
        # self.assertExpectedInline(
        #     self._format_allocations(lambda: profiled_code(add_empty_set=True)),
        #     """\
        #         0          1      Allocation
        #         0          1      Free
        #         0          2      Allocation
        #         0          2      Free""",
        # )

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl memory invalidation/lifecycle.")
    def test_tensorimpl_invalidation_keep_alive(self) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal memory management.")

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl memory invalidation/lifecycle.")
    def test_tensorimpl_invalidation_full(self) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal memory management.")

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl memory invalidation/lifecycle.")
    def test_tensorimpl_invalidation_scalar_args(self) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal memory management.")

    @pytest.mark.skip(reason="Tests PyTorch-specific internal Module/Optimizer parameter tracking and memory lifecycle.")
    def test_module_and_optimizer_ids(self) -> None:
        self.skipTest("This test relies on PyTorch's internal Module and Optimizer introspection, not directly translatable to TVM.")
        # Original logic:
        # model = torch.nn.Linear(2, 1, bias=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        # def check(cold_start: bool) -> None:
        #     with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
        #         x = torch.ones((1, 2))
        #         _ = x.sin()
        #         model(x).backward()
        #         optimizer.step()
        #         _ = optimizer.state[model.weight][
        #             "momentum_buffer"
        #         ].cos()
        #         _ = model.weight.grad.tan()

        #     nodes = p.profiler.kineto_results.experimental_event_tree()

        #     def get_fields(op_name, index):
        #         return self._get_tensor_fields(
        #             find_node_with_name(nodes, op_name), index
        #         )

        #     _, _, x_id = get_fields("aten::sin", 0)
        #     _, _, weight_momenumtum_id = get_fields("aten::cos", 0)
        #     _, _, weight_grad_id = get_fields("aten::tan", 0)
        #     self.assertNotEqual(x_id, weight_momenumtum_id)
        #     self.assertNotEqual(x_id, weight_grad_id)
        #     self.assertNotEqual(weight_momenumtum_id, weight_grad_id)

        #     linear_op_node = find_node_with_name(nodes, "aten::linear")
        #     self.assertIsNotNone(linear_op_node)
        #     x_metadata, weight_metadata, _ = linear_op_node.extra_fields.inputs
        #     self.assertEqual(x_id, x_metadata.id)

        #     linear_module_node = find_node_with_name(nodes, "nn.Module: Linear_0")
        #     self.assertIsNotNone(linear_module_node)
        #     self.assertIsNotNone(linear_module_node.extra_fields.module)
        #     self.assertIsNone(linear_module_node.extra_fields.optimizer)

        #     linear_parameters = linear_module_node.extra_fields.module.parameters
        #     name, weight, weight_grad = linear_parameters[0]
        #     self.assertEqual(name, "weight")
        #     self.assertEqual(weight.id, weight_metadata.id)

        #     self.assertEqual(weight_grad is None, cold_start)
        #     if not cold_start:
        #         self.assertEqual(weight_grad.id, weight_grad_id)

        #     step_node = find_node_with_regex(nodes, "_optimizer_step_code")
        #     self.assertIsNotNone(step_node)
        #     self.assertIsNone(step_node.extra_fields.module)
        #     self.assertIsNotNone(step_node.extra_fields.optimizer)
        #     optimizer_parameters = step_node.extra_fields.optimizer.parameters
        #     self.assertEqual(len(optimizer_parameters), 2)
        #     weight, weight_grad, state = optimizer_parameters[0]
        #     self.assertEqual(weight.id, weight_metadata.id)
        #     self.assertEqual(weight_grad.id, weight_grad_id)
        #     self.assertEqual(len(state), 1)
        #     self.assertEqual(state[0][0], "momentum_buffer")
        #     self.assertEqual(state[0][1].id, weight_momenumtum_id)

        # check(cold_start=True)
        # check(cold_start=False)

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl memory invalidation/lifecycle.")
    def _test_allocation_ids(self, before_fn, after_fn) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal memory management.")

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl memory invalidation/lifecycle.")
    def test_allocation_ids(self) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal memory management.")

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl memory invalidation/lifecycle.")
    def test_allocation_ids_with_other_ops(self) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal memory management.")

    @pytest.mark.skip(reason="Tests PyTorch-specific internal TensorImpl reuse behavior and allocation tracking.")
    def test_impl_reuse(self) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal TensorImpl reuse behavior.")
        # Original logic:
        # repeats = 1_000
        # with profile(profile_memory=True, record_shapes=True) as p:
        #     for _ in range(repeats):
        #         torch.ones((1,))
        #     gc.collect()

        # roots = p.profiler.kineto_results.experimental_event_tree()
        # tensor_impls = tuple(
        #     e.extra_fields.inputs[0].impl_ptr
        #     for e in _utils.traverse_dfs(roots)
        #     if e.name == "aten::fill_"
        # )

        # self.assertEqual(len(tensor_impls), repeats)
        # self.assertEqual(len(set(tensor_impls)), repeats)

    @pytest.mark.skip(reason="Tests PyTorch-specific internal allocation ID uniqueness and tracking.")
    def test_allocation_id_uniqueness(self) -> None:
        self.skipTest("This test is highly PyTorch-specific due to internal allocation ID tracking.")
        # Original logic:
        # repeats = 1_000
        # with profile(profile_memory=True, record_shapes=True) as p:
        #     for _ in range(repeats):
        #         torch.ones((1,))
        #     gc.collect()

        # roots = p.profiler.kineto_results.experimental_event_tree()
        # id_set = set()
        # for e in _utils.traverse_dfs(roots):
        #     fields = e.extra_fields
        #     if isinstance(fields, torch._C._profiler._ExtraFields_TorchOp):
        #         id_set |= {
        #             t.allocation_id
        #             for t in fields.inputs
        #             if isinstance(t, _TensorMetadata)
        #         }

        #     elif isinstance(fields, torch._C._profiler._ExtraFields_Allocation):
        #         id_set.add(fields.allocation_id)

        # id_set.difference_update([None])
        # self.assertEqual(repeats, len(id_set))

    @pytest.mark.skip(reason="Tests PyTorch-specific internal profiler extra fields and event structure.")
    def test_extra_fields(self):
        self.skipTest("This test is highly PyTorch-specific due to profiler internal event structure.")
        # Original logic:
        # with profile(with_stack=True, profile_memory=True) as p:
        #     _ = torch.ones((1,))

        # nodes = p.profiler.kineto_results.experimental_event_tree()
        # node = find_node_with_name(nodes, "aten::ones")
        # self.assertIsNotNone(node)

        # self.assertIsInstance(
        #     node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        # )

        # self.assertIsInstance(
        #     node.parent.extra_fields, torch._C._profiler._ExtraFields_PyCCall
        # )

        # self.assertEqual(node.children[0].name, "aten::empty")
        # self.assertEqual(node.children[0].children[0].name, "[memory]")
        # self.assertIsInstance(
        #     node.children[0].children[0].extra_fields,
        #     torch._C._profiler._ExtraFields_Allocation,
        # )

    @pytest.mark.skip(reason="Tests PyTorch-specific internal tensor properties and metadata tracking.")
    def test_tensor_properties(self):
        self.skipTest("This test is highly PyTorch-specific due to internal tensor metadata tracking.")
        # Original logic:
        # x = torch.ones(10, 10).as_strided([4, 4], [12, 3])
        # y = torch.ones(4, 1, requires_grad=True)

        # with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
        #     _ = x + y
        #     _ = x * y

        # nodes = p.profiler.kineto_results.experimental_event_tree()
        # node = find_node_with_name(nodes, "aten::add")
        # self.assertIsNotNone(node)

        # self.assertIsInstance(
        #     node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        # )

        # def getattr_inputs(name, default):
        #     return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # self.assertEqual(getattr_inputs("sizes", []), [[4, 4], [4, 1], []])
        # self.assertEqual(getattr_inputs("strides", []), [[12, 3], [1, 1], []])
        # self.assertEqual(
        #     getattr_inputs("layout", None), [torch.strided, torch.strided, None]
        # )
        # self.assertEqual(
        #     getattr_inputs("device", None),
        #     [torch.device("cpu"), torch.device("cpu"), None],
        # )
        # self.assertEqual(
        #     getattr_inputs("dtype", None), [torch.float32, torch.float32, None]
        # )
        # self.assertEqual(node.extra_fields.scope, torch.profiler.RecordScope.FUNCTION)

        # mul_node = find_node_with_name(nodes, "aten::mul")
        # self.assertIsNotNone(mul_node)
        # self.assertEqual(
        #     node.extra_fields.sequence_number + 1, mul_node.extra_fields.sequence_number
        # )

    @pytest.mark.skip(reason="Tests PyTorch-specific internal sparse tensor layout tracking.")
    def test_sparse_tensors(self):
        self.skipTest("This test is highly PyTorch-specific due to sparse tensor layout tracking.")
        # Original logic:
        # i = [[0, 1, 1], [2, 0, 2]]
        # v = [3, 4, 5]
        # s = torch.sparse_coo_tensor(i, v, (2, 3))

        # with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
        #     _ = s + s

        # nodes = p.profiler.kineto_results.experimental_event_tree()
        # node = find_node_with_name(nodes, "aten::add")
        # self.assertIsNotNone(node)

        # self.assertIsInstance(
        #     node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        # )

        # def getattr_inputs(name, default):
        #     return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # self.assertEqual(getattr_inputs("sizes", []), [[2, 3], [2, 3], []])
        # self.assertEqual(getattr_inputs("strides", []), [[], [], []])
        # self.assertEqual(
        #     getattr_inputs("layout", None), [torch.sparse_coo, torch.sparse_coo, None]
        # )
        # self.assertEqual(
        #     getattr_inputs("device", None),
        #     [torch.device("cpu"), torch.device("cpu"), None],
        # )

    @pytest.mark.skip(reason="Tests PyTorch-specific MKL-DNN tensor layout tracking.")
    def test_mkldnn_tensors(self):
        self.skipTest("This test is highly PyTorch-specific due to MKL-DNN tensor layout tracking.")
        # Original logic:
        # x = torch.ones(4, 3).to_mkldnn()

        # with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
        #     _ = x + x

        # nodes = p.profiler.kineto_results.experimental_event_tree()
        # node = find_node_with_name(nodes, "aten::add")
        # self.assertIsNotNone(node)

        # self.assertIsInstance(
        #     node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        # )

        # def getattr_inputs(name, default):
        #     return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # self.assertEqual(getattr_inputs("sizes", []), [[4, 3], [4, 3], []])
        # self.assertEqual(getattr_inputs("strides", []), [[], [], []])
        # self.assertEqual(
        #     getattr_inputs("layout", None), [torch._mkldnn, torch._mkldnn, None]
        # )
        # self.assertEqual(
        #     getattr_inputs("device", None),
        #     [torch.device("cpu"), torch.device("cpu"), None],
        # )

    @pytest.mark.skip(reason="Tests PyTorch-specific scalar promotion and profiler metadata.")
    def test_scalar_ins(self):
        self.skipTest("This test is highly PyTorch-specific due to scalar promotion and profiler metadata.")
        # Original logic:
        # x = torch.ones(5, 5)
        # alpha = 0.9

        # with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
        #     _ = torch.add(x, 9.1, alpha=alpha)

        # nodes = p.profiler.kineto_results.experimental_event_tree()
        # node = find_node_with_name(nodes, "aten::add")
        # self.assertIsNotNone(node)

        # def getattr_inputs(name, default):
        #     return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # self.assertEqual(
        #     getattr_inputs("dtype", None), [torch.float32, torch.float64, None]
        # )
        # self.assertEqual(getattr_inputs("sizes", []), [[5, 5], [], []])
        # self.assertEqual(node.extra_fields.inputs[2], alpha)

    @pytest.mark.skip(reason="Tests PyTorch-specific internal list of tensors handling in profiler.")
    def test_tensor_lists(self):
        self.skipTest("This test is highly PyTorch-specific due to profiler's handling of tensor lists.")
        # Original logic:
        # x = torch.ones((1,))
        # y = torch.ones((1,))
        # with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
        #     _ = torch.stack((x, y))

        # nodes = p.profiler.kineto_results.experimental_event_tree()
        # node = find_node_with_name(nodes, "aten::stack")
        # inputs = node.extra_fields.inputs
        # self.assertEqual(len(inputs), 2)
        # self.assertIsInstance(inputs[0], list)
        # self.assertEqual(len(inputs[0]), 2)
        # self.assertEqual(x.storage().data_ptr(), inputs[0][0].storage_data_ptr)
        # self.assertEqual(y.storage().data_ptr(), inputs[0][1].storage_data_ptr)

    # Helper function to find node by name in the dummy tree
    def _find_node_with_name(self, nodes_list, name):
        for node in nodes_list:
            if node.name == name:
                return node
            found = self._find_node_with_name(node.children, name)
            if found:
                return found
        return None

    @pytest.mark.skip(reason="Tests PyTorch-specific nn.Module parameter introspection by profiler.")
    def test_nnmodule_params(self):
        # This test relies on PyTorch's ability to introspect `nn.Module`'s `_parameters` and `grad`
        # which is not directly supported by TVM. We can mock the structure for some checks,
        # but the underlying `storage_data_ptr` and `grad` attributes are PyTorch runtime specific.
        self.skipTest("This test is highly PyTorch-specific due to nn.Module parameter introspection.")
        # Original logic:
        # def flat_out_extrafields(nodes, out=None):
        #     if out is None:
        #         out = []
        #     for node in nodes:
        #         if (
        #             isinstance(node.extra_fields, _ExtraFields_PyCall)
        #             and node.extra_fields.module
        #         ):
        #             if node.extra_fields.module.parameters:
        #                 out.append(node.extra_fields.module)
        #         flat_out_extrafields(node.children, out)
        #     return out

        # inputs_np = np.random.rand(10).astype("float32")
        # inputs = tvm_nd.array(inputs_np)
        # net = SimpleNet()

        # # Mock the profiling context
        # # For the purpose of this test, we construct a dummy event tree that simulates
        # # the structure the PyTorch profiler would capture.
        # mock_fc1_module = DummyModule("Linear_0")
        # mock_fc1_module.parameters = [
        #     ("weight", _TensorMetadata(storage_data_ptr=net.fc1_weight.data.data, id=id(net.fc1_weight)), _TensorMetadata(storage_data_ptr=0, id=0)),
        #     ("bias", _TensorMetadata(storage_data_ptr=net.fc1_bias.data.data, id=id(net.fc1_bias)), _TensorMetadata(storage_data_ptr=0, id=0)),
        # ]
        # mock_fc2_module = DummyModule("Linear_1")
        # mock_fc2_module.parameters = [
        #     ("weight", _TensorMetadata(storage_data_ptr=net.fc2_weight.data.data, id=id(net.fc2_weight)), _TensorMetadata(storage_data_ptr=0, id=0)),
        #     ("bias", _TensorMetadata(storage_data_ptr=net.fc2_bias.data.data, id=id(net.fc2_bias)), _TensorMetadata(storage_data_ptr=0, id=0)),
        # ]

        # dummy_nodes = [
        #     _Node(
        #         "nn.Module: SimpleNet_0",
        #         _ExtraFields_PyCCall(),
        #         children=[
        #             _Node(
        #                 "nn.Module: Linear_0",
        #                 _ExtraFields_PyCCall(module=mock_fc1_module),
        #                 children=[]
        #             ),
        #             _Node(
        #                 "nn.Module: Linear_1",
        #                 _ExtraFields_PyCCall(module=mock_fc2_module),
        #                 children=[]
        #             ),
        #         ]
        #     )
        # ]

        # with profile(with_stack=True, profile_memory=True) as p:
        #     p.profiler._event_tree = dummy_nodes

        # modules = flat_out_extrafields(
        #     p.profiler.kineto_results.experimental_event_tree()
        # )
        # self.assertEqual(
        #     len(modules), 2, f"Expected two parameter list, but got {len(modules)}"
        # )

        # params = [
        #     (n, p.storage_data_ptr, g.storage_data_ptr)
        #     for module in modules
        #     for (n, p, g) in module.parameters
        # ]
        # expected = [
        #     ("weight", net.fc1_weight.data.data, 0),
        #     ("bias", net.fc1_bias.data.data, 0),
        #     ("weight", net.fc2_weight.data.data, 0),
        #     ("bias", net.fc2_bias.data.data, 0),
        # ]
        # self.assertEqual(expected, params, f"{expected} vs. {params}")


    @pytest.mark.skip(reason="Tests PyTorch-specific optimizer state introspection by profiler.")
    def _flat_out_extrafields(self, nodes, out=None):
        self.skipTest("This test is highly PyTorch-specific due to optimizer introspection.")
        return []

    @pytest.mark.skip(reason="Tests PyTorch-specific optimizer state introspection by profiler.")
    def _check_results(self, opt, opts, check_items=False):
        self.skipTest("This test is highly PyTorch-specific due to optimizer introspection.")

    @pytest.mark.skip(reason="Tests PyTorch-specific optimizer state introspection by profiler.")
    def test_optimizer(self):
        self.skipTest("This test is highly PyTorch-specific due to optimizer introspection.")

    @pytest.mark.skip(reason="Tests PyTorch-specific optimizer state introspection by profiler.")
    def _test_optimizer_parameters(self, optimizer_factory):
        self.skipTest("This test is highly PyTorch-specific due to optimizer introspection.")

    @pytest.mark.skip(reason="Tests PyTorch-specific optimizer state introspection by profiler.")
    def test_optimizer_parameters_sgd(self):
        self.skipTest("This test is highly PyTorch-specific due to optimizer introspection.")

    @pytest.mark.skip(reason="Tests PyTorch-specific optimizer state introspection by profiler.")
    def test_optimizer_parameters_adam(self):
        self.skipTest("This test is highly PyTorch-specific due to optimizer introspection.")

    @pytest.mark.skip(reason="Tests PyTorch-specific internal allocation tracking and memory details.")
    def test_allocations(self):
        self.skipTest("This test is highly PyTorch-specific due to internal memory allocation tracking.")
        # Original logic:
        # gc.collect()
        # with profile(profile_memory=True) as p:
        #     x = torch.empty((3, 4))

        # nodes = p.profiler.kineto_results.experimental_event_tree()
        # node = find_node_with_name(nodes, "[memory]")
        # self.assertIsNotNone(node)

        # alloc_size = 3 * 4 * 4  # fp32 -> 4 bytes
        # ptr = node.extra_fields.ptr
        # self.assertGreater(ptr, 0)
        # self.assertEqual(node.extra_fields.alloc_size, alloc_size)
        # self.assertEqual(node.extra_fields.device, torch.device("cpu"))
        # total_allocated = node.extra_fields.total_allocated

        # self.assertEqual(node.extra_fields.total_reserved, 0)

        # with profile(profile_memory=True) as p:
        #     del x
        #     gc.collect()

        # nodes = p.profiler.kineto_results.experimental_event_tree()
        # node = find_node_with_name(nodes, "[memory]")
        # self.assertIsNotNone(node)

        # self.assertEqual(node.extra_fields.ptr, ptr)
        # self.assertEqual(node.extra_fields.alloc_size, -alloc_size)
        # self.assertEqual(node.extra_fields.device, torch.device("cpu"))
        # self.assertEqual(
        #     node.extra_fields.total_allocated, total_allocated - alloc_size
        # )

    @pytest.mark.skip(reason="Tests Python CPython object reference counting behavior, not TVM specific.")
    def test_refcounts(self):
        self.skipTest("Tests Python's ref counting and GC behavior, not TVM specific.")
        # Original logic:
        # class Sentinel:
        #     pass

        # def make():
        #     outer_sentinel = Sentinel()

        #     def outer():
        #         _ = outer_sentinel
        #         inner_sentinel = Sentinel()

        #         def inner():
        #             _ = inner_sentinel

        #         with profile(with_stack=True):
        #             inner()

        #         return weakref.ref(inner_sentinel)

        #     return outer, weakref.ref(outer_sentinel)

        # outer, outer_sentinel_ref = make()
        # inner_sentinel_ref = outer()

        # self.assertIsNone(inner_sentinel_ref())
        # self.assertIsNotNone(outer_sentinel_ref())

        # del outer
        # self.assertIsNone(outer_sentinel_ref())


if __name__ == "__main__":
    # To run this file, pytest is preferred to handle skips.
    # If running with unittest directly, the skips will still work.
    unittest.main()
