import os
import sys
from typing import List, Tuple, Union

import numpy as np
import pytest
import tvm
import tvm.relay as relay
import tvm.testing
from tvm.relay.testing import run_infer_type


# TVM does not have a direct equivalent for PyTorch's JitTestCase or its
# concept of TorchScript compilation for arbitrary Python code, especially
# for data structures like lists and tuples that are evaluated at Python runtime.
# Tests involving `torch.Tensor` are converted to TVM Relay graphs.
# Tests involving Python lists/strings/tuples are converted to direct Python assertions.
class TestSlice:

    # Helper function to compile and run a Relay expression for tensor-based tests
    def _check_relay_script(self, fn_expr_builder, inputs_np, expected_output_np):
        # inputs_np is a dictionary of input_name: numpy_array
        
        relay_vars = []
        params = {}
        
        for name, val_np in inputs_np.items():
            var = relay.var(name, relay.TensorType(val_np.shape, str(val_np.dtype)))
            relay_vars.append(var)
            params[name] = tvm.nd.array(val_np)

        # Build the Relay function expression using the provided builder
        if len(relay_vars) == 1:
            fn_expr = fn_expr_builder(relay_vars[0])
        elif len(relay_vars) > 1:
            fn_expr = fn_expr_builder(*relay_vars)
        else: # No tensor inputs, should not happen for _check_relay_script
            raise ValueError("No tensor inputs provided for Relay compilation.")

        func = relay.Function(relay_vars, fn_expr)
        mod = tvm.IRModule.from_expr(func)
        mod = run_infer_type(mod) # Infer types to ensure graph is valid

        target = "llvm"
        dev = tvm.cpu(0)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        module = tvm.runtime.GraphModule(lib["default"](dev))
        for name, val_nd in params.items():
            module.set_input(name, val_nd)
        
        module.run()
        actual_output = module.get_output(0).numpy()

        tvm.testing.assert_allclose(actual_output, expected_output_np, rtol=1e-5, atol=1e-5)


    def test_slice_kwarg(self):
        # PyTorch specific: TorchScript does not accept keyword arguments for slice.
        # This test checks a TorchScript compilation error.
        # TVM's Relay doesn't have a direct equivalent of this restriction at the
        # graph construction level, as Python 'slice' objects with kwargs would
        # fail in Python itself before Relay compilation, or if used to build
        # a Relay op, the op API would not accept kwargs.
        # We skip this as it tests a very specific TorchScript behavior.
        pytest.skip("TODO: TorchScript-specific compilation error not directly convertible to TVM Relay.")

    def test_slice_three_nones(self):
        # This tests Python list slicing, which is executed directly in Python.
        def three_nones_python(x: List[int]):
            return x[slice(None, None, None)]
        
        input_list = list(range(10))
        expected = three_nones_python(input_list)
        assert expected == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_slice_two_nones(self):
        # This tests Python list slicing, which is executed directly in Python.
        def two_nones_python(x: List[int]):
            return x[slice(None, None)]

        input_list = list(range(10))
        expected = two_nones_python(input_list)
        assert expected == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_slice_one_none(self):
        # This tests Python list slicing, which is executed directly in Python.
        def one_none_python(x: List[int]):
            return x[slice(None)]
        
        input_list = list(range(10))
        expected = one_none_python(input_list)
        assert expected == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_slice_stop_only(self):
        # This tests Python list slicing, which is executed directly in Python.
        def fn_python(x: List[int]):
            return x[slice(5)]
        
        input_list = list(range(10))
        expected = fn_python(input_list)
        assert expected == [0, 1, 2, 3, 4]

    def test_slice_stop_only_with_nones(self):
        # This tests Python list slicing, which is executed directly in Python.
        def fn_python(x: List[int]):
            return x[slice(None, 5, None)]
        
        input_list = list(range(10))
        expected = fn_python(input_list)
        assert expected == [0, 1, 2, 3, 4]

    def test_slice_start_stop(self):
        # This tests Python list slicing, which is executed directly in Python.
        def fn_python(x: List[int]):
            return x[slice(1, 5)]

        input_list = list(range(10))
        expected = fn_python(input_list)
        assert expected == [1, 2, 3, 4]

    def test_slice_start_stop_with_none(self):
        # This tests Python list slicing, which is executed directly in Python.
        def fn_python(x: List[int]):
            return x[slice(1, 5, None)]

        input_list = list(range(10))
        expected = fn_python(input_list)
        assert expected == [1, 2, 3, 4]

    def test_slice_start_stop_step(self):
        # This tests Python list slicing, which is executed directly in Python.
        def fn_python(x: List[int]):
            return x[slice(0, 6, 2)]

        input_list = list(range(10))
        expected = fn_python(input_list)
        assert expected == [0, 2, 4]

    def test_slice_string(self):
        # This tests Python string slicing, which is executed directly in Python.
        def fn_python(x: str):
            return x[slice(None, 3, 1)]

        input_string = "foo_bar"
        expected = fn_python(input_string)
        assert expected == "foo"

    def test_slice_tensor(self):
        # This tests tensor slicing, converted to Relay.
        def fn_relay(x_var):
            # x[slice(None, 3, 1)] is equivalent to x[0:3:1]
            return relay.op.transform.strided_slice(x_var, begin=[0], end=[3], strides=[1], axes=[0])

        input_np = np.ones(10, dtype=np.float32)
        expected_output_np = input_np[0:3:1]
        self._check_relay_script(fn_relay, {"x": input_np}, expected_output_np)

    def test_slice_tensor_multidim(self):
        # This tests multi-dimensional tensor slicing, converted to Relay.
        def fn_relay(x_var):
            # x[slice(None, 3, 1), 0]
            # Slices the first dimension from 0 to 3 with step 1.
            # Takes the 0-th index of the second dimension.
            # A strided_slice for both dimensions, then squeeze to remove the singleton dimension.
            sliced_x = relay.op.transform.strided_slice(
                x_var, begin=[0, 0], end=[3, 1], strides=[1, 1], axes=[0, 1]
            )
            # The result of strided_slice on (10, 10) with end=(3,1) for axes=(0,1) would be (3,1)
            # Squeeze the second dimension (axis=1) to get (3,)
            return relay.op.transform.squeeze(sliced_x, axis=[1])

        input_np = np.ones((10, 10), dtype=np.float32)
        expected_output_np = input_np[0:3:1, 0]
        self._check_relay_script(fn_relay, {"x": input_np}, expected_output_np)

    def test_slice_tensor_multidim_with_dots(self):
        # This tests multi-dimensional tensor slicing with '...', converted to Relay.
        def fn_relay(x_var):
            # x[slice(None, 3, 1), ...] is equivalent to x[0:3:1, :, :]
            # Applying strided_slice to the first axis explicitly covers this.
            return relay.op.transform.strided_slice(x_var, begin=[0], end=[3], strides=[1], axes=[0])

        input_np = np.ones((10, 10), dtype=np.float32)
        expected_output_np = input_np[0:3:1, ...] # equivalent to input_np[0:3:1, :]
        self._check_relay_script(fn_relay, {"x": input_np}, expected_output_np)

    def test_slice_as_variable(self):
        # This tests Python list slicing with a slice object stored in a variable,
        # executed directly in Python.
        def fn_python(x: List[int]):
            a = slice(1)
            return x[a]

        input_list = list(range(10))
        expected = fn_python(input_list)
        assert expected == [0]

    def test_slice_stop_clipped(self):
        # This tests Python list slicing behavior where stop index is clipped,
        # executed directly in Python.
        def fn_python(x: List[int]):
            return x[slice(1000)] # Python list slicing clips automatically
        
        input_list = list(range(10))
        expected = fn_python(input_list)
        assert expected == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_slice_dynamic_index(self):
        # This tests tensor slicing with dynamically computed indices (in Python),
        # converted to Relay.
        def t_relay(x_var):
            # `zero` and `one` are constant values determined at Python graph construction time.
            # Their Python values are embedded into the Relay strided_slice operator.
            zero = 0
            one = zero + 1
            slice1 = relay.op.transform.strided_slice(x_var, begin=[0], end=[1], strides=[1], axes=[0])
            slice2 = relay.op.transform.strided_slice(x_var, begin=[zero], end=[one], strides=[1], axes=[0])
            return relay.op.tensor.add(slice1, slice2)

        input_np = np.zeros((3, 2, 3), dtype=np.float32)
        
        # Eager Python equivalent for expected value calculation
        slice1_eager = input_np[0:1]
        zero_eager = 0
        one_eager = zero_eager + 1
        slice2_eager = input_np[zero_eager:one_eager]
        expected_output_np = slice1_eager + slice2_eager
        
        self._check_relay_script(t_relay, {"x": input_np}, expected_output_np)

    def test_tuple_slicing(self):
        # This test checks Python tuple slicing and TorchScript's tuple optimization.
        # TVM Relay operates on computation graphs, not dynamic Python tuples or classes directly.
        # We convert to direct Python function execution and assertions, skipping TorchScript-specific IR checks.
        
        def tuple_slice_python(a_val: bool) -> Tuple[int, ...]:
            if a_val: 
                b = (1, 2, 3, 4)
            else:
                b = (4, 3, 2, 1)
            c = b[-4:4] # c is equivalent to b for both cases
            e = c[1:-1]
            return e

        # Test with a_val = True (mimicking torch.tensor([1]) which evaluates to True)
        actual_true = tuple_slice_python(True)
        assert actual_true == (2, 3)

        # Test with a_val = False
        actual_false = tuple_slice_python(False)
        assert actual_false == (3, 2)

        # The original TorchScript test included assertions about graph structure
        # (e.g., `findAllNodes`, `output().type().elements()`, `lower_all_tuples`).
        # These are specific to TorchScript's internal IR and optimization passes and
        # do not have direct equivalents in TVM Relay for this type of test.
        # Therefore, these specific assertions are not converted.

    def test_module_list_slicing(self):
        # This test checks TorchScript's ability to compile Python classes
        # (torch.nn.Module and ModuleList) and their slicing behavior.
        # TVM Relay operates on computational graphs and does not directly
        # compile Python classes with their methods and attributes.
        # The core of the test is Python list slicing and attribute access
        # on objects within that list. We can replicate the functional Python behavior
        # without involving Relay compilation.

        class Bar: # Simplified to a plain Python class
            def __init__(self, identifier: str):
                self.identifier = identifier

            def forward(self):
                return 0

        class Foo: # Simplified to a plain Python class
            def __init__(self) -> None:
                # Simulate ModuleList with a regular Python list of Bar instances
                module_list = [Bar("A"), Bar("B"), Bar("C"), Bar("D"), Bar("E")]
                self.test = module_list 

            def forward(self) -> Tuple[List[Bar], List[Bar]]:
                # Python list slicing behavior
                return self.test[::-2], self.test[1:4:]

        foo_instance = Foo()
        result1, result2 = foo_instance.forward()

        assert len(result1) == 3
        assert result1[0].identifier == "E"
        assert result1[1].identifier == "C"
        assert result1[2].identifier == "A"

        assert len(result2) == 3
        assert result2[0].identifier == "B"
        assert result2[1].identifier == "C"
        assert result2[2].identifier == "D"
