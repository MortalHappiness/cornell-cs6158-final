import itertools
import operator
import types
import unittest
import weakref
from collections import defaultdict, namedtuple, OrderedDict, UserDict
from typing import Any

import numpy as np
import pytest
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import tvm.testing


# Helper for `same` function from PyTorch
def same(a, b, rtol=1e-5, atol=1e-8):
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(same(x, y, rtol, atol) for x, y in zip(a, b))
    if isinstance(a, (dict, OrderedDict, defaultdict, UserDict, SimpleDict, DummyUserDict)):
        if a.keys() != b.keys():
            return False
        return all(same(a[k], b[k], rtol, atol) for k in a.keys())
    if isinstance(a, (int, float, bool, np.number)):
        return a == pytest.approx(b, rel=rtol, abs=atol)
    if isinstance(a, tvm.runtime.ndarray.NDArray):
        # Convert to numpy for comparison with tvm.testing.assert_allclose
        return tvm.testing.assert_allclose(a.numpy(), b.numpy() if isinstance(b, tvm.runtime.ndarray.NDArray) else b, rtol=rtol, atol=atol)
    if isinstance(a, np.ndarray):
        return tvm.testing.assert_allclose(a, b.numpy() if isinstance(b, tvm.runtime.ndarray.NDArray) else b, rtol=rtol, atol=atol)
    return a == b


def _run_relay_graph(relay_func_expr, inputs_dict):
    """A helper to build and run a Relay function for tests."""
    if isinstance(relay_func_expr, relay.Function):
        mod = tvm.IRModule.from_expr(relay_func_expr)
    elif isinstance(relay_func_expr, tvm.ir.IRModule):
        mod = relay_func_expr
    else:
        raise TypeError(f"Expected relay.Function or IRModule, got {type(relay_func_expr)}")

    target = "llvm"
    dev = tvm.cpu(0)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=None)

    rt_mod = graph_executor.GraphModule(lib["default"](dev))

    for name, val in inputs_dict.items():
        if isinstance(val, np.ndarray):
            rt_mod.set_input(name, tvm.nd.array(val, device=dev))
        elif isinstance(val, (int, float, bool, np.number)):
            # Convert Python scalars to NDArray for input (defaulting to float32 for safety)
            rt_mod.set_input(name, tvm.nd.array(np.array(val, dtype="float32"), device=dev))
        else:
            raise TypeError(f"Unsupported input type for Relay graph: {type(val)} for input {name}")
    
    rt_mod.run()

    num_outputs = rt_mod.get_num_outputs()
    if num_outputs == 1:
        return rt_mod.get_output(0)
    else:
        # If the output is a tuple, it will be wrapped in a Python list.
        return [rt_mod.get_output(i) for i in range(num_outputs)]


# Placeholder for torch._dynamo.test_case.TestCase
class TestCase:
    def assertEqual(self, a, b, msg=None):
        assert same(a, b), msg or f"Expected {a} to be equal to {b}"

    def assertTrue(self, condition, msg=None):
        assert condition, msg or f"Expected {condition} to be True"

    def assertFalse(self, condition, msg=None):
        assert not condition, msg or f"Expected {condition} to be False"

    def assertIs(self, a, b, msg=None):
        assert a is b, msg or f"Expected {a} to be {b} (identity)"

    def assertIsNone(self, obj, msg=None):
        assert obj is None, msg or f"Expected {obj} to be None"

    def assertIn(self, member, container, msg=None):
        assert member in container, msg or f"Expected {member} to be in {container}"

    def assertRaises(self, expected_exception, callable_obj=None, *args, **kwargs):
        if callable_obj is None:
            return pytest.raises(expected_exception)
        with pytest.raises(expected_exception):
            callable_obj(*args, **kwargs)

    def assertRaisesRegex(self, expected_exception, expected_regex, callable_obj=None, *args, **kwargs):
        if callable_obj is None:
            return pytest.raises(expected_exception, match=expected_regex)
        with pytest.raises(expected_exception, match=expected_regex):
            callable_obj(*args, **kwargs)

# Dummy placeholders for PyTorch internal tools that are not directly portable.
class CompileCounter:
    # Not directly mappable in TVM; Dynamo-specific graph recompilation counter.
    # Its assertions are removed from the converted tests.
    def __init__(self):
        self.frame_count = 0
        self.op_count = 0
    def __call__(self, *args, **kwargs):
        self.frame_count += 1
        self.op_count += 1
        # In a real Dynamo test, this would return the compiled model.
        # Here, it's a no-op as the compilation mechanism is not replicated.
        pass

# `dict_items` is the type returned by `dict.items()` in Python 3+.
# For PyTorch, this was `torch._dynamo.utils.dict_items`.
dict_items = type({}.items())

# Placeholder for LoggingTestCase
class LoggingTestCase(TestCase):
    # Logging-related assertions are not directly portable to TVM graph compilation.
    # Records will be empty.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_records = []

    def getRecord(self, records, key):
        class DummyRecord:
            def getMessage(self):
                return "Dummy message for TVM conversion"
        return DummyRecord()

# Placeholder for make_logging_test, munge_exc, parametrize, instantiate_parametrized_tests
# These decorators/functions are removed or simplified to pass through.
def make_dynamo_test(func): return func
def make_logging_test(recompiles=True): return lambda func: func
def parametrize(*args, **kwargs): return pytest.mark.parametrize(*args, **kwargs)
def instantiate_parametrized_tests(cls): pass

# Dummy for Unsupported exception (used in test_builtin_or_with_invalid_types)
class Unsupported(Exception):
    pass


class SimpleDict(dict):
    pass


class DummyUserDict(UserDict):
    pass


class DictTests(TestCase):
    def test_dict_subclass_instantiation(self):
        # Original PyTorch fn:
        # def fn(x):
        #     sd = SimpleDict(x=5)
        #     return sd["x"] * x

        # Convert to a Relay function builder
        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            sd_x_val = relay.const(5, "float32")
            result = relay.op.multiply(sd_x_val, x)
            return relay.Function([x], result)

        x_np = np.random.randn(4).astype("float32")

        # Reference execution (using Python objects + numpy)
        class _LocalSimpleDict(dict):
            pass
        sd_ref = _LocalSimpleDict(x=5)
        ref_output = sd_ref["x"] * x_np

        # TVM execution
        relay_func = _get_relay_func()
        tvm_output = _run_relay_graph(relay_func, {"x": x_np})

        self.assertEqual(ref_output, tvm_output)

    def test_dict_subclass_local_mutation(self):
        # Original PyTorch fn:
        # def fn(x):
        #     sd = SimpleDict(x=5)
        #     z = sd["x"] * x
        #     sd["x"] = 10
        #     return z * sd["x"]

        # This test relies on mutation side effects. In TVM, we model the final tensor computation.
        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            
            sd_x_initial = relay.const(5, "float32")
            z = relay.op.multiply(sd_x_initial, x)

            sd_x_final = relay.const(10, "float32")
            result = relay.op.multiply(z, sd_x_final)
            return relay.Function([x], result)

        x_np = np.random.randn(4).astype("float32")

        # Reference execution (using Python objects + numpy)
        class _LocalSimpleDict(dict):
            pass
        sd_ref = _LocalSimpleDict(x=5)
        z_ref = sd_ref["x"] * x_np
        sd_ref["x"] = 10
        ref_output = z_ref * sd_ref["x"]

        # TVM execution
        relay_func = _get_relay_func()
        tvm_output = _run_relay_graph(relay_func, {"x": x_np})

        self.assertEqual(ref_output, tvm_output)

    def test_dict_subclass_local_with_non_dict_method(self):
        # Original PyTorch fn:
        # class MethodDict(dict):
        #     def add_1(self, x):
        #         return x + 1
        # def fn(x):
        #     sd = MethodDict(x=5)
        #     z = sd["x"] * x
        #     sd["x"] = 10
        #     return sd.add_1(z * sd["x"])

        class _LocalMethodDict(dict):
            def add_1(self, x_val):
                return x_val + 1

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            
            sd_x_initial = relay.const(5, "float32")
            z = relay.op.multiply(sd_x_initial, x)

            sd_x_final = relay.const(10, "float32")
            
            intermediate_product = relay.op.multiply(z, sd_x_final)
            result = relay.op.add(intermediate_product, relay.const(1, "float32"))
            return relay.Function([x], result)

        x_np = np.random.randn(4).astype("float32")

        # Reference execution
        sd_ref = _LocalMethodDict(x=5)
        z_ref = sd_ref["x"] * x_np
        sd_ref["x"] = 10
        ref_output = sd_ref.add_1(z_ref * sd_ref["x"])

        # TVM execution
        relay_func = _get_relay_func()
        tvm_output = _run_relay_graph(relay_func, {"x": x_np})

        self.assertEqual(ref_output, tvm_output)

    def test_dict_contains(self):
        # This test relies on Python's `in` operator and potential recompilation.
        # We model the final behavior of the conditional.
        
        # Initial state for sd
        sd_initial = {2: 5, 4: 10}

        # First run: 1 not in sd
        def _get_relay_func_case1():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(3, "float32"))
            return relay.Function([x], result)

        x_np = np.random.randn(4).astype("float32")

        # Reference for case 1
        ref_output_case1 = x_np * 3

        # TVM execution for case 1
        relay_func_case1 = _get_relay_func_case1()
        tvm_output_case1 = _run_relay_graph(relay_func_case1, {"x": x_np})
        self.assertEqual(ref_output_case1, tvm_output_case1)

        # Second run: sd is mutated to include 1. This would cause recompilation in Dynamo.
        # In TVM, we build a new Relay graph for the changed condition.
        sd_initial[1] = 15 # Simulate mutation

        def _get_relay_func_case2():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(2, "float32"))
            return relay.Function([x], result)

        # Reference for case 2 (1 in sd)
        ref_output_case2 = x_np * 2

        # TVM execution for case 2
        relay_func_case2 = _get_relay_func_case2()
        tvm_output_case2 = _run_relay_graph(relay_func_case2, {"x": x_np})
        self.assertEqual(ref_output_case2, tvm_output_case2)

        # "Ensure not recompilation..." is Dynamo-specific and not directly mappable.
        sd_initial[2] = 10 # This change doesn't affect `1 in sd`, so `fn` produces same result.
        ref_output_case3 = x_np * 2
        tvm_output_case3 = _run_relay_graph(relay_func_case2, {"x": x_np})
        self.assertEqual(ref_output_case3, tvm_output_case3)

    def test_dict_subclass_methods_fallback_readonly(self):
        # This test relies on Python dict methods and attribute access.
        # We run the dict logic in Python and capture the final scalar multiplier for Relay.
        
        sd_initial = SimpleDict()
        sd_initial[2] = 5
        sd_initial[4] = 10
        sd_initial.attr = 4

        def fn_ref(x_np, sd_param):
            x_val = x_np
            for value in sd_param.values():
                x_val = x_val * value
            for key in sd_param.keys():
                x_val = x_val * key
            for k, v in sd_param.items():
                x_val = x_val * k
                x_val = x_val * v
            
            if 1 in sd_param:
                x_val = x_val * 2
            else:
                x_val = x_val * 3

            x_val = x_val * sd_param.get(2, 0)
            x_val = x_val * sd_param.get(3, 4)
            x_val = len(sd_param) * x_val
            x_val = x_val * sd_param.attr
            return x_val

        # Compute the scalar multiplier based on `sd_initial` for the first run
        current_x_np = np.array(1.0, dtype="float32")
        temp_sd = sd_initial.copy() # Use a copy to avoid mutating the original dict for later checks
        for value in temp_sd.values(): current_x_np = current_x_np * value
        for key in temp_sd.keys(): current_x_np = current_x_np * key
        for k, v in temp_sd.items(): current_x_np = current_x_np * k; current_x_np = current_x_np * v
        if 1 in temp_sd: current_x_np = current_x_np * 2
        else: current_x_np = current_x_np * 3
        current_x_np = current_x_np * temp_sd.get(2, 0)
        current_x_np = current_x_np * temp_sd.get(3, 4)
        current_x_np = len(temp_sd) * current_x_np
        current_x_np = current_x_np * temp_sd.attr
        scalar_multiplier_run1 = current_x_np.item()

        def _get_relay_func_run1():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(scalar_multiplier_run1, "float32"))
            return relay.Function([x], result)

        x_np = np.random.randn(4).astype("float32")
        
        ref_output = fn_ref(x_np, sd_initial)
        tvm_output = _run_relay_graph(_get_relay_func_run1(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)

        # "Ensure a recompilation" test: sd is mutated.
        sd_initial[6] = 15
        # Re-compute the scalar multiplier after mutation
        current_x_np_recomp = np.array(1.0, dtype="float32")
        temp_sd_recomp = sd_initial.copy()
        for value in temp_sd_recomp.values(): current_x_np_recomp = current_x_np_recomp * value
        for key in temp_sd_recomp.keys(): current_x_np_recomp = current_x_np_recomp * key
        for k, v in temp_sd_recomp.items(): current_x_np_recomp = current_x_np_recomp * k; current_x_np_recomp = current_x_np_recomp * v
        if 1 in temp_sd_recomp: current_x_np_recomp = current_x_np_recomp * 2
        else: current_x_np_recomp = current_x_np_recomp * 3
        current_x_np_recomp = current_x_np_recomp * temp_sd_recomp.get(2, 0)
        current_x_np_recomp = current_x_np_recomp * temp_sd_recomp.get(3, 4)
        current_x_np_recomp = len(temp_sd_recomp) * current_x_np_recomp
        current_x_np_recomp = current_x_np_recomp * temp_sd_recomp.attr
        scalar_multiplier_recomp = current_x_np_recomp.item()
        
        def _get_relay_func_recomp():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(scalar_multiplier_recomp, "float32"))
            return relay.Function([x], result)

        ref_output_recomp = fn_ref(x_np, sd_initial)
        tvm_output_recomp = _run_relay_graph(_get_relay_func_recomp(), {"x": x_np})
        self.assertEqual(ref_output_recomp, tvm_output_recomp)


    def test_dict_subclass_instantiation_return(self):
        # Original PyTorch fn:
        # def fn(x):
        #     sd = SimpleDict(x=5 * x)
        #     sd["y"] = 10
        #     return sd

        # TVM Relay can only return tensor values, not Python dicts directly.
        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            sd_x_val = relay.op.multiply(relay.const(5, "float32"), x)
            sd_y_val = relay.const(10, "float32")
            return relay.Function([x], relay.expr.Tuple([sd_x_val, sd_y_val]))

        x_np = np.random.randn(4).astype("float32")

        # Reference execution (Python dict creation)
        class _LocalSimpleDict(dict):
            pass
        sd_ref = _LocalSimpleDict(x=5 * x_np)
        sd_ref["y"] = 10
        
        # Extract the tensor values for comparison from the reference
        ref_x = sd_ref["x"]
        ref_y = sd_ref["y"]

        # TVM execution returns a tuple of NDArrays
        tvm_outputs = _run_relay_graph(_get_relay_func(), {"x": x_np})
        tvm_x, tvm_y = tvm_outputs[0], tvm_outputs[1]

        self.assertEqual(ref_x, tvm_x)
        self.assertEqual(ref_y, tvm_y)
        # Original `self.assertEqual(type(ref), type(res))` is not mappable.

    def test_dict_subclass_methods_fallback_mutation(self):
        # This test relies on mutation of the input dict (`sd`).
        # We treat the dict operations as Python constant folding at graph build time.
        
        def fn_ref(sd_param, x_np):
            x_val = x_np
            for value in sd_param.values():
                x_val = x_val * value
            sd_param[6] = 14 # Mutation
            for key in sd_param.keys():
                x_val = x_val * key
            for k, v in sd_param.items():
                x_val = x_val * k
                x_val = x_val * v
            
            if 1 in sd_param:
                x_val = x_val * 2
            else:
                x_val = x_val * 3

            x_val = x_val * sd_param.get(2, 0)
            x_val = x_val * sd_param.get(3, 4)
            x_val = len(sd_param) * x_val
            return x_val

        x_np = np.random.randn(4).astype("float32")

        sd1 = SimpleDict()
        sd1[2] = 5
        sd1[4] = 10

        sd2 = SimpleDict()
        sd2[2] = 5
        sd2[4] = 10
        self.assertTrue(sd1 == sd2)

        # Reference execution (with mutation)
        sd_for_ref = sd1.copy()
        ref_output = fn_ref(sd_for_ref, x_np)
        
        # TVM execution: We need to determine the final scalar multiplier based on the *mutated* state.
        sd_for_relay_state = sd2.copy()
        sd_for_relay_state[6] = 14 # Simulate the mutation from fn

        scalar_multiplier_for_relay = np.array(1.0, dtype="float32")
        temp_sd = sd_for_relay_state.copy()
        for value in temp_sd.values(): scalar_multiplier_for_relay = scalar_multiplier_for_relay * value
        for key in temp_sd.keys(): scalar_multiplier_for_relay = scalar_multiplier_for_relay * key
        for k, v in temp_sd.items(): scalar_multiplier_for_relay = scalar_multiplier_for_relay * k; scalar_multiplier_for_relay = scalar_multiplier_for_relay * v
        if 1 in temp_sd: scalar_multiplier_for_relay = scalar_multiplier_for_relay * 2
        else: scalar_multiplier_for_relay = scalar_multiplier_for_relay * 3
        scalar_multiplier_for_relay = scalar_multiplier_for_relay * temp_sd.get(2, 0)
        scalar_multiplier_for_relay = scalar_multiplier_for_relay * temp_sd.get(3, 4)
        scalar_multiplier_for_relay = len(temp_sd) * scalar_multiplier_for_relay
        final_scalar = scalar_multiplier_for_relay.item()

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(final_scalar, "float32"))
            return relay.Function([x], result)

        tvm_output = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)
        self.assertTrue(sd1 == sd2) # Checks that both original dicts are in the same state (mutated by fn_ref and conceptually by _get_relay_func build logic)


    def test_dict_subclass_setitem(self):
        # Original PyTorch fn:
        # class SetItemDict(dict):
        #     def __setitem__(self, key, value):
        #         super().__setitem__(key, value + 1)
        # def fn(x):
        #     sd = SetItemDict(x=5 * x)
        #     sd["y"] = 10
        #     return sd

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            
            sd_x_val = relay.op.add(relay.op.multiply(relay.const(5, "float32"), x), relay.const(1, "float32"))
            sd_y_val = relay.const(11, "float32") # 10 + 1
            
            return relay.Function([x], relay.expr.Tuple([sd_x_val, sd_y_val]))

        x_np = np.random.randn(4).astype("float32")

        # Reference execution
        class _LocalSetItemDict(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value + 1)

        sd_ref = _LocalSetItemDict(x=5 * x_np)
        sd_ref["y"] = 10
        
        ref_x = sd_ref["x"]
        ref_y = sd_ref["y"]

        # TVM execution
        tvm_outputs = _run_relay_graph(_get_relay_func(), {"x": x_np})
        tvm_x, tvm_y = tvm_outputs[0], tvm_outputs[1]

        self.assertEqual(ref_x, tvm_x)
        self.assertEqual(ref_y, tvm_y)

    def test_custom_iter_dict(self):
        # This tests iteration order and dict mutation triggering recompilation.
        # TVM won't trace Python iteration logic or detect recompilations.
        # We simulate the *final state* of `d` for the tensor computation.
        
        class _LocalReversedDict(dict):
            def __iter__(self):
                return reversed(list(self.keys()))

        d_initial = _LocalReversedDict({"foo": 1, "bar": 2})

        def fn_ref(x_np, d_param):
            d_param.sample = 1 # side effect
            d_param["baz"] = 4 # side effect
            return x_np * d_param["foo"] * d_param["bar"]

        x_np = np.random.randn(4).astype("float32")

        # Simulate initial call to get reference output and mutated dict state
        d_for_ref1 = d_initial.copy()
        ref_output1 = fn_ref(x_np, d_for_ref1)

        # For Relay graph, capture the final state of `d`
        scalar_foo = d_for_ref1["foo"]
        scalar_bar = d_for_ref1["bar"]

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(scalar_foo, "float32"))
            result = relay.op.multiply(result, relay.const(scalar_bar, "float32"))
            return relay.Function([x], result)

        tvm_output1 = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output1, tvm_output1)

        # The subsequent calls to fn and recompilation checks are Dynamo-specific.
        # We ensure the *same numerical output* would be produced if the graph was rebuilt.
        ref_output2 = fn_ref(x_np, d_for_ref1)
        tvm_output2 = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output2, tvm_output2)

    def test_custom_keys_iter_dict(self):
        # This tests custom `keys()` method. The final computation is commutative.
        # TVM only sees the final scalar values.
        
        class _LocalCustomKeysDict(dict):
            def keys(self):
                return ["bar", "foo"]

        d_initial = _LocalCustomKeysDict({"foo": 1, "bar": 2})

        def fn_ref(x_np, d_param):
            return x_np * d_param["foo"] * d_param["bar"]

        x_np = np.random.randn(4).astype("float32")
        ref_output = fn_ref(x_np, d_initial)

        scalar_foo = d_initial["foo"]
        scalar_bar = d_initial["bar"]

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(scalar_foo, "float32"))
            result = relay.op.multiply(result, relay.const(scalar_bar, "float32"))
            return relay.Function([x], result)

        tvm_output = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)

    def test_dict_guard_on_keys_order(self):
        # This test hinges on dict key order. We build different Relay graphs for different dict orders.
        d_initial = {2: 4, 3: 5}

        def fn_ref(x_np, d_param):
            x_val = x_np
            for key, value in d_param.items():
                x_val = x_val * key + value
            return x_val

        x_np = np.random.randn(4).astype("float32")

        ref_output1 = fn_ref(x_np, d_initial)

        def _get_relay_func_order1():
            x = relay.var("x", shape=(4,), dtype="float32")
            x_val = relay.op.add(relay.op.multiply(x, relay.const(2, "float32")), relay.const(4, "float32"))
            x_val = relay.op.add(relay.op.multiply(x_val, relay.const(3, "float32")), relay.const(5, "float32"))
            return relay.Function([x], x_val)
        
        tvm_output1 = _run_relay_graph(_get_relay_func_order1(), {"x": x_np})
        self.assertEqual(ref_output1, tvm_output1)

        d_mutated = {2: 4, 3: 5}
        d_mutated[2] = d_mutated.pop(2) # Order: 3 then 2

        ref_output2 = fn_ref(x_np, d_mutated)

        def _get_relay_func_order2():
            x = relay.var("x", shape=(4,), dtype="float32")
            x_val = relay.op.add(relay.op.multiply(x, relay.const(3, "float32")), relay.const(5, "float32"))
            x_val = relay.op.add(relay.op.multiply(x_val, relay.const(2, "float32")), relay.const(4, "float32"))
            return relay.Function([x], x_val)

        tvm_output2 = _run_relay_graph(_get_relay_func_order2(), {"x": x_np})
        self.assertEqual(ref_output2, tvm_output2)

    def test_dict_guard_on_keys_order2(self):
        # Similar to test_dict_guard_on_keys_order.
        d_initial = {2: 4, 3: 5}

        def fn_ref(x_np, d_param):
            x_val = x_np
            for key in d_param:
                value = d_param[key]
                x_val = x_val * key + value
            return x_val

        x_np = np.random.randn(4).astype("float32")

        ref_output1 = fn_ref(x_np, d_initial)

        def _get_relay_func_order1():
            x = relay.var("x", shape=(4,), dtype="float32")
            x_val = relay.op.add(relay.op.multiply(x, relay.const(2, "float32")), relay.const(4, "float32"))
            x_val = relay.op.add(relay.op.multiply(x_val, relay.const(3, "float32")), relay.const(5, "float32"))
            return relay.Function([x], x_val)
        
        tvm_output1 = _run_relay_graph(_get_relay_func_order1(), {"x": x_np})
        self.assertEqual(ref_output1, tvm_output1)

        d_mutated = {2: 4, 3: 5}
        d_mutated[2] = d_mutated.pop(2) # Order: 3 then 2

        ref_output2 = fn_ref(x_np, d_mutated)

        def _get_relay_func_order2():
            x = relay.var("x", shape=(4,), dtype="float32")
            x_val = relay.op.add(relay.op.multiply(x, relay.const(3, "float32")), relay.const(5, "float32"))
            x_val = relay.op.add(relay.op.multiply(x_val, relay.const(2, "float32")), relay.const(4, "float32"))
            return relay.Function([x], x_val)

        tvm_output2 = _run_relay_graph(_get_relay_func_order2(), {"x": x_np})
        self.assertEqual(ref_output2, tvm_output2)

    def test_ordered_dict_reordered_keys(self):
        d = OrderedDict()
        d[2] = 4
        d[3] = 5
        d.move_to_end(2) # Final order: 3, 2

        def fn_ref(x_np, d_param):
            y_val = relay.const(0.0, "float32") # Will be computed in Python then used
            for idx, (key, value) in enumerate(d_param.items()):
                if idx == 0:
                    y_val = y_val + np.sin(x_np * value)
                else:
                    y_val = y_val + np.cos(x_np * value)
            return y_val

        x_np = np.random.randn(4).astype("float32")
        ref_output = fn_ref(x_np, d)

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            y_val = relay.const(0.0, "float32")
            
            # First item: (3, 5)
            y_val = relay.op.add(y_val, relay.op.sin(relay.op.multiply(x, relay.const(5, "float32"))))
            
            # Second item: (2, 4)
            y_val = relay.op.add(y_val, relay.op.cos(relay.op.multiply(x, relay.const(4, "float32"))))
            
            return relay.Function([x], y_val)
        
        tvm_output = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)

    def test_ordered_dict_subclass_reordered_keys(self):
        class ODSubclass(OrderedDict):
            def keys(self):
                return super().keys()

        d = ODSubclass()
        d[2] = 4
        d[3] = 5
        d.move_to_end(2) # Final order: 3, 2

        def fn_ref(x_np, d_param):
            y_val = relay.const(0.0, "float32")
            for idx, (key, value) in enumerate(d_param.items()):
                if idx == 0:
                    y_val = y_val + np.sin(x_np * value)
                else:
                    y_val = y_val + np.cos(x_np * value)
            return y_val

        x_np = np.random.randn(4).astype("float32")
        ref_output = fn_ref(x_np, d)

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            y_val = relay.const(0.0, "float32")
            
            y_val = relay.op.add(y_val, relay.op.sin(relay.op.multiply(x, relay.const(5, "float32"))))
            y_val = relay.op.add(y_val, relay.op.cos(relay.op.multiply(x, relay.const(4, "float32"))))
            
            return relay.Function([x], y_val)
        
        tvm_output = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)

    def test_lazy_key_guarding(self):
        d_initial = {"a": 2, "b": 3, "c": 5}

        def fn_ref(x_np, d_param):
            return x_np * d_param["a"]

        x_np = np.random.randn(4).astype("float32")

        # Run 1: initial state
        ref_output1 = fn_ref(x_np, d_initial)

        def _get_relay_func_state1():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(d_initial["a"], "float32"))
            return relay.Function([x], result)

        tvm_output1 = _run_relay_graph(_get_relay_func_state1(), {"x": x_np})
        self.assertEqual(ref_output1, tvm_output1)

        # Mutate `d` (pop "c", add "d"). Key "a" is still 2.
        d_initial.pop("c")
        d_initial["d"] = 10
        
        # Run 2: functional output should be the same as 'a' didn't change.
        ref_output2 = fn_ref(x_np, d_initial)
        
        # Reuse the same Relay function as the logic for `d["a"]` is unchanged.
        tvm_output2 = _run_relay_graph(_get_relay_func_state1(), {"x": x_np})
        self.assertEqual(ref_output2, tvm_output2)


    def test_lazy_key_non_const_guarding(self):
        d_initial = {list: 2, dict: 3, OrderedDict: 5, namedtuple: 7}

        def fn_ref(x_np, d_param):
            return x_np * d_param[list]

        x_np = np.random.randn(4).astype("float32")

        # Run 1: initial state
        ref_output1 = fn_ref(x_np, d_initial)

        def _get_relay_func_state1():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(d_initial[list], "float32"))
            return relay.Function([x], result)

        tvm_output1 = _run_relay_graph(_get_relay_func_state1(), {"x": x_np})
        self.assertEqual(ref_output1, tvm_output1)

        # Mutate `d` (pop `dict`, add `defaultdict`). Key `list` is still 2.
        d_initial.pop(dict)
        d_initial[defaultdict] = 10
        
        # Run 2: functional output should be the same as `d[list]` remains constant.
        ref_output2 = fn_ref(x_np, d_initial)
        
        # Reuse same Relay function
        tvm_output2 = _run_relay_graph(_get_relay_func_state1(), {"x": x_np})
        self.assertEqual(ref_output2, tvm_output2)

    def test_dict_mutation_side_effect(self):
        args1_np = {"a": np.random.randn(10).astype("float32"), "b": np.random.randn(10).astype("float32")}
        
        def fn_ref(d_param):
            d_out = dict(d_param)
            val_a = d_out["a"]
            val_b = d_out.pop("b")
            d_out["c"] = val_a + val_b
            return d_out

        ref_result_dict = fn_ref(args1_np)
        
        # Build Relay function that computes the final values for a, b, c
        def _get_relay_func():
            a = relay.var("a", shape=(10,), dtype="float32")
            b = relay.var("b", shape=(10,), dtype="float32")
            
            val_a_relay = a
            val_b_relay = b
            c_val_relay = relay.op.add(val_a_relay, val_b_relay)
            
            # Since the dict itself is returned, we need to return its elements.
            # Only 'a' and 'c' are present in the output dict. 'b' is popped.
            return relay.Function([a, b], relay.expr.Tuple([val_a_relay, c_val_relay]))

        tvm_outputs = _run_relay_graph(_get_relay_func(), {"a": args1_np["a"], "b": args1_np["b"]})
        tvm_a_final, tvm_c_final = tvm_outputs[0], tvm_outputs[1]

        self.assertEqual(ref_result_dict["a"], tvm_a_final)
        self.assertEqual(ref_result_dict["c"], tvm_c_final)
        
    def test_dict_copy_alias(self):
        def run_ref(x_np, d0_param):
            d1_copy = d0_param.copy()
            d1_copy[0] = 1
            return x_np + 1, d1_copy

        x_np = np.zeros(1).astype("float32")
        d0_initial = {}

        ref_x_plus_1, ref_d1 = run_ref(x_np, d0_initial)

        def _get_relay_func():
            x = relay.var("x", shape=(1,), dtype="float32")
            
            x_plus_1 = relay.op.add(x, relay.const(1, "float32"))
            d1_val_0 = relay.const(1, "float32")
            return relay.Function([x], relay.expr.Tuple([x_plus_1, d1_val_0]))

        tvm_outputs = _run_relay_graph(_get_relay_func(), {"x": x_np})
        tvm_x_plus_1, tvm_d1_val = tvm_outputs[0], tvm_outputs[1]

        self.assertTrue(same(ref_x_plus_1, tvm_x_plus_1))
        self.assertEqual(ref_d1, {0:1})
        self.assertEqual(tvm_d1_val, 1)
        
        self.assertEqual(d0_initial, {})

    def test_dict_subclass_get_method(self):
        class _Localdotdict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        config_initial = _Localdotdict({"a": 1, "b": 2})

        def fn_ref(x_np):
            x3 = x_np * config_initial.get("a", 3)
            return x3

        x_np = np.random.randn(2).astype("float32")
        ref_output = fn_ref(x_np)

        def _get_relay_func():
            x = relay.var("x", shape=(2,), dtype="float32")
            result = relay.op.multiply(x, relay.const(config_initial.get("a", 3), "float32"))
            return relay.Function([x], result)
        
        tvm_output = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)

    def test_dict_order_keys(self):
        def fn_ref(d_param):
            c = np.array(0.0, dtype="float32")
            for v in d_param.values():
                c = c + v
            return c

        args1_d = {}
        args1_d["a"] = np.random.rand(10).astype("float32")
        args1_d["b"] = np.random.rand(10).astype("float32")

        ref_output1 = fn_ref(args1_d)

        def _get_relay_func_order1(a_val_relay, b_val_relay):
            result = relay.op.add(a_val_relay, b_val_relay)
            return relay.Function([a_val_relay, b_val_relay], result)
        
        a_relay_var = relay.var("a", shape=(10,), dtype="float32")
        b_relay_var = relay.var("b", shape=(10,), dtype="float32")
        tvm_output1 = _run_relay_graph(_get_relay_func_order1(a_relay_var, b_relay_var), {"a": args1_d["a"], "b": args1_d["b"]})
        self.assertEqual(ref_output1, tvm_output1)

        args2_d = {}
        args2_d["b"] = args1_d["b"]
        args2_d["a"] = args1_d["a"]

        ref_output2 = fn_ref(args2_d)
        
        # `CompileCounter` assertions are removed. This checks numerical output consistency.
        b_relay_var = relay.var("b", shape=(10,), dtype="float32")
        a_relay_var = relay.var("a", shape=(10,), dtype="float32")
        tvm_output2 = _run_relay_graph(_get_relay_func_order1(a_relay_var, b_relay_var), {"a": args2_d["a"], "b": args2_d["b"]}) # Same `_get_relay_func_order1` is fine since addition is commutative.
        self.assertEqual(ref_output2, tvm_output2)

    def test_dict_namedtuple(self):
        def fn_ref(d_param):
            if namedtuple in d_param:
                return d_param[3] * 2
            else:
                return d_param[3] * 3

        args1_d = {namedtuple: None, 3: np.random.randn(3).astype("float32")}
        ref_output1 = fn_ref(args1_d)

        def _get_relay_func_case1():
            val_3 = relay.var("val_3", shape=(3,), dtype="float32")
            result = relay.op.multiply(val_3, relay.const(2, "float32"))
            return relay.Function([val_3], result)
        
        tvm_output1 = _run_relay_graph(_get_relay_func_case1(), {"val_3": args1_d[3]})
        self.assertEqual(ref_output1, tvm_output1)

        args2_d = {2: None, 3: np.random.randn(3).astype("float32")}
        ref_output2 = fn_ref(args2_d)

        def _get_relay_func_case2():
            val_3 = relay.var("val_3", shape=(3,), dtype="float32")
            result = relay.op.multiply(val_3, relay.const(3, "float32"))
            return relay.Function([val_3], result)
        
        tvm_output2 = _run_relay_graph(_get_relay_func_case2(), {"val_3": args2_d[3]})
        self.assertEqual(ref_output2, tvm_output2)

    def test_dict_order_keys_tensors(self):
        # Tensors as keys are complex for non-PyTorch backends.
        # This test is about dict key ordering. We use simple Python objects for keys.
        class TensorKeyProxy:
            _id_counter = itertools.count()
            def __init__(self, name):
                self.name = name
                self._id = next(self._id_counter)
            
            def __hash__(self):
                return self._id
            
            def __eq__(self, other):
                return isinstance(other, TensorKeyProxy) and self._id == other._id
            
            def __repr__(self):
                return f"TensorKeyProxy({self.name})"

        x_val_for_key = TensorKeyProxy("x_key")
        y_val_for_dict = np.random.randn(10).astype("float32")
        z_val_for_dict = np.random.randn(10).astype("float32")

        args1_d = {}
        args1_d[x_val_for_key] = y_val_for_dict
        args1_d[3] = z_val_for_dict

        def fn_ref(d_param, key_param):
            return d_param[key_param] + 3

        ref_output1 = fn_ref(args1_d, x_val_for_key)

        def _get_relay_func():
            y = relay.var("y", shape=(10,), dtype="float32")
            result = relay.op.add(y, relay.const(3, "float32"))
            return relay.Function([y], result)
        
        tvm_output1 = _run_relay_graph(_get_relay_func(), {"y": y_val_for_dict})
        self.assertEqual(ref_output1, tvm_output1)

        tvm_output_again = _run_relay_graph(_get_relay_func(), {"y": y_val_for_dict})
        self.assertEqual(ref_output1, tvm_output_again)

        args2_d = {}
        args2_d[3] = z_val_for_dict
        args2_d[x_val_for_key] = y_val_for_dict

        ref_output2 = fn_ref(args2_d, x_val_for_key)
        tvm_output2 = _run_relay_graph(_get_relay_func(), {"y": y_val_for_dict})
        self.assertEqual(ref_output2, tvm_output2)

    def test_dict_order_keys_modules(self):
        # Module objects as keys are hashed by identity.
        class ModuleKeyProxy:
            _id_counter = itertools.count()
            def __init__(self, name):
                self.name = name
                self._id = next(self._id_counter)
            
            def __hash__(self):
                return self._id
            
            def __eq__(self, other):
                return isinstance(other, ModuleKeyProxy) and self._id == other._id
            
            def __repr__(self):
                return f"ModuleKeyProxy({self.name})"
            
            def __call__(self, input_tensor): # Simulates Module's forward pass
                return input_tensor + 0.5 # A dummy operation

        x_key_proxy = ModuleKeyProxy("x_linear")
        y_val_proxy = ModuleKeyProxy("y_linear")
        z_val_proxy = ModuleKeyProxy("z_linear")

        args1_d = {}
        args1_d[x_key_proxy] = y_val_proxy
        args1_d[3] = z_val_proxy

        input_tensor_np = np.ones((2, 2)).astype("float32")

        def fn_ref(d_param, key_param):
            return d_param[key_param](input_tensor_np)

        ref_output1 = fn_ref(args1_d, x_key_proxy)

        def _get_relay_func():
            input_relay = relay.var("input_tensor", shape=(2, 2), dtype="float32")
            result = relay.op.add(input_relay, relay.const(0.5, "float32"))
            return relay.Function([input_relay], result)
        
        tvm_output1 = _run_relay_graph(_get_relay_func(), {"input_tensor": input_tensor_np})
        self.assertEqual(ref_output1, tvm_output1)

        tvm_output_again = _run_relay_graph(_get_relay_func(), {"input_tensor": input_tensor_np})
        self.assertEqual(ref_output1, tvm_output_again)

        args2_d = {}
        args2_d[3] = z_val_proxy
        args2_d[x_key_proxy] = y_val_proxy

        ref_output2 = fn_ref(args2_d, x_key_proxy)
        tvm_output2 = _run_relay_graph(_get_relay_func(), {"input_tensor": input_tensor_np})
        self.assertEqual(ref_output2, tvm_output2)

    def test_contains_dunder_dict(self):
        class _LocalUserDefined:
            def __init__(self) -> None:
                self.a = 3
                self.b = 5

            def run(self, x_val):
                if "a" in self.__dict__:
                    x_val = x_val * self.a
                if "b" in self.__dict__:
                    x_val = x_val * self.b
                self.c = 7
                if "c" in self.__dict__:
                    x_val = x_val * self.c
                return x_val * self.__dict__.get("a") * self.__dict__.get("z", 2)

        obj_ref = _LocalUserDefined()
        x_np = np.random.randn(4).astype("float32")
        ref_output = obj_ref.run(x_np)

        # Simulate `obj_ref.run` to get final constants
        dummy_x_for_constants = np.array(1.0, dtype="float32")
        obj_for_constants = _LocalUserDefined()
        _ = obj_for_constants.run(dummy_x_for_constants)

        final_multiplier = 1.0
        if "a" in obj_for_constants.__dict__: final_multiplier *= obj_for_constants.a
        if "b" in obj_for_constants.__dict__: final_multiplier *= obj_for_constants.b
        if "c" in obj_for_constants.__dict__: final_multiplier *= obj_for_constants.c
        
        final_multiplier *= obj_for_constants.__dict__.get("a")
        final_multiplier *= obj_for_constants.__dict__.get("z", 2)

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            result = relay.op.multiply(x, relay.const(final_multiplier, "float32"))
            return relay.Function([x], result)
        
        tvm_output = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)

    def test_contains_module_dunder_dict(self):
        class _LocalMyModule:
            def __init__(self) -> None:
                self.foo = 1
                self.bar = 2
                self.baz = 3

            def forward(self, x_val):
                if "foo" in self.__dict__:
                    return x_val * self.bar
                return x_val * self.baz

        mod_ref = _LocalMyModule()
        x_np = np.random.randn(10).astype("float32")
        ref_output = mod_ref.forward(x_np)

        def _get_relay_func():
            x = relay.var("x", shape=(10,), dtype="float32")
            result = relay.op.multiply(x, relay.const(mod_ref.bar, "float32"))
            return relay.Function([x], result)
        
        tvm_output = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)

    def test_update_dunder_dict(self):
        class _LocalUserDefined:
            def run(self, x_val):
                self.__dict__["a"] = 10
                return x_val * self.a + self.__dict__["a"]

        obj1_ref = _LocalUserDefined()
        x_np = np.random.randn(4).astype("float32")
        ref_output = obj1_ref.run(x_np)

        def _get_relay_func():
            x = relay.var("x", shape=(4,), dtype="float32")
            a_val = relay.const(10, "float32")
            result = relay.op.add(relay.op.multiply(x, a_val), a_val)
            return relay.Function([x], result)
        
        tvm_output = _run_relay_graph(_get_relay_func(), {"x": x_np})
        self.assertEqual(ref_output, tvm_output)
        self.assertEqual(obj1_ref.a, 10) # Python-level check for side-effect
