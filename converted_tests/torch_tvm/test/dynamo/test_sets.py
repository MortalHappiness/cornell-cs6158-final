import math
import unittest
from collections.abc import Iterable

import pytest
import numpy as np

# PyTorch internal classes are replaced with dummy classes for TVM conversion
# since they don't have direct functional equivalents in TVM and are used as
# opaque objects in sets for guard testing.
class SetSubclass(set):
    pass


class FrozenstSubclass(frozenset):
    pass


# Replaced torch._dynamo.test_case.TestCase with unittest.TestCase
class _BaseSetTests(unittest.TestCase):
    def setUp(self):
        # torch._dynamo.config.enable_trace_unittest is a PyTorch-specific configuration
        # that has no direct equivalent in TVM. Removed.
        super().setUp()

    def tearDown(self):
        # torch._dynamo.config.enable_trace_unittest is a PyTorch-specific configuration
        # that has no direct equivalent in TVM. Removed.
        return super().tearDown()

    def assertEqual(self, a, b):
        # For numpy arrays, use np.array_equal or np.allclose.
        # For scalar results, direct equality is fine.
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return self.assertTrue(np.array_equal(a, b), f"{a} != {b}")
        return self.assertTrue(a == b, f"{a} != {b}")

    def assertNotEqual(self, a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return self.assertFalse(np.array_equal(a, b), f"{a} == {b}")
        return self.assertTrue(a != b, f"{a} == {b}")


class CustomSetTests(_BaseSetTests):
    class CustomSet(set):
        def add(self, item):
            return super().add(item + 1)

        def contains(self, item):
            return True

    thetype = CustomSet

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_custom_add(self):
        s = self.thetype([1, 2])
        s.add(3)
        self.assertTrue(s == {1, 2, 4})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_custom_contains(self):
        s = self.thetype([1, 2])
        self.assertTrue(s.contains(3))


# Replaced torch._dynamo.test_case.TestCase with unittest.TestCase
class MiscTests(unittest.TestCase):
    def test_isdisjoint_with_generator(self):
        n = 0

        def gen():
            nonlocal n
            n += 1
            yield 1
            n += 2
            yield 2
            n += 3
            yield 3

        # torch.compile is a PyTorch-specific API, removed.
        def fn(x):
            nonlocal n
            s = {2, 4, 5}
            s.isdisjoint(gen())
            # Replaced torch.sin/torch.cos with numpy equivalents.
            # Assuming x is a numpy array for direct computation outside a Relay graph.
            if n == 3:
                return np.sin(x)
            return np.cos(x)

        # Replaced torch.randn with numpy equivalent.
        x = np.random.randn(1).astype(np.float32)
        y = fn(x)
        self.assertEqual(y, np.sin(x))


# LoggingTestCase is a PyTorch-specific class, replaced with unittest.TestCase.
class TestSetGuards(unittest.TestCase):
    # Dummy classes for PyTorch internal objects used in sets for guard testing.
    # These mimic the behavior of being hashable and comparable.
    class DummyTorchInternal:
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return f"<DummyTorchInternal {self.name}>"
        def __eq__(self, other):
            return isinstance(other, TestSetGuards.DummyTorchInternal) and self.name == other.name
        def __hash__(self):
            return hash(self.name)

    _set_grad_enabled = DummyTorchInternal("_set_grad_enabled")
    _exit_autocast = DummyTorchInternal("_exit_autocast")
    _set_fwd_grad_enabled = DummyTorchInternal("_set_fwd_grad_enabled")
    _enter_autocast = DummyTorchInternal("_enter_autocast")
    _set_autograd_fallback_mode = DummyTorchInternal("_set_autograd_fallback_mode")

    def test_set_with_function(self):
        s = {
            self._set_grad_enabled,
            "hello",
            self._exit_autocast,
        }
        # CompileCounter is a PyTorch-specific testing utility, removed.

        # torch.compile is a PyTorch-specific API, removed.
        def fn(x, s):
            if self._exit_autocast in s:
                return np.sin(x)
            return np.cos(x)

        # Replaced torch.randn with numpy equivalent.
        x = np.random.randn(2).astype(np.float32)
        y = fn(x, s)
        self.assertEqual(y, np.sin(x))
        # assertEqual(cnts.frame_count, X) is a PyTorch-specific assertion, removed.

        s.remove(self._exit_autocast)
        s.add(self._set_fwd_grad_enabled)
        y = fn(x, s)
        self.assertEqual(y, np.cos(x))
        # assertEqual(cnts.frame_count, X) is a PyTorch-specific assertion, removed.

    # make_logging_test is a PyTorch-specific decorator, removed.
    def test_in_guard(self, records=None): # records parameter removed as LoggingTestCase is gone
        s = {
            "Dynamo",
            "Inductor",
            "PyTorch",
            np.sin, # Replaced torch.sin with numpy.sin
        }
        # CompileCounter is a PyTorch-specific testing utility, removed.

        # torch.compile is a PyTorch-specific API, removed.
        def fn(x, s):
            if "PyTorch" in s:
                return np.sin(x)
            return np.cos(x)

        # Replaced torch.randn with numpy equivalent.
        x = np.random.randn(2).astype(np.float32)
        y = fn(x, s)
        self.assertEqual(y, np.sin(x))
        # assertEqual(cnts.frame_count, X) is a PyTorch-specific assertion, removed.

        s.remove("PyTorch")
        s.add("Cuda")
        y = fn(x, s)
        self.assertEqual(y, np.cos(x))
        # assertEqual(cnts.frame_count, X) is a PyTorch-specific assertion, removed.
        # Logging-specific assertions removed.

    def test_set_with_tensors(self):
        # This test originally demonstrated a PyTorch Dynamo graph break when sets contain tensors.
        # Without torch.compile, the Python code executes normally.
        # For TVM, directly embedding numpy arrays in a Python set and expecting Relay graph
        # tracing to handle iteration over them as graph ops would also lead to issues.
        # Here, it's converted to plain Python with numpy arrays, which runs successfully.
        # The `Unsupported` exception handling is removed.

        # Replaced torch.ones, torch.tensor, torch.zeros with numpy equivalents.
        s = {
            np.ones(1).astype(np.float32),
            np.array([1.0]).astype(np.float32),
            np.zeros(1).astype(np.float32),
        }
        # CompileCounter removed.

        # torch.compile removed.
        def fn(x, s):
            z = np.zeros(1).astype(np.float32)
            for i in s:
                z += i
            return x + z

        # Replaced torch.tensor with numpy equivalent.
        x = np.array([1.0]).astype(np.float32)

        # Removed assertExpectedInlineMunged(Unsupported, ...)
        result = fn(x, s)
        expected_z = np.ones(1).astype(np.float32) + np.array([1.0]).astype(np.float32) + np.zeros(1).astype(np.float32)
        expected_result = x + expected_z
        np.testing.assert_allclose(result, expected_result)
        # TODO: This test originally captured a Dynamo `Unsupported` error for sets containing `torch.Tensor`s.
        # Without `torch.compile`, the Python code executes normally with `numpy` arrays.
        # A true TVM equivalent would likely involve trying to trace a Relay graph with such a Python set,
        # which would also lead to graph break or unsupported error. For now, it runs as pure Python.

    def test_set_multiple_types(self):
        s = {
            "PyTorch",
            3.3,
            1j,
            math.nan,
        }
        # CompileCounter removed.

        # torch.compile removed.
        def fn(x, s):
            if "PyTorch" in s:
                return np.sin(x)
            return np.cos(x)

        # Replaced torch.tensor with numpy equivalent.
        x = np.array(1.0).astype(np.float32)
        y = fn(x, s)
        self.assertEqual(y, np.sin(x))
        # assertEqual(cnts.frame_count, X) removed.

        s.remove("PyTorch")
        y = fn(x, s)
        self.assertEqual(y, np.cos(x))
        # assertEqual(cnts.frame_count, X) removed.

    def test_set_recompile_on_key_pop(self):
        s = {
            self._set_grad_enabled,
            self._enter_autocast,
            self._exit_autocast,
        }

        # CompileCounter removed.

        def fn(x, s):
            if self._exit_autocast in s:
                return np.sin(x)
            return np.cos(x)

        # Replaced torch.randn with numpy equivalent.
        x = np.random.randn(4).astype(np.float32)
        opt_fn = fn # Replaced torch.compile with direct function call
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # assertEqual(cnts.frame_count, X) removed.

        s.remove(self._exit_autocast)

        res = opt_fn(x, s)
        # assertEqual(cnts.frame_count, X) removed.
        self.assertEqual(res, fn(x, s))

    def test_set_recompile_on_key_change(self):
        s = {
            self._set_grad_enabled,
            self._enter_autocast,
            self._exit_autocast,
        }

        # CompileCounter removed.

        def fn(x, s):
            if self._exit_autocast in s:
                return np.sin(x)
            return np.cos(x)

        # Replaced torch.randn with numpy equivalent.
        x = np.random.randn(4).astype(np.float32)
        opt_fn = fn # Replaced torch.compile with direct function call
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # assertEqual(cnts.frame_count, X) removed.

        s.remove(self._exit_autocast)
        s.add(self._set_autograd_fallback_mode)

        res = opt_fn(x, s)
        # assertEqual(cnts.frame_count, X) removed.
        self.assertEqual(res, fn(x, s))

    @unittest.skip("random failures on Python 3.9")
    def test_set_guard_on_keys_change(self):
        s = {
            self._set_grad_enabled,
            self._enter_autocast,
            self._exit_autocast,
        }

        # CompileCounter removed.

        def fn(x, s):
            for e in s:
                x = x * len(str(e))
            return x

        opt_fn = fn # Replaced torch.compile with direct function call
        opt_fn(np.random.randn(4).astype(np.float32), s) # Replaced torch.randn with numpy equivalent.
        opt_fn(np.random.randn(4).astype(np.float32), s) # Replaced torch.randn with numpy equivalent.
        # assertEqual(cnts.frame_count, X) removed.

        s.remove(self._exit_autocast)
        s.add(self._exit_autocast)

        x = np.random.randn(4).astype(np.float32) # Replaced torch.randn with numpy equivalent.
        res = opt_fn(x, s)
        # assertEqual(cnts.frame_count, X) removed.
        self.assertEqual(res, fn(x, s))


class _FrozensetBase:
    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_binop_sub(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p - p, self.thetype())
        self.assertEqual(p - q, self.thetype("ac"))
        self.assertEqual(q - p, self.thetype("ef"))
        # Using pytest.raises for TypeError, as self.assertRaises is for unittest.TestCase
        with pytest.raises(TypeError):
            p - 1
        self.assertEqual(self.thetype.__sub__(p, q), set("ac"))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_binop_or(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p | p, self.thetype("abc"))
        self.assertEqual(p | q, self.thetype("abcef"))
        self.assertEqual(self.thetype.__or__(p, q), set("abcef"))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_binop_and(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p & p, self.thetype("abc"))
        self.assertEqual(p & q, self.thetype("b"))
        self.assertEqual(self.thetype.__and__(p, q), set("b"))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_binop_xor(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p ^ p, self.thetype())
        self.assertEqual(p ^ q, self.thetype("acef"))
        self.assertEqual(self.thetype.__xor__(p, q), set("acef"))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_cmp_eq(self):
        p = self.thetype("abc")
        self.assertEqual(p, p)
        for C in set, frozenset, SetSubclass:
            self.assertEqual(p, C("abc"))
            self.assertEqual(p, C(p))
        self.assertTrue(self.thetype.__eq__(p, p))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_cmp_ne(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertNotEqual(p, q)
        self.assertNotEqual(q, p)
        for C in set, frozenset, SetSubclass, dict.fromkeys, str, list, tuple:
            self.assertNotEqual(p, C("abe"))
        self.assertNotEqual(p, 1)
        self.assertTrue(self.thetype.__ne__(p, q))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_cmp_less_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p < p)
        self.assertFalse(p < q)
        self.assertTrue(r < p)
        self.assertFalse(r < q)
        self.assertFalse(self.thetype.__lt__(p, p))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_cmp_greater_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p > p)
        self.assertFalse(p > q)
        self.assertTrue(p > r)
        self.assertFalse(q > r)
        self.assertFalse(self.thetype.__gt__(p, p))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_cmp_less_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p <= p)
        self.assertFalse(p <= q)
        self.assertTrue(r <= p)
        self.assertFalse(r <= q)
        self.assertTrue(self.thetype.__le__(p, p))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_cmp_greater_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p >= p)
        self.assertFalse(p >= q)
        self.assertTrue(p >= r)
        self.assertFalse(q >= r)
        self.assertTrue(self.thetype.__ge__(p, p))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_copy(self):
        p = self.thetype("abc")
        q = p.copy()
        self.assertEqual(p, q)
        with pytest.raises(TypeError):
            p.copy(1)
        self.assertEqual(self.thetype.copy(p), p)

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_issubset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(q.issubset(p))
        self.assertFalse(r.issubset(p))
        with pytest.raises(TypeError):
            p.issubset()
        with pytest.raises(TypeError):
            p.issubset(1)
        with pytest.raises(TypeError):
            p.issubset([[]])
        self.assertTrue(self.thetype.issubset(q, p))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_issuperset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(p.issuperset(q))
        self.assertFalse(p.issuperset(r))
        with pytest.raises(TypeError):
            p.issuperset()
        with pytest.raises(TypeError):
            p.issuperset(1)
        with pytest.raises(TypeError):
            p.issuperset([[]])
        self.assertTrue(self.thetype.issuperset(p, q))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_constructor_iterable(self):
        p = self.thetype("abc")
        self.assertIsInstance(p, self.thetype)
        self.assertIsInstance(p, Iterable)

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_equality(self):
        a = self.thetype("abc")
        for typ in (self.thetype, set, frozenset):
            self.assertEqual(a, typ(a))
            self.assertTrue(a == typ(a))
            self.assertTrue(a.__eq__(typ(a)))
            self.assertTrue(self.thetype.__eq__(a, typ(a)))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_in_frozenset(self):
        item_content = "abc"
        item = self.thetype(item_content)
        container = self.thetype([frozenset(item_content)]) # noqa: C405
        self.assertIn(item, container)

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_contains(self):
        s = self.thetype(["a", "b", "c"])
        self.assertIn("a", s)
        self.assertNotIn("d", s)
        self.assertTrue(s.__contains__("a"))
        self.assertTrue(self.thetype.__contains__(s, "b"))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_isdisjoint(self):
        x = self.thetype({"apple", "banana", "cherry"})
        y = self.thetype({"google", "microsoft", "apple"})
        z = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertFalse(x.isdisjoint(y))
        self.assertTrue(x.isdisjoint(z))
        with pytest.raises(TypeError):
            x.isdisjoint()
        with pytest.raises(TypeError):
            x.isdisjoint(1)
        with pytest.raises(TypeError):
            x.isdisjoint([[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertFalse(self.thetype.isdisjoint(p, q))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_intersection(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        intersection_set = set1.intersection(set2, set3)
        self.assertEqual(intersection_set, {"apple"})
        with pytest.raises(TypeError):
            set1.intersection(1)
        with pytest.raises(TypeError):
            set1.intersection([[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.intersection(p, q), {"b"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_union(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        union_set = p.union(q, r)
        self.assertEqual(union_set, {"a", "b", "c", "e", "f"})
        with pytest.raises(TypeError):
            p.union(1)
        with pytest.raises(TypeError):
            p.union([[]])
        s = self.thetype.union(q, r)
        self.assertEqual(s, {"b", "c", "e", "f"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_difference(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        difference_set = set1.difference(set2, set3)
        self.assertEqual(difference_set, {"banana", "cherry"})
        with pytest.raises(TypeError):
            set1.difference(1)
        with pytest.raises(TypeError):
            set1.difference([[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.difference(p, q), {"a", "c"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_symmetric_difference(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        # Original PyTorch test had a bug here, calling .difference() instead of .symmetric_difference()
        # Corrected to symmetric_difference.
        symmetric_diff_set = set1.symmetric_difference(set2)
        self.assertEqual(symmetric_diff_set, {"banana", "cherry", "google", "microsoft"})
        with pytest.raises(TypeError):
            set1.symmetric_difference()
        with pytest.raises(TypeError):
            set1.symmetric_difference(1)
        with pytest.raises(TypeError):
            set1.symmetric_difference([[]])
        p, q = map(self.thetype, ["abc", "bef"])
        symmetric_diff_set = self.thetype.symmetric_difference(p, q)
        self.assertEqual(symmetric_diff_set, {"a", "c", "e", "f"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_to_frozenset(self):
        input_set = self.thetype({"apple", "banana", "cherry"})
        set1 = frozenset(input_set)
        self.assertIsInstance(set1, frozenset)
        self.assertEqual(len(set1), 3)

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_to_set(self):
        input_set = self.thetype({"apple", "banana", "cherry"})
        set1 = set(input_set)
        self.assertIsInstance(set1, set)
        self.assertEqual(len(set1), 3)


class _SetBase(_FrozensetBase):
    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_add(self):
        p = self.thetype("abc")
        p.add("d")
        self.assertEqual(p, {"a", "b", "c", "d"})
        p.add("a")
        self.assertEqual(p, {"a", "b", "c", "d"})
        with pytest.raises(TypeError):
            p.add(["ab"])
        with pytest.raises(TypeError):
            p.add()
        set.add(p, "e")
        self.assertEqual(p, {"a", "b", "c", "d", "e"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_clear(self):
        p = self.thetype("abc")
        p.clear()
        self.assertEqual(p, set())
        p = self.thetype("abc")
        self.thetype.clear(p)
        self.assertEqual(len(p), 0)

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_remove(self):
        p = self.thetype("abc")
        self.assertEqual(p.remove("a"), None)
        self.assertEqual(p, {"b", "c"})
        with pytest.raises(KeyError):
            p.remove("a")
        p = self.thetype("abc")
        self.thetype.remove(p, "b")
        self.assertEqual(p, self.thetype({"a", "c"}))

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_intersection_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        self.assertIsNone(set1.intersection_update(set2, set3))
        self.assertEqual(set1, {"apple"})
        with pytest.raises(TypeError):
            set1.intersection_update([[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.intersection_update(p, q)
        self.assertEqual(p, {"b"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_difference_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertIsNone(set1.difference_update(set2, set3))
        self.assertEqual(set1, {"banana", "cherry"})
        with pytest.raises(TypeError):
            set1.difference_update([[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.difference_update(p, q)
        self.assertEqual(p, {"a", "c"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_symmetric_difference_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        self.assertIsNone(set1.symmetric_difference_update(set2))
        self.assertEqual(set1, {"banana", "cherry", "google", "microsoft"})
        with pytest.raises(TypeError):
            set1.symmetric_difference_update()
        with pytest.raises(TypeError):
            set1.symmetric_difference_update(1)
        with pytest.raises(TypeError):
            set1.symmetric_difference_update([[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.symmetric_difference_update(p, q)
        self.assertEqual(p, {"a", "c", "e", "f"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_pop(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        e = set1.pop()
        self.assertNotIn(e, set1)
        s = self.thetype()
        with pytest.raises(KeyError):
            s.pop()
        p = self.thetype("a")
        self.assertEqual(self.thetype.pop(p), "a")

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_update(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        p.update(q, r)
        self.assertEqual(p, {"a", "b", "c", "e", "f"})
        with pytest.raises(TypeError):
            p.update([[]])
        self.thetype.update(q, r)
        self.assertEqual(q, {"b", "c", "e", "f"})

    # make_dynamo_test is a PyTorch-specific decorator, removed.
    def test_discard(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set1.discard("banana")
        set2.discard("cherry")
        self.assertEqual(set1, {"apple", "cherry"})
        self.assertEqual(set2, {"google", "microsoft", "apple"})
        p = self.thetype("abc")
        self.thetype.discard(p, "a")
        self.assertEqual(p, {"b", "c"})


class FrozensetTests(_FrozensetBase, _BaseSetTests):
    thetype = frozenset


class SetTests(_SetBase, _BaseSetTests):
    thetype = set

    def test_in_frozenset(self):
        super().test_in_frozenset()


class UserDefinedSetTests(_SetBase, _BaseSetTests):
    class CustomSet(set):
        pass

    thetype = CustomSet

    def test_in_frozenset(self):
        super().test_in_frozenset()

    def test_equality(self):
        super().test_equality()


class UserDefinedFrozensetTests(_FrozensetBase, _BaseSetTests):
    class CustomFrozenset(frozenset):
        pass

    thetype = CustomFrozenset

    def test_in_frozenset(self):
        super().test_in_frozenset()


if __name__ == "__main__":
    # torch._dynamo.test_case.run_tests is a PyTorch-specific test runner.
    # Replaced with standard unittest.main()
    unittest.main()
