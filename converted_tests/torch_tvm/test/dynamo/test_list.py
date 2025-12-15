import unittest
# No torch imports needed, as these tests are for Python built-in list/tuple behavior.

# lst will be a global variable, as in the original test
lst = []


class TupleTests(unittest.TestCase):
    # Tuple methods
    # + count
    # + index
    # BinOps:
    # +, <, >, <=, >=, ==, !=
    # Dunder methods:
    # + __getitem__
    # + __contains__
    # + __delitem__ # This is not directly applicable to tuples, but in the original test it's just listed.

    thetype = tuple

    # Removed setUp and tearDown related to torch._dynamo.config
    # assertEqual and assertNotEqual are part of unittest.TestCase, no need to override unless behavior is custom.
    # The original overrides simply wrap self.assertTrue(a == b) or self.assertTrue(a != b), which is standard.
    # So, we rely on unittest.TestCase's methods.

    def test_count(self):
        p = self.thetype("abcab")
        self.assertEqual(p.count("a"), 2)
        self.assertEqual(p.count("ab"), 0)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.count)
        self.assertRaises(TypeError, p.count, 2, 3)

    def test_index(self):
        p = self.thetype("abc")
        self.assertEqual(p.index("a"), 0)
        self.assertRaises(ValueError, p.index, "e")

        # Wrong number of arguments
        self.assertRaises(TypeError, p.index)

    def test_binop_imul(self):
        p = self.thetype([1, 2, 3])
        r = p.__mul__(2)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype([1, 2, 3, 1, 2, 3]))
        self.assertEqual(p, self.thetype([1, 2, 3]))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__mul__)

        # can only multiply list by an integer
        self.assertRaises(TypeError, p.__mul__, 2.2)

    def test_binop_add(self):
        p, q = map(self.thetype, ["abc", "bcd"])
        self.assertIsInstance(p + q, self.thetype)
        self.assertEqual(p + q, self.thetype("abcbcd"))
        self.assertEqual(p.__add__(q), self.thetype("abcbcd"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__add__)

        # can only concatenate items of the same type
        self.assertRaises(TypeError, p.__add__, dict.fromkeys(q))

    def test_cmp_eq(self):
        p, q, r = map(self.thetype, ["ab", "abc", "ab"])
        self.assertTrue(p == p)
        self.assertTrue(p == r)
        self.assertEqual(p, p)
        self.assertEqual(p, r)
        self.assertNotEqual(p, q) # Use standard unittest.TestCase method
        self.assertTrue(p.__eq__(r))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__eq__)

    def test_cmp_ne(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p != q)
        self.assertNotEqual(p, q) # Use standard unittest.TestCase method
        self.assertTrue(p.__ne__(q))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__ne__)

    def test_cmp_less_than(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p < q)
        self.assertTrue(p.__lt__(q))
        self.assertFalse(q < p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__lt__)

    def test_cmp_greater_than(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(q > p)
        self.assertTrue(q.__gt__(p))
        self.assertFalse(p > q)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__gt__)

    def test_cmp_less_than_or_equal(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p <= q)
        self.assertTrue(p.__le__(q))
        self.assertFalse(q <= p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__le__)

    def test_cmp_greater_than_or_equal(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(q >= p)
        self.assertTrue(q.__ge__(p))
        self.assertFalse(p >= q)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__ge__)

    def test___getitem__(self):
        p = self.thetype("abc")
        self.assertEqual(p.__getitem__(2), "c")
        self.assertRaises(IndexError, p.__getitem__, 10)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__getitem__)
        self.assertRaises(TypeError, p.__getitem__, 1, 2)

    def test___contains__(self):
        p = self.thetype("abc")
        self.assertTrue(p.__contains__("a"))
        self.assertIsInstance(p.__contains__("c"), bool)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__contains__)
        self.assertRaises(TypeError, p.__contains__, 1, 2)


class ListTests(TupleTests):
    thetype = list

    def test_append(self):
        p = self.thetype("abc")
        self.assertIsNone(p.append("d"))
        self.assertEqual(p, ["a", "b", "c", "d"])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.append)
        self.assertRaises(TypeError, p.append, 2, 3)

    def test_copy(self):
        p = self.thetype("abc")
        self.assertEqual(p.copy(), p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.copy, 1)

    def test_clear(self):
        p = self.thetype("abc")
        self.assertIsNone(p.clear())
        self.assertEqual(p, [])
        self.assertEqual(len(p), 0)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.clear, 1)

    def test_extend(self):
        p, q = map(self.thetype, ["ab", "cd"])
        self.assertIsNone(p.extend(q))
        self.assertEqual(p, self.thetype("abcd"))

        # extend needs an iterable
        self.assertRaises(TypeError, p.extend, 1)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.extend)
        self.assertRaises(TypeError, p.extend, 2, 3)

    def test_insert(self):
        p = self.thetype("abc")
        self.assertIsNone(p.insert(1, "ef"))
        self.assertEqual(p, ["a", "ef", "b", "c"])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.insert)
        self.assertRaises(TypeError, p.insert, 1)
        self.assertRaises(TypeError, p.insert, 1, 2, 3)

    def test_pop(self):
        p = self.thetype("abcd")
        self.assertEqual(p.pop(), "d")
        self.assertEqual(p.pop(1), "b")
        self.assertRaises(IndexError, p.pop, 10)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.pop, 2, 3)

    def test_remove(self):
        p = self.thetype("abad")
        self.assertIsNone(p.remove("a"))
        self.assertEqual(p, ["b", "a", "d"])
        self.assertRaises(ValueError, p.remove, "x")

        # Wrong number of arguments
        self.assertRaises(TypeError, p.remove)
        self.assertRaises(TypeError, p.remove, 2, 3)

    def test_reverse(self):
        p = self.thetype("abcd")
        self.assertIsNone(p.reverse())
        self.assertEqual(p, self.thetype("dcba"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.reverse, 1)

    def test_sort(self):
        p = self.thetype("dbca")
        self.assertIsNone(p.sort())
        self.assertEqual(p, self.thetype("abcd"))

    def test_binop_imul(self):
        p = self.thetype([1, 2, 3])
        r = p.__imul__(2)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype([1, 2, 3, 1, 2, 3]))
        self.assertEqual(p, self.thetype([1, 2, 3, 1, 2, 3]))

        p = self.thetype("ab")
        p *= 2
        self.assertEqual(p, self.thetype("abab"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__imul__)

        # can only multiply list by an integer
        self.assertRaises(TypeError, p.__imul__, 2.2)

    def test_binop_imul_global_list(self):
        global lst
        lst = self.thetype(["a", "b"])

        # @torch.compile(backend="eager", fullgraph=True) # REMOVED
        def fn_imul_global_list():
            global lst
            lst *= 2
            lst.__imul__(3)
            # Original test returned x.sin(), which is now irrelevant as torch.compile is removed.
            return None

        # x = torch.tensor(1.0) # REMOVED
        # self.assertEqual(fn(x), x.sin()) # MODIFIED
        fn_imul_global_list() # Call the function directly
        self.assertEqual(lst, ["a", "b"] * 6)

    def test_binop_iadd(self):
        p, q = map(self.thetype, ["abc", "bcd"])
        r = p.__iadd__(q)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype("abcbcd"))
        self.assertEqual(p, self.thetype("abcbcd"))

        p = self.thetype("ab")
        p += "cd"
        self.assertEqual(p, self.thetype("abcd"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__iadd__)

        # can only concatenate items of the same type
        self.assertRaises(TypeError, p.__add__, dict.fromkeys(q))

    def test_binop_iadd_global_list(self):
        global lst
        lst = self.thetype([])

        # @torch.compile(backend="eager", fullgraph=True) # REMOVED
        def fn_iadd_global_list():
            global lst
            lst += ["a"]
            lst.__iadd__(["b"])
            # Original test returned x.sin(), which is now irrelevant.
            return None

        # x = torch.tensor(1.0) # REMOVED
        # self.assertEqual(fn(x), x.sin()) # MODIFIED
        fn_iadd_global_list() # Call the function directly
        self.assertEqual(lst, ["a", "b"])

    def test_binop_delitem_global_list(self):
        global lst
        lst = self.thetype(["a", "b", "c"])

        # @torch.compile(backend="eager", fullgraph=True) # REMOVED
        def fn_delitem_global_list():
            global lst
            del lst[1]
            # Original test returned x.sin(), which is now irrelevant.
            return None

        # x = torch.tensor(1.0) # REMOVED
        # self.assertEqual(fn(x), x.sin()) # MODIFIED
        fn_delitem_global_list() # Call the function directly
        self.assertEqual(lst, ["a", "c"])

    def test___setitem__(self):
        p = self.thetype("abc")
        self.assertIsNone(p.__setitem__(2, "a"))
        self.assertEqual(p, self.thetype("aba"))

        p[0:] = []
        self.assertEqual(p, [])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__setitem__)
        self.assertRaises(TypeError, p.__setitem__, 1)
        self.assertRaises(TypeError, p.__setitem__, 1, 2, 3)

    def test___delitem__(self):
        p = self.thetype("abcdef")
        self.assertIsNone(p.__delitem__(1))
        self.assertEqual(p, self.thetype("acdef"))

        self.assertIsNone(p.__delitem__(slice(1, 3)))
        self.assertEqual(p, self.thetype("aef"))

        # Slice step == 0
        self.assertRaises(ValueError, p.__delitem__, slice(1, 1, 0))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__delitem__)
        self.assertRaises(TypeError, p.__delitem__, 1.1)
        self.assertRaises(TypeError, p.__delitem__, 1, 2)


if __name__ == "__main__":
    unittest.main()
