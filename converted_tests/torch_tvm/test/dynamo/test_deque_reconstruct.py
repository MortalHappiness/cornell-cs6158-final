# Owner(s): ["module: dynamo"]

import collections
import contextlib
import unittest
import numpy as np
# import torch # Removed: Cannot import torch or any torch.* symbols
# import torch._inductor.test_case # Replaced with unittest.TestCase

# TODO: This test specifically targets PyTorch's torch.compile behavior
# regarding the global state of 'deque'. Since torch.compile has no direct
# TVM equivalent, this test cannot be fully converted while preserving
# its original intent. The @torch.compile decorator and related PyTorch-specific
# runtime logic are either commented out or replaced with standard Python/NumPy
# equivalents. This changes the core test semantics from testing PyTorch's
# compilation to testing basic Python/NumPy behavior.

class TestDequeReconstruct(unittest.TestCase): # Changed base class
    UNSET = object()

    @contextlib.contextmanager
    def set_deque_in_globals(self, value):
        # This context manager directly manipulates Python globals.
        # Its original purpose was to test torch.compile's interaction
        # with these globals. Since torch.compile is not convertible,
        # the relevance of this context manager for TVM is limited,
        # but kept for structural consistency.
        prev = globals().pop("deque", self.UNSET)
        self.assertNotIn("deque", globals()) # Ensure 'deque' is not already there, consistent with pop

        try:
            if value is not self.UNSET:
                globals()["deque"] = value
            yield
        finally:
            if prev is self.UNSET:
                globals().pop("deque", None)
                self.assertNotIn("deque", globals())
            else:
                globals()["deque"] = prev

    def test_deque_reconstruct_not_in_globals(self):
        with self.set_deque_in_globals(self.UNSET):

            # @torch.compile(backend="eager", fullgraph=True) # TODO: torch.compile has no direct TVM equivalent. Test will run as pure Python.
            def func(x):
                # x is now a numpy array. Operations like x+1 will be NumPy element-wise additions.
                # The function still returns a collections.deque.
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = np.random.randn(3, 4).astype(np.float32) # Changed from torch.randn, added dtype
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            # For collections.deque of numpy arrays, `assertEqual` correctly compares contents element-wise.
            # NumPy arrays define __eq__ for element-wise comparison, and for `assertEqual` to pass,
            # all elements must be equal.
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))

    def test_deque_reconstruct_in_globals(self):
        with self.set_deque_in_globals(collections.deque):
            # This does not emit a NameError because 'deque' is explicitly set in globals.
            dummy = collections.deque([0, 1, 2], maxlen=2)
            self.assertIsInstance(dummy, collections.deque)
            self.assertEqual(list(dummy), [1, 2])

            # @torch.compile(backend="eager", fullgraph=True) # TODO: torch.compile has no direct TVM equivalent. Test will run as pure Python.
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = np.random.randn(3, 4).astype(np.float32) # Changed from torch.randn, added dtype
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))

    def test_deque_reconstruct_shallows_globals(self):
        with self.set_deque_in_globals(None):
            # This does not emit a NameError, but `deque` resolves to `None`.
            # Accessing `deque` directly would cause a NameError if it was never defined globally.
            # `globals().get("deque")` is safer for checking its value.
            self.assertIsNone(globals().get("deque"))

            # @torch.compile(backend="eager", fullgraph=True) # TODO: torch.compile has no direct TVM equivalent. Test will run as pure Python.
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = np.random.randn(3, 4).astype(np.float32) # Changed from torch.randn, added dtype
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))


if __name__ == "__main__":
    # from torch._dynamo.test_case import run_tests # Removed
    # run_tests() # Replaced with standard unittest.main()
    unittest.main()
