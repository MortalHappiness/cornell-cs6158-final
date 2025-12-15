import pytest
import numpy as np
import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr_functor import rewrite_call
from tvm.relay.op.tensor import ones, zeros
from tvm.testing import assert_allclose


class TestBuffersOverride:
    def test_buffers_override(self):
        # TODO: This test case is highly specific to PyTorch's `torch.compile`
        # interacting with `torch.nn.Module`'s internal `buffers` attribute.
        # TVM does not have an equivalent `nn.Module` abstraction or a
        # compilation process that directly inspects and handles Python-level
        # attribute overrides in the same manner.
        #
        # To port this, one would need to:
        # 1. Represent the `SomeModel` as a Relay IRModule.
        # 2. Simulate the effect of `self.register_buffer` by including `A` as a `relay.Constant`
        #    or parameter in the IRModule.
        # 3. The override `self.buffers = []` is a Python-level manipulation
        #    of the `nn.Module` object *before* compilation. This exact interaction
        #    (how `torch.compile` handles an overridden `model.buffers` list but
        #    still respects `named_buffers` from `register_buffer`) is a PyTorch-specific
        #    behavior that doesn't have a direct analogy in TVM's graph compilation.
        # 4. The assertion `self.assertEqual(compiled_model.A, torch.ones(3, 3))`
        #    checks an attribute of the *wrapped PyTorch module* returned by `torch.compile`,
        #    not the behavior of the compiled graph itself.
        #
        # Due to these PyTorch-specific runtime and compilation semantics, a direct
        # or composite mapping to an equivalent TVM test is not feasible without
        # reimplementing large parts of PyTorch's `torch.compile` behavior.
        #
        # Keeping this test as a placeholder with a clear explanation.
        pytest.skip("Test relies on PyTorch's torch.compile and nn.Module internal attribute handling, no direct TVM equivalent.")

    def test_named_buffers_override(self):
        # TODO: Similar to `test_buffers_override`, this test case is highly
        # specific to PyTorch's `torch.compile` interacting with a
        # `torch.nn.Module`'s `named_buffers` attribute.
        #
        # Overriding `model.named_buffers = []` prevents `register_buffer`
        # from correctly tracking the buffer `B`, and `torch.compile` would
        # then not find it through the standard introspection. The test verifies
        # that `torch.compile` still allows accessing `compiled_model.B`
        # (which is a `torch.Tensor` from the original module instance) and
        # that the forward pass still runs.
        #
        # TVM's compilation process is fundamentally different, operating on
        # a static graph (IRModule) and a dictionary of parameters/constants.
        # It does not have equivalent mechanisms for Python-level `nn.Module`
        # attribute introspection or manipulation in the same way `torch.compile` does.
        #
        # Thus, a direct or composite mapping to an equivalent TVM test is not feasible.
        pytest.skip("Test relies on PyTorch's torch.compile and nn.Module internal attribute handling, no direct TVM equivalent.")


if __name__ == "__main__":
    pytest.main([__file__])
