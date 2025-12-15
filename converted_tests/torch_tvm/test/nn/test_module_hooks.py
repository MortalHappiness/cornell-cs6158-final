import gc
import math
import pickle
import unittest
import warnings
import weakref
from collections import namedtuple, OrderedDict
from copy import deepcopy
from functools import partial
import sys # For IS_WINDOWS
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np # For random data, tensor conversion
import tvm
import tvm.relay as relay
import tvm.topi as topi
import tvm.testing
# Removed torch, torch.nn, torch.testing._internal.common_nn, torch.testing._internal.common_utils


# Replace PyTorch's common_utils.TestCase with unittest.TestCase for standalone execution
class TestCase(unittest.TestCase):
    pass

# Helper to convert numpy array to TVM Relay Constant for graph construction
def to_relay_const(np_array):
    return relay.Constant(tvm.nd.array(np_array))

# Helper to get device context
def get_tvm_device(device_str):
    if device_str == 'cpu':
        return tvm.cpu(0)
    elif device_str == 'cuda':
        return tvm.cuda(0)
    # Add other devices as needed
    raise ValueError(f"Unsupported device: {device_str}")


# In TVM Relay, computation graphs are functional.
# The concept of `nn.Module` objects with mutable state and runtime Python hooks
# (like `register_forward_hook`, `register_backward_hook`, `state_dict` management)
# does not have a direct equivalent.
# These classes are defined as placeholders, but tests relying on PyTorch's
# Module-specific features (especially hooks and `id` comparison) will be skipped.

class Net: # Changed from nn.Module
    def __init__(self) -> None:
        # In PyTorch, this initializes nn.Sequential and nn.Linear layers.
        # In TVM, the network structure would be defined as a Relay function.
        # For these hook tests, the internal structure is not directly relevant for TVM conversion.
        pass

    def forward(self, x: relay.Expr) -> relay.Expr: # Type hints changed
        # Placeholder for computation. The original tests focus on side effects of hooks, not graph execution.
        return x


ToyNamedTuple = namedtuple("ToyNamedTuple", "content")


class ToyModel: # Changed from nn.Module
    def __init__(self, with_named_tuple=False) -> None:
        self.net1 = Net()
        self.net2 = Net()
        self.with_named_tuple = with_named_tuple

    def forward(self, x: relay.Expr) -> Any: # Type hints changed
        # Placeholder for computation that would call Net's forward.
        res = x
        if self.with_named_tuple:
            return ToyNamedTuple(res)
        else:
            return (res,)


# --- Hook functions (these are PyTorch-specific and won't have direct TVM equivalents for registration/execution) ---
# Kept as Python functions, but their invocation mechanism is not mappable.

def forward_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: Any, # Represents a PyTorch nn.Module
    hook_id: int,
    module: Any, # Represents a PyTorch nn.Module
    inp: tuple[Any], # Represents tuple of torch.Tensor
    out: Any, # Represents torch.Tensor
) -> None:
    fired_hooks.append(hook_id)
    # self.assertEqual(id(module), id(expected_module)) # PyTorch-specific object identity check
    self.assertEqual(len(inp), 1)


def forward_pre_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: Any, # Represents a PyTorch nn.Module
    hook_id: int,
    module: Any, # Represents a PyTorch nn.Module
    inp: tuple[Any], # Represents tuple of torch.Tensor
) -> None:
    fired_hooks.append(hook_id)
    # self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(inp), 1)


def full_backward_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: Any, # Represents a PyTorch nn.Module
    hook_id: int,
    module: Any, # Represents a PyTorch nn.Module
    grad_input: tuple[Any], # Represents tuple of torch.Tensor
    grad_output: tuple[Any], # Represents tuple of torch.Tensor
) -> None:
    fired_hooks.append(hook_id)
    # self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(grad_input), 1)
    self.assertEqual(len(grad_output), 1)


def full_backward_pre_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: Any, # Represents a PyTorch nn.Module
    hook_id: int,
    module: Any, # Represents a PyTorch nn.Module
    grad_input: tuple[Any], # Represents tuple of torch.Tensor
) -> None:
    fired_hooks.append(hook_id)
    # self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(grad_input), 1)


class KwargModel: # Changed from nn.Module
    def __init__(self) -> None:
        # super().__init__()
        self.net1 = Net()
        self.net2 = Net()
        # Internal PyTorch hook dicts, no TVM equivalent
        self._forward_hooks_with_kwargs = OrderedDict()
        self._forward_pre_hooks_with_kwargs = OrderedDict()

    def forward(self, x: relay.Expr, bias: relay.Expr = None) -> relay.Expr: # Type hints changed
        if bias is not None:
            x = relay.op.tensor.add(x, bias)
        return x

    def internal_forward_hook(
        self,
        module: Any, # Represents a PyTorch nn.Module
        args: tuple[relay.Expr], # Represents tuple of torch.Tensor
        kwargs: dict[str, Any],
        out: relay.Expr, # Represents torch.Tensor
    ):
        return relay.op.tensor.add(out, kwargs["bias"])


class FailsInForwardModel: # Changed from nn.Module
    def __init__(self) -> None:
        # super().__init__()
        self.net1 = Net()

    def forward(self, x: relay.Expr, fail: bool = True) -> relay.Expr: # Type hints changed
        if fail:
            raise RuntimeError("failing in forward")
        # Placeholder for computation.
        return x


def kwarg_forward_pre_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: Any, # Represents a PyTorch nn.Module
    hook_id: int,
    module: Any, # Represents a PyTorch nn.Module
    args: tuple[relay.Expr], # Represents tuple of torch.Tensor
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    fired_hooks.append(hook_id)
    # self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(args), 1)
    # For TVM, kwargs values would need to be Relay expressions or constants
    kwargs["bias"] = relay.op.tensor.multiply(relay.const(2.0, "float32"), kwargs["bias"])
    return args, kwargs


def kwarg_forward_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: Any, # Represents a PyTorch nn.Module
    hook_id: int,
    module: Any, # Represents a PyTorch nn.Module
    args: tuple[relay.Expr], # Represents tuple of torch.Tensor
    kwargs: dict[str, Any],
    out: relay.Expr, # Represents torch.Tensor
) -> Any:
    fired_hooks.append(hook_id)
    # self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(args), 1)

    out = relay.op.tensor.add(out, kwargs["bias"])
    return out


class DummyContextManager:
    def __init__(self, inp):
        self.input = inp

    def __enter__(self, *args, **kwargs):
        self.input.append(2)

    def __exit__(self, *args, **kwargs):
        self.input.append(-1)


# The majority of these tests rely on PyTorch's `nn.Module` hook system
# (forward, backward, state_dict hooks) which is fundamentally tied to
# PyTorch's object model and dynamic autograd graph.
# TVM Relay is a static, functional IR and does not have direct equivalents
# for these runtime Python callbacks on module objects.
# Therefore, these tests are marked as skipped.
class TestModuleHooks(TestCase):
    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_forward_hooks(self, named_tuple):
        # This test relies on `nn.Module.register_forward_hook` and execution flow.
        # This is specific to PyTorch's runtime and object model.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_forward_pre_hooks(self, named_tuple):
        # This test relies on `nn.Module.register_forward_pre_hook` and execution flow.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_full_backward_hooks(self, named_tuple):
        # This test relies on `nn.Module.register_full_backward_hook` and PyTorch's autograd system.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_full_backward_pre_hooks(self, named_tuple):
        # This test relies on `nn.Module.register_full_backward_pre_hook` and PyTorch's autograd system.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_mixed_hooks(self, named_tuple):
        # This test combines different PyTorch `nn.Module` hooks.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_kwarg_hooks(self):
        # This test uses PyTorch's `nn.Module` hooks with `with_kwargs=True`.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_remove_kwarg_hooks(self):
        # This test verifies removal of PyTorch `nn.Module` hooks.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_always_called_forward_hooks(self):
        # This test relies on PyTorch's `nn.Module` hooks with `always_call=True` for error handling.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_bw_hook_warning_for_non_tensor_or_tuple(self):
        # This test involves PyTorch's `nn.Module` backward hooks and type checking.
        pass


class TestStateDictHooks(TestCase):
    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_load_state_dict_pre_hook(self):
        # This test uses PyTorch's `nn.Module.register_load_state_dict_pre_hook`.
        # TVM manages parameters explicitly, not through module-level hooks.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_no_extra_ref_to_module(self):
        # This test relies on PyTorch's object model and weak references with hooks.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_pickled_hook(self):
        # This test relies on PyTorch's serialization mechanism for modules with hooks.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_load_state_dict_module_pre_hook(self):
        # This test uses PyTorch's `nn.Module`'s `_register_load_state_dict_pre_hook` and module instance methods.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_load_state_dict_post_hook(self):
        # This test uses PyTorch's `nn.Module.register_load_state_dict_post_hook` to modify `incompatible_keys`.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_load_state_dict_post_hook_backward_compatibility(self):
        # This test checks backward compatibility for `_load_state_dict_post_hooks` attribute in PyTorch.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def _test_register_state_dict_pre_hook(self, model, submodule):
        # This internal helper function registers state_dict pre-hooks on PyTorch modules.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_register_state_dict_pre_hook(self):
        # This test uses `_test_register_state_dict_pre_hook` with a standard PyTorch module.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_register_state_dict_pre_hook_lazy_module(self):
        # This test uses `_test_register_state_dict_pre_hook` with a PyTorch lazy module.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_register_state_dict_pre_hook_backward_compat(self):
        # This test checks backward compatibility for `_state_dict_pre_hooks` attribute in PyTorch.
        pass

    @unittest.skip("PyTorch state_dict hooks are not directly mappable to TVM's parameter handling.")
    def test_register_state_dict_post_hook(self, private):
        # This test checks `_register_state_dict_hook` and `register_state_dict_post_hook` in PyTorch.
        pass


class TestModuleGlobalHooks(TestCase):
    # These tests involve PyTorch's global module hook registry.
    # TVM Relay's functional nature does not have a global registry for Python callbacks on IR operations.
    def tearDown(self):
        # No direct TVM equivalent for clearing global PyTorch hooks.
        # This would be specific to PyTorch's internal state.
        pass

    @unittest.skip("PyTorch global nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_module_global_hooks(self):
        # This test uses `nn.modules.module.register_module_forward_hook` and `register_module_backward_hook`.
        pass

    @unittest.skip("PyTorch global nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_module_global_hook_invalid_outputs(self):
        # This tests error handling for invalid outputs from global backward hooks.
        pass

    @unittest.skip("PyTorch global nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_module_backward_global_hook_writeable(self):
        # This tests modifying gradients via global backward hooks.
        pass

    @unittest.skip("PyTorch global nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_module_global_forward_preforward_hook_writeable(self):
        # This tests modifying inputs/outputs via global forward pre/post hooks.
        pass

    @unittest.skip("PyTorch global nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_module_forward_preforward_hook_removable(self):
        # This tests dynamic removal of global forward pre-hooks.
        pass

    @unittest.skip("PyTorch global nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_module_forward_forward_hook_removable(self):
        # This tests dynamic removal of global forward hooks.
        pass

    @unittest.skip("PyTorch global nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_global_and_local_hooks_order(self):
        # This test verifies the execution order of global and local PyTorch hooks.
        pass

    @unittest.skip("PyTorch global nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_module_global_hooks_with_kwargs(self):
        # This tests global hooks with kwargs.
        pass


# Replaced torch.testing._internal.common_nn.NNTestCase with a local placeholder inheriting unittest.TestCase.
# The internal helper `_create_basic_net` from common_nn also has no direct TVM equivalent for creating modules.
class NNTestCase(TestCase):
    # PyTorch specific checks related to CUDA memory, no direct TVM equivalent for these test-level flags.
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _create_basic_net(self):
        # This function creates PyTorch nn.Modules, which don't map directly to TVM.
        # It's not possible to translate it to TVM Relay in a meaningful way for hook tests.
        return None, None, None

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def _test_hooks(self, backward_register_fn):
        # This is an internal helper that tests different types of backward hooks.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hooks(self):
        # This calls `_test_hooks` for various backward hook registration functions.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_cpp(self):
        # This tests hooks on a C++ implemented module (`nn.BatchNorm1d`).
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_backward_hooks_interaction(self):
        # This tests interaction between pre and post backward hooks in PyTorch.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_invalid_outputs(self):
        # This tests error handling for invalid return types from backward hooks.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_requires_grad(self):
        # This tests `requires_grad` propagation with hooks.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_no_requires_grad(self):
        # This tests hooks when no inputs require gradients.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_last_arg_requires_grad(self):
        # This tests a specific case with `L1Loss` and `requires_grad`.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_extra_input(self):
        # This tests a hook with an extra non-tensor input to forward.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_inplace(self):
        # This tests behavior of hooks with inplace operations, and errors.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_non_full_warning(self):
        # This tests warnings for non-full (legacy) backward hooks.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_backward_size(self):
        # This tests backward hook with varying input/output sizes.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_backward_writeable(self):
        # This tests modifying gradients via backward hooks.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_forward_preforward_writable(self):
        # This tests modifying inputs/outputs via forward pre/post hooks.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_buffer_registration(self):
        # This tests PyTorch's `register_module_buffer_registration_hook`.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_submodule_registration(self):
        # This tests PyTorch's `register_module_module_registration_hook`.
        pass

    @unittest.skip("PyTorch nn.Module hooks are not directly mappable to TVM Relay's functional IR.")
    def test_hook_parameter_registration(self):
        # This tests PyTorch's `register_module_parameter_registration_hook`.
        pass


# instantiate_parametrized_tests and parametrize_test are PyTorch testing utilities
# not directly applicable in a TVM context when most tests are skipped.
# If there were convertible parametrized tests, they would need manual expansion
# or conversion to pytest.mark.parametrize.

# Renamed to avoid name collision with standard `unittest.main()`
def run_tvm_tests():
    unittest.main()

if __name__ == "__main__":
    # Most tests in this file are fundamentally about PyTorch's nn.Module API
    # and its dynamic hook system, which do not have direct equivalents in
    # TVM Relay's functional, static graph paradigm.
    # Therefore, the relevant tests are skipped with explanatory messages.
    print("Skipping most tests in test_module_hooks.py due to fundamental differences in PyTorch nn.Module hook system vs TVM Relay functional IR.")
    run_tvm_tests()
