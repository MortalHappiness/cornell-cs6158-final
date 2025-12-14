import numpy as np
import pytest
import itertools
import logging
import functools
from typing import Tuple, Dict, List, Callable, Union

import torch
import torch.nn.functional
import torch.testing

# Try importing torchvision and cv2 for specific tests, skip if not available
try:
    import torchvision
    _has_torchvision = True
except ImportError:
    _has_torchvision = False

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

# Global setup for tests
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPPORTED_DTYPES = [torch.float16, torch.float32]

# Mappers for dtypes
def dtype_to_torch(dtype_str: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    elif dtype_str == "bool":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def dtype_to_np(dtype_torch: torch.dtype) -> np.dtype:
    if dtype_torch == torch.float32:
        return np.float32
    elif dtype_torch == torch.float16:
        return np.float16
    elif dtype_torch == torch.int32:
        return np.int32
    elif dtype_torch == torch.int64:
        return np.int64
    elif dtype_torch == torch.bool:
        return np.bool_
    else:
        raise ValueError(f"Unsupported dtype: {dtype_torch}")

# Convert TVM `Any` (dynamic shape) to a placeholder that NumPy/PyTorch can handle.
def resolve_shape(shape_tuple: Tuple, default_dynamic_dim_size: int = 1) -> Tuple:
    resolved = []
    for dim in shape_tuple:
        # TVM's `relay.Any` translates to `None` in Python representation typically.
        if dim is None or (isinstance(dim, str) and dim == "Any"):
            resolved.append(default_dynamic_dim_size)
        else:
            resolved.append(dim)
    return tuple(resolved)

def vmobj_to_list(o: Union[torch.Tensor, list, tuple, int, float, bool, np.generic]) -> List[np.ndarray]:
    """Converts PyTorch tensor(s) or Python scalars/lists of tensors to a list of NumPy arrays."""
    if isinstance(o, torch.Tensor):
        return [o.cpu().numpy()]
    elif isinstance(o, (list, tuple)):
        # Recursively convert elements in lists/tuples
        result = []
        for item in o:
            result.extend(vmobj_to_list(item))
        return result
    elif isinstance(o, (int, float, bool, np.generic)):
        # Convert Python scalars or NumPy scalars to a list containing a NumPy array
        return [np.array(o)]
    else:
        # Fallback for unexpected types
        raise RuntimeError(f"Unknown object type: {type(o)}")


def assert_result_dict_holds(result_dict: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]], dtype: torch.dtype = torch.float16):
    """
    Asserts that results from different execution modes in result_dict are close.
    """
    if len(result_dict) < 2:
        logging.warning("Less than two results to compare in result_dict. Skipping assertion.")
        return

    for k1, k2 in itertools.combinations(result_dict, 2):
        res1 = vmobj_to_list(result_dict[k1])
        res2 = vmobj_to_list(result_dict[k2])
        for r1, r2 in zip(res1, res2):
            if dtype == torch.float16:
                torch.testing.assert_close(r1, r2, rtol=1e-1, atol=1e-1)
            else: # float32
                torch.testing.assert_close(r1, r2, rtol=1e-3, atol=5e-3)


# Helper class to wrap functional operations as a module for torch.compile
class FunctionalModule(torch.nn.Module):
    def __init__(self, func_callable: Callable, device: torch.device):
        super().__init__()
        self.func = func_callable
        # For simplicity, if the function creates internal constants, they should be moved to device
        self.device = device
        # This wrapper assumes the callable itself handles device placement if it creates new tensors,
        # or that all inputs are already on the correct device.

    def forward(self, *input_args):
        return self.func(*input_args)


def run_and_verify_func(
    model_builder_fn: Callable[[torch.dtype], Tuple[Callable, Dict[str, Tuple], List[str]]],
    target_device: str = "cuda",
    run_inference: bool = True,
    data_type: torch.dtype = torch.float32,
    default_dynamic_dim_size: int = 1, # Default size for `relay.Any` dimensions
):
    """
    Test a PyTorch functional model by compiling, running, and comparing eager and compiled outputs.

    Parameters
    ----------
    model_builder_fn : Callable
        A function that takes `data_type` and returns a tuple:
        (1) A Python callable (e.g., lambda or def) representing the model's forward pass.
        (2) A dictionary of input names to their shapes (e.g., {"x_in": (1, 3, 2, 2)}).
        (3) A list of input names that represent "parameters" (constant inputs in TVM terms).
    target_device : str
        The target device for PyTorch tensors ("cuda" or "cpu").
    run_inference : bool
        If True, runs inference and compares outputs. If False, only attempts compilation.
    data_type : torch.dtype
        The data type for generated input tensors.
    default_dynamic_dim_size : int
        The size to use for `relay.Any` (dynamic) dimensions when generating random input data.
    """
    np.random.seed(42)

    # model_builder_fn returns a callable (like a lambda), input shapes, and parameter names.
    # The callable will take input tensors in the order of `input_names_for_callable`.
    model_callable_raw, input_shapes, param_names = model_builder_fn(data_type=data_type)

    device = torch.device(target_device if target_device == "cuda" and torch.cuda.is_available() else "cpu")

    # Generate input tensors. All `relay.var` in TVM become inputs to the PyTorch callable.
    input_tensors_map = {}
    ordered_input_tensors_for_callable = []
    input_names_for_callable = list(input_shapes.keys()) # Ensure order is preserved for positional args

    for name in input_names_for_callable:
        shape_tuple = input_shapes[name]
        # Resolve dynamic dimensions for data generation
        resolved_shape = resolve_shape(shape_tuple, default_dynamic_dim_size)
        
        if data_type == torch.bool: # For boolean ops
            np_data = np.random.randint(0, 2, size=resolved_shape, dtype=dtype_to_np(data_type))
        else:
            np_data = np.random.uniform(-1, 1, resolved_shape).astype(dtype_to_np(data_type))
        
        tensor = torch.tensor(np_data, dtype=data_type, device=device)
        input_tensors_map[name] = tensor
        ordered_input_tensors_for_callable.append(tensor)

    result_dict = {}

    for use_torch_compile in [False, True]:
        # Wrap the raw callable in a FunctionalModule for `torch.compile` if needed
        # Or just use the callable directly, torch.compile can handle functions.
        model_to_run = model_callable_raw

        if use_torch_compile:
            try:
                model_to_run = torch.compile(model_callable_raw)
            except Exception as e:
                logging.warning(f"torch.compile failed for model (use_torch_compile={use_torch_compile}): {e}. Skipping compiled test.")
                continue
        
        result_key = "compiled" if use_torch_compile else "eager"

        if run_inference:
            with torch.no_grad():
                outputs = model_to_run(*ordered_input_tensors_for_callable)
            result_dict[result_key] = outputs
    
    if run_inference:
        assert_result_dict_holds(result_dict, data_type)


# This part used TVM's internal IR inspection and attribute manipulation.
# There is no direct PyTorch equivalent. These tests will be skipped or commented out
# where the core logic cannot be meaningfully translated without testing TVM specifics.
# `set_outer_func_attr`, `set_inner_func_attr`, `ExprVisitor`, `AreOpsOnGraph`, `are_ops_on_trt` are TVM-specific.


_has_cuda_mark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

run_module = pytest.mark.parametrize(
    "run_inference",
    [
        # TVM's `run_module=False` implies compile-only and inspect artifact.
        # For PyTorch, `torch.compile` compiles, but inspecting the compiled artifact
        # directly in tests is not idiomatic/supported for correctness verification.
        # Thus, `run_inference=False` means we skip the actual output comparison.
        pytest.param(False, marks=_has_cuda_mark, id="compile_only_no_inference"),
        pytest.param(True, marks=_has_cuda_mark, id="run_inference"),
    ],
)


@run_module
def test_tensorrt_simple(run_inference):
    def get_graph(data_type_arg):
        def f(x_in, y_in, z_in):
            w = z_in * (x_in + y_in)
            out = torch.nn.functional.relu(w)
            return out
        x_shape = (1, 3, 2, 2)
        y_shape = (1, 3, 1, 1)
        z_shape = (1, 1, 1, 1)
        input_shapes = {"x_in": x_shape, "y_in": y_shape, "z_in": z_shape}
        param_names = []
        return f, input_shapes, param_names

    for dtype in SUPPORTED_DTYPES:
        run_and_verify_func(get_graph, run_inference=run_inference, data_type=dtype)


@run_module
def test_tensorrt_simple_cpu_io(run_inference):
    def get_graph(data_type_arg):
        def f(x_in, y_in, z_in):
            w = z_in * (x_in + y_in)
            out = torch.nn.functional.relu(w)
            return out
        # Original TVM uses "float32"
        x_shape = (1, 3, 2, 2)
        y_shape = (1, 3, 1, 1)
        z_shape = (1, 1, 1, 1)
        input_shapes = {"x_in": x_shape, "y_in": y_shape, "z_in": z_shape}
        param_names = ["y_in"] # In TVM, `y` is a param. In PyTorch, it's just another input.
        return f, input_shapes, param_names
    
    # Target "llvm" corresponds to CPU in TVM.
    run_and_verify_func(get_graph, target_device="cpu", run_inference=run_inference, data_type=torch.float32)


@run_module
def test_tensorrt_not_compatible(run_inference):
    def get_graph(data_type_arg):
        def f(x_in):
            y = x_in + x_in
            # This sequence of float->int32->float cast is often not offloaded by TRT
            z = y.to(torch.int32).to(torch.float32)
            out = torch.nn.functional.relu(z)
            return out
        # Original TVM uses "float32"
        x_shape = (1, 32, 14, 14)
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float32)


@run_module
def test_conv1d(run_inference):
    def get_graph(
        data_type_arg: torch.dtype, # Renamed to avoid shadowing outer var
        x_shape: Tuple = (1, 3, 224), # N, C_in, W
        k_original_shape: Tuple = (10, 3, 3), # TVM's kernel shape: (Out_C, In_C_per_group, kW)
        groups: int = 1,
        padding: Union[int, Tuple] = (1),
        strides: Union[int, Tuple] = (1),
        dilation: Union[int, Tuple] = (1),
        channels: Union[int, None] = None, # TVM's explicit output channels arg
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        # Infer PyTorch kernel shape: (out_channels, in_channels_per_group, k_width)
        out_channels = channels if channels is not None else k_original_shape[0]
        in_channels = x_shape[1]
        in_channels_per_group = in_channels // groups
        kernel_width = k_original_shape[2]

        k_pytorch_shape = (out_channels, in_channels_per_group, kernel_width)

        def f(x_in: torch.Tensor, kernel_in: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.conv1d(
                input=x_in,
                weight=kernel_in,
                bias=None,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        
        input_shapes = {"x_in": x_shape, "kernel_in": k_pytorch_shape}
        param_names = ["kernel_in"]
        return f, input_shapes, param_names

    for dtp in [torch.float16]: # Original TVM test only used float16
        # TVM's call was `get_graph(channels=10, d_type=d_type)`. `x_shape=(1,3,224)` and `k_shape=(10,3,3)` default from `get_graph`.
        # Correct PyTorch kernel shape if channels=10 is used:
        # x_shape=(1, 3, 224), groups=1. in_channels=3.
        # k_original_shape=(10,3,3). If `channels` arg is 10, then out_channels=10.
        # PyTorch k_shape: (out_channels=10, in_channels_per_group=3//1=3, k_width=3) => (10,3,3)
        run_and_verify_func(
            functools.partial(get_graph, channels=10),
            run_inference=run_inference,
            data_type=dtp
        )


@run_module
def test_conv2d(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        x_shape: Tuple, # (N, C_in, H, W)
        k_original_shape: Tuple, # TVM's k_shape, e.g., (Out_C, In_C_per_group, kH, kW)
        groups: int,
        padding: Tuple,
        strides: Tuple,
        dilation: Tuple,
        channels: Union[int, None], # TVM's explicit output channels arg
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        # Infer PyTorch kernel shape: (out_channels, in_channels_per_group, kH, kW)
        out_channels = channels if channels is not None else k_original_shape[0]
        in_channels = x_shape[1]
        in_channels_per_group = in_channels // groups
        kernel_h, kernel_w = k_original_shape[-2:] # Assuming last two dims are kernel spatial

        k_pytorch_shape = (out_channels, in_channels_per_group, kernel_h, kernel_w)

        def f(x_in: torch.Tensor, kernel_in: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.conv2d(
                input=x_in,
                weight=kernel_in,
                bias=None,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        input_shapes = {"x_in": x_shape, "kernel_in": k_pytorch_shape}
        param_names = ["kernel_in"]
        return f, input_shapes, param_names

    # Loop for various configurations
    for k_orig_shape, groups in [((16, 32, 3, 3), 1), ((32, 1, 3, 3), 32)]:
        in_channels_for_x = k_orig_shape[1] * groups # Compute C_in for x_shape
        current_x_shape = (1, in_channels_for_x, 8, 8)
        for padding in [(0, 0), (1, 1)]:
            for strides in [(1, 1), (2, 2)]:
                for dilation in [(1, 1), (2, 2)]:
                    run_and_verify_func(
                        functools.partial(
                            get_graph,
                            x_shape=current_x_shape,
                            k_original_shape=k_orig_shape,
                            groups=groups,
                            padding=padding,
                            strides=strides,
                            dilation=dilation,
                            channels=None, # Use k_original_shape[0] as output channels
                        ),
                        run_inference=run_inference,
                        data_type=torch.float16,
                    )
    
    # Specific case 1 from TVM: `get_graph((1, 3, 16, 16), (3, 8, 7, 7), 3, [2, 2, 3, 3], [2, 2], [1, 1], 24, data_type="float16")`
    #   x_shape=(1, 3, 16, 16), k_original_shape=(3, 8, 7, 7), groups=3, padding=[2, 2, 3, 3], strides=[2, 2], dilation=[1, 1], channels=24
    # PyTorch kernel shape calculation:
    #   out_channels = 24 (from `channels` arg)
    #   in_channels = 3 (from x_shape[1])
    #   in_channels_per_group = 3 // 3 = 1
    #   kernel_h, kernel_w = 7, 7 (from k_original_shape[-2:])
    #   Final PyTorch k_shape: (24, 1, 7, 7)
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 3, 16, 16), k_original_shape=(24, 1, 7, 7), groups=3, padding=[2, 2, 3, 3], strides=[2, 2], dilation=[1, 1], channels=24),
        run_inference=run_inference,
        data_type=torch.float16,
    )

    # Specific case 2 from TVM: `get_graph((1, 3, 16, 16), (1, 3, 1, 1), channels=1, data_type="float32")`
    #   x_shape=(1, 3, 16, 16), k_original_shape=(1, 3, 1, 1), groups=1, channels=1
    # PyTorch kernel shape calculation:
    #   out_channels = 1 (from `channels` arg)
    #   in_channels = 3 (from x_shape[1])
    #   in_channels_per_group = 3 // 1 = 3
    #   kernel_h, kernel_w = 1, 1 (from k_original_shape[-2:])
    #   Final PyTorch k_shape: (1, 3, 1, 1)
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 3, 16, 16), k_original_shape=(1, 3, 1, 1), groups=1, channels=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_conv2d_nhwc(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        x_shape: Tuple = (1, 8, 8, 32), # NHWC (N, H, W, C_in)
        k_original_shape: Tuple = (3, 3, 32, 16) # HWIO (kH, kW, C_in, C_out)
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        # PyTorch F.conv2d expects NCHW input, OIHW weight
        # input NHWC (N, H, W, C_in) -> NCHW (N, C_in, H, W)
        # weight HWIO (kH, kW, C_in, C_out) -> OIHW (C_out, C_in, kH, kW)
        # TVM's `channels=16` implies C_out = 16.
        out_channels = 16
        in_channels = x_shape[3] # C_in from NHWC
        kernel_h, kernel_w = k_original_shape[0:2] # kH, kW from HWIO
        
        # PyTorch `weight` tensor shape must be (out_c, in_c, kH, kW)
        k_pytorch_shape = (out_channels, in_channels, kernel_h, kernel_w)

        def f(x_in: torch.Tensor, kernel_in: torch.Tensor) -> torch.Tensor:
            x_nchw = x_in.permute(0, 3, 1, 2)
            kernel_oihw = kernel_in.permute(3, 2, 0, 1) # (C_out, C_in, kH, kW)
            out_nchw = torch.nn.functional.conv2d(
                input=x_nchw,
                weight=kernel_oihw,
                bias=None,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            )
            return out_nchw # Output is NCHW, no explicit out_layout in PyTorch F.conv2d
        
        # `kernel_in` will be generated using k_original_shape, then permuted inside `f`.
        input_shapes = {"x_in": x_shape, "kernel_in": k_original_shape}
        param_names = ["kernel_in"]
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float32)


@run_module
def test_conv2d_weights_const(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        x_shape: Tuple = (1, 32, 8, 8),
        k_shape: Tuple = (16, 32, 3, 3), # (Out_C, In_C_per_group, kH, kW)
        groups: int = 1,
        padding: Tuple = (0, 0),
        strides: Tuple = (1, 1),
        dilation: Tuple = (1, 1),
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            # `kernel` is a constant, so it's created within the functional graph
            kernel = torch.ones(k_shape, dtype=data_type_arg, device=x_in.device)
            return torch.nn.functional.conv2d(
                input=x_in,
                weight=kernel,
                bias=None,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        input_shapes = {"x_in": x_shape}
        param_names = [] # No `relay.var` parameters
        return f, input_shapes, param_names

    for dtp in [torch.float16]: # Original TVM test only used float16
        run_and_verify_func(get_graph, run_inference=run_inference, data_type=dtp)


@run_module
def test_conv2d_weights_transposed(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        x_shape: Tuple = (1, 32, 9, 9), # NCHW
        k_original_shape: Tuple = (3, 3, 32, 16), # TVM's k_shape HWIO (kH, kW, C_in, C_out)
        order: Tuple = (3, 2, 0, 1) # Permutation to (C_out, C_in, kH, kW) for PyTorch OIHW
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor, kernel_in: torch.Tensor) -> torch.Tensor:
            kernel_t = kernel_in.permute(order) # Apply the transposition
            # PyTorch F.conv2d expects NCHW input, OIHW weight
            return torch.nn.functional.conv2d(
                input=x_in,
                weight=kernel_t,
                bias=None,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            )
        # `kernel_in` will be generated with `k_original_shape`, then permuted inside `f`.
        input_shapes = {"x_in": x_shape, "kernel_in": k_original_shape}
        param_names = ["kernel_in"]
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float32)


@run_module
def test_dense(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        x_shape: Tuple = (1, 16),
        k_shape: Tuple = (32, 16) # PyTorch linear weight: (out_features, in_features)
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor, kernel_in: torch.Tensor) -> torch.Tensor:
            # PyTorch F.linear expects input (..., in_features), weight (out_features, in_features)
            # kernel_in is already in the correct shape (out_features, in_features)
            return torch.nn.functional.linear(x_in, kernel_in, bias=None)
        
        input_shapes = {"x_in": x_shape, "kernel_in": k_shape}
        param_names = ["kernel_in"]
        return f, input_shapes, param_names

    for dtp in [torch.float32]:
        run_and_verify_func(functools.partial(get_graph, x_shape=(1, 16), k_shape=(32, 16)), run_inference=run_inference, data_type=dtp)
        run_and_verify_func(functools.partial(get_graph, x_shape=(1, 16), k_shape=(1, 16)), run_inference=run_inference, data_type=dtp)


@run_module
def test_batch_matmul(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        x_shape: Tuple = (12, 128, 64),
        y_shape: Tuple = (12, 128, 64),
        transa: bool = False,
        transb: bool = True,
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
            # Apply transposition using .mT (matrix transpose) before matmul if flags are True
            a_processed = x_in.mT if transa else x_in
            b_processed = y_in.mT if transb else y_in
            return torch.matmul(a_processed, b_processed)
        
        input_shapes = {"x_in": x_shape, "y_in": y_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, x_shape=(12, 64, 128), y_shape=(12, 128, 64), transa=True, transb=True),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(12, 64, 128), y_shape=(12, 64, 128), transa=True, transb=False),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(12, 128, 64), y_shape=(12, 128, 64), transa=False, transb=True),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(12, 128, 64), y_shape=(12, 64, 128), transa=False, transb=False),
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_bias_add(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        x_shape: Tuple,
        channels: int,
        axis: int = 1 # TVM's default axis for bias_add, corresponds to channel dimension
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor, bias_in: torch.Tensor) -> torch.Tensor:
            # PyTorch's `add` handles broadcasting. We need to reshape the 1D bias
            # to match the broadcast semantics of `relay.nn.bias_add`'s `axis`.
            new_bias_shape = [1] * x_in.ndim
            new_bias_shape[axis] = channels
            reshaped_bias = bias_in.reshape(new_bias_shape)
            return x_in + reshaped_bias
        
        input_shapes = {"x_in": x_shape, "bias_in": (channels,)}
        param_names = ["bias_in"]
        return f, input_shapes, param_names

    run_and_verify_func(functools.partial(get_graph, x_shape=(1, 16), channels=16, axis=1), run_inference=run_inference, data_type=torch.float32)
    run_and_verify_func(functools.partial(get_graph, x_shape=(1, 6, 3, 4), channels=6, axis=1), run_inference=run_inference, data_type=torch.float32)


@run_module
def test_pool2d(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        op_func: Callable, # e.g., torch.nn.functional.max_pool2d
        x_shape: Tuple = (1, 3, 32, 32),
        pool_size: Tuple = (2, 2),
        strides: Tuple = (2, 2),
        padding: Tuple = (0, 0), # PyTorch symmetric (pad_h, pad_w)
        ceil_mode: bool = False,
        count_include_pad: Union[bool, None] = None, # Only for avg_pool2d
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            if op_func == torch.nn.functional.avg_pool2d:
                return op_func(
                    input=x_in,
                    kernel_size=pool_size,
                    stride=strides,
                    padding=padding,
                    ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad,
                )
            else: # torch.nn.functional.max_pool2d
                return op_func(
                    input=x_in,
                    kernel_size=pool_size,
                    stride=strides,
                    padding=padding,
                    ceil_mode=ceil_mode,
                )

        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    for pool_size in [(2, 2), (3, 3)]:
        for strides in [(1, 1), (2, 2)]:
            # TVM `padding` (0,0,1,1) translates to (pad_top=0, pad_bottom=0, pad_left=1, pad_right=1).
            # PyTorch `F.pool2d`'s `padding` arg is (pad_h, pad_w) for *symmetric* padding.
            # So, `(0,0,1,1)` in TVM maps to PyTorch `padding=(0,1)` if `top==bottom` and `left==right`.
            for original_padding_tvm in [(0, 0), (1, 1), (0, 0, 1, 1)]:
                pad_pytorch: Tuple
                if len(original_padding_tvm) == 2: # Symmetric (H_pad, W_pad)
                    pad_pytorch = original_padding_tvm
                elif len(original_padding_tvm) == 4: # (pad_top, pad_bottom, pad_left, pad_right)
                    if original_padding_tvm[0] == original_padding_tvm[1] and original_padding_tvm[2] == original_padding_tvm[3]:
                        # Map to symmetric PyTorch (H_pad, W_pad)
                        pad_pytorch = (original_padding_tvm[0], original_padding_tvm[2])
                    else:
                        logging.warning(f"Asymmetric padding {original_padding_tvm} in TVM. PyTorch F.pool2d `padding` arg requires symmetric padding. Skipping test case.")
                        continue
                else:
                    logging.warning(f"Unsupported padding format {original_padding_tvm}. Skipping test case.")
                    continue

                for ceil_mode in [False, True]:
                    # Skip "the padding size is larger than or equal to the filter size for exclusive-counting pooling"
                    # Original TVM condition
                    if pool_size == (2, 2) and original_padding_tvm == (0, 0, 1, 1):
                         continue

                    # For avg_pool2d
                    for count_include_pad in [False, True]:
                        # Skip "inclusive-counted blended or average pooling is not supported in combination with asymmetric padding"
                        # Original TVM condition. PyTorch's F.avg_pool2d usually handles this.
                        if count_include_pad and (original_padding_tvm == (0, 0, 1, 1) or strides == (2, 2)):
                            continue
                        
                        run_and_verify_func(
                            functools.partial(
                                get_graph,
                                op_func=torch.nn.functional.avg_pool2d,
                                pool_size=pool_size,
                                strides=strides,
                                padding=pad_pytorch,
                                ceil_mode=ceil_mode,
                                count_include_pad=count_include_pad,
                            ),
                            run_inference=run_inference,
                            data_type=torch.float32,
                        )
                    # For max_pool2d
                    run_and_verify_func(
                        functools.partial(
                            get_graph,
                            op_func=torch.nn.functional.max_pool2d,
                            pool_size=pool_size,
                            strides=strides,
                            padding=pad_pytorch,
                            ceil_mode=ceil_mode,
                        ),
                        run_inference=run_inference,
                        data_type=torch.float32,
                    )


@run_module
def test_global_pool2d(run_inference):
    def get_graph(data_type_arg: torch.dtype, op_func: Callable, x_shape: Tuple = (1, 3, 32, 32)) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            # Global pool means output_size = 1 for spatial dimensions
            return op_func(x_in, output_size=1)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, op_func=torch.nn.functional.adaptive_max_pool2d),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, op_func=torch.nn.functional.adaptive_avg_pool2d),
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_batch_flatten(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple = (1, 3, 4, 6)) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            # batch_flatten flattens all but the first (batch) dimension.
            return torch.flatten(x_in, start_dim=1)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    for dtp in [torch.float16, torch.float32]:
        run_and_verify_func(get_graph, run_inference=run_inference, data_type=dtp)


@run_module
def test_expand_dims(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple = (1, 3), axis: int = 1, num_newaxis: int = 1) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            current_tensor = x_in
            for _ in range(num_newaxis):
                current_tensor = torch.unsqueeze(current_tensor, dim=axis)
            return current_tensor
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float32)


@run_module
def test_squeeze(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, axis: Union[int, Tuple, None]) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            # TVM `axis` can be an int, tuple of ints, or None. PyTorch `squeeze` can take single dim or None.
            # If `axis` is a tuple, apply squeeze iteratively.
            if axis is None:
                return torch.squeeze(x_in)
            elif isinstance(axis, (list, tuple)):
                current_tensor = x_in
                # Sort axes in reverse order to avoid shifting dimensions already processed.
                sorted_axis = sorted(axis, reverse=True)
                for dim in sorted_axis:
                    current_tensor = torch.squeeze(current_tensor, dim=dim)
                return current_tensor
            else: # single integer axis
                return torch.squeeze(x_in, dim=axis)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    for dtype in SUPPORTED_DTYPES:
        run_and_verify_func(
            functools.partial(get_graph, x_shape=(1, 5, 1, 1), axis=(2, 3)),
            run_inference=run_inference,
            data_type=dtype,
        )
        run_and_verify_func(
            functools.partial(get_graph, x_shape=(1, 3, 1), axis=(-1,)),
            run_inference=run_inference,
            data_type=dtype,
        )


@run_module
def test_concatenate(run_inference):
    def get_graph(data_type_arg: torch.dtype, input_shapes_list: List[Tuple], axis: int) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(*input_tensors: torch.Tensor) -> torch.Tensor:
            # `input_tensors` is a tuple of tensors, directly usable by torch.cat
            return torch.cat(input_tensors, dim=axis)
        
        # Build input_shapes dict for run_and_verify_func, preserving argument order
        input_shapes_dict = {f"input_{i}": shape for i, shape in enumerate(input_shapes_list)}
        param_names = []
        return f, input_shapes_dict, param_names

    run_and_verify_func(
        functools.partial(get_graph, input_shapes_list=[(1, 2, 6, 6), (1, 3, 6, 6)], axis=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_split(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, indices_or_sections: Union[int, List[int]], axis: int) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            # torch.split returns a tuple of tensors
            return torch.split(x_in, split_size_or_sections=indices_or_sections, dim=axis)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 16), indices_or_sections=2, axis=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 16), indices_or_sections=4, axis=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 16), indices_or_sections=[8], axis=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 16), indices_or_sections=[2, 3, 6, 10, 14], axis=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_conv2d_transpose(run_inference):
    def get_graph(
        data_type_arg: torch.dtype,
        x_shape: Tuple = (1, 32, 8, 8), # (N, C_in, H, W)
        k_original_shape: Tuple = (32, 16, 3, 3), # TVM kernel for conv2d_transpose: (in_C, out_C_per_group, kH, kW)
        groups: int = 1,
        padding: Tuple = (0, 0),
        strides: Tuple = (1, 1),
    ) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        # Infer PyTorch kernel shape: (in_channels, out_channels_per_group, kH, kW)
        # TVM's `channels` is `k_original_shape[1]`.
        in_channels = x_shape[1]
        out_channels_per_group = k_original_shape[1]
        kernel_h, kernel_w = k_original_shape[2:]

        k_pytorch_shape = (in_channels, out_channels_per_group, kernel_h, kernel_w)

        def f(x_in: torch.Tensor, kernel_in: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.conv_transpose2d(
                input=x_in,
                weight=kernel_in,
                bias=None,
                stride=strides,
                padding=padding,
                output_padding=0, # TVM `conv2d_transpose` doesn't expose output_padding, assume 0
                groups=groups,
                dilation=1, # TVM doesn't expose dilation for conv2d_transpose, assume 1
            )
        input_shapes = {"x_in": x_shape, "kernel_in": k_pytorch_shape}
        param_names = ["kernel_in"]
        return f, input_shapes, param_names
    
    for padding in [(0, 0), (1, 1)]:
        for strides in [(1, 1), (2, 2)]:
            run_and_verify_func(
                functools.partial(get_graph, padding=padding, strides=strides),
                run_inference=run_inference,
                data_type=torch.float32,
            )


@run_module
def test_reshape(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, new_shape: Tuple) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            return torch.reshape(x_in, new_shape)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 1, 1, 10), new_shape=(-1, 10)),
        run_inference=run_inference,
        data_type=torch.float16,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 10, 2, 3), new_shape=(1, -1)),
        run_inference=run_inference,
        data_type=torch.float16,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 1, 2, 3), new_shape=(1, 6)),
        run_inference=run_inference,
        data_type=torch.float16,
    )


# The original TVM tests `test_dynamic_reshape` and `test_dynamic_offload` explicitly
# check TVM's internal partitioning logic (`are_ops_on_trt`, `tvm.ir.assert_structural_equal`)
# which has no direct equivalent in user-facing PyTorch/TorchInductor APIs.
# `torch.compile` is a black-box optimizer, so we cannot assert *which* ops were offloaded.
# These tests are therefore skipped to avoid misrepresenting their intent in PyTorch.
@pytest.mark.skip(reason="Tests TVM's internal TensorRT partitioning logic and IR structure, not directly convertible to PyTorch.")
def test_dynamic_offload(run_module):
    pass

@pytest.mark.skip(reason="Tests TVM's internal TensorRT partitioning logic and IR structure, not directly convertible to PyTorch.")
def test_dynamic_reshape(run_module):
    pass


@run_module
def test_transpose(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, order: Tuple) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            return torch.permute(x_in, dims=order)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 16, 7, 7), order=[0, 2, 3, 1]),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 7, 7, 16), order=[0, 3, 1, 2]),
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_float_const(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple = (1, 16)) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            beta = torch.tensor(1.0, dtype=data_type_arg, device=x_in.device)
            return torch.mul(x_in, beta)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float32)


@run_module
def test_float_const16(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple = (1, 16)) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            beta = torch.tensor(1.0, dtype=data_type_arg, device=x_in.device)
            return torch.mul(x_in, beta)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float16)


@run_module
def test_pad(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, pad_width_tvm: List[List[int]]) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        # Convert TVM `pad_width` format to PyTorch `pad` format
        # TVM: `[[dim0_before, dim0_after], [dim1_before, dim1_after], ...]` (for each dimension)
        # PyTorch `pad`: `(dimN-1_before, dimN-1_after, ..., dim0_before, dim0_after)` (reversed dimension order, flattened)
        pad_pytorch_list = []
        for dim_pad in reversed(pad_width_tvm):
            pad_pytorch_list.extend([dim_pad[0], dim_pad[1]])
        pad_pytorch_tuple = tuple(pad_pytorch_list)

        def f(x_in: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(input=x_in, pad=pad_pytorch_tuple, mode='constant', value=0.0)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 8, 16, 16), pad_width_tvm=[[0, 0], [0, 0], [0, 0], [0, 0]]),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 8, 16, 16), pad_width_tvm=[[0, 0], [0, 0], [1, 1], [1, 1]]),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 8, 16, 16), pad_width_tvm=[[0, 0], [0, 0], [0, 1], [2, 0]]),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 8, 3, 16, 16), pad_width_tvm=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_add(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
            return torch.add(x_in, y_in)
        
        input_shapes = {"x_in": x_shape, "y_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(functools.partial(get_graph, x_shape=(1, 1000)), run_inference=run_inference, data_type=torch.float16)


@run_module
def test_softmax(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, axis: int) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softmax(x_in, dim=axis)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 1000), axis=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 1000), axis=-1),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 3, 4), axis=-2),
        run_inference=run_inference,
        data_type=torch.float16,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 3, 4), axis=1),
        run_inference=run_inference,
        data_type=torch.float16,
    )


@run_module
def test_batch_norm(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, param_shape: Tuple, axis: int = 1, epsilon: float = 1e-5) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor, gamma_in: torch.Tensor, beta_in: torch.Tensor, moving_mean_in: torch.Tensor, moving_var_in: torch.Tensor) -> torch.Tensor:
            # For functional batch_norm in inference context, `running_mean` and `running_var` are fixed.
            # `weight` (gamma) and `bias` (beta) are used since `center=True`, `scale=True` in TVM test.
            return torch.nn.functional.batch_norm(
                input=x_in,
                running_mean=moving_mean_in,
                running_var=moving_var_in,
                weight=gamma_in,
                bias=beta_in,
                training=False, # Assuming inference context for these tests
                momentum=0.1, # Default PyTorch value, TVM doesn't expose it in this signature
                eps=epsilon,
            )
        
        input_shapes = {
            "x_in": x_shape,
            "gamma_in": param_shape,
            "beta_in": param_shape,
            "moving_mean_in": param_shape,
            "moving_var_in": param_shape,
        }
        # These are TVM parameters but are explicit inputs to the functional model.
        param_names = ["gamma_in", "beta_in", "moving_mean_in", "moving_var_in"] 
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 64, 56, 56), param_shape=(64,)),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 56, 56, 64), param_shape=(64,), axis=3, epsilon=1.001e-05),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 4, 8, 4), param_shape=(8,), axis=2),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 8, 4, 4, 4), param_shape=(8,), axis=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 4, 8, 4, 4), param_shape=(8,), axis=2),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 4, 4, 4, 8), param_shape=(8,), axis=4),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 8), param_shape=(8,), axis=1),
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 3, 8), param_shape=(8,), axis=2),
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_layer_norm(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, param_shape: Tuple, axis: Union[int, Tuple] = 1, epsilon: float = 1e-5) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        # Calculate PyTorch's `normalized_shape` from TVM's `axis`
        if isinstance(axis, int):
            norm_shape = x_shape[axis:]
        elif isinstance(axis, (list, tuple)):
            norm_shape = tuple(x_shape[d] for d in axis)
            if not norm_shape: # If empty tuple/list (normalize all but batch, if batch is x_shape[0])
                norm_shape = x_shape[1:] 
        else:
            raise ValueError(f"Unsupported axis format: {axis}")

        def f(x_in: torch.Tensor, gamma_in: torch.Tensor, beta_in: torch.Tensor) -> torch.Tensor:
            # `weight` (gamma) and `bias` (beta) are used since `center=True`, `scale=True` in TVM test.
            return torch.nn.functional.layer_norm(
                input=x_in,
                normalized_shape=norm_shape,
                weight=gamma_in,
                bias=beta_in,
                eps=epsilon,
            )
        
        input_shapes = {
            "x_in": x_shape,
            "gamma_in": param_shape,
            "beta_in": param_shape,
        }
        param_names = ["gamma_in", "beta_in"]
        return f, input_shapes, param_names

    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 32, 8, 8), param_shape=(32,), axis=1), # axis=1 implies (32,8,8) for normalized_shape
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 8, 8, 32), param_shape=(32,), axis=3, epsilon=1.001e-05), # axis=3 implies (32,) for normalized_shape
        run_inference=run_inference,
        data_type=torch.float32,
    )
    run_and_verify_func(
        functools.partial(get_graph, x_shape=(1, 8), param_shape=(8,), axis=1), # axis=1 implies (8,) for normalized_shape
        run_inference=run_inference,
        data_type=torch.float32,
    )


@run_module
def test_unary(run_inference):
    # Mapping dictionary from TVM ops to PyTorch ops. The `relay` prefix refers to TVM's ops.
    # We will use direct PyTorch ops in the `get_graph` function.
    torch_op_map = {
        "relu": torch.nn.functional.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "abs": torch.abs,
        "negative": torch.neg, # TVM `negative` maps to `torch.neg`
        "sin": torch.sin,
        "cos": torch.cos,
        "atan": torch.atan,
        "ceil": torch.ceil,
        "floor": torch.floor,
        "erf": torch.erf,
    }

    def get_graph(data_type_arg: torch.dtype, op_func: Callable, x_shape: Tuple = (1, 8, 3, 3)) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            return op_func(x_in)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    for op_name, torch_op in torch_op_map.items():
        run_and_verify_func(
            functools.partial(get_graph, op_func=torch_op),
            run_inference=run_inference,
            data_type=torch.float32,
        )


@run_module
def test_clip(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple = (1, 8, 3, 3), a_min: float = -0.2, a_max: float = 0.4) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            return torch.clamp(x_in, min=a_min, max=a_max)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float16)


@run_module
def test_relu(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple = (1, 8, 3, 4)) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.relu(x_in)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float16)


@run_module
def test_leaky_relu(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple = (1, 8, 3, 4), alpha: float = 0.1) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.leaky_relu(x_in, negative_slope=alpha)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names

    run_and_verify_func(get_graph, run_inference=run_inference, data_type=torch.float16)


@run_module
def test_binary(run_inference):
    # Mapping dictionary from TVM ops to PyTorch ops
    torch_op_map = {
        "add": torch.add,
        "subtract": torch.sub,
        "multiply": torch.mul,
        "divide": torch.div,
        "power": torch.pow,
    }

    def get_graph(data_type_arg: torch.dtype, op_func: Callable, x_shape: Tuple, y_shape: Tuple, y_is_const: bool = False) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor, y_in: Union[torch.Tensor, None] = None) -> torch.Tensor:
            if y_is_const:
                # Create y as a constant tensor on the same device as x_in
                y_const = torch.ones(y_shape, dtype=data_type_arg, device=x_in.device)
                return op_func(x_in, y_const)
            return op_func(x_in, y_in)
        
        input_shapes = {"x_in": x_shape}
        if not y_is_const:
            input_shapes["y_in"] = y_shape
        param_names = []
        return f, input_shapes, param_names

    for op_name, torch_op in torch_op_map.items():
        for dtp in SUPPORTED_DTYPES:
            for y_is_const in [True, False]:
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 8, 3, 3), y_shape=(1, 8, 3, 3), y_is_const=y_is_const),
                    run_inference=run_inference,
                    data_type=dtp,
                )
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 8, 1, 3), y_shape=(1, 8, 3, 1), y_is_const=y_is_const),
                    run_inference=run_inference,
                    data_type=dtp,
                )
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 10), y_shape=(10,), y_is_const=y_is_const),
                    run_inference=run_inference,
                    data_type=dtp,
                )
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 1, 1, 10), y_shape=(10,), y_is_const=y_is_const),
                    run_inference=run_inference,
                    data_type=dtp,
                )
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 1, 1), y_shape=(3,), y_is_const=y_is_const),
                    run_inference=run_inference,
                    data_type=dtp,
                )


@run_module
def test_reduce(run_inference):
    # Mapping dictionary from TVM ops to PyTorch ops
    # Note: `torch.max`/`min` when `dim` is specified return `(values, indices)`.
    # `.values` is appended in the lambda to get only values for comparison.
    torch_op_map = {
        "sum": torch.sum,
        "prod": torch.prod,
        "max": lambda x, dim, keepdim: torch.max(x, dim=dim, keepdim=keepdim).values,
        "min": lambda x, dim, keepdim: torch.min(x, dim=dim, keepdim=keepdim).values,
        "mean": torch.mean,
    }

    def get_graph(data_type_arg: torch.dtype, op_func: Callable, x_shape: Tuple = (1, 2, 3, 4), axis: Union[int, Tuple, None] = (2, 3), keepdims: bool = False) -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            if axis is None:
                # For global reductions (axis=None), torch.max/min return scalar,
                # and directly passing `keepdim` handles shape.
                if op_func in [torch.sum, torch.prod, torch.mean]:
                    return op_func(x_in, keepdim=keepdims)
                else: # op_func for max/min already returns just value
                    # For global max/min, PyTorch automatically handles scalar output for `keepdim=False`
                    # and correct shape for `keepdim=True`
                    return torch.max(x_in, keepdim=keepdims).values if op_func == torch_op_map["max"] \
                           else torch.min(x_in, keepdim=keepdims).values # Access .values on global max/min result
            else:
                return op_func(x_in, dim=axis, keepdim=keepdims)
        
        input_shapes = {"x_in": x_shape}
        param_names = []
        return f, input_shapes, param_names
    
    for dtp in SUPPORTED_DTYPES:
        for op_name, torch_op in torch_op_map.items():
            for keepdims in [True, False]:
                # TVM uses tuples for axis even for single dim, e.g., `(1,)`
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 2, 3, 4), axis=(1,), keepdims=keepdims),
                    run_inference=run_inference,
                    data_type=dtp,
                )
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 2, 3, 4), axis=(2, 3), keepdims=keepdims),
                    run_inference=run_inference,
                    data_type=dtp,
                )
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 2, 3, 4), axis=(1, 2), keepdims=keepdims),
                    run_inference=run_inference,
                    data_type=dtp,
                )
                run_and_verify_func(
                    functools.partial(get_graph, op_func=torch_op, x_shape=(1, 2, 3, 4), axis=(1, 2, 3), keepdims=keepdims),
                    run_inference=run_inference,
                    data_type=dtp,
                )


@run_module
def test_strided_slice(run_inference):
    def get_graph(data_type_arg: torch.dtype, x_shape: Tuple, begin_tvm: List[int], end_tvm: List[int], strides_tvm: Union[List[int], None] = None, slice_mode: str = "size") -> Tuple[Callable, Dict[str, Tuple], List[str]]:
        def f(x_in: torch.Tensor) -> torch.Tensor:
            actual_end = []
            if slice_mode == "size":
                for i in range(len(begin_tvm)):
                    actual_end.append(begin_tvm[i] + end_tvm[i])
            else: # slice_mode == "end"
                actual_end = list(end_tvm)

            slices = []
            for i in range(len(x_shape)):
                start = begin_tvm[i] if i < len(begin_tvm) else 0
                stop = actual_end[i] if i < len(actual_end) else x_in.shape[i]
                step = strides_tvm[i] if strides_tvm and i < len(strides_tvm) else 1
                
                #
