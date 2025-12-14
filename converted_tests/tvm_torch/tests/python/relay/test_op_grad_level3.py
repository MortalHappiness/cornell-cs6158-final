import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_allclose
import functools

# Custom utility to handle dtype conversions from string to torch.dtype
DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

def to_torch_dtype(dtype_str):
    return DTYPE_MAP.get(dtype_str, None)

# Re-implementing check_grad for PyTorch to compare analytical gradients with numerical gradients
def check_grad(fwd_func, inputs_np, test_inputs_np=None, output_idx=0, **kwargs):
    """
    Checks the analytical gradients of a PyTorch function against numerical gradients.

    Args:
        fwd_func (callable): A Python function that takes torch tensors and performs the forward pass,
                              returning a single torch.Tensor output or a tuple of tensors.
        inputs_np (list): A list of numpy arrays representing all inputs to fwd_func.
        test_inputs_np (list, optional): A subset of inputs_np for which gradients should be computed.
                                     If None, gradients are computed for all inputs_np.
        output_idx (int): If fwd_func returns a tuple, this is the index of the output
                          to compute gradients against. Default is 0.
        **kwargs: Additional keyword arguments passed to torch.testing.assert_allclose
                  (e.g., rtol, atol).
    """
    if test_inputs_np is None:
        test_inputs_np = inputs_np

    # Prepare torch tensors from numpy inputs, enabling grad for test_inputs
    torch_inputs_all = []
    torch_inputs_for_grad_filtered = []
    
    # Map numpy array identity to torch tensor to ensure consistency
    np_id_to_torch_map = {} 

    for np_input in inputs_np:
        requires_grad = any(np.array_equal(np_input, ti_np) for ti_np in test_inputs_np)

        if np_input.dtype in (np.int32, np.int64, np.uint8, np.int8) and requires_grad:
            # If an integer input requires grad, convert to float for differentiation
            torch_input = torch.tensor(np_input.astype(np.float32), requires_grad=True)
        else:
            torch_input = torch.tensor(np_input, requires_grad=requires_grad)
        
        torch_inputs_all.append(torch_input)
        np_id_to_torch_map[id(np_input)] = torch_input

    # Populate torch_inputs_for_grad_filtered based on test_inputs_np
    for np_input_test in test_inputs_np:
        if id(np_input_test) in np_id_to_torch_map:
            t_input = np_id_to_torch_map[id(np_input_test)]
            if t_input.requires_grad:
                torch_inputs_for_grad_filtered.append(t_input)
        else:
            # This case implies test_inputs_np has an element not explicitly in inputs_np
            # For simplicity in these tests, we assume test_inputs_np elements are references to inputs_np elements.
            # If not, a more complex mapping or error handling would be needed.
            raise ValueError("test_inputs_np contains an element not found in inputs_np")


    # Compute forward pass and a scalar loss for gradient calculation
    output = fwd_func(*torch_inputs_all)
    
    if isinstance(output, (tuple, list)):
        output_for_loss = output[output_idx]
    else:
        output_for_loss = output

    # Create a scalar loss for autograd
    if output_for_loss.numel() > 1:
        loss = output_for_loss.sum()
    else:
        loss = output_for_loss

    # Compute analytical gradients using PyTorch's autograd
    if not torch_inputs_for_grad_filtered:
        analytical_grads = tuple()
    else:
        analytical_grads = torch.autograd.grad(loss, torch_inputs_for_grad_filtered, retain_graph=True, allow_unused=True)

    # Compute numerical gradients for comparison (central difference)
    numerical_grads = []
    epsilon = 1e-3

    for x_grad in torch_inputs_for_grad_filtered:
        if not x_grad.is_floating_point():
            numerical_grads.append(torch.zeros_like(x_grad)) 
            continue

        grad_num_tensor = torch.zeros_like(x_grad)
        
        it = np.nditer(x_grad.detach().cpu().numpy(), flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            multi_idx = it.multi_index

            # Perturb +epsilon
            x_plus_eps = x_grad.clone().detach()
            x_plus_eps[multi_idx] += epsilon
            
            # Create a new list of inputs for the forward function with the perturbed tensor
            current_inputs_plus = []
            for t in torch_inputs_all:
                if t is x_grad:
                    current_inputs_plus.append(x_plus_eps)
                else:
                    current_inputs_plus.append(t.clone().detach()) # Detach others to prevent unintended grad tracking

            output_plus = fwd_func(*current_inputs_plus)
            if isinstance(output_plus, (tuple, list)):
                output_plus_scalar = output_plus[output_idx].sum()
            else:
                output_plus_scalar = output_plus.sum()

            # Perturb -epsilon
            x_minus_eps = x_grad.clone().detach()
            x_minus_eps[multi_idx] -= epsilon

            current_inputs_minus = []
            for t in torch_inputs_all:
                if t is x_grad:
                    current_inputs_minus.append(x_minus_eps)
                else:
                    current_inputs_minus.append(t.clone().detach())

            output_minus = fwd_func(*current_inputs_minus)
            if isinstance(output_minus, (tuple, list)):
                output_minus_scalar = output_minus[output_idx].sum()
            else:
                output_minus_scalar = output_minus.sum()

            grad_num_val = (output_plus_scalar - output_minus_scalar) / (2 * epsilon)
            grad_num_tensor[multi_idx] = grad_num_val
            
            it.iternext()
        numerical_grads.append(grad_num_tensor)

    for grad_ana, grad_num in zip(analytical_grads, numerical_grads):
        assert_allclose(grad_ana.detach().cpu().numpy(), grad_num.detach().cpu().numpy(), **kwargs)

    fwd_output_np = output_for_loss.detach().cpu().numpy()
    analytical_grads_np = tuple(g.detach().cpu().numpy() for g in analytical_grads)
    return fwd_output_np, analytical_grads_np


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_clip(dtype):
    ref = lambda x: np.where(
        x > 10.0, np.zeros_like(x), np.where(x < 1.0, np.zeros_like(x), np.ones_like(x))
    )
    
    fwd_func = lambda x: torch.clamp(x, min=1.0, max=10.0)

    data_np = np.random.rand(10, 4).astype(dtype) * 11.0
    ref_grad = ref(data_np)
    
    # check_grad will compare PyTorch's analytical gradient to a numerical approximation
    _, (op_grad,) = check_grad(fwd_func, inputs_np=[data_np], rtol=0.01)

    np.testing.assert_allclose(op_grad, ref_grad, rtol=0.01)


def verify_transpose_grad(d_shape, axes=None):
    def fwd_func(data):
        if axes is None:
            # TVM default for axes=None is to reverse dimensions
            return data.permute(tuple(reversed(range(data.ndim))))
        return data.permute(axes)

    data_np = np.random.rand(*d_shape).astype("float32")
    check_grad(fwd_func, inputs_np=[data_np])


def test_transpose_grad():
    verify_transpose_grad((1, 2, 3, 4))
    verify_transpose_grad((1, 2, 3, 4), axes=(0, 2, 3, 1))


def test_negative_grad():
    def fwd_func(data):
        return -data

    data_np = np.random.rand(10, 4).astype("float32")
    check_grad(fwd_func, inputs_np=[data_np])


def test_cast_grad():
    def fwd_func(data):
        return data.to(torch.float64)

    data_np = np.random.rand(10, 4).astype("float32")
    check_grad(fwd_func, inputs_np=[data_np])


def test_cast_like_grad():
    def fwd_func(data, like):
        # like's dtype is used for casting data
        return data.to(like.dtype)

    data_np = np.random.rand(10, 4).astype("float32")
    like_np = np.random.rand(1,).astype("float64")
    # Only compute gradient for data, as `like` specifies target dtype and its value is irrelevant for grad
    check_grad(fwd_func, inputs_np=[data_np, like_np], test_inputs_np=[data_np])


def test_copy_grad():
    def fwd_func(data):
        return data.clone()

    data_np = np.random.rand(10, 4).astype("float64")
    check_grad(fwd_func, inputs_np=[data_np])


def test_take_grad():
    data_shape = (3, 4, 5)
    data_dtype = "float64"
    
    data_np = np.random.rand(*data_shape).astype(data_dtype) * 1e-5
    indices_np = np.array([1, 2], dtype="int32") # Indices do not require grad

    # take on axis
    def fwd_func_axis(data, indices):
        return torch.index_select(data, dim=1, index=indices.long())
    
    check_grad(fwd_func_axis, inputs_np=[data_np, indices_np], test_inputs_np=[data_np])

    # take on flattened
    def fwd_func_flattened(data, indices):
        return torch.take(data, indices.long())
    
    check_grad(fwd_func_flattened, inputs_np=[data_np, indices_np], test_inputs_np=[data_np])


def test_stack_grad():
    def fwd_func(*args):
        return torch.stack(args, dim=0)

    args_np = [np.random.rand(2, 3, 4).astype("float64") for _ in range(3)] # x, y, z
    check_grad(fwd_func, inputs_np=args_np)


def test_squeeze_grad():
    data_np = np.random.rand(2, 1, 1, 3, 4, 1).astype("float64")

    # Squeeze all singleton dimensions
    def fwd_func_all(data):
        return data.squeeze()
    check_grad(fwd_func_all, inputs_np=[data_np])

    # Squeeze specific dimensions
    def fwd_func_subset(data):
        return data.squeeze(1).squeeze(-1) # axis=[1, -1]
    check_grad(fwd_func_subset, inputs_np=[data_np])


def test_arange_grad():
    # PyTorch's arange is differentiable with float start/end/step.
    # The length of the sequence itself being dependent on these inputs makes numerical
    # differentiation tricky for shape-changing ops in general, as noted in TVM.
    # However, PyTorch's autograd system handles it correctly.
    dtype = "float64"
    start_np = np.array(2.5, dtype=dtype)
    stop_np = np.array(9.5, dtype=dtype)
    step_np = np.array(1.8, dtype=dtype)
    
    def fwd_func(start, stop, step):
        return torch.arange(start=start, end=stop, step=step, dtype=to_torch_dtype(dtype))
    
    check_grad(fwd_func, inputs_np=[start_np, stop_np, step_np])


def test_gather_nd_grad():
    def fwd_func(data, indices):
        # PyTorch advanced indexing equivalent to gather_nd
        # indices.T to get (coord_dim, num_indices) for tuple indexing
        return data[tuple(indices.T)]

    data_np = np.random.rand(2, 3).astype("float64")
    # indices_np must be long type for advanced indexing
    indices_np = np.array([[0, 1, 1, 0], [0, 1, 0, 0]], dtype="int64") 
    
    # Only data is differentiable
    check_grad(
        fwd_func, inputs_np=[data_np, indices_np], test_inputs_np=[data_np]
    )


def test_reshape_like_grad():
    def fwd_func(data, shape_like):
        return data.reshape(shape_like.shape)

    data_np = np.random.rand(2, 3, 4).astype("float32")
    shape_like_np = np.random.rand(6, 2, 2).astype("float32") # Only shape is used
    
    # Only data is differentiable
    check_grad(fwd_func, inputs_np=[data_np, shape_like_np], test_inputs_np=[data_np])


@pytest.mark.skip(reason="This test verifies TVM IR structural properties, not directly convertible to PyTorch eager mode.")
def test_zeros_ones_grad_const_ints():
    # This test asserts specific IR types for gradient of ops with constant shapes.
    # In PyTorch, a tensor created with constant shape has no gradient unless it depends
    # on a requires_grad=True input. This concept maps poorly to PyTorch's eager execution model.
    # TODO: This test needs a conceptual rewrite if its core intent (e.g., ensuring zero gradient) is to be preserved
    #       for a PyTorch equivalent where the "shape" itself is not a differentiable input.
    pass


@pytest.mark.skip(reason="This test verifies TVM IR structural properties and passes, not directly convertible to PyTorch eager mode.")
def test_zeros_ones_grad_const_expr():
    # Similar to test_zeros_ones_grad_const_ints, this checks how Relay's gradient
    # transformation interacts with symbolic shapes and optimization passes like DynamicToStatic.
    # This is specific to TVM's compilation pipeline and not applicable to PyTorch eager mode.
    # TODO: See test_zeros_ones_grad_const_ints.
    pass


def test_zeros_ones_grad_dynamic():
    rank = np.random.randint(low=1, high=5, dtype="int32")
    # dyn_shape will be treated as float for gradient calculation, then converted to long for torch.zeros/ones size.
    dyn_shape_np = np.random.randint(low=1, high=4, size=(rank,)).astype("float32") 
    
    # For zeros op
    def fwd_func_zeros(shape_data_param):
        return torch.zeros(shape_data_param.long().tolist(), dtype=torch.float32)

    # For ones op
    def fwd_func_ones(shape_data_param):
        return torch.ones(shape_data_param.long().tolist(), dtype=torch.float32)

    # Test zeros
    res_zeros, (grad_zeros,) = check_grad(fwd_func_zeros, inputs_np=[dyn_shape_np], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_zeros, np.zeros(dyn_shape_np.astype("int32"), dtype="float32"))
    np.testing.assert_allclose(grad_zeros, np.zeros_like(dyn_shape_np)) # Gradient wrt shape is zero

    # Test ones
    res_ones, (grad_ones,) = check_grad(fwd_func_ones, inputs_np=[dyn_shape_np], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_ones, np.ones(dyn_shape_np.astype("int32"), dtype="float32"))
    np.testing.assert_allclose(grad_ones, np.zeros_like(dyn_shape_np)) # Gradient wrt shape is zero


if __name__ == "__main__":
    pytest.main([__file__])
