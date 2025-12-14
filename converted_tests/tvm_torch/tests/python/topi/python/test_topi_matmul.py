import numpy as np
import torch
import torch.testing
import functools # For composite mappings that might use reduce

# A helper function that replaces the TVM compilation and execution boilerplate
# with direct PyTorch tensor operations.
def with_torch(lam, *args):
    """Take numpy arrays as args, convert them to PyTorch tensors and call `lam`.
    Result of lambda is converted back to numpy array and returned.
    """
    # Convert numpy arrays to torch tensors.
    # Assuming float32 based on the original TVM test's data generation.
    torch_args = [torch.tensor(arg, dtype=torch.float32) for arg in args]

    # Call the lambda with PyTorch tensors
    out = lam(*torch_args)

    # Convert result back to numpy array and return.
    # If `out` is a scalar tensor, use .item()
    if isinstance(out, torch.Tensor):
        return out.cpu().numpy()
    # If the lambda returns multiple outputs (e.g., tuple of tensors),
    # this helper assumes a single tensor output for simplicity,
    # consistent with matmul/tensordot. Adjust if needed for other ops.
    return out


# Helper for tensordot axes conversion as PyTorch's dims argument can be specific.
def convert_tensordot_axes(axes_tvm):
    if isinstance(axes_tvm, int):
        # PyTorch `dims` can take an int directly for common cases
        return axes_tvm
    if isinstance(axes_tvm, (tuple, list)) and len(axes_tvm) == 2:
        # If TVM's axes is (int, int), convert to ([int], [int]) for PyTorch
        if isinstance(axes_tvm[0], int) and isinstance(axes_tvm[1], int):
            return ([axes_tvm[0]], [axes_tvm[1]])
        # If already (list, list), return as is
        if isinstance(axes_tvm[0], (tuple, list)) and isinstance(axes_tvm[1], (tuple, list)):
            return axes_tvm
    # Fallback for unexpected or direct pass-through
    return axes_tvm


def verify_nn_matmul(sa, sb, transp_a, transp_b):
    a = np.random.uniform(low=-1.0, high=1.0, size=sa).astype(np.float32)
    b = np.random.uniform(low=-1.0, high=1.0, size=sb).astype(np.float32)
    c1 = np.matmul(np.transpose(a) if transp_a else a, np.transpose(b) if transp_b else b)
    
    # TVM: topi.nn.matmul(A, B, transpose_a=transp_a, transpose_b=transp_b)
    # PyTorch: apply transpose using .mT before torch.matmul
    c2 = with_torch(
        lambda A_torch, B_torch: torch.matmul(
            A_torch.mT if transp_a else A_torch,
            B_torch.mT if transp_b else B_torch
        ),
        a,
        b,
    )
    torch.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_nn_matmul():
    verify_nn_matmul((1, 1), (1, 1), False, False)
    verify_nn_matmul((1, 1), (1, 1), True, True)
    verify_nn_matmul((2, 2), (2, 2), False, False)
    verify_nn_matmul((2, 2), (2, 2), True, True)
    verify_nn_matmul((2, 3), (3, 5), False, False)
    verify_nn_matmul((5, 3), (3, 2), False, False)
    verify_nn_matmul((3, 5), (3, 2), True, False)
    verify_nn_matmul((3, 5), (2, 3), True, True)
    verify_nn_matmul((3, 5), (3, 2), True, False) # Duplicate test case, keep for fidelity
    verify_nn_matmul((5, 3), (2, 3), False, True)


def verify_matmul(sa, sb, transp_a, transp_b):
    a = np.random.uniform(low=-1.0, high=1.0, size=sa).astype(np.float32)
    b = np.random.uniform(low=-1.0, high=1.0, size=sb).astype(np.float32)
    c1 = np.matmul(np.transpose(a) if transp_a else a, np.transpose(b) if transp_b else b)
    
    # TVM: topi.matmul(A, B, transp_a, transp_b)
    # PyTorch: apply transpose using .mT before torch.matmul
    c2 = with_torch(lambda A_torch, B_torch: torch.matmul(
            A_torch.mT if transp_a else A_torch,
            B_torch.mT if transp_b else B_torch
        ),
        a,
        b,
    )
    torch.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_matmul():
    verify_matmul((1, 1), (1, 1), False, False)
    verify_matmul((1, 1), (1, 1), True, True)
    verify_matmul((2, 2), (2, 2), False, False)
    verify_matmul((2, 2), (2, 2), True, True)
    verify_matmul((2, 3), (3, 5), False, False)
    verify_matmul((5, 3), (3, 2), False, False)
    verify_matmul((3, 5), (3, 2), True, False)
    verify_matmul((3, 5), (2, 3), True, True)


def verify_tensordot(sa, sb, axes):
    a = np.random.uniform(low=-1.0, high=1.0, size=sa).astype(np.float32)
    b = np.random.uniform(low=-1.0, high=1.0, size=sb).astype(np.float32)
    c1 = np.tensordot(a, b, axes)
    
    # TVM: topi.tensordot(A, B, axes)
    # PyTorch: torch.tensordot(A, B, dims=convert_tensordot_axes(axes))
    c2 = with_torch(lambda A_torch, B_torch: torch.tensordot(A_torch, B_torch, dims=convert_tensordot_axes(axes)), a, b)
    torch.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_tensordot():
    verify_tensordot((3,), (3,), 0) # Note: numpy often flattens 1D for scalar product, PyTorch requires ([0],[0]) for 0-axis tensordot
    verify_tensordot((2, 3), (3, 5), 1)
    verify_tensordot((2, 2, 3), (2, 3, 5), 2)
    verify_tensordot((2, 2, 3, 4), (2, 3, 4, 5), 3)
    verify_tensordot((3, 2, 2), (2, 3, 5), (1, 0)) # tuple of 2 ints -> converted to ([1],[0])
    verify_tensordot((3, 2, 2), (2, 3, 5), ((1, 0), (0, 1)))
    verify_tensordot((4, 3, 2, 2), (2, 4, 3, 5), ((1, 2, 0), (2, 0, 1)))


if __name__ == "__main__":
    test_nn_matmul()
    test_matmul()
    test_tensordot()
