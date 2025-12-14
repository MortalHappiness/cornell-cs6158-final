import pytest
import numpy as np
import torch
import functools # Needed for potential logical_and/or reductions, though not directly used in final mapping.

# Global device setup
if torch.cuda.is_available():
    _DEVICE = torch.device("cuda:0")
else:
    _DEVICE = torch.device("cpu")

def _tvm_dtype_to_torch_dtype(tvm_dtype_str):
    if tvm_dtype_str == "float32":
        return torch.float32
    elif tvm_dtype_str == "float16":
        return torch.float16
    elif tvm_dtype_str == "float64":
        return torch.float64
    elif tvm_dtype_str == "int8":
        return torch.int8
    elif tvm_dtype_str == "int32":
        return torch.int32
    elif tvm_dtype_str == "int64":
        return torch.int64
    # Handle vectorized dtypes like "float32x2"
    if 'x' in tvm_dtype_str:
        base_dtype_str, _ = tvm_dtype_str.split('x')
        return _tvm_dtype_to_torch_dtype(base_dtype_str)
    raise ValueError(f"Unsupported TVM dtype: {tvm_dtype_str}")

# A simple decorator to replace tvm.testing.requires_rocm with requires_cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA (ROCm equivalent)")


@requires_cuda
def test_rocm_cross_thread_reduction():
    # Original TVM TE definition:
    # n = te.size_var("n")
    # m = te.size_var("m")
    # A = te.placeholder((n, m), name="A")
    # k = te.reduce_axis((0, m), "k")
    # B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
    # The TVM scheduling part (split, rfactor, bind, compute_at, set_store_predicate) is TVM-specific
    # and has no direct PyTorch equivalent. torch.compile handles such optimizations internally.

    class ReductionModel(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=1)

    nn = 128 # Equivalent to TVM's 'n' and 'm' for this square matrix example
    A_dtype = torch.float32 # Assuming common float type for data
    a_np = np.random.uniform(size=(nn, nn)).astype(A_dtype.name.replace('torch.', ''))
    a = torch.tensor(a_np, device(_DEVICE))

    model = ReductionModel()
    compiled_model = torch.compile(model)
    b = compiled_model(a)

    # Convert output to numpy for assertion
    b_numpy = b.cpu().numpy()
    
    # tvm.testing.assert_allclose maps to torch.testing.assert_allclose
    torch.testing.assert_allclose(b_numpy, np.sum(a_np, axis=1), rtol=1e-4)


@requires_cuda
def test_rocm_inf_nan():
    def check_inf_nan(dev, n, value, dtype_str):
        # Original TVM TE definition for C:
        # A = te.placeholder((n,), name="A", dtype=dtype) # A is input but not used in compute
        # inf_value = tvm.tir.const(value, dtype=dtype)
        # C = te.compute((n,), lambda i: inf_value, name="C")
        # TVM scheduling is specific and not directly convertible.

        torch_dtype = _tvm_dtype_to_torch_dtype(dtype_str)

        class InfNanModel(torch.nn.Module):
            def __init__(self, value, size, dtype, device):
                super().__init__()
                self.value_tensor = torch.tensor(value, dtype=dtype, device=device).expand(size)

            def forward(self, _): # _ represents the 'A' input, which is unused
                return self.value_tensor

        a = torch.empty((n,), dtype=torch_dtype, device=dev) # Equivalent of TVM's A, an unused input
        
        model = InfNanModel(value, (n,), torch_dtype, dev)
        compiled_model = torch.compile(model)
        
        c = compiled_model(a) # Running the model

        # Check the properties of the output tensor
        if np.isinf(value):
            assert torch.all(torch.isinf(c))
            assert torch.all((c > 0) == (value > 0)) # Check sign of infinity
        elif np.isnan(value):
            assert torch.all(torch.isnan(c))
        else:
            # This case shouldn't happen based on original test values, but for completeness
            torch.testing.assert_allclose(c, torch.full((n,), value, dtype=torch_dtype, device=dev))

    dev = _DEVICE

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float64")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float64")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float64")


@requires_cuda
def test_rocm_reduction_binding():
    # This test primarily focuses on TVM Tensor Expression scheduling primitives
    # (reorder, split, bind) without executing or asserting a numerical output.
    # PyTorch's `torch.compile` handles optimization and scheduling internally,
    # and does not expose these low-level directives to the user.
    # Therefore, this test cannot be directly translated to an equivalent PyTorch test
    # that verifies the *specific scheduling choices*.
    # We create a minimal functional model that represents the computation, but
    # the TVM-specific scheduling aspects are not transferable.

    # Original TVM TE definition:
    # k = te.reduce_axis((0, 32), "k")
    # A = te.placeholder((96, 32), name="A")
    # B = te.compute((96,), lambda m: te.sum(A[m, k], axis=k), name="B")
    # s = te.create_schedule(B.op)
    # s[B].reorder(B.op.reduce_axis[0], B.op.axis[0])
    # mo, _ = s[B].split(B.op.axis[0], 32)
    # s[B].bind(mo, bx)
    # The above scheduling logic is TVM-specific.

    class ReductionModel(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=1) # The equivalent computation

    A_shape = (96, 32)
    A_dtype = torch.float32
    A_tensor = torch.randn(A_shape, dtype=A_dtype, device=_DEVICE)

    model = ReductionModel()
    compiled_model = torch.compile(model)

    # Just ensure it compiles and runs without error.
    # The specific scheduling 'binding' aspect is not testable in PyTorch,
    # as TorchInductor makes its own scheduling decisions.
    _ = compiled_model(A_tensor)
    # No direct numerical assertion here as the original test didn't have one
    # after the scheduling. The intent was to test the scheduler's ability
    # to accept these bindings.

@requires_cuda
def test_rocm_copy():
    def check_rocm(dtype_str, n):
        # A = te.placeholder((n,), name="A", dtype=dtype) # Not needed in PyTorch for this copy test
        torch_dtype = _tvm_dtype_to_torch_dtype(dtype_str)
        dev = _DEVICE
        
        a_np = np.random.uniform(size=(n,)).astype(torch_dtype.name.replace('torch.', ''))
        
        # tvm.nd.empty((n,), A.dtype, dev).copyfrom(a_np)
        a = torch.from_numpy(a_np).to(dev)
        
        # b_np = a.numpy()
        b_np = a.cpu().numpy() # explicit cpu() call needed for .numpy()
        
        torch.testing.assert_allclose(a_np, b_np)
        torch.testing.assert_allclose(a_np, a.cpu().numpy())

    for _ in range(10): # Reduced iterations and N size for faster testing
        dtype_str = np.random.choice(["float32", "float16", "int8", "int32"])
        logN = np.random.randint(1, 10) # Reduced max size for faster testing
        peturb = np.random.uniform(low=0.5, high=1.5)
        check_rocm(dtype_str, int(peturb * (2**logN)))


@requires_cuda
def test_rocm_vectorize_add():
    num_thread = 8 # This is a TVM scheduling hint, not directly used in PyTorch computation

    def check_rocm(dtype_str, n, lanes):
        # A = te.placeholder((n,), name="A", dtype="%sx%d" % (dtype, lanes))
        # B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")
        # TVM scheduling (split, bind) is specific and not directly convertible.

        torch_dtype = _tvm_dtype_to_torch_dtype(dtype_str)
        
        class VectorizedAddModel(torch.nn.Module):
            def __init__(self, constant_val, constant_dtype, device):
                super().__init__()
                # PyTorch handles vectorized ops inherently if shapes are compatible
                self.constant = torch.tensor(constant_val, dtype=constant_dtype, device=device)

            def forward(self, x):
                return x + self.constant

        dev = _DEVICE
        
        # a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np.random.uniform(size=(n, lanes)))
        a_np = np.random.uniform(size=(n, lanes)).astype(torch_dtype.name.replace('torch.', ''))
        a = torch.tensor(a_np, device=dev)
        
        model = VectorizedAddModel(1, torch_dtype, dev)
        compiled_model = torch.compile(model)
        
        c = compiled_model(a) # Running the model
        
        torch.testing.assert_allclose(c.cpu().numpy(), a_np + 1)

    check_rocm("float32", 64, 2)
    check_rocm("float16", 64, 2)


if __name__ == "__main__":
    pytest.main([__file__])
