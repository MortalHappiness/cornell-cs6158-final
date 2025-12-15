import unittest
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm.runtime import ndarray as tvm_ndarray

# Assume these environment variables or pytest fixtures for TVM setup
# For standalone runnability, we'll use placeholder values and check actual device existence.
# In a real TVM testing setup, these would typically come from pytest fixtures.

HAS_CUDA = tvm.cuda().exist
HAS_NUMPY = True  # numpy is always available in this context
HAS_MULTIGPU = HAS_CUDA and tvm.cuda().max_num_device() > 1

# TVM does not have a direct "NUMBA_CUDA" integration similar to PyTorch's.
# However, Numba DeviceNDArray implements DLPack, so we can test TVM's DLPack
# interoperability with Numba's GPU arrays.
# We will consider `HAS_NUMBA_CUDA` as `HAS_CUDA` for enabling these tests,
# assuming Numba CUDA is installed and works if CUDA is available.
HAS_NUMBA_CUDA = HAS_CUDA

if HAS_NUMBA_CUDA:
    try:
        import numba.cuda
        _HAS_NUMBA_CUDA_ACTUAL = True
    except ImportError:
        _HAS_NUMBA_CUDA_ACTUAL = False
else:
    _HAS_NUMBA_CUDA_ACTUAL = False

# Helper function to convert numpy dtypes to TVM dtypes string representation
def _np_dtype_to_tvm_dtype(np_dtype):
    return str(np_dtype)


class TestNumbaIntegration(unittest.TestCase):

    @pytest.mark.skipif(not HAS_NUMPY, reason="No numpy")
    @pytest.mark.skipif(not HAS_CUDA, reason="No cuda")
    def test_cuda_array_interface(self):
        """PyTorch tests for `__cuda_array_interface__` which is specific to PyTorch's
        internal mechanism for Numba integration. TVM NDArray implements `__dlpack__`
        for interoperability. This test cannot be directly translated to TVM.
        """
        self.skipTest("PyTorch-specific __cuda_array_interface__ test not applicable to TVM NDArray.")


    @pytest.mark.skipif(not HAS_CUDA, reason="No cuda")
    @pytest.mark.skipif(not _HAS_NUMBA_CUDA_ACTUAL, reason="No numba.cuda")
    def test_array_adaptor(self):
        """Test TVM NDArray interoperability with numba.cuda via DLPack."""

        tvm_dev = tvm.cuda(0)

        # NumPy dtypes for Numba and TVM consistency
        np_dtypes = [
            np.complex64,
            np.complex128,
            np.float16,
            np.float32,
            np.float64,
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.uint32,
            np.int32,
            np.uint64,
            np.int64,
            np.bool_,
        ]

        for np_dt in np_dtypes:
            tvm_dt = _np_dtype_to_tvm_dtype(np_dt)

            # CPU NumPy arrays do not register as Numba CUDA arrays.
            cput_np = np.arange(10).astype(np_dt)
            self.assertFalse(numba.cuda.is_cuda_array(cput_np))
            with self.assertRaises(TypeError):
                numba.cuda.as_cuda_array(cput_np)

            # Numba CUDA DeviceNDArray:
            # Create a Numba CUDA DeviceNDArray directly from a NumPy array.
            numba_cudat = numba.cuda.to_device(cput_np)
            self.assertTrue(numba.cuda.is_cuda_array(numba_cudat))

            # TVM can wrap this Numba CUDA array via DLPack (zero-copy).
            tvm_cudat = tvm_ndarray.from_dlpack(numba_cudat)

            # Check properties (dtype, shape)
            self.assertEqual(tvm_cudat.dtype, tvm_dt)
            self.assertEqual(tvm_cudat.shape, numba_cudat.shape)

            # Strides: Numba reports strides in bytes. TVM NDArray's `is_contiguous()` checks if it's C-contiguous.
            # DLPack conversion should preserve contiguity.
            self.assertTrue(numba_cudat.is_c_contiguous())
            self.assertTrue(tvm_cudat.is_contiguous())
            # For `numba_cudat.strides` (bytes), we compare against expected NumPy byte strides.
            if tvm_cudat.ndim > 0:
                expected_numba_strides = tuple(cput_np.strides)
                self.assertEqual(numba_cudat.strides, expected_numba_strides)
            else: # Scalar case
                self.assertEqual(numba_cudat.strides, ())

            # The data is identical in the shared view.
            tvm.testing.assert_allclose(tvm_cudat.numpy(), np.asarray(numba_cudat))

            # Writes to the TVM NDArray should be reflected in the Numba array.
            new_val = np.array(11, dtype=np_dt) if np_dt != np.bool_ else np.bool_(1)
            if tvm_cudat.size > 0:
                # Modify part of the TVM array in-place.
                # Assuming `tvm_cudat` is a view and allows direct mutation for DLPack sharing test.
                # Direct slice assignment with copyfrom is one way.
                slice_len = min(5, tvm_cudat.size)
                if slice_len > 0:
                    new_slice_data = tvm_ndarray.array(np.full(slice_len, new_val, dtype=np_dt), device=tvm_dev)
                    tvm_cudat.copyfrom(np.concatenate([new_slice_data.numpy(), tvm_cudat.numpy()[slice_len:]]))

            tvm.cuda(0).sync() # Synchronize TVM's CUDA stream
            numba.cuda.synchronize() # Synchronize Numba's CUDA stream
            tvm.testing.assert_allclose(tvm_cudat.numpy(), np.asarray(numba_cudat))

            # Strided arrays are supported.
            # Create a strided NumPy array, convert to Numba, then to TVM via DLPack.
            strided_cput_np = cput_np[::2]
            strided_numba_cudat = numba.cuda.to_device(strided_cput_np)
            strided_tvm_cudat = tvm_ndarray.from_dlpack(strided_numba_cudat)

            self.assertEqual(strided_tvm_cudat.dtype, tvm_dt)
            self.assertEqual(strided_tvm_cudat.shape, strided_cput_np.shape)

            # Compare Numba strides (bytes) to NumPy's strides
            expected_numba_strides_strided = tuple(strided_cput_np.strides)
            self.assertEqual(strided_numba_cudat.strides, expected_numba_strides_strided)

            # The data is identical in the shared view.
            tvm.testing.assert_allclose(strided_tvm_cudat.numpy(), np.asarray(strided_numba_cudat))


    @pytest.mark.skipif(not HAS_CUDA, reason="No cuda")
    @pytest.mark.skipif(not _HAS_NUMBA_CUDA_ACTUAL, reason="No numba.cuda")
    def test_conversion_errors(self):
        """TVM NDArray DLPack interaction with Numba for invalid cases."""

        tvm_dev_cpu = tvm.cpu(0)
        tvm_dev_cuda = tvm.cuda(0)

        # CPU tensors are not Numba CUDA arrays.
        cput_np = np.arange(100)
        self.assertFalse(numba.cuda.is_cuda_array(cput_np))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cput_np)

        cput_tvm = tvm_ndarray.array(np.arange(100), device=tvm_dev_cpu)
        self.assertFalse(numba.cuda.is_cuda_array(cput_tvm)) # TVM CPU array is not a CUDA array for Numba.
        with self.assertRaises(TypeError):
             numba.cuda.as_cuda_array(cput_tvm) # Numba as_cuda_array expects CUDA-backed DLTensor

        # Sparse tensors: TVM NDArray itself is dense.
        # PyTorch's `torch.sparse_coo_tensor` has no direct DLPack-compatible TVM NDArray equivalent.
        # So, this portion is not directly translatable and is skipped.
        # TODO: Sparse tensor mapping is complex and not directly applicable to generic TVM NDArray DLPack.
        # If a TVM sparse tensor type with DLPack existed, this could be adapted.
        # For now, mark as a conceptual gap due to differing sparse tensor representations.
        self.skipTest("Sparse tensor part of test_conversion_errors is PyTorch-specific and complex for TVM.")

        # Gradient tracking: TVM Relay is a functional graph IR. NDArrays are runtime data buffers
        # and do not track gradients directly like PyTorch tensors with `requires_grad=True`.
        # Therefore, the RuntimeError from `cuda_gradt` in PyTorch due to autograd is not applicable to TVM NDArrays.
        cuda_t = tvm_ndarray.zeros((100,), dtype='float32', device=tvm_dev_cuda)
        self.assertTrue(numba.cuda.is_cuda_array(cuda_t)) # TVM NDArray has DLPack
        numba_view = numba.cuda.as_cuda_array(cuda_t) # This should work without RuntimeError
        self.assertIsInstance(numba_view, numba.cuda.devicearray.DeviceNDArray)
        # No RuntimeError expected for TVM NDArray due to gradient tracking, as it doesn't track gradients.


    @pytest.mark.skipif(not HAS_CUDA, reason="No cuda")
    @pytest.mark.skipif(not _HAS_NUMBA_CUDA_ACTUAL, reason="No numba.cuda")
    @pytest.mark.skipif(not HAS_MULTIGPU, reason="No multigpu")
    def test_active_device(self):
        """TVM `from_dlpack` with Numba array respects device IDs."""

        tvm_dev0 = tvm.cuda(0)
        tvm_dev1 = tvm.cuda(1)

        # Both TVM/Numba default to device 0.
        cudat_np = np.arange(10, dtype='int64')
        numba_cudat_dev0 = numba.cuda.to_device(cudat_np, device=0)
        tvm_cudat_dev0 = tvm_ndarray.from_dlpack(numba_cudat_dev0)

        self.assertEqual(tvm_cudat_dev0.device.device_id, 0)
        self.assertIsInstance(tvm_cudat_dev0, tvm_ndarray.NDArray)

        # Numba array on a non-default device (device 1).
        numba_cudat_dev1 = numba.cuda.to_device(cudat_np, device=1)
        tvm_cudat_dev1 = tvm_ndarray.from_dlpack(numba_cudat_dev1)

        self.assertEqual(tvm_cudat_dev1.device.device_id, 1)
        self.assertIsInstance(tvm_cudat_dev1, tvm_ndarray.NDArray)

        # The original PyTorch test checks for `numba.cuda.driver.CudaAPIError`
        # if the PyTorch tensor's device doesn't match Numba's *active context*.
        # TVM's `from_dlpack` directly consumes the DLTensor, which includes device metadata,
        # so it doesn't rely on the *current active CUDA context* in the same way PyTorch might.
        # It should correctly wrap the DLTensor regardless of the Python process's current CUDA device context.
        with numba.cuda.devices.gpus[0]: # Numba's active context is device 0
            # TVM can still create an NDArray from a Numba array on device 1,
            # and the resulting TVM NDArray will correctly report its device as 1.
            tvm_cudat_from_dev1_in_ctx0 = tvm_ndarray.from_dlpack(numba_cudat_dev1)
            self.assertEqual(tvm_cudat_from_dev1_in_ctx0.device.device_id, 1)
            self.assertIsInstance(tvm_cudat_from_dev1_in_ctx0, tvm_ndarray.NDArray)

            # Attempting to convert `numba_cudat_dev1` to a Numba CUDA array within the
            # device 0 context *might* raise a Numba error, but this is testing Numba's
            # internal behavior, not TVM's `from_dlpack` directly.
            with self.assertRaises(numba.cuda.driver.CudaAPIError):
                numba.cuda.as_cuda_array(numba_cudat_dev1)


    @pytest.mark.skip(reason="PyTorch issue #54418 - temporary disabled in PyTorch, will revisit for TVM.")
    @pytest.mark.skipif(not HAS_NUMPY, reason="No numpy")
    @pytest.mark.skipif(not HAS_CUDA, reason="No cuda")
    @pytest.mark.skipif(not _HAS_NUMBA_CUDA_ACTUAL, reason="No numba.cuda")
    def test_from_cuda_array_interface(self):
        """tvm.runtime.ndarray.from_dlpack() supports Numba's DLPack protocol.

        Numba DeviceNDArray objects implement __dlpack__ for device memory.
        """
        tvm_dev_cuda = tvm.cuda(0)

        np_dtypes = [
            np.complex64,
            np.complex128,
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            np.bool_,
        ]

        for dtype in np_dtypes:
            numpy_arys = [
                np.ones((), dtype=dtype),
                np.arange(6).reshape(2, 3).astype(dtype),
                np.arange(6)
                .reshape(2, 3)
                .astype(dtype)[1:],  # View offset should be handled by DLPack
                np.arange(6)
                .reshape(2, 3)
                .astype(dtype)[:, None],  # change the strides but still contiguous for DLPack
            ]

            # Zero-copy when using `tvm_ndarray.from_dlpack()`
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                tvm_ary = tvm_ndarray.from_dlpack(numba_ary)

                # DLPack means sharing memory, so the data pointers should be the same.
                self.assertEqual(tvm_ary.handle, numba_ary.device_ctypes_pointer.value)
                tvm.testing.assert_allclose(tvm_ary.numpy(), np.asarray(numba_ary, dtype=dtype))

                # Check that `tvm_ary` and `numba_ary` point to the same device memory.
                # Writes to TVM NDArray should be reflected in Numba array and vice-versa.
                new_val = np.array(42, dtype=dtype) if dtype != np.bool_ else np.bool_(1)
                if tvm_ary.size > 0:
                    modified_np_ary = tvm_ary.numpy()
                    if dtype == np.bool_:
                        modified_np_ary[0] = not modified_np_ary[0] # Boolean toggle
                    else:
                        modified_np_ary[0] += new_val
                    tvm_ary.copyfrom(modified_np_ary)

                tvm.cuda(0).sync()
                numba.cuda.synchronize()

                tvm.testing.assert_allclose(tvm_ary.numpy(), np.asarray(numba_ary, dtype=dtype))

            # Implicit-copy (when output is desired on CPU device)
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                # This explicitly copies from Numba GPU to host NumPy, then to TVM CPU.
                tvm_ary_cpu_copy = tvm_ndarray.array(np.asarray(numba_ary), device=tvm.cpu(0))

                tvm.testing.assert_allclose(tvm_ary_cpu_copy.numpy(), np.asarray(numba_ary, dtype=dtype))

                # Check that `tvm_ary_cpu_copy` and `numba_ary` points to different memory.
                new_val = np.array(42, dtype=dtype) if dtype != np.bool_ else np.bool_(1)
                if tvm_ary_cpu_copy.size > 0:
                    modified_np_ary = tvm_ary_cpu_copy.numpy()
                    if dtype == np.bool_:
                        modified_np_ary[0] = not modified_np_ary[0]
                    else:
                        modified_np_ary[0] += new_val
                    tvm_ary_cpu_copy.copyfrom(modified_np_ary)
                tvm.cuda(0).sync()
                numba.cuda.synchronize()

                # Numba array should remain unchanged, TVM CPU array should be modified.
                tvm.testing.assert_allclose(np.asarray(numba_ary, dtype=dtype), numba_ary.copy_to_host())
                tvm.testing.assert_allclose(tvm_ary_cpu_copy.numpy(), np.asarray(numba_ary, dtype=dtype) + (new_val if dtype != np.bool_ else 0)) # simplified for bool toggle if needed

            # Explicit-copy (when output is desired on CUDA device but not zero-copy)
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                # Explicitly copy from Numba GPU to host NumPy, then to TVM GPU.
                tvm_ary_copy = tvm_ndarray.array(np.asarray(numba_ary), device=tvm_dev_cuda)

                tvm.testing.assert_allclose(tvm_ary_copy.numpy(), np.asarray(numba_ary, dtype=dtype))

                # Check that `tvm_ary_copy` and `numba_ary` points to different memory.
                new_val = np.array(42, dtype=dtype) if dtype != np.bool_ else np.bool_(1)
                if tvm_ary_copy.size > 0:
                    modified_np_ary = tvm_ary_copy.numpy()
                    if dtype == np.bool_:
                        modified_np_ary[0] = not modified_np_ary[0]
                    else:
                        modified_np_ary[0] += new_val
                    tvm_ary_copy.copyfrom(modified_np_ary)
                tvm.cuda(0).sync()
                numba.cuda.synchronize()

                # Numba array should remain unchanged, TVM GPU array should be modified.
                tvm.testing.assert_allclose(np.asarray(numba_ary, dtype=dtype), numba_ary.copy_to_host())
                tvm.testing.assert_allclose(tvm_ary_copy.numpy(), np.asarray(numba_ary, dtype=dtype) + (new_val if dtype != np.bool_ else 0))


    @pytest.mark.skipif(not HAS_NUMPY, reason="No numpy")
    @pytest.mark.skipif(not HAS_CUDA, reason="No cuda")
    @pytest.mark.skipif(not _HAS_NUMBA_CUDA_ACTUAL, reason="No numba.cuda")
    def test_from_cuda_array_interface_inferred_strides(self):
        """tvm.ndarray.from_dlpack(numba_ary) should have correct inferred (contiguous) strides."""
        tvm_dev_cuda = tvm.cuda(0)

        np_dtypes = [
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
        ]
        for dtype in np_dtypes:
            numpy_ary = np.arange(6).reshape(2, 3).astype(dtype)
            numba_ary = numba.cuda.to_device(numpy_ary)
            self.assertTrue(numba_ary.is_c_contiguous())
            tvm_ary = tvm_ndarray.from_dlpack(numba_ary)
            # TVM NDArray created from DLPack of a C-contiguous array should also be contiguous.
            self.assertTrue(tvm_ary.is_contiguous())


    @pytest.mark.skip(reason="PyTorch issue #54418 - temporary disabled in PyTorch, will revisit for TVM.")
    @pytest.mark.skipif(not HAS_NUMPY, reason="No numpy")
    @pytest.mark.skipif(not HAS_CUDA, reason="No cuda")
    @pytest.mark.skipif(not _HAS_NUMBA_CUDA_ACTUAL, reason="No numba.cuda")
    def test_from_cuda_array_interface_lifetime(self):
        """tvm.runtime.ndarray.from_dlpack(obj) tensor grabs a reference to obj to extend its lifetime."""
        numba_ary = numba.cuda.to_device(np.arange(6))
        tvm_ary = tvm_ndarray.from_dlpack(numba_ary)
        # Check that they point to the same memory initially
        self.assertEqual(tvm_ary.handle, numba_ary.device_ctypes_pointer.value)

        del numba_ary # Delete the original Numba array. The DLPack ownership model means tvm_ary keeps the memory alive.
        tvm.testing.assert_allclose(tvm_ary.numpy(), np.arange(6)) # `tvm_ary` is still alive and accessible.


    @pytest.mark.skip(reason="PyTorch issue #54418 - temporary disabled in PyTorch, will revisit for TVM.")
    @pytest.mark.skipif(not HAS_NUMPY, reason="No numpy")
    @pytest.mark.skipif(not HAS_CUDA, reason="No cuda")
    @pytest.mark.skipif(not _HAS_NUMBA_CUDA_ACTUAL, reason="No numba.cuda")
    @pytest.mark.skipif(not HAS_MULTIGPU, reason="No multigpu")
    def test_from_cuda_array_interface_active_device(self):
        """tvm.ndarray.from_dlpack() handles Numba arrays from various devices."""
        tvm_dev_cuda0 = tvm.cuda(0)
        tvm_dev_cuda1 = tvm.cuda(1)

        # Zero-copy: Numba on device 0, TVM also wraps for device 0.
        numba_ary_dev0 = numba.cuda.to_device(np.arange(6), device=0)
        tvm_ary_dev0 = tvm_ndarray.from_dlpack(numba_ary_dev0)
        self.assertEqual(tvm_ary_dev0.device.device_id, 0)
        tvm.testing.assert_allclose(tvm_ary_dev0.numpy(), np.asarray(numba_ary_dev0))
        self.assertEqual(tvm_ary_dev0.handle, numba_ary_dev0.device_ctypes_pointer.value)

        # Implicit-copy (when the Numba and TVM target device differ for a copy operation).
        # This explicitly copies from Numba GPU (dev0) to TVM GPU (dev1).
        numba_ary_dev0_for_copy = numba.cuda.to_device(np.arange(6), device=0)
        tvm_ary_dev1_copy = tvm_ndarray.array(np.asarray(numba_ary_dev0_for_copy.copy_to_host()), device=tvm_dev_cuda1)
        # Note: `numba_ary_dev0_for_copy.copy_to_host()` copies from GPU0 to host NumPy.
        # Then `tvm_ndarray.array` copies from NumPy to TVM NDArray on device 1.
        # This is a full copy chain, reflecting "implicit copy" to a different device.

        self.assertEqual(tvm_ary_dev1_copy.device.device_id, 1)
        tvm.testing.assert_allclose(tvm_ary_dev1_copy.numpy(), np.asarray(numba_ary_dev0_for_copy))
        # Verify that the data pointers are NOT equal, indicating a copy.
        self.assertNotEqual(tvm_ary_dev1_copy.handle, numba_ary_dev0_for_copy.device_ctypes_pointer.value)

        # The original test checked for `if1["data"]` and `if2["data"]` from `__cuda_array_interface__`.
        # For DLPack, we check `handle` (TVM's internal pointer) against Numba's ctypes pointer.


if __name__ == "__main__":
    # To run with pytest, install pytest and run `pytest your_file.py`
    # For CI/standalone, we use pytest to handle skips and discovery.
    import sys
    pytest.main(sys.argv)
