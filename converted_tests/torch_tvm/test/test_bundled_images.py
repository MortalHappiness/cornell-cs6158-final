#!/usr/bin/env python3
import io
import os
import cv2  # External dependency for image processing
import numpy as np
import pytest
import unittest

import tvm
from tvm import relay
from tvm.runtime import NDArray
from tvm.relay import transform as _transform
from tvm.relay import op as _op
from tvm.testing import assert_allclose

# NO_MAPPING: torch.ops.load_library is for loading PyTorch custom C++ operators.
# There is no direct TVM equivalent for loading PyTorch-specific custom ops.
# The functionality of these ops will be simulated using OpenCV and NumPy externally.
# The original PyTorch code: torch.ops.load_library("//caffe2/torch/fb/operators:decode_bundled_image")

# Helper function to simulate the behavior of PyTorch's custom image decoding ops.
# This assumes the custom ops decode a JPEG byte array to an NCHW float32 tensor
# and apply a channel-wise scale and bias.
def _decode_jpeg_and_normalize(jpeg_bytes: np.ndarray, weight_val: np.ndarray, bias_val: np.ndarray) -> np.ndarray:
    """
    Simulates the core decoding and normalization logic of
    `torch.ops.fb.image_decode_to_NCHW` or `torch.ops.fb.decode_bundled_image`.
    Input: JPEG bytes (numpy uint8 array), weight (numpy float32 1D array), bias (numpy float32 1D array).
    Output: Decoded and normalized image as NCHW float32 numpy array.
    """
    # 1. Decode JPEG bytes to BGR, HWC (uint8)
    decoded_bgr_hwc = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if decoded_bgr_hwc is None:
        raise ValueError("Failed to decode image bytes. Check if the input is a valid JPEG byte array.")

    # 2. Convert to RGB, HWC (uint8)
    decoded_rgb_hwc = cv2.cvtColor(decoded_bgr_hwc, cv2.COLOR_BGR2RGB)

    # 3. Convert to float32 NCHW: (H, W, C) -> (C, H, W) -> (N, C, H, W)
    decoded_nchw_float = np.expand_dims(np.transpose(decoded_rgb_hwc, (2, 0, 1)), axis=0).astype(np.float32)

    # 4. Apply channel-wise scale (weight) and bias
    # Assuming weight/bias are (C,) and are broadcasted across H, W
    output_nchw = np.zeros_like(decoded_nchw_float)
    num_channels = decoded_nchw_float.shape[1]
    for c in range(num_channels):
        output_nchw[:, c, :, :] = decoded_nchw_float[:, c, :, :] * weight_val[c] + bias_val[c]
    return output_nchw


# NO_MAPPING: `model_size` and `save_and_load` are PyTorch JIT serialization functions.
# TVM has its own module serialization (e.g., `relay.build`, `lib.export_library`).
# The original test relies on `loaded.get_all_bundled_inputs()`, which is deeply
# integrated into PyTorch's JIT and has no direct TVM equivalent.


# `bundle_jpeg_image` adapted for TVM context:
# It no longer produces an `InflatableArg` (PyTorch-specific). Instead, it directly returns
# the encoded JPEG bytes (as a numpy array) which can then be decoded externally.
def bundle_jpeg_image(img_numpy: np.ndarray, quality: int) -> np.ndarray:
    """
    Encodes a NumPy image array (HWC or NCHW, uint8) into JPEG bytes.
    Returns the encoded bytes as a numpy uint8 array.
    """
    pixels = img_numpy
    if pixels.ndim == 4:
        assert pixels.shape[0] == 1, "NCHW image expected to have batch size 1"
        pixels = pixels[0].transpose(1, 2, 0)  # NCHW to HWC

    # Ensure pixels are uint8 for cv2.imencode
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode(".JPEG", pixels, encode_param)
    return enc_img  # Return the encoded bytes as a numpy array


# `get_tensor_from_raw_BGR` adapted for TVM context:
# Now returns a `tvm.runtime.NDArray` directly from a raw BGR image.
def get_tvm_ndarray_from_raw_BGR(im_numpy: np.ndarray) -> NDArray:
    """
    Transforms a raw BGR HWC numpy image (e.g., from cv2.imread)
    into a normalized NCHW float32 TVM NDArray.
    """
    raw_data_rgb_hwc = cv2.cvtColor(im_numpy, cv2.COLOR_BGR2RGB)
    raw_data_float_chw = np.transpose(raw_data_rgb_hwc, (2, 0, 1)).astype(np.float32)
    raw_data_normalized_chw = raw_data_float_chw / 255.0
    raw_data_nchw = np.expand_dims(raw_data_normalized_chw, axis=0) # Add batch dimension (N, C, H, W)
    return tvm.nd.array(raw_data_nchw)


class TestBundledImages(unittest.TestCase):
    def test_single_tensors(self):
        # In PyTorch, SingleTensorModel is an identity module.
        # In TVM, we represent this as a Relay identity function.
        # Assuming typical image input shape and dtype for tests.
        input_shape = (1, 3, 224, 224)
        input_dtype = "float32"
        data_var = relay.var("data", shape=input_shape, dtype=input_dtype)
        identity_func = relay.Function([data_var], data_var)
        mod = tvm.IRModule.from_expr(identity_func)

        target = tvm.target.Target("llvm")  # Use a default target, e.g., "llvm" or "cuda"
        dev = tvm.cpu(0)  # Use a default device
        # For testing, we can build and run the module
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target)
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        # --- Setup: Create a dummy image for testing if it doesn't exist ---
        test_img_dir = "caffe2/test/test_img"
        test_img_path = os.path.join(test_img_dir, "p1.jpg")
        try:
            im_np_raw = cv2.imread(test_img_path)
            if im_np_raw is None:
                raise FileNotFoundError(f"Test image not found at {test_img_path}")
        except FileNotFoundError:
            # Create a dummy image if the real one isn't found
            print(f"Warning: {test_img_path} not found. Creating a dummy image.")
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_img[50:150, 50:150, 0] = 255  # Blue square
            dummy_img[100:200, 100:200, 1] = 128  # Green square
            dummy_img[150:220, 150:220, 2] = 64   # Red square
            os.makedirs(test_img_dir, exist_ok=True)
            cv2.imwrite(test_img_path, dummy_img)
            im_np_raw = dummy_img
        
        # --- Simulate original PyTorch workflow for the first check ---
        # `bundle_jpeg_image` now returns encoded bytes (numpy array)
        encoded_img_bytes_np = bundle_jpeg_image(im_np_raw, 90)

        # In PyTorch, `inflated = loaded.get_all_bundled_inputs()` would call the decoder.
        # Here, we explicitly decode the bytes using our helper to get the float NCHW data.
        # The `decode_bundled_image` op often implies simple decoding to normalized RGB.
        # The reference `get_tensor_from_raw_BGR` normalizes by 255.
        # So we use weight=1/255 and bias=0 for `_decode_jpeg_and_normalize`.
        default_weight = np.full((3,), 1.0 / 255.0, dtype=np.float32)
        default_bias = np.zeros(3, dtype=np.float32)
        decoded_data_np = _decode_jpeg_and_normalize(encoded_img_bytes_np, default_weight, default_bias)
        decoded_data_tvm = tvm.nd.array(decoded_data_np)

        # Feed the decoded data to the TVM runtime and get output
        runtime.set_input("data", decoded_data_tvm)
        runtime.run()
        model_output_tvm = runtime.get_output(0)

        # `raw_data` in PyTorch is the reference NCHW float32 tensor
        raw_data_tvm = get_tvm_ndarray_from_raw_BGR(im_np_raw)

        # Assertions
        # NO_MAPPING: `len(inflated)` and `len(inflated[0])` are PyTorch-specific checks.
        self.assertEqual(raw_data_tvm.shape, model_output_tvm.shape)
        assert_allclose(raw_data_tvm.numpy(), model_output_tvm.numpy(), atol=0.1, rtol=1e-01)

        # --- Second part of the test: direct call to custom op simulation ---
        # Check if `fb::image_decode_to_NCHW` works as expected
        with open(test_img_path, "rb") as fp:
            jpeg_bytes_full = np.frombuffer(fp.read(), dtype=np.uint8)

            # Re-create PyTorch's `weight` and `bias` values for the custom op call
            # `weight = torch.full((3,), 1.0 / 255.0).diag()` for PyTorch is a (3,3) matrix `(1/255)*Identity`.
            # This implies a channel-wise scaling/biasing.
            # So, for simulation, the `weight` values are `1.0/255.0` for each channel.
            sim_weight_np = np.full((3,), 1.0 / 255.0, dtype=np.float32)
            sim_bias_np = np.zeros(3, dtype=np.float32)

            # Simulate `torch.ops.fb.image_decode_to_NCHW` using our helper
            im2_tensor_np = _decode_jpeg_and_normalize(jpeg_bytes_full, sim_weight_np, sim_bias_np)
            im2_tensor_tvm = tvm.nd.array(im2_tensor_np)

            self.assertEqual(raw_data_tvm.shape, im2_tensor_tvm.shape)
            assert_allclose(raw_data_tvm.numpy(), im2_tensor_tvm.numpy(), atol=0.1, rtol=1e-01)


if __name__ == "__main__":
    pytest.main([__file__]) # Run tests using pytest
    # The original __main__ had a RuntimeError to disable direct execution.
    # We replace it with standard pytest discovery to make the converted file runnable.
