import os
import unittest
import numpy as np
import PIL.Image
# No torchvision, no torch.
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay.op.algorithm import topk
from tvm.testing import assert_allclose
import pytest # For xfail/skip marking


# Helper to simulate torchvision.transforms for image preprocessing.
# This function is retained for illustrative purposes but its output
# is not directly used by the runnable test case, which instead uses
# simulated *model outputs*.
def _get_test_image_np_array():
    data_dir = os.path.join(os.path.dirname(__file__), "assets")
    img_path = os.path.join(data_dir, "grace_hopper_517x606.jpg")
    input_image = PIL.Image.open(img_path)

    # Replicate torchvision.transforms.Resize(256)
    input_image = input_image.resize((256, 256), PIL.Image.BILINEAR)
    # Replicate torchvision.transforms.CenterCrop(224)
    width, height = input_image.size
    new_width, new_height = 224, 224
    left = int((width - new_width) / 2)
    top = int((height - new_height) / 2)
    right = int((width + new_width) / 2)
    bottom = int((height + new_height) / 2)
    input_image = input_image.crop((left, top, right, bottom))

    # Replicate torchvision.transforms.ToTensor() -> HWC_uint8 to CHW_float32, scale [0,1]
    img_np = np.array(input_image).astype(np.float32) / 255.0
    # From HWC to CHW
    img_np = img_np.transpose((2, 0, 1))

    # Replicate torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3, 1, 1))
    img_np = (img_np - mean) / std

    # Replicate .unsqueeze(0) -> add batch dimension
    img_np = np.expand_dims(img_np, axis=0) # Shape (1, 3, 224, 224)
    return img_np


# Base class for TVM tests, replacing onnx_test_common._TestONNXRuntime.
# It focuses on running the _TopPredictor logic, as direct conversion of
# PyTorch quantized models (without torch imports) is not yet feasible.
class _TestTVMRuntime(unittest.TestCase):
    def setUp(self):
        self.target = tvm.target.Target("llvm")
        self.dev = tvm.cpu(0)

    def _get_tvm_topk_predictor_module(self, input_shape, input_dtype):
        """
        Creates a TVM Relay module that replicates the functionality of the
        PyTorch `_TopPredictor` class. This module expects the *output of the
        base model* as its input, and then applies `topk` to find the top prediction.
        """
        # Input to this Relay module: a tensor representing `base_model(x)`
        # Expected shape is typically (1, num_classes) for a classification model.
        base_model_output_var = relay.var("base_model_output", shape=input_shape, dtype=input_dtype)

        # Replicate `_TopPredictor` forward logic: `_, topk_id = torch.topk(x[0], 1)`
        # `x[0]` corresponds to slicing the batch dimension.
        # `relay.strided_slice` is used to get the first element of the batch.
        # Then, `relay.squeeze` removes the now-singleton batch dimension.
        slice_output = relay.strided_slice(base_model_output_var, begin=[0,0], end=[1, input_shape[1]])
        slice_output_squeezed = relay.squeeze(slice_output, axis=[0]) # Shape (num_classes,)

        # Apply `topk` to get the top-1 index.
        # TVM's `topk` returns a tuple (values, indices) if `ret_type="both"`.
        # We need the indices, which is the second element of the tuple.
        _, tvm_topk_id = topk(slice_output_squeezed, k=1, ret_type="both", is_ascend=False)

        # The final Relay graph returns just the `topk_id`
        tvm_mod = tvm.IRModule.from_expr(tvm_topk_id)
        tvm_params = {} # This simple graph has no parameters
        return tvm_mod, tvm_params

    def _run_topk_predictor_test(self, simulated_base_model_output_np):
        """
        Runs the TVM `_TopPredictor` logic with a simulated base model output.
        This function *does not* load any PyTorch models.
        It *assumes* the output of a (quantized) PyTorch model's forward pass
        is provided as `simulated_base_model_output_np`.
        """
        # 1. Calculate the expected output (top-1 index) from the simulated base model output using NumPy.
        # This simulates `torch.topk(x[0], 1)` on NumPy arrays.
        # `x[0]` is `simulated_base_model_output_np[0]`.
        # For k=1, np.argmax is sufficient.
        expected_topk_id_np = np.argmax(simulated_base_model_output_np[0]).astype(np.int64)
        expected_topk_id_np = np.array([expected_topk_id_np], dtype=np.int64) # Ensure it's (1,) and int64
        
        # 2. Prepare input for the TVM Relay module.
        base_model_output_shape = simulated_base_model_output_np.shape
        base_model_output_dtype = str(simulated_base_model_output_np.dtype)

        # 3. Get the TVM Relay module for the _TopPredictor logic
        tvm_mod, tvm_params = self._get_tvm_topk_predictor_module(
            base_model_output_shape, base_model_output_dtype
        )

        # 4. Compile the TVM module
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(tvm_mod, self.target, params=tvm_params)

        # 5. Run TVM inference
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](self.dev))
        runtime.set_input("base_model_output", simulated_base_model_output_np)
        runtime.run()
        tvm_output_np = runtime.get_output(0).numpy()

        # 6. Compare outputs
        assert_allclose(tvm_output_np, expected_topk_id_np, rtol=1e-5, atol=1e-8)


# The original `parameterized` decorator is removed.
class TestQuantizedModelsTVM(_TestTVMRuntime):
    # The original tests aim to test torchvision.models.quantization models by loading
    # them (pretrained and quantized) and then exporting to ONNX.
    # Without importing `torch` and `torchvision`, directly converting these
    # *pretrained, already quantized* PyTorch models to TVM Relay is not feasible
    # with `tvm.relay.frontend.from_pytorch`.
    #
    # `tvm.relay.frontend.from_pytorch` currently targets float PyTorch models.
    # To test quantized models in TVM, one would typically:
    # 1. Load a *float* PyTorch model.
    # 2. Convert it to a float TVM Relay module.
    # 3. Apply TVM's own quantization passes (`relay.quantize.quantize`) on the float Relay module.
    # 4. Then compile and run the quantized TVM module.
    # This is a different workflow than what the original PyTorch tests do (loading pre-quantized models).
    #
    # Therefore, all model-specific tests are skipped with a TODO.
    # A single runnable test case (`test_mobilenet_v3_topk_logic_only`) is provided
    # to ensure the `_TestTVMRuntime` infrastructure and `_TopPredictor`'s `topk` logic
    # are runnable and demonstrate the conversion principles.

    # This is the runnable test case, demonstrating only the `_TopPredictor` logic.
    # It does NOT load or validate any actual torchvision quantized model.
    def test_mobilenet_v3_topk_logic_only(self):
        num_classes = 1000
        # Simulate the output of a classification model (e.g., Mobilenet V3 output before topk)
        # Shape (1, num_classes), float32.
        simulated_output = np.random.rand(1, num_classes).astype(np.float32)
        # Ensure a unique and predictable maximum for a clear top-1 index.
        # We'll explicitly set a value to ensure `argmax` returns a specific index.
        target_top_index = 50
        simulated_output[0, :] = np.arange(num_classes).astype(np.float32) / num_classes # Baseline values
        simulated_output[0, target_top_index] = 1.0001 # Make target_top_index the highest value

        self._run_topk_predictor_test(simulated_output)

    @unittest.skip(
        "TODO: Conversion for torchvision.models.quantization.inception_v3("
        "pretrained=True, quantize=True) to TVM Relay requires a specialized frontend "
        "or a TVM-side quantization workflow (without torch imports)."
    )
    def test_inception_v3(self):
        pass

    @unittest.skip(
        "TODO: Conversion for torchvision.models.quantization.googlenet("
        "pretrained=True, quantize=True) to TVM Relay requires a specialized frontend "
        "or a TVM-side quantization workflow (without torch imports)."
    )
    def test_googlenet(self):
        pass

    @unittest.skip(
        "TODO: Conversion for torchvision.models.quantization.shufflenet_v2_x0_5("
        "pretrained=True, quantize=True) to TVM Relay requires a specialized frontend "
        "or a TVM-side quantization workflow (without torch imports)."
    )
    def test_shufflenet_v2_x0_5(self):
        pass

    @unittest.skip(
        "TODO: Conversion for torchvision.models.quantization.resnet18("
        "pretrained=True, quantize=True) to TVM Relay requires a specialized frontend "
        "or a TVM-side quantization workflow (without torch imports)."
    )
    def test_resnet18(self):
        pass

    @unittest.skip(
        "TODO: Conversion for torchvision.models.quantization.resnet50("
        "pretrained=True, quantize=True) to TVM Relay requires a specialized frontend "
        "or a TVM-side quantization workflow (without torch imports)."
    )
    def test_resnet50(self):
        pass

    @unittest.skip(
        "TODO: Conversion for torchvision.models.quantization.resnext101_32x8d("
        "pretrained=True, quantize=True) to TVM Relay requires a specialized frontend "
        "or a TVM-side quantization workflow (without torch imports)."
    )
    def test_resnext101_32x8d(self):
        pass


if __name__ == "__main__":
    unittest.main()
