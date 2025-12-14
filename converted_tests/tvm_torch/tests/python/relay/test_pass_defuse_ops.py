import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest

def test_defuse_simple():
    """Simple testcase.
    The TVM test verifies that `DefuseOps` correctly reverses `FuseOps`,
    resulting in a structurally identical graph to the original.
    In PyTorch, we model the original computation and use `torch.compile`
    to implicitly handle fusion/defusion, then assert numerical equivalence.
    The "defused" state is simply the eager-mode computation.
    """

    # Define the computation as a Python function using PyTorch ops
    def simple_computation(x):
        y = x + 1.0  # Equivalent to relay.add(x, relay.const(1, "float32"))
        z = torch.exp(y) # Equivalent to relay.exp(y)
        w = torch.squeeze(z) # Equivalent to relay.squeeze(z)
        return w

    # Prepare a concrete input tensor for PyTorch
    input_data = torch.randn(10, 20, dtype=torch.float32)

    # Run the "original" computation in eager mode
    original_output = simple_computation(input_data)

    # Compile the function and run it. TorchInductor will perform its own
    # fusion/defusion internally. We verify that the compiled version
    # produces the same numerical result as the eager one.
    compiled_computation = torch.compile(simple_computation, fullgraph=True)
    compiled_output = compiled_computation(input_data)

    # Assert numerical equality. The TVM structural_equal check is
    # abstracted away by ensuring numerical correctness through compilation.
    torch.testing.assert_close(original_output, compiled_output)


def test_inception_like():
    """Testcase with an Inception-like block.
    Similar to `test_defuse_simple`, we model the original (unfused)
    computation in PyTorch and verify numerical equivalence after `torch.compile`.
    """
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        def forward(self, data):
            y = self.conv(data)
            return F.relu(y)

    class InceptionLike(nn.Module):
        def __init__(self, in_channels, out_channels_per_branch, kernel_size, padding):
            super().__init__()
            # In TVM, `channels=16` was specified for each `conv`.
            # This means each conv branch outputs 16 channels.
            self.conv0 = ConvBlock(in_channels, out_channels_per_branch, kernel_size, padding)
            self.conv1 = ConvBlock(in_channels, out_channels_per_branch, kernel_size, padding)

        def forward(self, data):
            c0 = self.conv0(data)
            c1 = self.conv1(data)
            # Concat along channel dimension (axis=1)
            return torch.cat((c0, c1), dim=1)

    class BeforeModel(nn.Module):
        def __init__(self, dshape):
            super().__init__()
            batch_size, in_channels, H, W = dshape
            # First InceptionLike block: input `x` has `in_channels`
            self.in1 = InceptionLike(in_channels, 16, kernel_size=(3, 3), padding=(1, 1))
            # Second InceptionLike block: input `in1` has 16+16=32 channels from previous concat
            self.in2 = InceptionLike(32, 16, kernel_size=(3, 3), padding=(1, 1))

        def forward(self, x):
            y1 = self.in1(x)
            y2 = self.in2(y1)
            return y2

    dshape = (1, 16, 64, 64) # Input shape in NCHW format
    # Create an instance of the model
    model = BeforeModel(dshape)

    # Generate a dummy input
    input_data = torch.randn(dshape, dtype=torch.float32)

    # Run the eager model
    eager_output = model(input_data)

    # Compile the model and run
    compiled_model = torch.compile(model, fullgraph=True)
    compiled_output = compiled_model(input_data)

    # Assert numerical equality
    torch.testing.assert_close(eager_output, compiled_output)


def test_defuse_complex():
    """Complex defuse testcase.
    This test involves various operations including layout transformations.
    The TVM `golden_defused` function represents the desired unfused graph.
    We implement this graph directly as a PyTorch `nn.Module` and verify
    numerical outputs when run eagerly vs. compiled.
    Care must be taken with layout assumptions (NCHW vs NHWC) and parameter initialization.
    """
    class GoldenDefusedModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Weights for conv2d layers. TVM uses OHWI, PyTorch uses OIHW.
            # conv_layer1_weight: (out_channels=64, kH=7, kW=7, in_channels=3) OHWI
            # PyTorch: (out_channels=64, in_channels=3, kH=7, kW=7) OIHW
            self.conv_layer1_weight = nn.Parameter(torch.randn(64, 3, 7, 7))
            # conv_layer2_weight: (out_channels=64, kH=3, kW=3, in_channels=64) OHWI
            # PyTorch: (out_channels=64, in_channels=64, kH=3, kW=3) OIHW
            self.conv_layer2_weight = nn.Parameter(torch.randn(64, 64, 3, 3))

            # Batch Norm parameters for 64 channels.
            # Initialized with random for gamma/beta and zeros/ones for moving stats.
            self.bn_gamma0 = nn.Parameter(torch.randn(64))  # Scale
            self.bn_beta0 = nn.Parameter(torch.randn(64))   # Shift
            # Running mean and variance are buffers, typically not trainable (training=False below)
            self.register_buffer("bn_mmean0", torch.zeros(64))
            self.register_buffer("bn_mvar0", torch.ones(64))

        def forward(self, data):
            # 1. Input `data`: (1, 3, 224, 224) NCHW (from relay.var in golden_defused)

            # 2. `data1 = relay.layout_transform(data, src_layout="NCHW", dst_layout="NHWC")`
            data1_nhwc = data.permute(0, 2, 3, 1) # (1, 224, 224, 3) NHWC

            # 3. `c0 = relay.nn.conv2d(data1, conv_layer1_weight, ..., data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC")`
            # PyTorch `F.conv2d` expects NCHW input and OIHW kernel.
            # Convert `data1_nhwc` to NCHW for conv input:
            data1_nchw_for_conv = data1_nhwc.permute(0, 3, 1, 2) # (1, 3, 224, 224) NCHW

            c0_nchw = F.conv2d(
                data1_nchw_for_conv,
                self.conv_layer1_weight, # Already in OIHW format
                stride=(2, 2),
                padding=(3, 3), # TVM (3,3,3,3) corresponds to symmetric (3,3) in PyTorch
                dilation=(1, 1), # Default
                groups=1 # Default
            ) # Output is (1, 64, 112, 112) NCHW

            # 4. `c1 = relay.nn.batch_norm(c0, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0, axis=3)`
            # PyTorch `F.batch_norm` expects NCHW input and parameters for `dim=1` (channels).
            # TVM's `axis=3` with NHWC `c0` means normalizing channels. For PyTorch's NCHW `c0_nchw`,
            # this corresponds to `dim=1`.
            c1_nchw = F.batch_norm(
                c0_nchw,
                self.bn_mmean0,
                self.bn_mvar0,
                self.bn_gamma0,
                self.bn_beta0,
                training=False, # Assuming inference mode as typical for Relay-compiled graphs
                momentum=0.1,   # PyTorch default
                eps=1e-5        # TVM default
            ) # Output is (1, 64, 112, 112) NCHW
            c2_nchw = c1_nchw # TVM bn returns tuple, we take 0th element (output)

            # 5. `c3 = relay.nn.max_pool2d(c2, pool_size=(3, 3), strides=(2, 2), padding=(1, 1, 1, 1), layout="NHWC", out_layout="NHWC")`
            # PyTorch `F.max_pool2d` expects NCHW input.
            c3_nchw = F.max_pool2d(
                c2_nchw,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1) # TVM (1,1,1,1) corresponds to symmetric (1,1) in PyTorch
            ) # Output is (1, 64, 56, 56) NCHW

            # 6. `c4 = relay.nn.conv2d(c3, conv_layer2_weight, ..., data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC")`
            # `c3_nchw` is `(1, 64, 56, 56) NCHW`. It's the input `data2` conceptually in TVM graph.
            # PyTorch `F.conv2d` expects NCHW input.
            c4_nchw = F.conv2d(
                c3_nchw, # input from max_pool2d, already NCHW
                self.conv_layer2_weight, # Already in OIHW format
                stride=(1, 1), # Default
                padding=(1, 1), # TVM (1,1,1,1) corresponds to symmetric (1,1)
                dilation=(1, 1), # Default
                groups=1 # Default
            ) # Output is (1, 64, 56, 56) NCHW

            # 7. `c5 = relay.nn.batch_norm(c4, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0, axis=3)`
            c5_nchw = F.batch_norm(
                c4_nchw,
                self.bn_mmean0,
                self.bn_mvar0,
                self.bn_gamma0,
                self.bn_beta0,
                training=False,
                momentum=0.1,
                eps=1e-5
            ) # Output is (1, 64, 56, 56) NCHW
            c6_nchw = c5_nchw

            # 8. `c7 = relay.nn.relu(c6)`
            c7_nchw = F.relu(c6_nchw) # Output is (1, 64, 56, 56) NCHW

            # 9. `c8 = relay.add(c3, c7)`
            # Both `c3_nchw` (output of max_pool2d) and `c7_nchw` (output of relu) are
            # (1, 64, 56, 56) NCHW. They can be added directly.
            c8_nchw = c3_nchw + c7_nchw # Output is (1, 64, 56, 56) NCHW
            
            # 10. `c9 = relay.nn.relu(c8)`
            c9_nchw = F.relu(c8_nchw) # Output is (1, 64, 56, 56) NCHW
            
            # The TVM `golden_defused` model eventually returns `c9`. All its intermediate
            # ops using NHWC layout produce NHWC output. So the final output should be NHWC.
            final_output_nhwc = c9_nchw.permute(0, 2, 3, 1) # (1, 56, 56, 64) NHWC
            return final_output_nhwc

    # Instantiate the model
    model = GoldenDefusedModel()

    # Generate input data for the model. TVM `data` var had shape (1, 3, 224, 224) NCHW.
    input_shape = (1, 3, 224, 224)
    input_data = torch.randn(input_shape, dtype=torch.float32)

    # Run the eager model
    eager_output = model(input_data)

    # Compile the model and run
    compiled_model = torch.compile(model, fullgraph=True)
    compiled_output = compiled_model(input_data)

    # Assert numerical equality. Relax tolerance slightly for floating point differences
    # that can arise from different compiler/runtime implementations.
    torch.testing.assert_close(eager_output, compiled_output, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
