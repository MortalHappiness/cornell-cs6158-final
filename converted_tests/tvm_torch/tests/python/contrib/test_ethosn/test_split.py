import torch
import numpy as np
import pytest

# Mocking tvm.testing.requires_ethosn and other Ethos-N infrastructure
# In a real PyTorch environment, this decorator would typically be removed
# or replaced with a PyTorch-specific device/backend requirement if applicable.
class EthosNRequired:
    def __call__(self, f):
        return f
requires_ethosn = EthosNRequired()

class TeiMock:
    """Mock for Ethos-N infrastructure interactions."""

    def make_module(self, model_func, _):
        """Mock for creating a module. In PyTorch, the 'model_func' is already callable."""
        return model_func

    def build_and_run(self, pytorch_model_func, inputs, output_count, _, npu=False, additional_config_args=None):
        """
        Mock for building and running.
        Simulates running on CPU and an 'NPU' (represented by torch.compile).
        """
        input_tensor = inputs["a"]
        if npu:
            # Simulate NPU path conceptually with torch.compile
            compiled_model = torch.compile(pytorch_model_func)
            output = compiled_model(input_tensor)
        else:
            # Simulate CPU path
            output = pytorch_model_func(input_tensor)
        
        # torch.split always returns a tuple of tensors.
        # This aligns with TVM's astuple() behavior after split.
        return output

    def verify(self, outputs, dtype, tolerance):
        """
        Mock for verifying outputs.
        Compares results from plain PyTorch and compiled PyTorch.
        """
        assert len(outputs) == 2, "Expected outputs from both plain and compiled runs."
        plain_outputs = outputs[0]
        compiled_outputs = outputs[1]

        assert len(plain_outputs) == len(compiled_outputs), "Mismatch in number of output tensors."

        for i in range(len(plain_outputs)):
            # For integer dtypes, we expect exact equality.
            # For float dtypes, we use assert_allclose with a reasonable tolerance.
            if "int" in dtype:
                torch.testing.assert_allclose(plain_outputs[i], compiled_outputs[i], rtol=0, atol=0)
            else:
                torch.testing.assert_allclose(plain_outputs[i], compiled_outputs[i], rtol=1e-5, atol=1e-5)

    def make_ethosn_partition(self, model_func):
        """
        Mock for creating an Ethos-N partition.
        No direct PyTorch equivalent as this is backend-specific.
        """
        return model_func # Returns the model_func itself, as it's just a proxy for the computational graph.

    def test_error(self, pytorch_model_func, _, err_msg):
        """
        Mock for testing Ethos-N specific errors.
        Since these are backend-specific hardware/compilation constraints not inherent
        to PyTorch's `torch.split`, the PyTorch tests would typically succeed.
        Therefore, we skip these tests to avoid false positives or complex simulations.
        """
        pytest.skip(f"Ethos-N specific error test: '{err_msg}' is not applicable to standard PyTorch behavior.")

tei = TeiMock()

def convert_dtype_string_to_torch_dtype(dtype_str):
    """Converts a NumPy-style dtype string to a PyTorch dtype object."""
    if dtype_str == "uint8":
        return torch.uint8
    elif dtype_str == "int8":
        return torch.int8
    elif dtype_str == "int16":
        return torch.int16
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def _get_torch_split_sizes_from_tvm_indices_or_sections(input_shape, tvm_splits, axis):
    """
    Converts TVM's `indices_or_sections` to PyTorch's `split_size_or_sections`.
    """
    dim_size = input_shape[axis]
    if isinstance(tvm_splits, int):
        # If an integer, TVM means number of equal sections. PyTorch handles this directly.
        return tvm_splits
    else: # tvm_splits is a list of indices
        # TVM: `indices_or_sections=[idx1, idx2]` means split at these indices.
        # PyTorch: `split_size_or_sections=[size1, size2, ...]` means list of chunk sizes.
        current_idx = 0
        sizes = []
        for split_idx in tvm_splits:
            if split_idx <= current_idx or split_idx > dim_size:
                # Malformed indices will cause errors in torch.split later.
                # For example, if indices are not increasing or out of bounds.
                # This conversion assumes valid TVM split points.
                pass 
            sizes.append(split_idx - current_idx)
            current_idx = split_idx
        sizes.append(dim_size - current_idx)
        return sizes

def _get_pytorch_model(shape, dtype, splits, axis):
    """
    Defines the PyTorch model (a callable function) that performs the split operation.
    This is analogous to defining a Relay expression.
    """
    def model_func(input_tensor):
        split_size_or_sections = _get_torch_split_sizes_from_tvm_indices_or_sections(
            input_tensor.shape, splits, axis
        )
        return torch.split(input_tensor, split_size_or_sections=split_size_or_sections, dim=axis)
    return model_func


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape,splits,axis",
    [
        ((1, 16, 16, 32), (2, 7, 10), 2),
        ((1, 12, 8, 16), 3, 1),
    ],
)
def test_split(dtype, shape, splits, axis):
    """Compare Split output between plain PyTorch and TorchInductor."""
    np.random.seed(0)

    torch_dtype = convert_dtype_string_to_torch_dtype(dtype)

    # Prepare input tensor with random data respecting dtype limits
    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max + 1
        numpy_data = np.random.randint(low, high, size=shape, dtype=dtype)
    else: # float type
        numpy_data = np.random.rand(*shape).astype(dtype)

    input_tensor = torch.tensor(numpy_data, dtype=torch_dtype)

    # Define the PyTorch model (functional representation of the computation)
    model_func = _get_pytorch_model(shape, dtype, splits, axis)

    # Collect outputs from both plain PyTorch and compiled PyTorch
    outputs_from_runs = []
    
    # Run plain PyTorch (npu=False equivalent)
    plain_output = model_func(input_tensor)
    outputs_from_runs.append(plain_output)

    # Run compiled PyTorch (npu=True equivalent using TorchInductor)
    compiled_output = tei.build_and_run(
        model_func,
        {"a": input_tensor},
        None, # output_count is inferred by torch.split, so None here is fine for mock
        {},
        npu=True,
        additional_config_args={"inline_non_compute_intensive_partitions": False},
    )
    outputs_from_runs.append(compiled_output)

    # Verify the results are close (or exact for int dtypes)
    tei.verify(outputs_from_runs, dtype, 0)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,dtype,splits,axis,err_msg",
    [
        ((1, 4, 4, 4, 4), "uint8", 4, 2, "dimensions=5, dimensions must be <= 4;"),
        ((1, 4, 4, 4), "int16", 4, 2, "dtype='int16', dtype must be either uint8, int8 or int32;"),
        ((2, 4, 4, 4), "uint8", 4, 2, "batch size=2, batch size must = 1;"),
        ((1, 4, 4, 4), "uint8", 1, 0, "Split cannot be performed along batch axis (axis 0);"),
        (
            (1, 4, 4, 4), # Input shape
            "uint8",
            # If splits is an int, it's number of sections. 4 sections along axis 3 (size 4) -> 4 chunks of size 1.
            # Ethos-N specific constraint requires multiples of 16 for channel dimension.
            4, 
            3,
            "Split along the channels dimension (axis 3) requires all output sizes "
            "(specified in splitInfo.m_Sizes) to be multiples of 16;",
        ),
    ],
)
def test_split_failure(shape, dtype, splits, axis, err_msg):
    """Check Split error messages (Ethos-N specific - skipped for PyTorch)."""
    # The original TVM test `test_split_failure` targets specific error messages
    # that arise from the Arm Ethos-N backend's constraints (e.g., maximum dimensions,
    # supported dtypes, fixed batch size, or channel alignment requirements).
    #
    # Standard PyTorch operations (like `torch.split`) do not inherently enforce
    # these same backend-specific hardware or compilation limitations. For example:
    # - `torch.split` readily operates on 5D tensors.
    # - `torch.split` supports `torch.int16` dtype.
    # - `torch.split` works correctly with batch sizes greater than 1.
    # - `torch.split` can split along any valid dimension, including the batch axis (dim=0).
    # - `torch.split` has no "multiples of 16" constraint for channel output sizes.
    #
    # Consequently, running the equivalent PyTorch code for these parameters would
    # typically *succeed* where the Ethos-N compilation would have failed.
    # To maintain the semantic intent of these *failure tests* (i.e., to show that
    # Ethos-N rejects certain configurations), and because PyTorch's generic behavior
    # does not match these specific rejection criteria, the most appropriate conversion
    # is to explicitly skip these tests in the PyTorch context.
    
    # We pass a dummy model function to the mock `tei.test_error`, as the actual PyTorch
    # computation for these arguments would likely not raise the expected Ethos-N errors.
    dummy_model_func = _get_pytorch_model(shape, dtype, splits, axis)
    tei.test_error(
        dummy_model_func,
        {}, 
        err_msg
    )
