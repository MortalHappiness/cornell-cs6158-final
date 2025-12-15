import os
import shutil
import tempfile
import glob
import math
import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.relay.testing import run_infer_type
from tvm.relay import transform

# TVM requires pydot and graphviz's 'dot' for .dot and .svg dumping via to_graph
try:
    import pydot  # noqa: F401
    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

HAS_DOT = True if shutil.which("dot") is not None else False


class TestGraphTransformObserver:
    def test_sdpa_rewriter(self):
        # Check for CUDA device
        if not tvm.runtime.enabled("cuda"):
            pytest.skip("CUDA not enabled or device not found.")

        # Check for pydot and dot for graph visualization, as in original test
        if not (HAS_PYDOT and HAS_DOT):
            pytest.skip("pydot or dot executable not found for graph visualization.")

        # Define tensor shapes and dtypes
        batch_size, n_head, seq_len, embed_dim = (4, 2, 16, 32)
        tensor_shape = (batch_size, n_head, seq_len, embed_dim)

        # Define Relay variables for inputs
        q = relay.var("query", shape=tensor_shape, dtype="float32")
        k = relay.var("key", shape=tensor_shape, dtype="float32")
        v = relay.var("value", shape=tensor_shape, dtype="float32")

        # Translate dot_prod_attention logic to Relay
        # key.transpose(-2, -1) equivalent to permuting last two axes
        # For a 4D tensor (N, H, S, E), transposing -2 and -1 means (N, H, E, S)
        key_t = relay.transpose(k, axes=(0, 1, 3, 2))

        # torch.matmul(query, key.transpose(-2, -1))
        matmul1 = relay.nn.matmul(q, key_t)

        # .div(math.sqrt(key.shape[-1]))
        # The original PyTorch code calculates sqrt of the *original* key's last dim (embed_dim)
        scale_factor = relay.const(1.0 / math.sqrt(float(embed_dim)), dtype="float32")
        scaled_matmul = relay.op.tensor.multiply(matmul1, scale_factor)

        # .softmax(dim=-1)
        softmax_out = relay.nn.softmax(scaled_matmul, axis=-1)

        # .matmul(value)
        result = relay.nn.matmul(softmax_out, v)

        # Create a Relay function and module
        func = relay.Function([q, k, v], result)
        mod = tvm.IRModule.from_expr(func)
        mod = run_infer_type(mod) # Infer types to make the graph explicit

        # Create a temporary directory for graph dumps
        tmpdir = tempfile.mkdtemp()
        try:
            # Dump the initial (unoptimized) graph
            input_graph_dot_path = os.path.join(tmpdir, "input_graph.dot")
            input_graph_svg_path = os.path.join(tmpdir, "input_graph.svg")
            
            # Convert Relay function to Graphviz Digraph and save
            initial_digraph = tvm.relay.transform.to_graph(mod["main"])
            pydot.graph_from_dot_data(str(initial_digraph))[0].write_dot(input_graph_dot_path)
            pydot.graph_from_dot_data(str(initial_digraph))[0].write_svg(input_graph_svg_path)

            # Apply Relay optimizations.
            # Torch.compile's SDPA rewriter would replace this sequence with a fused op.
            # In TVM, we can apply fusion passes which would attempt to combine operations.
            # For demonstration, we use a sequence of standard Relay passes including FuseOps.
            with transform.PassContext(opt_level=3): # opt_level=3 includes fusion passes
                optimized_mod = transform.Sequential(
                    [
                        transform.EliminateCommonSubexpr(),
                        transform.SimplifyInference(),
                        transform.FoldConstant(),
                        transform.FuseOps(fuse_opt_level=2), # Higher fuse_opt_level encourages more aggressive fusion
                    ]
                )(mod)
            optimized_mod = run_infer_type(optimized_mod)

            # Dump the optimized graph
            output_graph_dot_path = os.path.join(tmpdir, "output_graph.dot")
            output_graph_svg_path = os.path.join(tmpdir, "output_graph.svg")

            optimized_digraph = tvm.relay.transform.to_graph(optimized_mod["main"])
            pydot.graph_from_dot_data(str(optimized_digraph))[0].write_dot(output_graph_dot_path)
            pydot.graph_from_dot_data(str(optimized_digraph))[0].write_svg(output_graph_svg_path)


            # Verify that graph dump files were created
            found_input_dot = False
            found_output_dot = False
            found_input_svg = False
            found_output_svg = False
            
            for filepath_object in glob.glob(tmpdir + "/*"):
                if os.path.isfile(filepath_object):
                    if filepath_object.endswith("input_graph.dot"):
                        found_input_dot = True
                    elif filepath_object.endswith("output_graph.dot"):
                        found_output_dot = True
                    elif filepath_object.endswith("input_graph.svg"):
                        found_input_svg = True
                    elif filepath_object.endswith("output_graph.svg"):
                        found_output_svg = True

            assert found_input_dot
            assert found_output_svg # The original test asserts both .svg files
            assert found_input_svg
            assert found_output_dot # Added for completeness


        finally:
            shutil.rmtree(tmpdir) # Clean up temporary directory
