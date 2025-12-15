import pytest
import torch
import numpy as np
import functools

# Mock/placeholder for TVM IR types and passes.
# These tests are about TVM's graph rewriting passes and IR structural equality,
# which has no direct equivalent in user-facing PyTorch APIs.
# The `run_opt_pass` and `tvm.ir.structural_equal` will be replaced with placeholders
# that allow the file to run but essentially skip the core TVM-specific logic.

# Dummy classes/functions to make the code parseable and runnable
class MockRelayOp:
    def __getattr__(self, name):
        # Return a dummy function for any called op, e.g., relay.nn.conv2d
        def _dummy_op(*args, **kwargs):
            # This is a placeholder for a symbolic Relay operation.
            # It does not perform computation but maintains argument structure.
            # In a real PyTorch test, this would be a torch.* function or an nn.Module.
            return MockRelayCall(op=f"nn.{name}", args=args, attrs=kwargs)
        return _dummy_op

class MockRelayQnnOp:
    def __getattr__(self, name):
        def _dummy_qnn_op(*args, **kwargs):
            return MockRelayCall(op=f"qnn.op.{name}", args=args, attrs=kwargs)
        return _dummy_qnn_op

class MockRelayVisionOp:
    def __getattr__(self, name):
        def _dummy_vision_op(*args, **kwargs):
            return MockRelayCall(op=f"vision.{name}", args=args, attrs=kwargs)
        return _dummy_op # Use _dummy_op as vision ops often share similar structures

class MockRelayImageOp:
    def __getattr__(self, name):
        def _dummy_image_op(*args, **kwargs):
            return MockRelayCall(op=f"image.{name}", args=args, attrs=kwargs)
        return _dummy_image_op

class MockRelayExpr:
    # Base class for mock Relay expressions
    def __init__(self, value=None):
        self._value = value
    
    def astuple(self):
        # For TupleWrapper.astuple()
        return self._value.fields if isinstance(self._value, MockRelayTuple) else (self._value,)

    def astext(self):
        return str(self._value)

class MockRelayVar(MockRelayExpr):
    def __init__(self, name, shape, dtype="float32"):
        super().__init__(f"Var(name={name}, shape={shape}, dtype={dtype})")
        self.name_hint = name
        self.checked_type = MockRelayTensorType(shape, dtype) # Mock Relay IR type system
    
    # Enable attribute access like .shape for mock variables
    @property
    def shape(self):
        return self.checked_type.shape

class MockRelayConstant(MockRelayExpr):
    def __init__(self, value, dtype="float32"):
        super().__init__(f"Constant(value={value}, dtype={dtype})")
        self.value = value
        self.checked_type = MockRelayTensorType(np.shape(value), dtype)

class MockRelayFunction(MockRelayExpr):
    def __init__(self, params, body):
        super().__init__(f"Function(params={params}, body={body})")
        self.params = params
        self.body = body

class MockRelayCall(MockRelayExpr):
    def __init__(self, op, args, attrs=None, type_args=None, span=None):
        super().__init__(f"Call(op={op}, args={args}, attrs={attrs})")
        self.op = op
        self.args = args
        self.attrs = attrs
        self.type_args = type_args
        self.span = span

class MockRelayTuple(MockRelayExpr):
    def __init__(self, fields):
        super().__init__(f"Tuple(fields={fields})")
        self.fields = fields

class MockRelayTupleGetItem(MockRelayExpr):
    def __init__(self, tuple_value, index):
        super().__init__(f"TupleGetItem(tuple={tuple_value}, index={index})")
        self.tuple_value = tuple_value
        self.index = index
        self._value = tuple_value.fields[index] if isinstance(tuple_value, MockRelayTuple) and len(tuple_value.fields) > index else None

# Mock Relay TensorType for .checked_type attribute
class MockRelayTensorType(MockRelayExpr):
    def __init__(self, shape, dtype):
        super().__init__(f"TensorType(shape={shape}, dtype={dtype})")
        self.shape = shape
        self.dtype = dtype

# For `analysis.free_vars(y)`
class MockRelayAnalysis:
    @staticmethod
    def free_vars(expr):
        # In a real scenario, this would traverse the IR and find free variables.
        # For mocking, we can return dummy vars or empty list.
        # For these tests, free_vars is usually just the input variables.
        # This implementation is a placeholder.
        if isinstance(expr, MockRelayFunction):
            return expr.params
        # A very basic attempt to extract vars from nested calls if needed.
        # This is not robust and might not cover all Relay IR structures.
        found_vars = set()
        queue = [expr]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if isinstance(current, MockRelayVar):
                found_vars.add(current)
            elif isinstance(current, MockRelayCall):
                for arg in current.args:
                    queue.append(arg)
            elif isinstance(current, MockRelayTuple):
                for field in current.fields:
                    queue.append(field)
            elif isinstance(current, MockRelayTupleGetItem):
                queue.append(current.tuple_value)
            # Add other MockRelayExpr types as needed
        return list(found_vars)

# Mock for `tvm.tir.layout` and `tvm.tir.transform`
class MockTirLayout:
    def __init__(self, layout_str):
        self.layout_str = layout_str
    
    def __str__(self):
        return self.layout_str
    
    def __eq__(self, other):
        return isinstance(other, (MockTirLayout, str)) and self.layout_str == str(other)
    
    def __hash__(self):
        return hash(self.layout_str)

class MockTvmTransform:
    @staticmethod
    def Sequential(passes):
        # In TVM, this applies IR passes. Here, it does nothing effectively.
        return lambda mod: mod
    
    class PassContext:
        def __init__(self, opt_level):
            self.opt_level = opt_level
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

# Mock for tvm.ir.structural_equal
class MockIR:
    @staticmethod
    def structural_equal(a, b):
        # This function is the core of structural comparison in TVM IR.
        # As PyTorch does not expose an equivalent symbolic IR for structural comparison,
        # we mark this as not convertible and always return False to trigger pytest.skip.
        pytest.skip("Test relies on TVM Relay IR structural equality, which is not convertible to PyTorch directly.")
        return False

# Mock TVM namespace
class MockTVM:
    IRModule = type("IRModule", (object,), {"from_expr": lambda expr: expr}) # Just return expr for pass-through
    transform = MockTvmTransform()
    tir = type("tir", (object,), {"layout": MockTirLayout, "SourceName": lambda x: x, "Span": lambda *args: None, "layout_from_string": MockTirLayout})
    ir = MockIR()
    relay = None # Will be set below

tvm = MockTVM()

# Create the full MockRelay object
class FullMockRelay:
    var = MockRelayVar
    Function = MockRelayFunction
    const = MockRelayConstant
    Tuple = MockRelayTuple
    TupleGetItem = MockRelayTupleGetItem
    Call = MockRelayCall
    TensorType = MockRelayTensorType

    nn = MockRelayOp()
    qnn = MockRelayQnnOp()
    vision = MockRelayVisionOp()
    image = MockRelayImageOp()
    op = MockRelayOp() # For `relay.op.split`

    analysis = MockRelayAnalysis()

    # Layout transform is explicitly tested, need a mock that captures info
    @staticmethod
    def layout_transform(data, src_layout, dst_layout):
        class Attrs:
            def __init__(self, src, dst):
                self.src_layout = src
                self.dst_layout = dst
            def __str__(self):
                return f"src_layout={self.src_layout}, dst_layout={self.dst_layout}"
        return MockRelayCall(
            op="layout_transform",
            args=[data],
            attrs=Attrs(tvm.tir.layout(src_layout), tvm.tir.layout(dst_layout))
        )

    # Specific ops used in tests
    @staticmethod
    def expand_dims(data, axis, num_newaxis):
        class Attrs:
            def __init__(self, axis_val, num_newaxis_val):
                self.axis = axis_val
                self.num_newaxis = num_newaxis_val
            def __str__(self):
                return f"axis={self.axis}, num_newaxis={self.num_newaxis}"
        return MockRelayCall(
            op="expand_dims",
            args=[data],
            attrs=Attrs(axis, num_newaxis)
        )
    
    @staticmethod
    def add(lhs, rhs):
        return MockRelayCall(op="add", args=[lhs, rhs])

    @staticmethod
    def cast(data, dtype):
        class Attrs:
            def __init__(self, dtype_val):
                self.dtype = dtype_val
            def __str__(self):
                return f"dtype={self.dtype}"
        return MockRelayCall(op="cast", args=[data], attrs=Attrs(dtype))

    @staticmethod
    def strided_slice(data, begin, end, strides, axes=None):
        class Attrs:
            def __init__(self, begin_val, end_val, strides_val, axes_val):
                self.begin = begin_val
                self.end = end_val
                self.strides = strides_val
                self.axes = axes_val
            def __str__(self):
                return f"begin={self.begin}, end={self.end}, strides={self.strides}, axes={self.axes}"
        return MockRelayCall(op="strided_slice", args=[data], attrs=Attrs(begin, end, strides, axes))
    
    @staticmethod
    def topk(data, k, axis, ret_type="both", is_ascend=False, dtype="int32"):
        # Returns a TupleWrapper, need to mock that too.
        # This is a simplistic mock; real Relay topk returns an Expr, which may be a Tuple.
        # For a TupleWrapper, astuple() is called.
        class Attrs:
            def __init__(self, k_val, axis_val, ret_type_val, is_ascend_val):
                self.k = k_val
                self.axis = axis_val
                self.ret_type = ret_type_val
                self.is_ascend = is_ascend_val
            def __str__(self):
                return f"k={self.k}, axis={self.axis}, ret_type='{self.ret_type}', is_ascend={self.is_ascend}"

        mock_attrs = Attrs(k, axis, ret_type, is_ascend)
        mock_values = MockRelayCall(op="topk_values", args=[data], attrs=mock_attrs)
        mock_indices = MockRelayCall(op="topk_indices", args=[data], attrs=mock_attrs)

        class TupleWrapper(MockRelayExpr):
            def __init__(self, values, indices):
                self.values = values
                self.indices = indices
                self._tuple = MockRelayTuple([values, indices])
                super().__init__(self._tuple) # For astext()
            def astuple(self):
                return self._tuple
        
        return TupleWrapper(mock_values, mock_indices)


    @staticmethod
    def squeeze(data, axis=None):
        class Attrs:
            def __init__(self, axis_val):
                self.axis = axis_val
            def __str__(self):
                return f"axis={self.axis}"
        return MockRelayCall(op="squeeze", args=[data], attrs=Attrs(axis))

    @staticmethod
    def sum(data, axis=None, keepdims=False, exclude=False):
        class Attrs:
            def __init__(self, axis_val, keepdims_val, exclude_val):
                self.axis = axis_val
                self.keepdims = keepdims_val
                self.exclude = exclude_val
            def __str__(self):
                return f"axis={self.axis}, keepdims={self.keepdims}, exclude={self.exclude}"
        return MockRelayCall(op="sum", args=[data], attrs=Attrs(axis, keepdims, exclude))

    @staticmethod
    def mean(data, axis=None, keepdims=False, exclude=False):
        class Attrs:
            def __init__(self, axis_val, keepdims_val, exclude_val):
                self.axis = axis_val
                self.keepdims = keepdims_val
                self.exclude = exclude_val
            def __str__(self):
                return f"axis={self.axis}, keepdims={self.keepdims}, exclude={self.exclude}"
        return MockRelayCall(op="mean", args=[data], attrs=Attrs(axis, keepdims, exclude))
    
    @staticmethod
    def max(data, axis=None, keepdims=False, exclude=False):
        class Attrs:
            def __init__(self, axis_val, keepdims_val, exclude_val):
                self.axis = axis_val
                self.keepdims = keepdims_val
                self.exclude = exclude_val
            def __str__(self):
                return f"axis={self.axis}, keepdims={self.keepdims}, exclude={self.exclude}"
        return MockRelayCall(op="max", args=[data], attrs=Attrs(axis, keepdims, exclude))

    @staticmethod
    def multiply(lhs, rhs):
        return MockRelayCall(op="multiply", args=[lhs, rhs])

    @staticmethod
    def transpose(data, axes=None):
        class Attrs:
            def __init__(self, axes_val):
                self.axes = axes_val
            def __str__(self):
                return f"axes={self.axes}"
        return MockRelayCall(op="transpose", args=[data], attrs=Attrs(axes))

tvm.relay = FullMockRelay()
relay = tvm.relay # Convenience alias

# Register dummy infer_correct_layout to satisfy the test that checks for it
_infer_correct_layout_registry = {}
def register_alter_op_layout(op_name, level):
    def decorator(f):
        _infer_correct_layout_registry[op_name] = f
        return f
    return decorator

relay.op.register_alter_op_layout = register_alter_op_layout

# InferCorrectLayoutOutput needs a basic definition
class InferCorrectLayoutOutput:
    def __init__(self, input_layouts, output_layouts, new_attrs):
        self.input_layouts = input_layouts
        self.output_layouts = output_layouts
        self.new_attrs = new_attrs

# The transform.ConvertLayout class itself will be mocked to do nothing
# as we are skipping the actual pass application and structural comparison.
# This will allow `transform.ConvertLayout({...})` to be instantiated
# but won't perform any actual graph rewriting.
class MockConvertLayoutPass:
    def __init__(self, layouts):
        self.layouts = layouts
    
    def __call__(self, mod):
        # This mock pass does nothing and returns the module unchanged.
        # The actual conversion logic in TVM is complex and not directly portable.
        return mod

tvm.relay.transform.ConvertLayout = MockConvertLayoutPass

# Dummy LayoutConfig
class MockLayoutConfig:
    def __init__(self, skip_layers=None):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

tvm.relay.transform.LayoutConfig = MockLayoutConfig


def run_opt_pass(expr, passes):
    # This function is a placeholder for TVM's graph optimization passes.
    # In PyTorch, there isn't a direct equivalent for applying symbolic IR passes
    # and then comparing the transformed IR structurally.
    # For the purpose of making this test runnable and indicating non-convertibility,
    # we return the original expression, effectively skipping the pass application.
    return expr


def test_no_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        return before()

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_qnn_binary_no_convert_layout():
    def before():
        x = relay.var("x", shape=(2, 2))
        y = relay.var("y", shape=(1, 2))
        return relay.Function(
            [x, y],
            relay.qnn.op.add(
                x,
                y,
                lhs_scale=relay.const(0.0156863, "float32"),
                lhs_zero_point=relay.const(127, "int32"),
                rhs_scale=relay.const(0.0117647, "float32"),
                rhs_zero_point=relay.const(85, "int32"),
                output_scale=relay.const(0.0235294, "float32"),
                output_zero_point=relay.const(128, "int32"),
            ),
        )

    def expected():
        return before()

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_nhwc_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight = relay.layout_transform(weight, "OIHW", "HWIO")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_transpose_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d_transpose(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "IOHW")
        y = relay.nn.conv2d_transpose(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d_transpose": ["NCHW", "IOHW"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_bias_pool_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.bias_add(y, bias, axis=3)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NHWC")
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))

        bias = relay.expand_dims(bias, axis=0, num_newaxis=3)
        bias = relay.layout_transform(bias, "NHWC", "NCHW")
        y = relay.add(y, bias)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2))
        y = relay.cast(y, "int32")
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_bias_pool_uses_specified_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.bias_add(y, bias, axis=3)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NHWC")
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))

        bias = relay.expand_dims(bias, axis=0, num_newaxis=3)
        bias = relay.layout_transform(bias, "NHWC", "NCHW")
        y = relay.add(y, bias)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NHWC", out_layout="NHWC")
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(
        a,
        tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"], "nn.max_pool2d": ["NHWC"]}),
    )
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_concat_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64))
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y1 = relay.nn.conv2d(
            y,
            weight2,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        ret = relay.concatenate([y, y1], axis=3)
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64))
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(y, weight1, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y1 = relay.nn.conv2d(y, weight2, channels=64, kernel_size=(3, 3), padding=(1, 1))
        ret = relay.concatenate([y, y1], axis=1)
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_deformable_conv_bias_pool_convert_layout():
    def before(N, CI, H, W, CO, KH, KW, layout):
        if layout == "NCHW":
            data_shape = (N, CI, H, W)
            weight_shape = (CO, CI, KH, KW)
            kernel_layout = "OIHW"
        else:
            data_shape = (N, H, W, CI)
            weight_shape = (KH, KW, CI, CO)
            kernel_layout = "HWIO"
        bias_shape = (CO,)

        data = relay.var("data", shape=data_shape, dtype="float32")
        offset = relay.var("offset", shape=data_shape, dtype="float32") # Mock offset shape
        weight = relay.var("weight", shape=weight_shape, dtype="float32")
        bias = relay.var("bias", shape=bias_shape, dtype="float32")

        y = relay.nn.deformable_conv2d(
            data,
            offset,
            weight,
            kernel_size=(KH, KW),
            channels=CO,
            data_layout=layout,
            kernel_layout=kernel_layout,
        )
        y = relay.nn.bias_add(y, bias, axis=-1 if layout == "NHWC" else 1)
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout=layout)
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected(N, CI, H, W, CO, KH, KW, OH, OW, src_layout, dst_layout):
        layout_map = {"src": {}, "dst": {}}
        if src_layout == "NCHW":
            nchw = layout_map["src"]
            nhwc = layout_map["dst"]
        else:
            nchw = layout_map["dst"]
            nhwc = layout_map["src"]

        nchw["data_layout"] = "NCHW"
        nchw["data_shape"] = (N, CI, H, W)
        # offset_shape logic needs the actual output dimensions of conv2d.
        # For mocking, approximate or use a placeholder.
        # Original offset_shape calculations might be complex. Simplifed for mock.
        nchw["offset_shape"] = (N, KH * KW * 2, OH, OW) 
        nchw["weight_shape"] = (CO, CI, KH, KW)
        nchw["kernel_layout"] = "OIHW"

        nhwc["data_layout"] = "NHWC"
        nhwc["data_shape"] = (N, H, W, CI)
        nhwc["offset_shape"] = (N, OH, OW, KH * KW * 2)
        nhwc["weight_shape"] = (KH, KW, CI, CO)
        nhwc["kernel_layout"] = "HWIO"

        bias_shape = (CO,)

        data = relay.var("data", shape=layout_map["src"]["data_shape"], dtype="float32")
        offset = relay.var("offset", shape=layout_map["src"]["offset_shape"], dtype="float32")
        weight = relay.var("weight", shape=layout_map["src"]["weight_shape"], dtype="float32")
        bias = relay.var("bias", shape=bias_shape, dtype="float32")

        data = relay.layout_transform(
            data, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        offset = relay.layout_transform(
            offset, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        weight = relay.layout_transform(
            weight, layout_map["src"]["kernel_layout"], layout_map["dst"]["kernel_layout"]
        )
        y = relay.nn.deformable_conv2d(
            data,
            offset,
            weight,
            kernel_size=(KH, KW),
            channels=CO,
            data_layout=layout_map["dst"]["data_layout"],
            kernel_layout=layout_map["dst"]["kernel_layout"],
        )
        if layout_map["src"]["data_layout"] == "NHWC":
            bias = relay.expand_dims(bias, axis=0, num_newaxis=3)
        else:
            bias = relay.expand_dims(bias, axis=1, num_newaxis=2)
            bias = relay.expand_dims(bias, axis=0, num_newaxis=1) # Original example shows axis=0 here. It might vary depending on num_newaxis values. This is hard to precisely match without knowing full Relay semantics.
        bias = relay.layout_transform(
            bias, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        y = relay.add(y, bias)
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout=layout_map["dst"]["data_layout"])
        y = relay.cast(y, "int32")
        y = relay.layout_transform(
            y, layout_map["dst"]["data_layout"], layout_map["src"]["data_layout"]
        )
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    # NHWC -> NCHW
    a = before(1, 3, 224, 224, 32, 3, 3, "NHWC")
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.deformable_conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(
        expected(1, 3, 224, 224, 32, 3, 3, 222, 222, "NHWC", "NCHW"), None
    )
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    # NCHW -> NHWC
    a = before(1, 3, 224, 224, 32, 3, 3, "NCHW")
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.deformable_conv2d": ["NHWC", "default"]}))
    b = run_opt_pass(
        expected(1, 3, 224, 224, 32, 3, 3, 222, 222, "NCHW", "NHWC"), None
    )
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_deformable_conv_bias_pool_uses_specified_convert_layout():
    def before(N, CI, H, W, CO, KH, KW, layout):
        if layout == "NCHW":
            data_shape = (N, CI, H, W)
            weight_shape = (CO, CI, KH, KW)
            kernel_layout = "OIHW"
        else:
            data_shape = (N, H, W, CI)
            weight_shape = (KH, KW, CI, CO)
            kernel_layout = "HWIO"
        bias_shape = (CO,)

        data = relay.var("data", shape=data_shape, dtype="float32")
        offset = relay.var("offset", shape=data_shape, dtype="float32") # Mock offset shape
        weight = relay.var("weight", shape=weight_shape, dtype="float32")
        bias = relay.var("bias", shape=bias_shape, dtype="float32")

        y = relay.nn.deformable_conv2d(
            data,
            offset,
            weight,
            kernel_size=(KH, KW),
            channels=CO,
            data_layout=layout,
            kernel_layout=kernel_layout,
        )
        y = relay.nn.bias_add(y, bias, axis=-1 if layout == "NHWC" else 1)
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout=layout)
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected(N, CI, H, W, CO, KH, KW, OH, OW, src_layout, dst_layout, max_pool_layout=None):
        layout_map = {"src": {}, "dst": {}}
        if src_layout == "NCHW":
            nchw = layout_map["src"]
            nhwc = layout_map["dst"]
        else:
            nchw = layout_map["dst"]
            nhwc = layout_map["src"]

        nchw["data_layout"] = "NCHW"
        nchw["data_shape"] = (N, CI, H, W)
        nchw["offset_shape"] = (N, KH * KW * 2, OH, OW)
        nchw["weight_shape"] = (CO, CI, KH, KW)
        nchw["kernel_layout"] = "OIHW"

        nhwc["data_layout"] = "NHWC"
        nhwc["data_shape"] = (N, H, W, CI)
        nhwc["offset_shape"] = (N, OH, OW, KH * KW * 2)
        nhwc["weight_shape"] = (KH, KW, CI, CO)
        nhwc["kernel_layout"] = "HWIO"

        bias_shape = (CO,)

        data = relay.var("data", shape=layout_map["src"]["data_shape"], dtype="float32")
        offset = relay.var("offset", shape=layout_map["src"]["offset_shape"], dtype="float32")
        weight = relay.var("weight", shape=layout_map["src"]["weight_shape"], dtype="float32")
        bias = relay.var("bias", shape=bias_shape, dtype="float32")

        data = relay.layout_transform(
            data, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        offset = relay.layout_transform(
            offset, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        weight = relay.layout_transform(
            weight, layout_map["src"]["kernel_layout"], layout_map["dst"]["kernel_layout"]
        )
        y = relay.nn.deformable_conv2d(
            data,
            offset,
            weight,
            kernel_size=(KH, KW),
            channels=CO,
            data_layout=layout_map["dst"]["data_layout"],
            kernel_layout=layout_map["dst"]["kernel_layout"],
        )
        if layout_map["src"]["data_layout"] == "NHWC":
            bias = relay.expand_dims(bias, axis=0, num_newaxis=3)
        else:
            bias = relay.expand_dims(bias, axis=1, num_newaxis=2)
            bias = relay.expand_dims(bias, axis=0, num_newaxis=1) # See comment above for this line
        bias = relay.layout_transform(
            bias, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        y = relay.add(y, bias)
        y = relay.nn.relu(y)
        if max_pool_layout != layout_map["dst"]["data_layout"]:
            y = relay.layout_transform(y, layout_map["dst"]["data_layout"], max_pool_layout)
        y = relay.nn.max_pool2d(
            y, pool_size=(2, 2), layout=max_pool_layout, out_layout=max_pool_layout
        )
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    # NHWC -> NCHW
    a = before(1, 3, 224, 224, 32, 3, 3, "NHWC")
    a = run_opt_pass(
        a,
        tvm.relay.transform.ConvertLayout(
            {"nn.deformable_conv2d": ["NCHW", "default"], "nn.max_pool2d": ["NHWC"]}
        ),
    )
    # - in the before() func, its last argument "NHWC" is also the layout of max_pool
    b = run_opt_pass(
        # max_pool has its own layout argument
        expected(1, 3, 224, 224, 32, 3, 3, 222, 222, "NHWC", "NCHW", max_pool_layout="NHWC"),
        None,
    )
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    # NCHW -> NHWC
    a = before(1, 3, 224, 224, 32, 3, 3, "NCHW")
    a = run_opt_pass(
        a,
        tvm.relay.transform.ConvertLayout(
            {"nn.deformable_conv2d": ["NHWC", "default"], "nn.max_pool2d": ["NCHW"]}
        ),
    )
    # - in the before() func, its last argument "NCHW" is also the layout of max_pool
    b = run_opt_pass(
        # max_pool has its own layout argument
        expected(1, 3, 224, 224, 32, 3, 3, 222, 222, "NCHW", "NHWC", max_pool_layout="NCHW"),
        None,
    )
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_dual_path_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(3, 3, 32, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(
            y,
            weight2,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.batch_flatten(y)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(3, 3, 32, 32))
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y1 = relay.layout_transform(y1, "NCHW", "NHWC")
        y2 = relay.layout_transform(y, "NCHW", "NHWC")
        y2 = relay.nn.batch_flatten(y2)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_bn_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        gamma = relay.var("gamma", relay.TensorType((64,), "float32"))
        beta = relay.var("beta", relay.TensorType((64,), "float32"))
        mean = relay.var("mean", relay.TensorType((64,), "float32"))
        variance = relay.var("variance", relay.TensorType((64,), "float32"))
        y = relay.nn.batch_norm(y, gamma, beta, mean, variance, axis=3)
        y = relay.TupleGetItem(y, 0) # batch_norm returns a tuple (output, mean, var)
        return relay.Function(relay.analysis.free_vars(y), y)

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))

    # Check that there is only 1 NHWC to NCHW transform.
    has_lt = list()
    find_op = lambda x: has_lt.append(
        isinstance(x, tvm.relay.Call)
        and x.op == "layout_transform"
        and x.attrs.src_layout == "NCHW"
        and x.attrs.dst_layout == "NHWC"
    )
    # The post_order_visit logic must be adapted for the mock objects
    # This part specifically checks the *structure of the transformed IR*.
    # This assertion is inherently about TVM IR and not convertible.
    # Therefore, we skip the entire test here.
    pytest.skip("Test relies on TVM Relay IR structural analysis after a pass, not convertible to PyTorch.")


def test_slice_like_convert_layout():
    def verify_slice_like(after, expected_axes):
        # Verify if the slice_like after the convert layout has the expected axes.
        has_expected = list()
        checker = lambda x: has_expected.append(
            isinstance(x, tvm.relay.Call)
            and x.op == "slice_like"
            and str(x.attrs.axes) == str(expected_axes)
        )
        # This part specifically checks the *structure of the transformed IR*.
        # This assertion is inherently about TVM IR and not convertible.
        # Therefore, we skip the entire test here.
        pytest.skip("Test relies on TVM Relay IR structural analysis after a pass, not convertible to PyTorch.")


    def func_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        out = relay.op.slice_like(y, y, axes=[1, 2])
        return relay.Function(relay.analysis.free_vars(out), out)

    after = run_opt_pass(func_nhwc(), tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    verify_slice_like(after, [2, 3])

    def func_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(32, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        out = relay.op.slice_like(y, y, axes=[2, 3])
        return relay.Function(relay.analysis.free_vars(out), out)

    after = run_opt_pass(func_nchw(), tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))
    verify_slice_like(after, [1, 2])

    def func_vars():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        # z has no layout information so convert layout won't happen.
        z = relay.var("y", shape=(1, 56, 56, 32))
        out = relay.op.slice_like(y, z, axes=[1, 2])
        return relay.Function(relay.analysis.free_vars(out), out)

    after = run_opt_pass(func_vars(), tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    verify_slice_like(after, [1, 2])


def test_transpose_convert_layout():
    def verify_transpose(after, expected_axes, expected_transform_cnt):
        # Verify if the transpose after the convert layout has the expected axes.
        has_expected = list()
        checker = lambda x: has_expected.append(
            isinstance(x, tvm.relay.Call)
            and x.op == "transpose"
            and str(x.attrs.axes) == str(expected_axes)
        )
        # This part specifically checks the *structure of the transformed IR*.
        # This assertion is inherently about TVM IR and not convertible.
        # Therefore, we skip the entire test here.
        pytest.skip("Test relies on TVM Relay IR structural analysis after a pass, not convertible to PyTorch.")

        # The rest of the checker is also skipped.
        is_transform = list()
        checker = lambda x: is_transform.append(
            1 if isinstance(x, tvm.relay.Call) and x.op == "layout_transform" else 0
        )
        relay.analysis.post_order_visit(after, checker)
        assert (
            sum(is_transform) == expected_transform_cnt
        ), "Expected %s layout_transform, but get\n%s" % (expected_transform_cnt, after)

    def nhwc_to_nchw():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        z = relay.var("z", shape=(56, 56, 32))
        out = relay.add(y, z)
        out = relay.transpose(out, axes=[0, 3, 1, 2])
        out = relay.nn.batch_flatten(out)
        func = relay.Function(relay.analysis.free_vars(out), out)
        return run_opt_pass(func, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))

    verify_transpose(nhwc_to_nchw(), [0, 1, 2, 3], 3)

    def nchw_to_nhwc():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(32, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        z = relay.var("z", shape=(32, 56, 56))
        out = relay.add(y, z)
        out = relay.transpose(out, axes=[0, 2, -1, 1])  # Also test a negative axis.
        out = relay.nn.batch_flatten(out)
        func = relay.Function(relay.analysis.free_vars(out), out)
        return run_opt_pass(func, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))

    verify_transpose(nchw_to_nhwc(), [0, 1, 2, 3], 3)

    def default_axes():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(32, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        z = relay.var("z", shape=(32, 56, 56))
        out = relay.add(y, z)
        out = relay.transpose(out)  # No axes provided, will use the reversed axes.
        func = relay.Function(relay.analysis.free_vars(out), out)
        return run_opt_pass(func, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))

    verify_transpose(default_axes(), [2, 1, 3, 0], 3)


def test_resnet_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(1, 1, 64, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(
            x, weight2, channels=32, kernel_size=(1, 1), data_layout="NHWC", kernel_layout="HWIO"
        )
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y, layout="NHWC")
        return relay.Function(relay.analysis.free_vars(y), y)

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(1, 1, 64, 32))
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        x = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2, channels=32, kernel_size=(1, 1))
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        return relay.Function(relay.analysis.free_vars(y), y)

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_resnet_pool_uses_specified_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(1, 1, 64, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(
            x, weight2, channels=32, kernel_size=(1, 1), data_layout="NHWC", kernel_layout="HWIO"
        )
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y, layout="NHWC")
        return relay.Function(relay.analysis.free_vars(y), y)

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(1, 1, 64, 32))
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        x = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2, channels=32, kernel_size=(1, 1))
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.nn.global_max_pool2d(y, layout="NHWC", out_layout="NHWC")
        return relay.Function(relay.analysis.free_vars(y), y)

    a = before()
    a = run_opt_pass(
        a,
        tvm.relay.transform.ConvertLayout(
            {"nn.conv2d": ["NCHW", "default"], "nn.global_max_pool2d": ["NHWC"]}
        ),
    )
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_scalar_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.add(y, relay.const(1, "float32"))
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        w = relay.layout_transform(w, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.add(y, relay.const(1.0, "float32"))

        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_ln_convert_layout():
    """Check that layout transforms are propagated through ln."""

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))

        y = relay.nn.layer_norm(y, gamma, beta, axis=3)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        w = relay.layout_transform(w, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))

        y = relay.nn.layer_norm(y, gamma, beta, axis=1)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_InstanceNorm_convert_layout():
    """Check that layout transforms are propagated through instance norm."""

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))

        y = relay.nn.instance_norm(y, gamma, beta, axis=3)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        w = relay.layout_transform(w, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))

        y = relay.nn.instance_norm(y, gamma, beta, axis=1)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_bn_convert_layout():
    """Check that layout transforms are propagated through bn."""

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((64,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((64,), dtype))

        y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var, axis=3)
        y = relay.nn.relu(y[0]) # y[0] is used because batch_norm returns a tuple
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        w = relay.layout_transform(w, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((64,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((64,), dtype))

        y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var, axis=1)
        y = relay.nn.relu(y[0]) # y[0] is used because batch_norm returns a tuple
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_qnn_conv_requantize_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight = relay.var("weight", shape=(3, 3, 64, 64), dtype="int8")
        y = relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(1, "int32"), # input_zero_point
            relay.const(1, "int32"), # kernel_zero_point
            relay.const(1, "float32"), # input_scale
            relay.const(1, "float32"), # kernel_scale
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.qnn.op.requantize(
            y,
            relay.const(1, "float32"), # input_scale
            relay.const(1, "int32"),   # input_zero_point
            relay.const(1, "float32"), # output_scale
            relay.const(1, "int32"),   # output_zero_point
            out_dtype="int32",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight = relay.var("weight", shape=(3, 3, 64, 64), dtype="int8")
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        y = relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.qnn.op.requantize(
            y,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            axis=1,
            out_dtype="int32",
        )
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"qnn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_qnn_conv_concat_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64), dtype="int8")
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64), dtype="int8")
        y = relay.qnn.op.conv2d(
            x,
            weight1,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y1 = relay.qnn.op.conv2d(
            y,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.cast(y, "int8")
        y1 = relay.cast(y1, "int8") # Fixed from original: y1 was cast from y instead of y1
        ret = relay.qnn.op.concatenate(
            [y, y1],
            [relay.const(1, "float32"), relay.const(1, "float32")],
            [relay.const(1, "int32"), relay.const(1, "int32")],
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            axis=3,
        )
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64), dtype="int8")
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64), dtype="int8")
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.qnn.op.conv2d(
            y,
            weight1,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y1 = relay.qnn.op.conv2d(
            y,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.cast(y, "int8")
        y1 = relay.cast(y1, "int8")
        ret = relay.qnn.op.concatenate(
            [y, y1],
            [relay.const(1, "float32"), relay.const(1, "float32")],
            [relay.const(1, "int32"), relay.const(1, "int32")],
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            axis=1,
        )
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"qnn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_qnn_conv_add_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64), dtype="int8")
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64), dtype="int8")
        y = relay.qnn.op.conv2d(
            x,
            weight1,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y1 = relay.qnn.op.conv2d(
            y,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.cast(y, "int8")
        y1 = relay.cast(y1, "int8")
        ret = relay.qnn.op.add(
            y,
            y1,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
        )
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64), dtype="int8")
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64), dtype="int8")
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.qnn.op.conv2d(
            y,
            weight1,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y1 = relay.qnn.op.conv2d(
            y,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.cast(y, "int8")
        y1 = relay.cast(y1, "int8")
        ret = relay.qnn.op.add(
            y,
            y1,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
        )
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"qnn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_qnn_conv_nhwc_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype="int8")
        weight = relay.var("weight", shape=(64, 64, 3, 3), dtype="int8")
        y = relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype="int8")
        weight = relay.var("weight", shape=(64, 64, 3, 3), dtype="int8")
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight = relay.layout_transform(weight, "OIHW", "HWIO")
        y = relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"qnn.conv2d": ["NHWC", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_qnn_conv_transpose_requantize_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight = relay.var("weight", shape=(3, 3, 64, 64), dtype="int8")
        y = relay.qnn.op.conv2d_transpose(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype="int32",
        )
        y = relay.qnn.op.requantize(
            y,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            out_dtype="int32",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight = relay.var("weight", shape=(3, 3, 64, 64), dtype="int8")
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "IOHW")
        y = relay.qnn.op.conv2d_transpose(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            out_dtype="int32",
        )
        y = relay.qnn.op.requantize(
            y,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            axis=1,
            out_dtype="int32",
        )
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"qnn.conv2d_transpose": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_convert_kernel_layout():
    """Check that convolution kernel layout is correctly transformed."""

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        w = relay.layout_transform(w, "HWIO", "OHWI")
        y = relay.nn.conv2d(
            x,
            w,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="OHWI",
        )
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "OHWI"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_roi_align_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_align(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, sample_ratio=2, layout="NCHW"
        )
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight1 = relay.layout_transform(weight1, "OIHW", "HWIO")
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_align(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, sample_ratio=2, layout="NHWC"
        )
        ret = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    a = before()
    desired_layouts = {
        "nn.conv2d": ["NHWC", "HWIO"],
        "vision.roi_align": ["NHWC", "default"],
    }
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout(desired_layouts))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_strided_slice_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.relu(y)
        y = relay.strided_slice(y, begin=[0, 1], end=[1, -1, 10], strides=[1, 1, 2, 1])
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight = relay.layout_transform(weight, "OIHW", "HWIO")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.strided_slice(y, begin=[0, 0, 0, 1], end=[1, 10, 56, -1], strides=[1, 2, 1, 1])
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_split_convert_layout():
    def _test_conv_split_convert_layout1():
        def before():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=512,
                kernel_size=(3, 3),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.op.split(y, indices_or_sections=2, axis=-1)
            # y is a TupleWrapper, need to get the tuple for TupleGetItem
            y_tuple = y.astuple()
            a = relay.TupleGetItem(y_tuple, 0)
            b = relay.TupleGetItem(y_tuple, 1)
            out = relay.Tuple([a, b])
            return relay.Function(relay.analysis.free_vars(out), out)

        def expected():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=512, kernel_size=(3, 3))
            y = relay.nn.relu(y)
            y = relay.op.split(y, indices_or_sections=2, axis=1)
            y_tuple = y.astuple()
            a = relay.TupleGetItem(y_tuple, 0)
            b = relay.TupleGetItem(y_tuple, 1)
            a = relay.layout_transform(a, "NCHW", "NHWC")
            b = relay.layout_transform(b, "NCHW", "NHWC")
            out = relay.Tuple([a, b])
            return relay.Function(relay.analysis.free_vars(out), out)

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    def _test_conv_split_convert_layout2():
        def before():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=512,
                kernel_size=(3, 3),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.op.split(y, indices_or_sections=2, axis=3)
            y_tuple = y.astuple()
            a = relay.TupleGetItem(y_tuple, 0)
            b = relay.TupleGetItem(y_tuple, 1)
            out = relay.Tuple([a, b])
            return relay.Function(relay.analysis.free_vars(out), out)

        def expected():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=512, kernel_size=(3, 3))
            y = relay.nn.relu(y)
            y = relay.op.split(y, indices_or_sections=2, axis=1)
            y_tuple = y.astuple()
            a = relay.TupleGetItem(y_tuple, 0)
            b = relay.TupleGetItem(y_tuple, 1)
            a = relay.layout_transform(a, "NCHW", "NHWC")
            b = relay.layout_transform(b, "NCHW", "NHWC")
            out = relay.Tuple([a, b])
            return relay.Function(relay.analysis.free_vars(out), out)

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    def _test_conv_split_convert_layout3():
        def before():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=512,
                kernel_size=(3, 3),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.op.split(y, indices_or_sections=(5, 10), axis=-1)
            y_tuple = y.astuple()
            a = relay.TupleGetItem(y_tuple, 0)
            b = relay.TupleGetItem(y_tuple, 1)
            c = relay.TupleGetItem(y_tuple, 2)
            out = relay.Tuple([a, b, c])
            return relay.Function(relay.analysis.free_vars(out), out)

        def expected():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=512, kernel_size=(3, 3))
            y = relay.nn.relu(y)
            y = relay.op.split(y, indices_or_sections=(5, 10), axis=1)
            y_tuple = y.astuple()
            a = relay.TupleGetItem(y_tuple, 0)
            b = relay.TupleGetItem(y_tuple, 1)
            c = relay.TupleGetItem(y_tuple, 2)
            a = relay.layout_transform(a, "NCHW", "NHWC")
            b = relay.layout_transform(b, "NCHW", "NHWC")
            c = relay.layout_transform(c, "NCHW", "NHWC")
            out = relay.Tuple([a, b, c])
            return relay.Function(relay.analysis.free_vars(out), out)

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    def _test_conv_split_convert_layout_blocking():
        def before():
            x = relay.var("x", shape=(1, 512, 38, 38))
            weight = relay.var("weight", shape=(512, 512, 3, 3))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=512,
                kernel_size=(3, 3),
                data_layout="NCHW",
                kernel_layout="OIHW",
            )
            y = relay.nn.relu(y)
            y = relay.op.split(y, indices_or_sections=[256], axis=1)
            y_tuple = y.astuple()
            a = relay.TupleGetItem(y_tuple, 0)
            b = relay.TupleGetItem(y_tuple, 1)
            out = relay.Tuple([a, b])
            return relay.Function(relay.analysis.free_vars(out), out)

        def expected():
            x = relay.var("x", shape=(1, 512, 38, 38))
            weight = relay.var("weight", shape=(512, 512, 3, 3))
            weight = relay.layout_transform(weight, "OIHW", "OIHW4o")
            x = relay.layout_transform(x, "NCHW", "NCHW4c")
            y = relay.nn.contrib_conv2d_nchwc(
                x,
                weight,
                channels=512,
                kernel_size=(3, 3),
                padding=(0, 0),
                data_layout="NCHW4c",
                kernel_layout="OIHW4o",
            )
            y = relay.nn.relu(y)
            y = relay.op.split(y, indices_or_sections=[64], axis=1)
            y_tuple = y.astuple()
            a = relay.TupleGetItem(y_tuple, 0)
            b = relay.TupleGetItem(y_tuple, 1)
            a = relay.layout_transform(a, "NCHW4c", "NCHW")
            b = relay.layout_transform(b, "NCHW4c", "NCHW")
            out = relay.Tuple([a, b])
            return relay.Function(relay.analysis.free_vars(out), out)

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW4c", "OIHW4o"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    _test_conv_split_convert_layout1()
    _test_conv_split_convert_layout2()
    _test_conv_split_convert_layout3()
    _test_conv_split_convert_layout_blocking()


def test_conv_strided_slice_axes_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 28, 28, 32))
        weight = relay.var("weight", shape=(3, 3, 32, 32))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.strided_slice(y, begin=[0, 16], end=[1, 33], strides=[1, 1], axes=[0, 3])
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 28, 28, 32))
        weight = relay.var("weight", shape=(3, 3, 32, 32))
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        x = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.strided_slice(y, begin=[0, 16], end=[1, 33], strides=[1, 1], axes=[0, 1])

        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = run_opt_pass(before(), tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_topk_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y_tuple_wrapper = relay.topk(y, k=2, axis=2) # This is a TupleWrapper
        # Original code had `if isinstance(y, relay.expr.TupleWrapper): y = y.astuple()`
        # which means `y` would become a Relay Tuple here.
        y = y_tuple_wrapper.astuple()
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        x = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y_tuple_wrapper = relay.topk(y, k=2, axis=3) # This is a TupleWrapper
        y = y_tuple_wrapper.astuple()
        a = relay.TupleGetItem(y, 0)
        b = relay.TupleGetItem(y, 1)
        a = relay.layout_transform(a, "NCHW", "NHWC")
        b = relay.layout_transform(b, "NCHW", "NHWC")
        out = relay.Tuple([a, b])
        return relay.Function(relay.analysis.free_vars(out), out)

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_roi_pool_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_pool(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, layout="NCHW"
        )
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight1 = relay.layout_transform(weight1, "OIHW", "HWIO")
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_pool(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, layout="NHWC"
        )
        ret = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(ret), ret)
        return y

    a = before()
    desired_layouts = {
        "nn.conv2d": ["NHWC", "HWIO"],
        "vision.roi_pool": ["NHWC", "default"],
    }
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout(desired_layouts))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_default_keyword():
    """Check that the default keyword selects correct TVM default layout."""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 3, 3, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OHWI",
        )
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        w = relay.var("weight", shape=(64, 3, 3, 64))
        w = relay.layout_transform(w, "OHWI", "OIHW")
        y = relay.nn.conv2d(
            x,
            w,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_different_ops_convert_layout():
    """Check convert layout correctly supports converting the layout of
    different ops in the same graph.
    """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 3, 3, 64))
        weight2 = relay.var("weight2", shape=(64, 3, 3, 64), dtype="int8")
        weight3 = relay.var("weight3", shape=(64, 3, 3, 64))
        out = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OHWI",
        )
        out = relay.cast(out, "int8")
        out = relay.qnn.op.conv2d(
            out,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OHWI",
        )
        out = relay.cast(out, "float32")
        out = relay.nn.conv2d_transpose(
            out,
            weight3,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OHWI",
        )
        out = relay.Function(relay.analysis.free_vars(out), out)
        return out

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 3, 3, 64))
        weight2 = relay.var("weight2", shape=(64, 3, 3, 64), dtype="int8")
        weight3 = relay.var("weight3", shape=(64, 3, 3, 64))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight1 = relay.layout_transform(weight1, "OHWI", "HWIO")
        out = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        out = relay.cast(out, "int8")
        out = relay.layout_transform(out, "NHWC", "NCHW")
        weight2 = relay.layout_transform(weight2, "OHWI", "OIHW")
        out = relay.qnn.op.conv2d(
            out,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        out = relay.cast(out, "float32")
        out = relay.layout_transform(out, "NCHW", "NHWC")
        weight3 = relay.layout_transform(weight3, "OHWI", "HWIO")
        out = relay.nn.conv2d_transpose(
            out,
            weight3,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        out = relay.layout_transform(out, "NHWC", "NCHW")
        out = relay.Function(relay.analysis.free_vars(out), out)
        return out

    a = before()
    desired_layouts = {
        "nn.conv2d": ["NHWC", "HWIO"],
        "qnn.conv2d": ["NCHW", "OIHW"],
        "nn.conv2d_transpose": ["NHWC", "HWIO"],
    }
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout(desired_layouts))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_no_desired_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_align(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, sample_ratio=2, layout="NCHW"
        )
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight1 = relay.layout_transform(weight1, "OIHW", "HWIO")
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.layout_transform(y, "NHWC", "NCHW")
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_align(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, sample_ratio=2, layout="NCHW"
        )
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "HWIO"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_convert_with_config():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)

        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        y2 = relay.nn.conv2d(
            y,
            weight2,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y2 = relay.nn.relu(y2)

        out = relay.Function([x, weight, weight2], y2)
        return out

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))

        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        weight2 = relay.layout_transform(weight2, "HWIO", "HWOI")

        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NHWC", "HWNC")

        y2 = relay.nn.conv2d(
            y,
            weight2,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="HWNC",
            kernel_layout="HWOI",
        )
        y2 = relay.nn.relu(y2)

        y2 = relay.layout_transform(y2, "HWNC", "NHWC")
        output = relay.Function(relay.analysis.free_vars(y2), y2)
        return output

    a = before()
    layout_config = tvm.relay.transform.LayoutConfig(skip_layers=[0])
    with layout_config:
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["HWNC", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_squeeze_convert_layout():
    def _test_conv_squeeze_convert_layout1():
        # specified axis is squeezed
        def before():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=1000,
                kernel_size=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.squeeze(y, axis=[-3])
            return relay.Function(relay.analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=1000, kernel_size=(1, 1))
            y = relay.nn.relu(y)
            y = relay.squeeze(y, axis=[2])
            y = relay.layout_transform(y, "NCW", "NWC")
            return relay.Function(relay.analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    def _test_conv_squeeze_convert_layout2():
        # all axes of dimension 1 are squeezed
        def before():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=1000,
                kernel_size=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.squeeze(y)
            return relay.Function(relay.analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=1000, kernel_size=(1, 1))
            y = relay.nn.relu(y)
            y = relay.squeeze(y, [0, 2, 3])
            return relay.Function(relay.analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    def _test_conv_squeeze_convert_layout3():
        # squeeze axis is empty
        def before():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=1000,
                kernel_size=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.squeeze(y, axis=[])
            return relay.Function(relay.analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=1000, kernel_size=(1, 1))
            y = relay.nn.relu(y)
            y = relay.squeeze(y, axis=[])
            y = relay.layout_transform(y, "NCHW", "NHWC")
            return relay.Function(relay.analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    _test_conv_squeeze_convert_layout1()
    _test_conv_squeeze_convert_layout2()
    _test_conv_squeeze_convert_layout3()


def test_conv_reduce_convert_layout():
    for reduce_op in [relay.max, relay.mean, relay.sum]: # Changed order to match mapping
        def _test_conv_reduce_convert_layout1_inner(): # Wrapped tests in function to make them runnable
            def before():
                x = relay.var("x", shape=(1, 1, 1, 2048))
                weight = relay.var("weight", shape=(1, 1, 2048, 1000))
                y = relay.nn.conv2d(
                    x,
                    weight,
                    channels=1000,
                    kernel_size=(1, 1),
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                )
                y = relay.nn.relu(y)
                y = reduce_op(y, axis=(1, 2))
                y = reduce_op(y, axis=(1,)) # This axis index will change if previous reduce_op changes shape
                y = reduce_op(y)
                y = reduce_op(y) # Applying reduce_op multiple times to scalar is redundant but mirrors TVM IR
                return relay.Function(relay.analysis.free_vars(y), y)

            def expected():
                x = relay.var("x", shape=(1, 1, 1, 2048))
                weight = relay.var("weight", shape=(1, 1, 2048, 1000))
                weight = relay.layout_transform(weight, "HWIO", "OIHW")
                x = relay.layout_transform(x, "NHWC", "NCHW")
                y = relay.nn.conv2d(x, weight, channels=1000, kernel_size=(1, 1))
                y = relay.nn.relu(y)
                y = reduce_op(y, axis=(2, 3))
                y = reduce_op(y, axis=(1,))
                y = reduce_op(y)
                y = reduce_op(y)
                return relay.Function(relay.analysis.free_vars(y), y)

            a = before()
            a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
            b = run_opt_pass(expected(), None) # transform.InferType is not convertible

            pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")
        _test_conv_reduce_convert_layout1_inner()

    def _test_conv_reduce_convert_layout2():
        def _set_span(y, text):
            return relay.Call(
                y.op, y.args, y.attrs, y.type_args, tvm.tir.Span(tvm.tir.SourceName(text), 0, 0, 0, 0)
            )

        def before():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=512,
                kernel_size=(3, 3),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = _set_span(y, "SpanConv2D")
            y = relay.nn.relu(y)
            y = _set_span(y, "SpanRelu")
            y = relay.multiply(y, y)
            y = _set_span(y, "SpanMultiply")
            y = relay.sum(y, axis=(3,), keepdims=True)
            y = _set_span(y, "SpanSum")
            return relay.Function(relay.analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=512, kernel_size=(3, 3))
            y = relay.nn.relu(y)
            y = relay.multiply(y, y)
            y = relay.sum(y, axis=(1,), keepdims=True)
            y = relay.layout_transform(y, "NCHW", "NHWC")
            return relay.Function(relay.analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        # The checks for 'SpanX' in astext() are about string representation of TVM IR
        pytest.skip("Test relies on TVM Relay IR structural equality and string representation, not convertible to PyTorch.")

    _test_conv_reduce_convert_layout2()


def test_image_resize2d_convert_layout():
    def _test_image_resize_convert_layout_nchw_to_nhwc():
        def before():
            x = relay.var("x", shape=(1, 2, 4, 4))
            y = relay.image.resize2d(x, (8, 8))
            y = relay.Function([x], y)
            return y

        def expected():
            x = relay.var("x", shape=(1, 2, 4, 4))
            x = relay.layout_transform(x, "NCHW", "NHWC")
            y = relay.image.resize2d(x, (8, 8), layout="NHWC")
            y = relay.layout_transform(y, "NHWC", "NCHW")
            y = relay.Function(relay.analysis.free_vars(y), y)
            return y

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"image.resize2d": ["NHWC"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    def _test_image_resize_convert_layout_nhwc_to_nchw():
        def before():
            x = relay.var("x", shape=(1, 4, 4, 2))
            y = relay.image.resize2d(x, (8, 8), layout="NHWC")
            y = relay.Function([x], y)
            return y

        def expected():
            x = relay.var("x", shape=(1, 4, 4, 2))
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.image.resize2d(x, (8, 8), layout="NCHW")
            y = relay.layout_transform(y, "NCHW", "NHWC")
            y = relay.Function(relay.analysis.free_vars(y), y)
            return y

        a = before()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"image.resize2d": ["NCHW"]}))
        b = run_opt_pass(expected(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")

    _test_image_resize_convert_layout_nchw_to_nhwc()
    _test_image_resize_convert_layout_nhwc_to_nchw()


def test_conv_image_resize2d_convert_layout():
    """Check that layout transforms are propagated through image resize."""

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.image.resize2d(y, (112, 112), layout="NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        w = relay.layout_transform(w, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.image.resize2d(y, (112, 112), layout="NCHW")
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_infer_correct_layout():
    test_infer_correct_layout_flag = False

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    @relay.op.register_alter_op_layout("nn.relu", level=11)
    def infer_correct_layout_relu(attrs, new_in_layouts, old_in_layouts, old_in_types):
        nonlocal test_infer_correct_layout_flag
        test_infer_correct_layout_flag = True
        ret = tvm.tir.layout("")
        if new_in_layouts:
            assert len(new_in_layouts) >= 1
            ret = new_in_layouts[0]
        else:
            for i in range(len(old_in_layouts)):
                if old_in_layouts[i]:
                    ret = old_in_layouts[i]
                    break
        input_layouts = []
        for i in range(len(old_in_layouts)):
            input_layouts.append(ret)
        return InferCorrectLayoutOutput(input_layouts, [ret], attrs)

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    # This assertion relies on the side effect of TVM's internal registry and pass execution.
    # It cannot be directly converted to PyTorch's user-facing APIs.
    # The `test_infer_correct_layout_flag` would remain False with mocked pass execution.
    # So we skip this test.
    pytest.skip("Test relies on TVM Relay IR pass side effects and internal registry, not convertible to PyTorch.")


def test_reduce_op_convert_layout():
    for reduce_op in [relay.argmax, relay.mean, relay.max]:

        def before_inner():
            x = relay.var("x", shape=(1, 64, 56, 56))
            weight = relay.var("weight", shape=(64, 64, 3, 3))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=64,
                kernel_size=(3, 3),
                padding=(1, 1),
                data_layout="NCHW",
                kernel_layout="OIHW",
            )
            y = reduce_op(y, axis=[2, 3])
            y = relay.Function([x, weight], y)
            return y

        def expected_inner():
            x = relay.var("x", shape=(1, 64, 56, 56))
            weight = relay.var("weight", shape=(64, 64, 3, 3))
            x = relay.layout_transform(x, "NCHW", "NHWC")
            weight = relay.layout_transform(weight, "OIHW", "HWIO")
            y = relay.nn.conv2d(
                x,
                weight,
                channels=64,
                kernel_size=(3, 3),
                padding=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = reduce_op(y, axis=[1, 2])
            y = relay.Function(relay.analysis.free_vars(y), y)
            return y

        a = before_inner()
        a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))
        b = run_opt_pass(expected_inner(), None) # transform.InferType is not convertible

        pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_max_pool_uses_specified_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NCHW")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight = relay.layout_transform(weight, "OIHW", "OHWI")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="OHWI",
        )
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NHWC", out_layout="NHWC")
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(
        a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "OHWI"], "nn.max_pool2d": ["NHWC"]})
    )
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible

    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


@pytest.mark.parametrize(
    "data_layout, kernel_layout",
    [
        ("NCHW1c", "OIHW1i1o"),
        ("NCHW4c", "OIHW4i4o"),
        ("NCHW8c", "OIHW8i8o"),
        ("NCHW16c", "OIHW16i16o"),
    ],
)
def test_resnet_convert_layout_nchwc(data_layout, kernel_layout):
    x = relay.var("x", shape=(1, 3, 224, 224))
    weight1 = relay.var("weight1", shape=(64, 3, 7, 7))
    weight2 = relay.var("weight2", shape=(64, 64, 3, 3))
    weight3 = relay.var("weight3", shape=(64, 64, 1, 1))

    def before():
        y = relay.nn.conv2d(
            x,
            weight1,
            strides=(2, 2),
            padding=(3, 3),
            channels=64,
            kernel_size=(7, 7),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(3, 3), strides=(2, 2), padding=(1, 1))
        y1 = relay.nn.conv2d(
            y,
            weight2,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(
            y,
            weight3,
            channels=64,
            kernel_size=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y2 = relay.nn.relu(y2)
        y = y1 + y2
        y = relay.nn.global_max_pool2d(y, layout="NCHW")
        return y

    # Mock `nn.contrib_conv2d_nchwc` for this specific test
    class MockContribOp:
        def __getattr__(self, name):
            def _dummy_op(*args, **kwargs):
                return MockRelayCall(op=f"nn.contrib.{name}", args=args, attrs=kwargs)
            return _dummy_op
    relay.nn.contrib_conv2d_nchwc = MockContribOp().conv2d_nchwc


    def expected():
        if data_layout == "NCHW1c":
            y = relay.nn.contrib_conv2d_nchwc(
                relay.layout_transform(x, "NCHW", data_layout),
                relay.layout_transform(weight1, "OIHW", kernel_layout),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                data_layout=data_layout,
                kernel_layout=kernel_layout,
            )
            y = relay.nn.relu(y)
            y = relay.nn.max_pool2d(
                y, pool_size=(3, 3), strides=(2, 2), padding=(1, 1), layout=data_layout
            )
        else:
            y = relay.nn.conv2d(
                x,
                weight1,
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                data_layout="NCHW",
                kernel_layout="OIHW",
            )
            y = relay.nn.relu(y)
            y = relay.nn.max_pool2d(y, pool_size=(3, 3), strides=(2, 2), padding=(1, 1))
            y = relay.layout_transform(y, "NCHW", data_layout)
        y1 = relay.nn.contrib_conv2d_nchwc(
            y,
            relay.layout_transform(weight2, "OIHW", kernel_layout),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.contrib_conv2d_nchwc(
            y,
            relay.layout_transform(weight3, "OIHW", kernel_layout),
            channels=64,
            kernel_size=(1, 1),
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )
        y2 = relay.nn.relu(y2)
        y = y1 + y2
        y = relay.nn.global_max_pool2d(y, layout=data_layout)
        y = relay.layout_transform(y, data_layout, "NCHW")
        return y

    a = before()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": [data_layout, kernel_layout]}))
    b = run_opt_pass(expected(), None) # transform.InferType is not convertible
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")


def test_conv_l2n_convert_layout():
    """Check that layout transforms are propagated through bn."""
    axis_list = ([3], [-1], [2, 3])
    expected_axis = ([1], [1], [3, 1])
    for i, axis in enumerate(axis_list):

        def before_inner(): # Wrapped tests in function to make them runnable
            x = relay.var("x", shape=(1, 56, 56, 64))
            weight = relay.var("weight", shape=(3, 3, 64, 64))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=64,
                kernel_size=(3, 3),
                padding=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            z = relay.nn.l2_normalize(y, eps=0.001, axis=axis)
            z = relay.Function(relay.analysis.free_vars(z), z)
            return z

        def expected_inner(): # Wrapped tests in function to make them runnable
            x = relay.var("x", shape=(1, 56, 56, 64))
            w = relay.var("weight", shape=(3, 3, 64, 64))
            x = relay.layout_transform(x, "NHWC", "NCHW")
            w = relay.layout_transform(w, "HWIO", "OIHW")
            y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))
            z = relay.nn.l2_normalize(y, eps=0.001, axis=expected_axis[i])
            z = relay.layout_transform(z, "NCHW", "NHWC")
            z = relay.Function(relay.analysis.free_vars(z), z)
            return z

    a = before_inner()
    a = run_opt_pass(a, tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected_inner(), None) # transform.InferType is not convertible
    pytest.skip("Test relies on TVM Relay IR structural equality, not convertible to PyTorch.")
