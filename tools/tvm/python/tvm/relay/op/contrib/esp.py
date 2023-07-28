import math
import re
import tvm
import tvm.ir
from tvm import relay
from tvm._ffi import register_func
from tvm.relay import transform
# from ... import _ffi_api
from tvm.relay.dataflow_pattern import *
from .register import register_pattern_table, get_pattern_table
from ..strategy.generic import is_depthwise_conv2d
from tvm.relay.build_module import bind_params_by_name
tvm._ffi._init_api("relay.ext.esp.transform", __name__)


# preprocess
class TransformRewriterCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.data = wildcard()
        self.src_scl = is_constant()
        self.src_zp = is_constant()

        trans = is_op("transpose")(self.data).has_attr({"axes": [0, 3, 1, 2]})
        self.quant = is_op("qnn.quantize")(trans, self.src_scl, self.src_zp).has_attr({"axis": 1})
        out = is_op("layout_transform")(self.quant | trans).has_attr({"src_layout": "NCHW", "dst_layout": "NHWC"})
        
        self.pattern = out

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        quant = node_map.get(self.quant)[0] if node_map.get(self.quant) else None
        if quant:
            src_scl = node_map[self.src_scl][0]
            src_zp = node_map[self.src_zp][0]
            return relay.qnn.op.quantize(data, src_scl, src_zp, axis=3)
        else:
            final_dtype = node_map[self.pattern][0].checked_type.dtype
            return relay.op.cast(data, dtype=final_dtype)

def preprocess_onnx_for_esp(mod, params=None):
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    
    mod["main"] = rewrite(TransformRewriterCallback(), mod["main"])
    
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.MergeComposite(get_pattern_table("esp_onnx")),
            transform.AnnotateTarget("esp"),
            # transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            GenerateESPConstants(),
            ExtractConstantsFromPartitionedFunction(),
            transform.InferType(),
        ]
    )
    return seq(mod)


# pattern
def check_qnn_conv2d_onnx_support(conv2d):
    conv2d_input = conv2d.args[0]
    conv2d_weight = conv2d.args[1]

    # check if depthwise Conv2D
    kernel_layout = conv2d.attrs.kernel_layout
    kernel_pos_o = kernel_layout.index("O")
    kernel_pos_i = kernel_layout.index("I")
    groups = conv2d.attrs.groups
    is_depthwise = False
    # if groups == int(conv2d_input.checked_type.shape[3]) and groups == int(
    #     conv2d_weight.checked_type.shape[pos_o]
    # ):
    if conv2d.attrs.channels == conv2d_weight.checked_type.shape[kernel_pos_o] * conv2d_weight.checked_type.shape[kernel_pos_i]:
    #     int kernel_pos_dm = input_c == 1 ? kernel_pos_o : kernel_pos_i;
    #   depth_multiplier = qnn::get_const_int(filter_shape[kernel_pos_dm]);
        is_depthwise = True
        kernel_pos_dm = kernel_pos_o if int(conv2d_input.checked_type.shape[3]) == 1 else kernel_pos_i
        output_channel = conv2d_weight.checked_type.shape[2] * conv2d_weight.checked_type.shape[3]
        depth_multiplier = output_channel // conv2d_input.checked_type.shape[3]
        if depth_multiplier != 1:
            return False
        # if groups != 1:
        #     print("group dwconv2d not supported")
        #     return False # 组卷积不支持
        #HWIO (3,3,1,8) in_c = 1, depth_multipler = 8支持
        #HWOI (3,3,2,8) in_c = 2, depth_multipler = 8不支持
        #HWOI (3,3,8,1) out_c = 8, depth_multipler = 1支持
        # if int(output_channel) % 8 != 0:
        #     print("not 8x dw conv2d")
        #     return False #暂不支持不能整除的
    if not is_depthwise:
        if groups != 1:
            print("group conv2d not supported")
            return False # 组卷积不支持
        
        # if int(conv2d_input.checked_type.shape[3]) % 8 != 0:
        #     print("not 8x conv2d")
        #     return False #暂不支持不能整除的

    # input_zero_point should be 0
    input_zero_point = conv2d.args[2].data.numpy().item(0)
    if input_zero_point != 0:
        return False
    
    # kernel zero_point should be 0
    kernel_zp = conv2d.args[3].data.numpy()
    kernel_zp = [kernel_zp] if kernel_zp.ndim == 0 else kernel_zp
    if not all([zp == 0 for zp in kernel_zp]):
        return False

    return True

def make_qnn_conv_onnx_pattern():
    # optional_pad = is_op("nn.pad")(wildcard(), is_constant())
    qnn_conv2d = is_op("qnn.conv2d")(
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        ) # out_dtype="int32"
    req = is_op("qnn.requantize")(
        qnn_conv2d, is_constant(), is_constant(), is_constant(), is_constant()
    )
    
    bias_add = is_op("add")(req, is_constant())

    clip_or_req = req.optional(is_op("nn.relu")) | bias_add.optional(is_op("nn.relu"))
    
    return clip_or_req

def check_qnn_conv2d_onnx(pattern):
    """Check if the Conv2D is supported by esp."""
    if str(pattern.op.name) == "nn.relu":
        relu = pattern
        relu_input = relu.args[0]
        if str(relu_input.op.name) == "add":
            bias_add = relu_input
            requantize = bias_add.args[0]
    elif str(pattern.op.name) == "add":
        bias_add = pattern
        requantize = bias_add.args[0]
    else:  
        requantize = pattern
    
    conv2d = requantize.args[0]
    
    if not check_qnn_conv2d_onnx_support(conv2d):
        return False

    # check if dtypes are supported for the following entities
    # (input_dtype, weight_dtype, bias_dtype, out_dtype, pattern_dtype)
    conv2d_input_dtype = conv2d.args[0].checked_type.dtype
    conv2d_weight_dtype = conv2d.args[1].checked_type.dtype
    if bias_add:
        bias_dtype = bias_add.args[1].checked_type.dtype
    else:
        # this is only to enable to following check that validates all sorts of dtypes
        bias_dtype = "int8" if conv2d_input_dtype == "int8" else "int16"
    valid_dtypes = None
    if conv2d_input_dtype == "int8":
        valid_dtypes = ("int8", "int8", "int8", "int32", "int8")
    else:
        return False

    if (
        conv2d_input_dtype,
        conv2d_weight_dtype,
        bias_dtype,
        conv2d.attrs.out_dtype,
        pattern.checked_type.dtype,
    ) != valid_dtypes:
        return False
    
    return True
    
# QLinearAdd, QLinearMul
def make_two_fp_input_onnx_pattern(op_name):
    dequantize1 = is_op("qnn.dequantize")(
            wildcard(),
            is_constant(),
            is_constant(),
        )
    # tmp = wildcard()
    dequantize2 = is_op("qnn.dequantize")(
            wildcard(),
            is_constant(),
            is_constant(),
        ) 
    add = is_op(op_name)(dequantize1, dequantize2)
    quantize = is_op("qnn.quantize")(
        add, is_constant(), is_constant()
    )
    
    clip_or_req = quantize.optional(is_op("nn.relu"))
    
    return clip_or_req

# QLinearAveragePool
def make_one_fp_input_onnx_pattern(op_name):
    dequantize = is_op("qnn.dequantize")(
            wildcard(),
            is_constant(),
            is_constant(),
        )
    op = is_op(op_name)(dequantize)
    quantize = is_op("qnn.quantize")(
        op, is_constant(), is_constant()
    )
    return quantize

@register_pattern_table("esp_onnx")
def pattern_table():
  qnn_conv2d_pat_onnx = ("esp.qnn_conv2d_onnx", make_qnn_conv_onnx_pattern(), check_qnn_conv2d_onnx)
  qnn_add_pat_onnx = ("esp.qnn_add_onnx", make_two_fp_input_onnx_pattern("add"))
  qnn_mul_pat_onnx = ("esp.qnn_mul_onnx", make_two_fp_input_onnx_pattern("multiply"))
  qnn_avg_pool_onnx = ("esp.qnn_avg_pool_onnx", make_one_fp_input_onnx_pattern("nn.avg_pool2d"))
  qnn_leaky_relu_onnx = ("esp.qnn_leaky_relu_onnx", make_one_fp_input_onnx_pattern("nn.leaky_relu"))

#   esp_patterns = [qnn_conv2d_pat_onnx, qnn_add_pat_onnx, qnn_mul_pat_onnx, qnn_avg_pool_onnx, qnn_leaky_relu_onnx]
  esp_patterns = [qnn_conv2d_pat_onnx]
  return esp_patterns


# debugging
from tvm.relay.expr import const
class LegalizeQnnOpForESP(DFPatternCallback):
    """Legalize QNN based patterns to match DNNL
    """

    def __init__(self):
        super(LegalizeQnnOpForESP, self).__init__()
        self.src = wildcard()
        self.wgh = wildcard()
        self.bias = wildcard()

        self.src_scl = is_constant()
        self.src_zp = is_constant()
        self.wgh_scl = is_constant()
        self.wgh_zp = is_expr(const(0))

        self.rq_in_scl = is_constant()
        self.rq_in_zp = is_expr(const(0))
        self.rq_out_scl = is_constant()
        self.rq_out_zp = is_expr(const(0))

        self.root = is_op("qnn.conv2d")(
            self.src, self.wgh, self.src_zp, self.wgh_zp, self.src_scl, self.wgh_scl
        )
        req = is_op("qnn.requantize")(
            self.root, self.rq_in_scl, self.rq_in_zp, self.rq_out_scl, self.rq_out_zp
        )
        pat = is_op("add")(req, self.bias) | req  # optional bias
        
        self.relu = is_op("nn.relu")(pat)
        self.pattern = pat | self.relu

    def callback(self, pre, post, node_map):
        root = node_map[self.root][0]
        src = node_map[self.src][0]
        wgh = node_map[self.wgh][0]
        # bias = node_map.get(self.bias, default=[relay.const(0, dtype="int32")])[0]
        bias = node_map.get(self.bias)[0] if node_map.get(self.bias) else None
        relu = node_map.get(self.relu)[0] if node_map.get(self.relu) else None
        src_scl = node_map[self.src_scl][0]
        src_zp = node_map[self.src_zp][0]
        wgh_scl = node_map[self.wgh_scl][0]
        wgh_zp = node_map[self.wgh_zp][0]

        rq_in_scl = node_map[self.rq_in_scl][0]
        rq_out_scl = node_map[self.rq_out_scl][0]

        final_dtype = node_map[self.pattern][0].checked_type.dtype

        conv2d = tvm.relay.Call(
            root.op,
            [src, wgh, src_zp, wgh_zp, src_scl, wgh_scl],
            root.attrs,
            root.type_args,
            root.span,
        )
        right_shift = round(math.log2(rq_out_scl.data.asnumpy()/rq_in_scl.data.asnumpy()))
        shift = relay.op.right_shift(conv2d, const(right_shift))
        shift = relay.op.clip(shift, -128, 127)
        if bias:
            shift = relay.op.cast(shift, dtype="int16")
            new_bias = relay.op.cast(bias, dtype="int16")
            shift = relay.op.add(shift, new_bias)
            shift = relay.op.clip(shift, -128, 127)
            shift = relay.op.cast(shift, dtype="int8")

        if relu:
            shift = relay.op.nn.relu(shift)
        
        return shift

def evaluate_onnx_for_esp(mod, params=None):
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.RemoveUnusedFunctions(),
            transform.SimplifyInference(),
        ]
    )
    mod = seq(mod)

    mod["main"] = rewrite(LegalizeQnnOpForESP(), mod["main"])
    
    mod = transform.InferType()(mod)
    
    return mod

