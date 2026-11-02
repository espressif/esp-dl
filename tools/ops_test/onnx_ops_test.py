import onnx
import numpy as np
from onnx import ModelProto, TensorProto, helper, OperatorSetIdProto

# Largest head_dim / Lk the esp-dl Attention kernels accept. Mirror
# ``ATTENTION_MAX_HEAD_DIM`` and ``ATTENTION_MAX_KV_TOKENS`` on the device side
# and ``ESPDL_ATTENTION_MAX_HEAD_DIM`` / ``ESPDL_ATTENTION_MAX_KV_TOKENS`` in
# esp-ppq's espdl_attention.py. Kept here so a test config that exceeds a bound
# fails at generation rather than on device. Delegate to esp-ppq's env-driven
# bounds so ``ESPDL_ATTENTION_MAX_HEAD_DIM`` / ``ESPDL_ATTENTION_MAX_KV_TOKENS``
# opt test generation in as well; fall back to the defaults when esp-ppq is not
# installed (dev machine without the quantizer wheel).
try:
    from esp_ppq.executor.op.torch.espdl_attention import (
        ESPDL_ATTENTION_MAX_HEAD_DIM as _ESP_PPQ_MAX_ATTENTION_HEAD_DIM,
        ESPDL_ATTENTION_MAX_KV_TOKENS as _ESP_PPQ_MAX_ATTENTION_KV_TOKENS,
    )

    MAX_ATTENTION_HEAD_DIM = _ESP_PPQ_MAX_ATTENTION_HEAD_DIM
    MAX_ATTENTION_KV_TOKENS = _ESP_PPQ_MAX_ATTENTION_KV_TOKENS
except Exception:
    MAX_ATTENTION_HEAD_DIM = 64
    MAX_ATTENTION_KV_TOKENS = 2048


def IDENTITY_TEST(config) -> ModelProto:
    """
    Identity operator

    Inputs
    input (heterogeneous) - T:
        Input tensor

    Outputs
    output (heterogeneous) - T:
        Tensor to copy input into.
    """

    input_shape = config["input_shape"]
    export_name_prefix = config.get("export_name_prefix", "onnx-model")

    # Create ValueInfoProto
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, input_shape
    )

    # Create Identity Node
    node_def = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"],
    )

    # Create GraphProto
    graph_def = helper.make_graph(
        [node_def],
        "identity_model",
        [input_tensor],
        [output_tensor],
    )

    # Create ModelProto
    model_def = helper.make_model(graph_def, producer_name=export_name_prefix)

    # Check model
    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def SPACETODEPTH_TEST(config) -> ModelProto:
    """
    ONNX Operator: SpaceToDepth

    SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
    this op outputs a copy of the input tensor where values from the height and width dimensions
    are moved to the depth dimension. This is the inverse transformation of DepthToSpace.

    Inputs
    input (differentiable) - T
        Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.

    Outputs
    output (differentiable) - T
        Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].

    Attributes
    blocksize - INT (default is 1)
        Blocks along spatial dimension.
    """

    input_shape = config["input_shape"]
    blocksize = config.get("blocksize", 2)
    export_name_prefix = config.get("export_name_prefix", "onnx-model-spacetodepth")

    # Create ValueInfoProto
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )

    # Calculate output shape
    output_shape = [
        input_shape[0],  # batch
        input_shape[1] // blocksize,  # height
        input_shape[2] // blocksize,  # width
        input_shape[3] * (blocksize * blocksize),  # channels
    ]

    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )

    # Create SpaceToDepth Node
    node_def = helper.make_node(
        "SpaceToDepth", inputs=["input"], outputs=["output"], blocksize=blocksize
    )

    # Create GraphProto
    graph_def = helper.make_graph(
        [node_def],
        "space_to_depth_model",
        [input_tensor],
        [output_tensor],
    )

    # Create ModelProto
    model_def = helper.make_model(graph_def, producer_name=export_name_prefix)

    # Check model
    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def SCATTER_ND_TEST(config) -> ModelProto:
    """
    ONNX Operator: ScatterND

    ScatterND takes three inputs: data tensor, indices tensor, and updates tensor.
    The output is produced by creating a copy of the input data, and then updating
    its values to values specified by updates at specific index positions specified by indices.
    Supports reduction operations: none (default), add, mul, max, min.

    Inputs
    data (differentiable) - T:
        Tensor of rank r >= 1.
    indices (non-differentiable) - tensor(int64):
        Tensor of rank q >= 1.
    updates (differentiable) - T:
        Tensor of rank q + r - indices.shape[-1] - 1.

    Outputs
    output (differentiable) - T:
        Tensor of rank r >= 1.

    Attributes
    reduction - STRING (default is 'none'):
        Type of reduction to apply: none, add, mul, max, min.
    """

    data_shape = config["input_shape"][0]
    indices_shape = list(np.array(config["indices_shape"]).shape)
    indices = np.array(config["indices_shape"]).flatten().tolist()
    updates_shape = config["input_shape"][1]
    print("data_shape:", data_shape)
    print("indices_shape:", indices_shape, config["indices_shape"])
    print("updates_shape:", updates_shape)
    reduction = config.get("reduction", "none")
    export_name_prefix = config.get("export_name_prefix", "onnx-model-scatternd")

    # Create ValueInfoProto
    data_tensor = helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)

    indices_tensor = helper.make_tensor(
        name="indices",
        data_type=TensorProto.INT64,
        dims=indices_shape,
        vals=indices,
    )

    squeezed_axis = helper.make_tensor(
        name="squeezed_axis",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0],
    )

    updates_tensor = helper.make_tensor_value_info(
        "updates", TensorProto.FLOAT, updates_shape
    )
    # Create intermediate tensors after squeeze
    data_squeezed_tensor = helper.make_tensor_value_info(
        "data_squeezed", TensorProto.FLOAT, data_shape[1:]
    )

    updates_squeezed_tensor = helper.make_tensor_value_info(
        "updates_squeezed", TensorProto.FLOAT, updates_shape[1:]
    )

    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, data_shape
    )
    # Create Squeeze nodes
    squeeze_nodes = []

    data_squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["data", "squeezed_axis"],
        outputs=["data_squeezed"],
    )
    squeeze_nodes.append(data_squeeze_node)
    scatter_data_input = "data_squeezed"

    updates_squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["updates", "squeezed_axis"],
        outputs=["updates_squeezed"],
    )
    squeeze_nodes.append(updates_squeeze_node)
    scatter_updates_input = "updates_squeezed"

    # Create ScatterND Node
    if reduction == "none":
        scatter_node = helper.make_node(
            "ScatterND",
            inputs=[scatter_data_input, "indices", scatter_updates_input],
            outputs=["output"],
        )
    else:
        scatter_node = helper.make_node(
            "ScatterND",
            inputs=[scatter_data_input, "indices", scatter_updates_input],
            outputs=["output"],
            reduction=reduction,
        )

    # Create GraphProto with all nodes
    all_nodes = squeeze_nodes + [scatter_node]

    # Create GraphProto
    graph_def = helper.make_graph(
        all_nodes,
        "scatter_nd_model",
        [data_tensor, updates_tensor],
        [output_tensor],
        initializer=[indices_tensor, squeezed_axis],
    )

    # Create ModelProto
    model_def = helper.make_model(graph_def, producer_name=export_name_prefix)
    model_def.ir_version = 10  # Set IR version to 7 to support ScatterND with reduction

    # Check model
    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def REVERSESEQUENCE_TEST(config) -> ModelProto:
    """
    Attributes
    batch_axis - INT (default is '1'):

    (Optional) Specify which axis is batch axis. Must be one of 1 (default), or 0.

    time_axis - INT (default is '0'):

    (Optional) Specify which axis is time axis. Must be one of 0 (default), or 1.

    Inputs
    input (heterogeneous) - T:

    Tensor of rank r >= 2.

    Outputs
    Y (heterogeneous) - T:

    Tensor with same shape of input.
    """

    # Get Attributes from Config
    batch_axis = config.get("batch_axis", 1)
    time_axis = config.get("time_axis", 0)
    input_shape = config["input_shape"]
    sequence_lens = config["sequence_lens"]  # 获取常量数据
    export_name_prefix = config.get("export_name_prefix", "onnx-model")

    batch_size = input_shape[batch_axis]
    assert (
        len(sequence_lens) == batch_size
    ), f"sequence_lens length {len(sequence_lens)} does not match batch size {batch_size}"

    # Create value info
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    sequence_lens_tensor = helper.make_tensor_value_info(
        "sequence_lens", TensorProto.FLOAT, [batch_size]
    )
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)

    sequence_lens_tensor = helper.make_tensor(
        name="sequence_lens",
        data_type=TensorProto.INT64,
        dims=[batch_size],
        vals=sequence_lens,
    )

    node2 = helper.make_node(
        "ReverseSequence",
        inputs=["input", "sequence_lens"],  # 第二个输入指向initializer
        outputs=["Y"],
        batch_axis=batch_axis,
        time_axis=time_axis,
    )

    # Create GraphProto
    graph_def = helper.make_graph(
        [node2],
        "reverse_sequence_model",
        [input_tensor],  # 输入仅包含input
        [output_tensor],
        initializer=[sequence_lens_tensor],  # 添加常量initializer
    )

    # Create ModelProto
    model_def = helper.make_model(graph_def, producer_name=export_name_prefix)

    # Check model
    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def REDUCESUMSQUARE_TEST(config) -> ModelProto:
    """
    Attributes
    keepdims - INT (default is '1'):

        Keep the reduced dimension or not, default 1 means keep reduced dimension.

    noop_with_empty_axes - INT (default is '0'):

        Defines behavior when axes is not provided or is empty. If false (default),
        reduction happens over all axes. If true, no reduction is applied, but
        other operations will be performed.

    Inputs
    data (heterogeneous) - T:

        An input tensor.

    axes (optional, heterogeneous) - tensor(int64):

        Optional input list of integers, along which to reduce.

    Outputs
    reduced (heterogeneous) - T:

        Reduced output tensor.
    """

    # Get Attributes from Config
    keepdims = int(config.get("keepdim", False))
    noop_with_empty_axes = config.get("noop_with_empty_axes", 0)
    input_shape = config["input_shape"]
    dims = config["dim"]  # 获取常量数据
    export_name_prefix = config.get("export_name_prefix", "onnx-model")

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )

    # Create ReduceSumSquare node
    reduce_axes_tensor = helper.make_tensor(
        name="reduce_axes",
        data_type=TensorProto.INT64,
        dims=[len(dims)],
        vals=dims,
    )

    assert len(input_shape) >= len(
        dims
    ), f"input_shape length {len(input_shape)} must be greater than or equal to the length of dims {len(dims)}."
    reduce_output_shape = input_shape.copy()
    count = 0
    for dim in dims:
        if dim < 0:
            dim += len(input_shape)

        dim -= count
        print(f"xiewei debug, dim: {dim}")
        if keepdims == 1:
            reduce_output_shape[dim] = 1
        else:
            reduce_output_shape.pop(dim)
            count += 1

    # output value info
    reduce_output_tensor = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, reduce_output_shape
    )

    reduce_node = helper.make_node(
        "ReduceSumSquare",
        inputs=["input", "reduce_axes"],  # 第二个输入指向initializer
        outputs=["Y"],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
    )

    # Create GraphProto
    graph_def = helper.make_graph(
        nodes=[reduce_node],
        name="reduce_sum_square_model",
        inputs=[input_tensor],
        outputs=[reduce_output_tensor],
        initializer=[reduce_axes_tensor],
    )

    # 创建 OperatorSetIdProto 对象
    opset = OperatorSetIdProto()
    opset.domain = ""  # 默认域（核心ONNX算子）
    opset.version = 18  # 指定opset版本

    # Create ModelProto
    model_def = helper.make_model(
        graph_def, opset_imports=[opset], producer_name="espdl ops test"
    )

    # Check model
    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def REDUCELOGSUM_TEST(config) -> ModelProto:
    """
    Attributes
    keepdims - INT (default is '1'):

        Keep the reduced dimension or not, default 1 means keep reduced dimension.

    noop_with_empty_axes - INT (default is '0'):

        Defines behavior when axes is not provided or is empty. If false (default),
        reduction happens over all axes. If true, no reduction is applied, but
        other operations will be performed.

    Inputs
    data (heterogeneous) - T:

        An input tensor.

    axes (optional, heterogeneous) - tensor(int64):

        Optional input list of integers, along which to reduce.

    Outputs
    reduced (heterogeneous) - T:

        Reduced output tensor.
    """

    # Get Attributes from Config
    keepdims = int(config.get("keepdims", False))
    noop_with_empty_axes = config.get("noop_with_empty_axes", 0)
    input_shape = config["input_shape"]
    dims = config["dim"]  # 获取常量数据
    export_name_prefix = config.get("export_name_prefix", "onnx-model")

    # Create Exp node
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )

    exp_output_tensor = helper.make_tensor_value_info(
        "exp_output", TensorProto.FLOAT, input_shape
    )

    exp_node = helper.make_node(
        "Exp",
        inputs=["input"],
        outputs=["exp_output"],
    )

    # Create ReduceLogSum node
    reduce_axes_tensor = helper.make_tensor(
        name="reduce_axes",
        data_type=TensorProto.INT64,
        dims=[len(dims)],
        vals=dims,
    )

    assert len(input_shape) >= len(
        dims
    ), f"input_shape length {len(input_shape)} must be greater than or equal to the length of dims {len(dims)}."
    reduce_output_shape = input_shape.copy()
    count = 0
    for dim in dims:
        if dim < 0:
            dim += len(input_shape)

        dim -= count
        print(f"xiewei debug, dim: {dim}")
        if keepdims == 1:
            reduce_output_shape[dim] = 1
        else:
            reduce_output_shape.pop(dim)
            count += 1

    # output value info
    reduce_output_tensor = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, reduce_output_shape
    )

    reduce_node = helper.make_node(
        "ReduceLogSum",
        inputs=["exp_output", "reduce_axes"],  # 第二个输入指向initializer
        outputs=["Y"],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
    )

    # Create GraphProto
    graph_def = helper.make_graph(
        nodes=[exp_node, reduce_node],
        name="reduce_log_sum_model",
        inputs=[input_tensor],
        outputs=[reduce_output_tensor],
        initializer=[reduce_axes_tensor],
        value_info=[exp_output_tensor],
    )

    # 创建 OperatorSetIdProto 对象
    opset = OperatorSetIdProto()
    opset.domain = ""  # 默认域（核心ONNX算子）
    opset.version = 18  # 指定opset版本

    # Create ModelProto
    model_def = helper.make_model(
        graph_def, opset_imports=[opset], producer_name="espdl ops test"
    )

    # Check model
    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def MOD_TEST(config) -> ModelProto:
    """
    ONNX Operator: Mod

    Performs element-wise binary mod (with Numpy-style broadcasting support).
    When fmod=1, behaves like C fmod() where the result sign follows the dividend.

    Inputs
    A (differentiable) - T:
        Dividend tensor
    B (differentiable) - T:
        Divisor tensor

    Outputs
    C (differentiable) - T:
        Remainder tensor

    Attributes
    fmod - INT (default is 0):
        0: integer mod (Python %), 1: float mod (C fmod)
    """

    input_shape = config["input_shape"]
    fmod = config.get("fmod", 1)
    export_name_prefix = config.get("export_name_prefix", "onnx-model-mod")

    if isinstance(input_shape[0], list):
        input0_shape = input_shape[0]
        input1_shape = input_shape[1]
    else:
        input0_shape = input_shape
        input1_shape = input_shape

    input0_tensor = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, input0_shape
    )
    input1_tensor = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, input1_shape
    )

    # Use Exp on input1 to ensure divisor is always positive and non-zero
    exp_node = helper.make_node(
        "Exp",
        inputs=["input1"],
        outputs=["safe_input1"],
    )

    # Calculate output shape with broadcasting
    output_shape = list(np.broadcast_shapes(input0_shape, input1_shape))

    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )

    mod_node = helper.make_node(
        "Mod",
        inputs=["input0", "safe_input1"],
        outputs=["output"],
        fmod=fmod,
    )

    graph_def = helper.make_graph(
        [exp_node, mod_node],
        "mod_model",
        [input0_tensor, input1_tensor],
        [output_tensor],
    )

    opset = OperatorSetIdProto()
    opset.domain = ""
    opset.version = 13

    model_def = helper.make_model(
        graph_def, opset_imports=[opset], producer_name=export_name_prefix
    )

    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def DEPTHTOSPACE_TEST(config) -> ModelProto:
    """
    ONNX Operator: DepthToSpace

    DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
    This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
    the input tensor where values from the depth dimension are moved in spatial blocks to the height
    and width dimensions. By default, mode = DCR.

    Inputs
    input (differentiable) - T
        Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.

    Outputs
    output (differentiable) - T
        Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].

    Attributes
    blocksize - INT (default is 1)
        Blocks along spatial dimension.
    mode - STRING (default is 'DCR')
        DCR (depth-column-row) or CRD (column-row-depth) mode (both modes are identical for blocksize=2)
    """

    input_shape = config["input_shape"]
    blocksize = config.get("blocksize", 2)
    mode = config.get("mode", "DCR")
    export_name_prefix = config.get("export_name_prefix", "onnx-model-depthtospace")

    # Create ValueInfoProto
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )

    # Calculate output shape
    N, C, H, W = input_shape
    C_out = C // (blocksize * blocksize)
    H_out = H * blocksize
    W_out = W * blocksize
    output_shape = [N, C_out, H_out, W_out]

    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )

    # Create DepthToSpace Node
    node_def = helper.make_node(
        "DepthToSpace",
        inputs=["input"],
        outputs=["output"],
        blocksize=blocksize,
        mode=mode,
    )

    # Create GraphProto
    graph_def = helper.make_graph(
        [node_def],
        "depth_to_space_model",
        [input_tensor],
        [output_tensor],
    )

    # Create ModelProto
    model_def = helper.make_model(graph_def, producer_name=export_name_prefix)

    # Check model
    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def LPNORM_TEST(config) -> ModelProto:
    """
    ONNX Operator: LpNormalization

    Given a matrix, apply Lp-normalization along the provided axis.
    output = input / Lp_norm(input, axis).
    When the Lp norm is zero, the output is defined to be zero.

    Attributes
    axis - INT (default is -1): The axis on which to apply normalization.
    p - INT (default is 2): The order of the normalization, only 1 or 2 are supported.
    """

    input_shape = config["input_shape"]
    axis = config.get("axis", -1)
    p = config.get("p", 2)
    export_name_prefix = config.get("export_name_prefix", "onnx-model-lpnorm")

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, input_shape
    )

    node_def = helper.make_node(
        "LpNormalization",
        inputs=["input"],
        outputs=["output"],
        axis=axis,
        p=p,
    )

    graph_def = helper.make_graph(
        [node_def],
        "lpnorm_model",
        [input_tensor],
        [output_tensor],
    )

    model_def = helper.make_model(graph_def, producer_name=export_name_prefix)

    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def RMSNORMALIZATION_TEST(config) -> ModelProto:
    """
    RMSNormalization custom operator.

    Computes: y = x / sqrt(mean(x^2) + epsilon) * weight

    Inputs
    X (heterogeneous) - T:
        Input tensor
    Scale (heterogeneous) - T:
        Scale/weight tensor (1D, size = normalized dimension)

    Outputs
    Y (heterogeneous) - T:
        Normalized output tensor (same shape as X)

    Attributes
    axis - INT (default -1):
        The axis to normalize
    epsilon - FLOAT (default 1e-5):
        Small value to avoid division by zero
    """

    input_shape = config["input_shape"]
    axis = config.get("axis", -1)
    epsilon = config.get("epsilon", 1e-5)
    export_name_prefix = config.get("export_name_prefix", "rmsnorm-model")

    # Input tensor
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )

    # Weight/Scale initializer (shape = last dimension)
    valid_axis = axis if axis >= 0 else len(input_shape) + axis
    weight_shape = input_shape[valid_axis:]

    # Create weight with random values around 1.0
    np.random.seed(42)
    weight_size = 1
    for d in weight_shape:
        weight_size *= d
    weight_values = np.random.normal(1.0, 0.02, weight_size).astype(np.float32)

    weight_init = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=weight_shape,
        vals=weight_values,
    )

    # Output tensor
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, input_shape
    )

    node_def = helper.make_node(
        "RMSNormalization",
        inputs=["input", "weight"],
        outputs=["output"],
        axis=axis,
        epsilon=epsilon,
    )

    graph_def = helper.make_graph(
        [node_def],
        "rmsnorm_model",
        [input_tensor],
        [output_tensor],
        initializer=[weight_init],
    )

    opset = OperatorSetIdProto()
    opset.domain = ""
    opset.version = 13

    model_def = helper.make_model(
        graph_def, opset_imports=[opset], producer_name=export_name_prefix
    )

    try:
        onnx.checker.check_model(model_def)
        print("The model is checked!")
    except Exception as e:
        print(f"ONNX checker warning (expected for custom ops): {e}")

    return model_def


def ATTENTION_TEST(config) -> ModelProto:
    """Attention (v2) — fused Q · K^T -> scale -> [+mask/causal] -> softmax -> · V.

    Inputs
        q          : ``[B, Hq,  Lq, D]`` (BHND) or ``[B, Lq, Hq, D]`` (BNHD)
        k          : ``[B, Hkv, Lk, D]`` (BHND) or ``[B, Lk, Hkv, D]`` (BNHD)
        v          : ``[B, Hkv, Lk, D]`` (BHND) or ``[B, Lk, Hkv, D]`` (BNHD)
        attn_mask  : broadcastable to ``[B, Hq, Lq, Lk]``, optional. Either a
                     graph input (``input_shape[3]``) or, with
                     ``const_mask_shape``, a constant with -inf key padding.
                     ``const_mask_pattern = "tiles"`` instead cycles each
                     32x64 (query x key) block through all-zero / finite /
                     mixed -inf / all -inf, shifted per query block and head.

    ``head_dim`` must be in ``[1, MAX_ATTENTION_HEAD_DIM]`` and ``Lk`` in
    ``[1, MAX_ATTENTION_KV_TOKENS]`` — the device kernels size their scratch
    on those bounds and refuse anything larger.

    Attributes
        scale     : FLOAT, optional. Defaults to ``1 / sqrt(D)`` at runtime.
        is_causal : INT,   optional (0 by default).
        layout    : ESP-DL extension; ``BHND`` (default) or ``BNHD``.

    Output
        out : ``[B, Hq, Lq, D]``, same axis convention as Q.

    Matching esp-dl module:
    ``esp-dl/esp-dl/dl/module/include/dl_module_attention.hpp``.
    """
    input_shapes = config["input_shape"]
    export_name_prefix = config.get("export_name_prefix", "onnx-model-attention")
    layout = str(config.get("layout", "BHND")).upper()
    if layout not in ("BHND", "BNHD"):
        raise ValueError(
            f"Attention test config: layout must be 'BHND' or 'BNHD', got {layout!r}."
        )
    if onnx.defs.onnx_opset_version() < 23:
        raise RuntimeError("Attention ops tests require onnx>=1.18.0 (opset 23).")

    if len(input_shapes) < 3:
        raise ValueError("Attention needs at least 3 input shapes (Q, K, V)")
    q_shape, k_shape, v_shape = input_shapes[0], input_shapes[1], input_shapes[2]
    mask_shape = input_shapes[3] if len(input_shapes) > 3 else None

    assert (
        len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4
    ), "Q/K/V must all be 4D"
    assert q_shape[0] == k_shape[0] == v_shape[0], "B mismatch across Q/K/V"
    head_axis, seq_axis = (2, 1) if layout == "BNHD" else (1, 2)
    assert k_shape[head_axis] == v_shape[head_axis], "K.Hkv and V.Hkv must match"
    assert (
        k_shape[head_axis] > 0 and q_shape[head_axis] % k_shape[head_axis] == 0
    ), "GQA/MQA requires Hq % Hkv == 0"
    assert k_shape[seq_axis] == v_shape[seq_axis], "K.Lk and V.Lk must match"
    assert 1 <= k_shape[seq_axis] <= MAX_ATTENTION_KV_TOKENS, (
        f"Lk={k_shape[seq_axis]} outside [1, {MAX_ATTENTION_KV_TOKENS}]; the device "
        "sizes the shared K-head scratch on that bound"
    )
    D = q_shape[3]
    assert q_shape[3] == k_shape[3] == v_shape[3], "head_dim mismatch across Q/K/V"
    assert 1 <= D <= MAX_ATTENTION_HEAD_DIM, (
        f"head_dim={D} outside [1, {MAX_ATTENTION_HEAD_DIM}]; the device "
        "kernels size their scratch on that bound"
    )

    out_shape = list(q_shape)

    q_in = helper.make_tensor_value_info("q", TensorProto.FLOAT, q_shape)
    k_in = helper.make_tensor_value_info("k", TensorProto.FLOAT, k_shape)
    v_in = helper.make_tensor_value_info("v", TensorProto.FLOAT, v_shape)
    inputs = [q_in, k_in, v_in]
    node_inputs = ["q", "k", "v"]
    initializers = []
    const_mask_shape = config.get("const_mask_shape")
    if mask_shape is not None and const_mask_shape is not None:
        raise ValueError(
            "Attention test config: use input_shape[3] or const_mask_shape, not both"
        )
    if mask_shape is not None:
        mask_in = helper.make_tensor_value_info(
            "attn_mask", TensorProto.FLOAT, mask_shape
        )
        inputs.append(mask_in)
        node_inputs.append("attn_mask")
    elif const_mask_shape is not None:
        # Random calibration never produces -inf, so a constant mask is the
        # only way to cover key padding: keys past 3/4 of Lk are -inf and the
        # rest carry a finite relative bias.
        lk = const_mask_shape[-1]
        rng = np.random.default_rng(config.get("const_mask_seed", 0))
        mask_np = rng.uniform(-2.0, 2.0, const_mask_shape).astype(np.float32)
        pattern = config.get("const_mask_pattern", "key_padding")
        if pattern == "key_padding":
            mask_np[..., (3 * lk) // 4 :] = -np.inf
        elif pattern == "tiles":
            # Matches the device TILE_Q x TILE_K grid of the mask precompute.
            shape4 = [1] * (4 - len(const_mask_shape)) + list(const_mask_shape)
            m4 = mask_np.reshape(shape4)
            for h in range(shape4[1]):
                for i0 in range(0, shape4[2], 32):
                    for j0 in range(0, shape4[3], 64):
                        blk = m4[:, h, i0 : i0 + 32, j0 : j0 + 64]
                        kind = (h + i0 // 32 + j0 // 64) % 4
                        if kind == 0:
                            blk[...] = 0.0
                        elif kind == 2:
                            blk[..., 1::2] = -np.inf
                        elif kind == 3:
                            blk[...] = -np.inf
            mask_np = m4.reshape(const_mask_shape)
        else:
            raise ValueError(
                f"Attention test config: unknown const_mask_pattern {pattern!r}"
            )
        initializers.append(
            helper.make_tensor(
                "attn_mask",
                TensorProto.FLOAT,
                const_mask_shape,
                mask_np.flatten().tolist(),
            )
        )
        node_inputs.append("attn_mask")

    out_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)

    node_kwargs = {}
    if "scale" in config:
        node_kwargs["scale"] = float(config["scale"])
    if bool(config.get("is_causal", False)):
        node_kwargs["is_causal"] = 1
    node_domain = ""
    opsets = [helper.make_opsetid("", 23)]
    if layout != "BHND":
        node_kwargs["layout"] = layout
        node_domain = "com.espressif"
        opsets.append(helper.make_opsetid(node_domain, 1))
    node = helper.make_node(
        "Attention",
        inputs=node_inputs,
        outputs=["out"],
        domain=node_domain,
        **node_kwargs,
    )

    graph_def = helper.make_graph(
        [node],
        "attention_model",
        inputs,
        [out_info],
        initializer=initializers,
    )

    model_def = helper.make_model(
        graph_def, opset_imports=opsets, producer_name=export_name_prefix
    )

    onnx.checker.check_model(model_def)
    print("The model is checked!")

    return model_def


def ROTARY_EMBEDDING_TEST(config) -> ModelProto:
    """Build one ONNX RoPE node for the ESP-DL vision subset.

    The three floating-point graph inputs are ``X``, ``cos_cache`` and
    ``sin_cache``. The generic ops_test pipeline calibrates them, exports
    reference values into ``.espdl``, and the common device test executes
    ``Model::test()`` followed by the normal benchmark loop.

    Supported here deliberately matches the device module: 4-D
    ``X=[B,H,L,D]``, even ``D``, half-split full rotation, 2-D/3-D caches
    with last dimension ``D/2``, and no ``position_ids``.
    """
    input_shapes = config["input_shape"]
    export_name_prefix = config.get("export_name_prefix", "onnx-model-rotary-embedding")

    if len(input_shapes) != 3:
        raise ValueError(
            "RotaryEmbedding needs exactly 3 input shapes: [X, cos_cache, sin_cache]"
        )
    x_shape, cos_shape, sin_shape = (list(shape) for shape in input_shapes)
    if cos_shape != sin_shape:
        raise ValueError(
            f"cos_cache and sin_cache must have the same shape, got {cos_shape} vs {sin_shape}"
        )

    if any(
        isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or int(dim) <= 0
        for shape in (x_shape, cos_shape, sin_shape)
        for dim in shape
    ):
        raise ValueError(
            f"all RotaryEmbedding dimensions must be positive integers: {input_shapes}"
        )
    x_shape = [int(dim) for dim in x_shape]
    cos_shape = [int(dim) for dim in cos_shape]
    sin_shape = [int(dim) for dim in sin_shape]

    if len(x_shape) != 4:
        raise ValueError(
            f"ESP-DL RotaryEmbedding requires 4-D X [B,H,L,D], got {x_shape}"
        )
    B, H, L, D = x_shape
    if D % 2 != 0:
        raise ValueError(f"head_dim D must be even, got {D}")

    try:
        interleaved = int(config.get("interleaved", 0))
        rotary_embedding_dim = int(config.get("rotary_embedding_dim", 0))
        num_heads = int(config.get("num_heads", 0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("RotaryEmbedding attributes must be integers") from error
    if interleaved != 0:
        raise ValueError(
            "ESP-DL RotaryEmbedding supports only half-split layout (interleaved=0)"
        )
    if rotary_embedding_dim not in (0, D):
        raise ValueError(
            f"partial rotation is unsupported: rotary_embedding_dim={rotary_embedding_dim}, "
            f"head_dim D={D}"
        )
    if num_heads not in (0, H):
        raise ValueError(f"num_heads={num_heads} contradicts X's H axis {H}")
    if config.get("position_ids") is not None:
        raise ValueError("position_ids is not supported by ESP-DL RotaryEmbedding")

    D_half = D // 2
    if len(cos_shape) == 2:
        expected_cache_shape = [L, D_half]
        if cos_shape != expected_cache_shape:
            raise ValueError(
                f"2-D cache must be {expected_cache_shape}, got {cos_shape}"
            )
    elif len(cos_shape) == 3:
        if cos_shape[0] not in (1, B) or cos_shape[1:] != [L, D_half]:
            raise ValueError(
                f"3-D cache must be [1|B,L,D/2] for X={x_shape}, got {cos_shape}"
            )
    else:
        raise ValueError(f"cache must be 2-D or 3-D, got {cos_shape}")

    x_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    cos_in = helper.make_tensor_value_info("cos_cache", TensorProto.FLOAT, cos_shape)
    sin_in = helper.make_tensor_value_info("sin_cache", TensorProto.FLOAT, sin_shape)
    out_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, x_shape)

    node_kwargs = {"interleaved": interleaved}
    if rotary_embedding_dim:
        node_kwargs["rotary_embedding_dim"] = rotary_embedding_dim
    if num_heads:
        node_kwargs["num_heads"] = num_heads
    node = helper.make_node(
        "RotaryEmbedding",
        inputs=["x", "cos_cache", "sin_cache"],
        outputs=["out"],
        **node_kwargs,
    )

    graph_def = helper.make_graph(
        [node],
        "rotary_embedding_model",
        [x_in, cos_in, sin_in],
        [out_info],
    )

    opset = OperatorSetIdProto()
    opset.domain = ""
    # RotaryEmbedding is standardized in opset 23; onnx < 1.18 has no schema
    # for it, so check_model below reports it as unregistered. The protobuf
    # itself is well-formed either way, and esp-ppq only reads the version.
    opset.version = 23
    model_def = helper.make_model(
        graph_def, opset_imports=[opset], producer_name=export_name_prefix
    )

    try:
        onnx.checker.check_model(model_def)
        print("The model is checked!")
    except Exception as e:
        print(f"ONNX checker warning (needs onnx>=1.18 for the opset-23 schema): {e}")

    return model_def


def TOPK_TEST(config) -> ModelProto:
    """
    TopK operator

    Inputs
    X (heterogeneous) - T:
        Tensor of shape [a_1, a_2, ..., a_n, r]
    K (heterogeneous) - tensor(int64):
        A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve

    Outputs
    Values (heterogeneous) - T:
        Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing top K values
    Indices (heterogeneous) - I:
        Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing the corresponding indices
    """
    input_shape = config["input_shape"]
    k = int(config["k"])
    axis = int(config.get("axis", -1))
    largest = 1 if config.get("largest", True) else 0
    sorted_attr = 1 if config.get("sorted", True) else 0
    force_ties = bool(config.get("force_ties", False))
    export_name_prefix = config.get("export_name_prefix", "onnx-model-topk")

    rank = len(input_shape)
    if not -rank <= axis < rank:
        raise ValueError(
            f"axis={axis} is outside [-{rank}, {rank - 1}] for shape {input_shape}"
        )
    if not 0 < k <= input_shape[axis]:
        raise ValueError(
            f"k={k} must be in (0, input_shape[{axis}]={input_shape[axis]}]"
        )

    out_shape = list(input_shape)
    out_shape[axis] = k

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    values_tensor = helper.make_tensor_value_info(
        "values", TensorProto.FLOAT, out_shape
    )
    indices_tensor = helper.make_tensor_value_info(
        "indices", TensorProto.INT64, out_shape
    )
    k_init = helper.make_tensor("k", TensorProto.INT64, [1], [k])
    initializers = [k_init]
    nodes = []
    topk_input = "input"
    if force_ties:
        initializers += [
            helper.make_tensor("clip_min", TensorProto.FLOAT, [], [1.0]),
            helper.make_tensor("clip_max", TensorProto.FLOAT, [], [1.0]),
        ]
        nodes.append(
            helper.make_node("Clip", ["input", "clip_min", "clip_max"], ["clipped"])
        )
        topk_input = "clipped"

    nodes.append(
        helper.make_node(
            "TopK",
            inputs=[topk_input, "k"],
            outputs=["values", "indices"],
            axis=axis,
            largest=largest,
            sorted=sorted_attr,
        )
    )

    graph_def = helper.make_graph(
        nodes,
        "topk_model",
        [input_tensor],
        [values_tensor, indices_tensor],
        initializer=initializers,
    )

    opset = OperatorSetIdProto()
    opset.domain = ""
    opset.version = 13

    model_def = helper.make_model(
        graph_def, opset_imports=[opset], producer_name=export_name_prefix
    )
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    return model_def


def _inverse_sigmoid_const(name, value):
    return helper.make_tensor(name, TensorProto.FLOAT, [], [float(value)])


def INVERSE_SIGMOID_TEST(config) -> ModelProto:
    """InverseSigmoid — fused inverse-sigmoid."""
    input_shape = config["input_shape"]
    eps = float(config.get("eps", 1e-5))
    emit = config.get("emit", "node")
    export_name_prefix = config.get("export_name_prefix", "onnx-model-inverse-sigmoid")

    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps={eps} must lie in (0, 1)")
    if emit not in ("node", "subgraph"):
        raise ValueError(f"emit must be 'node' or 'subgraph', got {emit!r}")

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, input_shape
    )

    if emit == "node":
        node = helper.make_node(
            "InverseSigmoid", inputs=["input"], outputs=["output"], eps=eps
        )
        graph_def = helper.make_graph(
            [node], "inverse_sigmoid_model", [input_tensor], [output_tensor]
        )
    else:
        inits = [
            _inverse_sigmoid_const("c0", 0.0),
            _inverse_sigmoid_const("c1", 1.0),
            _inverse_sigmoid_const("ceps_n", eps),
            _inverse_sigmoid_const("ceps_d", eps),
            _inverse_sigmoid_const("one", 1.0),
        ]
        nodes = [
            helper.make_node("Clip", ["input", "c0", "c1"], ["xc"]),
            helper.make_node("Clip", ["xc", "ceps_n"], ["num"]),
            helper.make_node("Sub", ["one", "xc"], ["one_minus"]),
            helper.make_node("Clip", ["one_minus", "ceps_d"], ["den"]),
            helper.make_node("Div", ["num", "den"], ["ratio"]),
            helper.make_node("Log", ["ratio"], ["output"]),
        ]
        graph_def = helper.make_graph(
            nodes,
            "inverse_sigmoid_fusion_model",
            [input_tensor],
            [output_tensor],
            initializer=inits,
        )

    opset = OperatorSetIdProto()
    opset.domain = ""
    opset.version = 13
    model_def = helper.make_model(
        graph_def, opset_imports=[opset], producer_name=export_name_prefix
    )

    try:
        onnx.checker.check_model(model_def)
        print("The model is checked!")
    except Exception as e:
        print(f"ONNX checker warning (expected for custom InverseSigmoid): {e}")

    return model_def
