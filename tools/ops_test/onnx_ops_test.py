import onnx
import numpy as np
from onnx import ModelProto, TensorProto, helper, OperatorSetIdProto


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
    keepdims = int(config.get("keepdims", False))
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
