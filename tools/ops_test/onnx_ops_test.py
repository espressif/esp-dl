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
