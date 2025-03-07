import onnx
from onnx import ModelProto, TensorProto, helper


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
