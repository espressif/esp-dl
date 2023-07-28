import numpy as np
import argparse
import collections
import logging
import tempfile
from pathlib import Path
import onnx
import onnx.numpy_helper
from onnx import onnx_pb as onnx_proto
import onnxruntime
from onnxruntime.quantization import (
    CalibrationDataReader, 
    CalibrationMethod,
    QuantFormat,
    QuantizationMode,
    QuantType, 
    QDQQuantizer,
    create_calibrator
)
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.registry import QLinearOpsRegistry, CreateOpQuantizer
from onnxruntime.quantization.quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizedValue,
    QuantizedValueType,
    QuantType,
    __producer__,
    __version__,
    load_model,
    model_has_pre_process_metadata,
    compute_scale_zp,
    find_by_name,
    get_qmin_qmax_for_qType,
    quantize_nparray,
    tensor_proto_to_array,
)


def conver_scale_to_2_exponent(scale):
	power = np.ceil(np.log2(scale))
	new_scale = pow(2, power)

	return new_scale
  
class ESPQuantizer(ONNXQuantizer):
	def quantize_data(self, data, qType, symmetric, reduce_range=False): # from quant_utils
		"""
		:param data: data to quantize
		:param qType: data type to quantize to. Supported types UINT8 and INT8
		:param symmetric: whether symmetric quantization is used or not. This is applied to INT8.
		:return: minimum, maximum, zero point, scale, and quantized weights

		To pack weights, we compute a linear transformation

		- when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
		- when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
			`m = max(abs(rmin), abs(rmax))`

		and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation

		:math:`r = S(q-z)`, where

		- *r*: real original value
		- *q*: quantized value
		- *S*: scale
		- *z*: zero point
		"""

		rmin = 0
		rmax = 0
		zero_point = 0
		scale = 1.0
		if len(data):
			rmin = min(data)
			rmax = max(data)
			qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)

			zero_point, scale = compute_scale_zp(rmin, rmax, qmin, qmax, symmetric)
			scale = conver_scale_to_2_exponent(scale)

		quantized_data = quantize_nparray(qType, np.asarray(data), scale, zero_point)

		return rmin, rmax, zero_point, scale, quantized_data
	
	def calculate_quantization_params(self):
			if self.tensors_range is None:
				return

			# adjust tensor_ranges for input of Clip and Relu node
			for node in self.model.nodes():
				if node.op_type not in ["Clip", "Relu"]:
					continue
				if self.is_activation_symmetric:
					continue
				if not self.should_quantize_node(node):
					continue
				if len(self.model.input_name_to_nodes()[node.input[0]]) != 1:
					continue
				if node.input[0] not in self.tensors_range.keys() or node.output[0] not in self.tensors_range.keys():
					continue
				self.tensors_range[node.input[0]] = self.tensors_range[node.output[0]]
			quantization_params = {}
			for tensor_name in self.tensors_range.keys():
				rmin, rmax = self.tensors_range[tensor_name]
				qmin, qmax = get_qmin_qmax_for_qType(self.activation_qType, symmetric=self.is_activation_symmetric)

				quantization_params[tensor_name] = compute_scale_zp(rmin, rmax, qmin, qmax, self.is_activation_symmetric) #[z, s]
				quantization_params[tensor_name][1] = conver_scale_to_2_exponent(quantization_params[tensor_name][1])

			return quantization_params

	def quantize_initializer(self, weight, qType, reduce_range=False, keep_float_weight=False):
		"""
		:param weight: TensorProto initializer
		:param qType: type to quantize to
		:param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
									If keep_float_weight is False, quantize the weight, or don't quantize the weight.
		:return: quantized weight name, zero point name, scale name
		"""
		# Find if this input is already quantized
		if weight.name in self.quantized_value_map:
			quantized_value = self.quantized_value_map[weight.name]
			return (
				quantized_value.q_name,
				quantized_value.zp_name,
				quantized_value.scale_name,
			)

		q_weight_name = weight.name + TENSOR_NAME_QUANT_SUFFIX
		zp_name = weight.name + "_zero_point"
		scale_name = weight.name + "_scale"

		# Update packed weight, zero point, and scale initializers
		weight_data = tensor_proto_to_array(weight)
		_, _, zero_point, scale, q_weight_data = self.quantize_data(
			weight_data.flatten().tolist(),
			qType,
			self.is_weight_symmetric,
			self.reduce_range and reduce_range,
		)
		scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, [], [scale])
		zero_initializer = onnx.helper.make_tensor(zp_name, qType, [], [zero_point])
		self.model.initializer().extend([scale_initializer, zero_initializer])

		if not keep_float_weight:
			q_weight_data = np.asarray(q_weight_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[qType]).reshape(
				weight.dims
			)
			q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)
			self.model.initializer().extend([q_weight_initializer])

		# Log entry for this quantized weight
		quantized_value = QuantizedValue(
			weight.name,
			q_weight_name,
			scale_name,
			zp_name,
			QuantizedValueType.Initializer,
			None,
		)
		self.quantized_value_map[weight.name] = quantized_value

		return q_weight_name, zp_name, scale_name

	def quantize_weight_per_channel(
		self,
		weight_name,
		weight_qType,
		channel_axis,
		reduce_range=True,
		keep_float_weight=False,
	):
		# Find if this input is already quantized
		if weight_name in self.quantized_value_map:
			quantized_value = self.quantized_value_map[weight_name]
			return (
				quantized_value.q_name,
				quantized_value.zp_name,
				quantized_value.scale_name,
			)

		initializer = find_by_name(weight_name, self.model.initializer())
		if initializer is None:
			raise ValueError("{} is not an initializer", weight_name)

		weights = tensor_proto_to_array(initializer)
		channel_count = weights.shape[channel_axis]
		rmin_list = []
		rmax_list = []
		zero_point_list = []
		scale_list = []
		quantized_per_channel_data_list = []
		for i in range(channel_count):
			per_channel_data = weights.take(i, channel_axis)
			rmin, rmax, zero_point, scale, quantized_per_channel_data = self.quantize_data(
				per_channel_data.flatten().tolist(),
				weight_qType,
				self.is_weight_symmetric or weight_qType == onnx_proto.TensorProto.INT8,
				self.reduce_range and reduce_range,
			)
			rmin_list.append(rmin)
			rmax_list.append(rmax)
			zero_point_list.append(zero_point)
			scale_list.append(scale)
			quantized_per_channel_data_list.append(quantized_per_channel_data)

		# combine per_channel_data into one
		reshape_dims = list(weights.shape)  # deep copy
		reshape_dims[channel_axis] = 1  # only one per channel for reshape
		quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
		for i in range(1, len(quantized_per_channel_data_list)):
			channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
			quantized_weights = np.concatenate((quantized_weights, channel_weights), channel_axis)

		q_weight_name = weight_name + TENSOR_NAME_QUANT_SUFFIX
		zp_name = weight_name + "_zero_point"
		scale_name = weight_name + "_scale"

		quantized_value = QuantizedValue(
			weight_name,
			q_weight_name,
			scale_name,
			zp_name,
			QuantizedValueType.Initializer,
			None,
		)
		self.quantized_value_map[weight_name] = quantized_value

		# Update packed weight, zero point, and scale initializers
		zero_scale_shape = [initializer.dims[channel_axis]]
		scale_initializer = onnx.helper.make_tensor(
			scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape, scale_list
		)
		zero_initializer = onnx.helper.make_tensor(zp_name, weight_qType, zero_scale_shape, zero_point_list)

		self.model.initializer().extend([scale_initializer, zero_initializer])

		if not keep_float_weight:
			quantized_weights = np.asarray(
				quantized_weights,
				dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight_qType],
			).reshape(initializer.dims)
			q_weight_initializer = onnx.numpy_helper.from_array(quantized_weights, q_weight_name)
			self.model.initializer().extend([q_weight_initializer])

		return q_weight_name, zp_name, scale_name


	def quantize_model(self):
		if self.has_QDQ_nodes():
				logging.warning(
						"Please check if the model is already quantized."
						"Note you don't need to quantize a QAT model. OnnxRuntime support to run QAT model directly."
				)

		for node in self.model.nodes():
				# quantize subgraphes if have
				if self.enable_subgraph_quantization:
						node = self.quantize_node_with_sub_graph(node)

				number_of_existing_new_nodes = len(self.new_nodes)
				op_quantizer = CreateOpQuantizer(self, node)
				op_quantizer.quantize()
				for i in range(number_of_existing_new_nodes, len(self.new_nodes)):
						for output_name in self.new_nodes[i].output:
								self.generated_value_names.add(output_name)

		self._dequantize_outputs()

		
		# extend is used to append to the list for a protobuf fields
		# https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
		self.model.graph().ClearField("node")
		self.model.graph().node.extend(self.new_nodes)
  
		self.new_nodes = []
		self.handle_bias()
		self.model.graph().node.extend(self.new_nodes)
  
		# Remove ununsed initializers from graph, starting from the top level graph.
		if self.parent is None:
				_, initializers_not_found = self.model.clean_initializers()
				if len(initializers_not_found) > 0:
						raise RuntimeError("Invalid model with unknown initializers/tensors." + str(initializers_not_found))

		self.model.model.producer_name = __producer__
		self.model.model.producer_version = __version__

		return self.model.model
    
    
	def handle_bias(self):
		for node in self.model.model.graph.node:
			if node.op_type == 'QLinearConv':
				
				if len(node.input) == 9:
					# print(node.input[8])
					new_bias_name = node.input[8]
					old_bias_name = new_bias_name.replace(TENSOR_NAME_QUANT_SUFFIX, "")
     
					new_bias_initializer = find_by_name(new_bias_name, self.model.initializer())
					# new_bias_data = tensor_proto_to_array(new_bias_initializer)
					old_bias_initializer = find_by_name(old_bias_name, self.model.initializer())
					old_bias_data = tensor_proto_to_array(old_bias_initializer)
					outputscale_initializer = find_by_name(node.input[6], self.model.initializer())
					output_scale = tensor_proto_to_array(outputscale_initializer)
					quantized_data = (np.asarray(old_bias_data) / output_scale).round().astype(np.int8)
     
					# update bias initializer
					# fake_bias_name = new_bias_name + "_fake"
					# zero_bias_data = np.zeros(new_bias_initializer.dims, dtype=np.int32)
					# packed_fake_bias_initializer = onnx.numpy_helper.from_array(zero_bias_data, fake_bias_name)
					# self.model.initializer().extend([packed_fake_bias_initializer])
					# node.input[8] = fake_bias_name
					node.input.remove(node.input[8])
     
					bias_name = new_bias_name + "_dl"
					packed_bias_initializer = onnx.numpy_helper.from_array(quantized_data[:, np.newaxis, np.newaxis], bias_name)
					self.model.initializer().extend([packed_bias_initializer])

					ori_output_name  = node.output[0]
					new_output_name = node.output[0] + '/conv_bias:0'
					node.output[0] = new_output_name
					bias_add_node = onnx.helper.make_node(
						op_type='Add',
						name=node.name + '/conv_bias',
						inputs=[new_output_name, bias_name],
						outputs=[ori_output_name],
					)
					self.new_nodes.append(bias_add_node)
					# self.model.graph().node.extend(bias_add_node)
		
def check_static_quant_arguments(quant_format: QuantFormat, activation_type: QuantType, weight_type: QuantType):
    if activation_type == QuantType.QInt8 and weight_type == QuantType.QUInt8:
        raise ValueError(
            "ONNXRuntime quantization doesn't support data format:"
            "activation_type=QuantType.QInt8, weight_type = QuantType.QUInt8"
        )

    if activation_type == QuantType.QInt8 and weight_type == QuantType.QInt8 and quant_format != QuantFormat.QDQ:
        logging.warning(
            "Please use QuantFormat.QDQ for activation type QInt8 and weight type QInt8. "
            "Or it will lead to bad performance on x64."
        )

def fuse_dequantize_relu_quantize(onnx_model):
    model_proto = onnx_model.model
    nodes = model_proto.graph.node

    for node in nodes:
        if node.op_type == 'Relu':
            parents = onnx_model.get_parents(node)
            children = onnx_model.get_children(node)
            if len(parents) == 1 and len(children) == 1:
                if parents[0].op_type == 'DequantizeLinear' and children[0].op_type == 'QuantizeLinear':
                    pre = parents[0]
                    post = children[0]
                    pre_scale = find_by_name(pre.input[1], onnx_model.initializer()).float_data
                    pose_scale = find_by_name(post.input[1], onnx_model.initializer()).float_data
                    if pre_scale == pose_scale: # same scale
                       input_data = pre.input[0]
                       output_data = post.output[0]
                       node.input[0] = input_data
                       node.output[0] = output_data
                       model_proto.graph.node.remove(pre)
                       model_proto.graph.node.remove(post)
              
def esp_quantize_static(
    model_input,
    model_output,
    calibration_data_reader: CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    optimize_model=True,
    use_external_data_format=False,
    calibrate_method=CalibrationMethod.MinMax,
    extra_options=None,
):
    """
    Given an onnx model and calibration data reader, create a quantized onnx model and save it into a file
    It is recommended to use QuantFormat.QDQ format from 1.11 with activation_type = QuantType.QInt8 and weight_type
    = QuantType.QInt8. If model is targeted to GPU/TRT, symmetric activation and weight are required. If model is
    targeted to CPU, asymmetric activation and symmetric weight are recommended for balance of performance and
    accuracy.

    Args:

        model_input: file path of model to quantize
        model_output: file path of quantized model
        calibration_data_reader: a calibration data reader. It
            enumerates calibration data and generates inputs for the
            original model.
        quant_format: QuantFormat{QOperator, QDQ}.
            QOperator format quantizes the model with quantized operators directly.
            QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
        activation_type:
            quantization data type of activation. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        calibrate_method:
            Current calibration methods supported are MinMax and Entropy.
                Please use CalibrationMethod.MinMax or CalibrationMethod.Entropy as options.
        op_types_to_quantize:
                specify the types of operators to quantize, like ['Conv'] to quantize Conv only.
                It quantizes all supported operators by default.
        per_channel: quantize weights per channel
        reduce_range:
            quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine,
            especially for per-channel mode
        weight_type:
            quantization data type of weight. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        nodes_to_quantize:
            List of nodes names to quantize. When this list is not None only the nodes in this list
            are quantized.
            example:
            [
                'Conv__224',
                'Conv__252'
            ]
        nodes_to_exclude:
            List of nodes names to exclude. The nodes in this list will be excluded from quantization
            when it is not None.
        optimize_model: Deprecating Soon! Optimize model before quantization. NOT recommended, optimization will
            change the computation graph, making debugging of quantization loss difficult.
        use_external_data_format: option used for large size (>2GB) model. Set to False by default.
        extra_options:
            key value pair dictionary for various options in different case. Current used:
                extra.Sigmoid.nnapi = True/False  (Default is False)
                ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
                WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
                EnableSubgraph = True/False : Default is False. If enabled, subgraph will be quantized.
                                              Dyanmic mode currently is supported. Will support more in the future.
                ForceQuantizeNoInputCheck = True/False :
                    By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                    quantized already. Setting to True to force such operator always quantize input and so generate
                    quantized output. Also, the True behavior could be disabled per node using the nodes_to_exclude.
                MatMulConstBOnly = True/False:
                    Default is False for static mode. If enabled, only MatMul with const B will be quantized.
                AddQDQPairToWeight = True/False :
                    Default is False which quantizes floating-point weight and feeds it to solely inserted
                    DeQuantizeLinear node. If True, it remains floating-point weight and inserts both
                    QuantizeLinear/DeQuantizeLinear nodes to weight.
                OpTypesToExcludeOutputQuantization = list of op type :
                    Default is []. If any op type is specified, it won't quantize the output of ops with this
                    specific op types.
                DedicatedQDQPair = True/False :
                    Default is False. When inserting QDQ pair, multiple nodes can share a single QDQ pair as their
                    inputs. If True, it will create identical and dedicated QDQ pair for each node.
                QDQOpTypePerChannelSupportToAxis = dictionary :
                    Default is {}. Set channel axis for specific op type, for example: {'MatMul': 1}, and it's
                    effective only when per channel quantization is supported and per_channel is True. If specific
                    op type supports per channel quantization but not explicitly specified with channel axis,
                    default channel axis will be used.
                CalibTensorRangeSymmetric = True/False :
                    Default is False. If enabled, the final range of tensor during calibration will be explicitly
                    set to symmetric to central point "0".
                CalibMovingAverage = True/False :
                    Default is False. If enabled, the moving average of the minimum and maximum values will be
                    computed when the calibration method selected is MinMax.
                CalibMovingAverageConstant = float :
                    Default is 0.01. Constant smoothing factor to use when computing the moving average of the
                    minimum and maximum values. Effective only when the calibration method selected is MinMax and
                    when CalibMovingAverage is set to True.
    """

    extra_options = extra_options or {}
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_quantize = nodes_to_quantize or []
    op_types_to_quantize = op_types_to_quantize or []
    mode = QuantizationMode.QLinearOps
    # mode = QuantizationMode.IntegerOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)

    pre_processed: bool = model_has_pre_process_metadata(model)
    if not pre_processed:
        logging.warning(
            "Please consider pre-processing before quantization. See "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )

    calib_extra_options_keys = [
        ("CalibTensorRangeSymmetric", "symmetric"),
        # ("CalibMovingAverage", "moving_average"),
        # ("CalibMovingAverageConstant", "averaging_constant"),
    ]
    calib_extra_options = {
        key: extra_options.get(name) for (name, key) in calib_extra_options_keys if name in extra_options
    }

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        calibrator = create_calibrator(
            model,
            op_types_to_quantize,
            augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
            calibrate_method=calibrate_method,
            use_external_data_format=use_external_data_format,
            extra_options=calib_extra_options,
        )
        calibrator.collect_data(calibration_data_reader)
        tensors_range = calibrator.compute_range()
        del calibrator

    check_static_quant_arguments(quant_format, activation_type, weight_type)

    if quant_format is QuantFormat.QOperator:
        quantizer = ESPQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
    else:
        quantizer = QDQQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )

    quantizer.quantize_model()
    # quantizer.handle_bias()

  	# rename node name
    node_name_counter = collections.Counter()
    for node in quantizer.model.model.graph.node:
        node_name_counter[node.op_type] += 1
        node.name = str(node.op_type) + '_' + str(node_name_counter[node.op_type])
    
    fuse_dequantize_relu_quantize(quantizer.model)
  
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
    if not pre_processed:
        logging.warning(
            "Please consider pre-processing before quantization. See "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )
