import numpy as np
import argparse
import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from esp_quantizer import esp_quantize_static

class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_npy: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)

        self.calib_data_list = np.load(calibration_image_npy)
        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: calib_data[np.newaxis, ...]} for calib_data in self.calib_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def main(input_model_path, output_model_path, calibration_dataset_path, per_channel_):
    model_proto = onnx.load(input_model_path)
    
    dr = DataReader(
        calibration_dataset_path, input_model_path
    )

    extra_options_ = {
        "ActivationSymmetric": "True",
        "WeightSymmetric": "True",
    }
    
    not_supported_node = []
    tvm_not_support_list = ['Softmax']
    for node in model_proto.graph.node:
        if node.op_type in tvm_not_support_list:
            not_supported_node.append(node.name)
            
    esp_quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QOperator,
        calibrate_method=CalibrationMethod.Entropy,
        per_channel=per_channel_,
        weight_type=QuantType.QInt8,
        optimize_model=True,
        nodes_to_exclude=not_supported_node,
        extra_options=extra_options_
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset", required=True, help="calibration data set")

    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    
    main(args.input_model, args.output_model, args.calibrate_dataset, args.per_channel)