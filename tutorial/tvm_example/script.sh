python -m onnxruntime.quantization.preprocess --input model.onnx --output model_opt.onnx

python ../../tools/tvm/esp_quantize_onnx.py --input_model model_opt.onnx --output_model model_quant.onnx --calibrate_dataset calib_img.npy

python ../../tools/tvm/export_onnx_model.py --model_path model_quant.onnx --img_path input_sample.npy --target_chip esp32s3 --out_path "." --template_path "../../tools/tvm/template_project_for_model/"
