import numpy as np
import os
import shutil
import tarfile
import pathlib
from pathlib import Path
import re
import onnx
import tvm
from tvm.micro import export_model_library_format
from tvm.relay.op.contrib.esp import preprocess_onnx_for_esp, evaluate_onnx_for_esp

if 'ESP_DL_PATH' in os.environ:
    esp_dl_library_path = os.environ['ESP_DL_PATH']
else:
    current_script_path = Path(__file__).resolve()
    parent_parent_path = current_script_path.parent.parent.parent
    esp_dl_library_path = str(parent_parent_path)

def generate_project(template_dir, tar_file, output_dir, input_ndarray, input_shape, input_dtype, output_shape, output_dtype):    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    
    # shutil.copytree(template_dir, output_dir, dirs_exist_ok=True) #only python3.8
    shutil.copytree(template_dir, output_dir)
    
    
    # copy esp-dl lib as one of the component
    print(f"esp_dl_library_path: {esp_dl_library_path}")
    shutil.copytree(esp_dl_library_path + '/include', output_dir + '/components/esp-dl/include')
    shutil.copytree(esp_dl_library_path + '/lib', output_dir + '/components/esp-dl/lib')
    shutil.copy(esp_dl_library_path + '/CMakeLists.txt', output_dir + '/components/esp-dl/CMakeLists.txt')
    

    # tvm model files as one of the component
    target_dir = os.path.join(output_dir, 'components', 'tvm_model', 'model')
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(tar_file) as tar:
        tar.extractall(path=target_dir)
    
        
    with open(target_dir+'/codegen/host/include/tvmgen_default.h', 'r') as f:
        text = f.read()
    input_name, output_name=re.findall(r'void\*\s*(\w+)', text)

    file_path = os.path.join(output_dir, 'main', 'app_main.c')
    with open(file_path, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('.input = input_data', f'.{input_name} = input_data')
    filedata = filedata.replace('.output = output_data', f'.{output_name} = output_data')
    with open(file_path, 'w') as file:
        file.write(filedata)

    # store the input ndarray into a new "input_data.h" file
    with open(os.path.join(output_dir, 'main', 'input_data.h'), 'w') as file:
        file.write('#include <tvmgen_default.h>\n')
        file.write(f'const size_t input_len = {input_ndarray.size};\n')
        file.write(f'const static __attribute__((aligned(16))) {input_dtype} input_data[] = {{')
        file.write(', '.join(map(str, input_ndarray.flatten().tolist())))
        file.write('};\n')

    # allocate a array for the output in a new "output_data.h" file
    product = 1
    for num in output_shape:
        product *= num
    with open(os.path.join(output_dir, 'main', 'output_data.h'), 'w') as file:
        file.write('#include <tvmgen_default.h>\n')
        file.write(f'const size_t output_len = {product};\n')
        file.write(f'const static __attribute__((aligned(16))) {output_dtype} output_data[{product}]\n')
        file.write('={};\n')

    print(f"generated project in: {output_dir}")
    
    

def dump_tvm_mod(mod, params, model_tar_file):
    RUNTIME = tvm.relay.backend.Runtime("crt")
    TARGET = tvm.target.target.micro("esp32")

    executor = tvm.relay.backend.Executor("aot",
        {
            "workspace-byte-alignment": 16,
            "constant-byte-alignment": 16,
            "interface-api": 'c',
            "unpacked-api": 1,
        }
    )

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True, "tir.disable_storage_rewrite": True, "tir.usmp.enable":True,"tir.usmp.algorithm":"hill_climb"}):
        module = tvm.relay.build(mod, target=TARGET, runtime=RUNTIME, params=params, executor=executor)

    export_model_library_format(module, model_tar_file)

  

def get_shape(tensor):
    shape = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            # shape.append(dim.dim_param)
            shape.append(1)
    return tuple(shape)

TENSOR_TYPE_TO_C_TYPE = {1: 'float', 2: 'uint8_t', 3: 'int8_t', 4: 'uint16_t', 5: 'int16_t', 6: 'int'}
TENSOR_TYPE_TO_NP_TYPE = {1: np.dtype('float32'), 2: np.dtype('uint8'), 3: np.dtype('int8'), 4: np.dtype('uint16'), 5: np.dtype('int16'), 6: np.dtype('int32'), 7: np.dtype('int64'), 9: np.dtype('bool'), 10: np.dtype('float16'), 16: np.dtype('float32'), 11: np.dtype('float64'), 14: np.dtype('complex64'), 15: np.dtype('complex128'), 12: np.dtype('uint32'), 13: np.dtype('uint64'), 8: np.dtype('O')}

def get_c_type(tensor):
    elem_type = tensor.elem_type
    # type = onnx.TensorProto.DataType.items()[elem_type][0]
    assert elem_type in TENSOR_TYPE_TO_C_TYPE, f"{elem_type} not found in TENSOR_TYPE_TO_C_TYPE"
    return TENSOR_TYPE_TO_C_TYPE[elem_type]

def get_np_type(tensor):
    elem_type = tensor.elem_type
    assert elem_type in TENSOR_TYPE_TO_NP_TYPE, f"{elem_type} not found in TENSOR_TYPE_TO_NP_TYPE"
    return TENSOR_TYPE_TO_NP_TYPE[elem_type]

def get_model_info(onnx_model):
    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    input_shape = get_shape(onnx_model.graph.input[0].type.tensor_type)
    output_shape = get_shape(onnx_model.graph.output[0].type.tensor_type)
    input_dtype = get_c_type(onnx_model.graph.input[0].type.tensor_type)
    output_dtype = get_c_type(onnx_model.graph.output[0].type.tensor_type)
    
    return input_name, input_shape, input_dtype, output_name, output_shape, output_dtype
    
def print_model_info(input_name, input_shape, input_type, output_name, output_shape, output_type):
    print("Model Information:")
    print("------------------")
    print(f"Input Name: {input_name}")
    print(f"Input Shape: {input_shape}")
    print(f"Input DType: {input_type}")
    print(f"Output Name: {output_name}")
    print(f"Output Shape: {output_shape}")
    print(f"Output DType: {output_type}")

def dump_tvm_onnx_project(target_chip, model_path, img_path, template_path, generated_project_path):
    onnx_model = onnx.load(model_path)
    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    input_shape = get_shape(onnx_model.graph.input[0].type.tensor_type)
    output_shape = get_shape(onnx_model.graph.output[0].type.tensor_type)
    input_dtype = get_c_type(onnx_model.graph.input[0].type.tensor_type)
    output_dtype = get_c_type(onnx_model.graph.output[0].type.tensor_type)
    
    print_model_info(input_name, input_shape, input_dtype, output_name, output_shape, output_dtype)
    
    image_sample = np.load(img_path)
    image_shape = image_sample.shape
    assert input_shape == image_shape, f"input shape={input_shape}, image_shape={image_shape}: shapes must be same"
    
    mod, params = tvm.relay.frontend.from_onnx(onnx_model, shape={input_name: input_shape})
    
    if target_chip == 'esp32s3':
        desired_layouts = {'qnn.conv2d': ['NHWC', 'default'], 'nn.avg_pool2d': ['NHWC', 'default']}
        # Convert the layout to NCHW
        seq = tvm.transform.Sequential([tvm.relay.transform.RemoveUnusedFunctions(),
                                        tvm.relay.transform.ConvertLayout(desired_layouts)],
                                    )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)

        mod = preprocess_onnx_for_esp(mod, params)
    
    
    # print(mod)
    model_tar_file = generated_project_path + "/model.tar"
    dump_tvm_mod(mod, params, model_tar_file)
    output_dir = generated_project_path + "/new_project"
    generate_project(template_path, model_tar_file, output_dir, image_sample, input_shape, input_dtype, output_shape, output_dtype) 
    

def debug_onnx_model(target_chip, model_path, img_path):
    from tvm.contrib.debugger.debug_executor import GraphModuleDebug
    
    onnx_model = onnx.load(model_path)
    input_name = onnx_model.graph.input[0].name
    input_shape = get_shape(onnx_model.graph.input[0].type.tensor_type)
    output_shape = get_shape(onnx_model.graph.output[0].type.tensor_type)
    input_dtype = get_np_type(onnx_model.graph.input[0].type.tensor_type)
    output_dtype = get_np_type(onnx_model.graph.output[0].type.tensor_type)
    
    mod, params = tvm.relay.frontend.from_onnx(onnx_model, shape={input_name: input_shape})
    if target_chip == 'esp32s3':
        desired_layouts = {'qnn.conv2d': ['NHWC', 'default'], 'nn.avg_pool2d': ['NHWC', 'default']}
        # Convert the layout to NCHW
        seq = tvm.transform.Sequential([tvm.relay.transform.RemoveUnusedFunctions(),
                                        tvm.relay.transform.ConvertLayout(desired_layouts)],
                                    )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)

    mod = evaluate_onnx_for_esp(mod, params)
    mod = tvm.relay.qnn.transform.CanonicalizeOps()(mod)
    target = "llvm"
    dev = tvm.device("llvm")
    with tvm.transform.PassContext(opt_level=0):
        lib = tvm.relay.build(mod, target, params=params)

    m = GraphModuleDebug(
        lib["debug_create"]("default", dev),
        [dev],
        lib.graph_json,
        dump_root = os.path.dirname(os.path.abspath(model_path))+"/tvmdbg",
    )
    # set inputs
    image_sample = np.load(img_path)
    m.set_input(input_name, tvm.nd.array(image_sample))
    m.set_input(**params)
    m.run()
    # output of model
    tvm_out = m.get_output(0, tvm.nd.empty(output_shape, dtype=output_dtype))
    print(tvm_out.numpy())
    
    # output of first relu layer
    tvm_out = tvm.nd.empty((1,32,32,16),dtype="int8")
    m.debug_get_output("tvmgen_default_fused_nn_relu", tvm_out)
    print(tvm_out.numpy().flatten()[0:16])
    
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Assistant')
    parser.add_argument('--target_chip', help='esp32s3')
    parser.add_argument('--model_path', help='path of onnx model')
    parser.add_argument('--img_path', help='path of npy file which stores an input image of this model')
    parser.add_argument('--template_path', help='path of template project')
    parser.add_argument('--out_path', help='path of generated project')
    args = parser.parse_args()
    
    dump_tvm_onnx_project(target_chip=args.target_chip, model_path=args.model_path, img_path=args.img_path,  template_path=args.template_path, generated_project_path=args.out_path)
    # debug_onnx_model(args.target_chip, args.model_path, args.img_path)