from sin_model import generate_data, SinPredictor
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx


def collate_fn(batch):
    # TensorDataset 迭代的时候返回的是 Tuple(x, y), 量化的时候只需要x, 不需要label y。
    batch = batch[0].to(DEVICE)
    return batch


if __name__ == "__main__":

    ONNX_MODEL_PATH = "sin_model.onnx"
    ESPDL_MODEL_PATH = "sin_model.espdl"
    INPUT_SHAPE = [1, 1]  # 1 个输入特征
    TARGET = "esp32s3"  # 量化目标，可选 'esp32p4', 'esp32s31', 'esp32s3' or 'c'
    QUANT_TYPE = "w8a8"  # 量化方案，可选 'w8a8', 'w8a16', 'w16a16' or 'none'
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available

    x, y = generate_data()
    # dataloader shuffle必须设置为False。
    # 因为计算量化误差的时候会多次遍历数据集，如果shuffle是True的话，会得到错误的量化误差。
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_MODEL_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # 校准的步数
        input_shape=INPUT_SHAPE,  # 输入形状，批次为 1
        inputs=None,
        target=TARGET,  # 量化目标类型
        quant_type=QUANT_TYPE,  # 量化方案
        collate_fn=collate_fn,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # 输出详细日志信息
    )
