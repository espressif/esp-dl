from torch.utils.data import DataLoader, TensorDataset
import torch
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor
from models import TCN, TestModel_2
import os


def generate_data(input_shape):
    return torch.randn(*input_shape)


if __name__ == "__main__":

    TARGET_TEST_PATH = "../test_streaming_model/main/models"
    INPUT_SHAPE = [1, 16, 15]  # input feature
    TARGET = "esp32p4"  # Quantization target type, options: 'c', 'esp32s3' or 'esp32p4'
    NUM_OF_BITS = 8  # Number of quantization bits
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
    TARGET_TEST_PATH = os.path.join(TARGET_TEST_PATH, TARGET)
    ESPDL_MODEL_PATH = os.path.join(TARGET_TEST_PATH, "model.espdl")
    ESPDL_STEAMING_MODEL_PATH = os.path.join(TARGET_TEST_PATH, "streaming_model.espdl")

    x_test = generate_data(INPUT_SHAPE)
    dataset = TensorDataset(x_test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = TCN(
        in_channels=INPUT_SHAPE[1],
        expand_channels=32,
        out_channels=16,
        kernel_size=3,
        stride=1,
        dilation=2,
    )
    model.eval()

    # export non-streaming model
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # Number of calibration steps
        input_shape=INPUT_SHAPE,  # Input shape, batch size is 1
        inputs=None,
        target=TARGET,  # Quantization target type
        num_of_bits=NUM_OF_BITS,  # Number of quantization bits
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # Output detailed log information
    )

    # export streaming model
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_STEAMING_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=1,
        auto_streaming=True,  # Enable automatic streaming conversion
        streaming_input_shape=[1, 16, 3],  # Streaming input shape
        streaming_table=None,
    )
