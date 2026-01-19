from torch.utils.data import DataLoader, TensorDataset
import torch
from esp_ppq.api import espdl_quantize_torch, insert_streaming_cache_on_var
from esp_ppq.executor.torch import TorchExecutor
from models import TCN, TestModel_2
import os


def generate_data(input_shape):
    return torch.randn(*input_shape)


if __name__ == "__main__":

    TARGET_TEST_PATH = "../test_streaming_model/main/models"
    INPUT_SHAPE = [1, 16, 24]  # input feature: [batch_size, channels, sequence_length]
    TARGET = "esp32p4"  # Quantization target type, options: 'c', 'esp32s3' or 'esp32p4'
    NUM_OF_BITS = 8  # Number of quantization bits
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
    TARGET_TEST_PATH = os.path.join(TARGET_TEST_PATH, TARGET)
    ESPDL_MODEL_PATH = os.path.join(TARGET_TEST_PATH, "model.espdl")
    ESPDL_STEAMING_MODEL_PATH = os.path.join(TARGET_TEST_PATH, "streaming_model.espdl")

    x_test = generate_data(INPUT_SHAPE)
    dataset = TensorDataset(x_test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_frame_size = 3
    model = TCN(
        in_channels=INPUT_SHAPE[1],
        expand_channels=INPUT_SHAPE[1] * 2,
        out_channels=INPUT_SHAPE[1],
        kernel_size=3,
        stride=1,
        dilation=2,
        output_frame_size=output_frame_size,
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
    streaming_table = []
    # Manually specify cache attributes for variables that can not insert streamingCache automatically.
    # def insert_streaming_cache_on_var(
    #     var_name: str, window_size: int, op_name: str = None, frame_axis: int = 1
    # ) -> Dict[str, Any]
    streaming_table.append(
        insert_streaming_cache_on_var("/out_conv/Conv_output_0", output_frame_size - 1)
    )
    streaming_table.append(insert_streaming_cache_on_var("PPQ_Variable_0", 1, "/Slice"))

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
        streaming_input_shape=[
            1,
            16,
            3,
        ],  # Streaming input shape, [batch_size, channels, chunk_size]
        streaming_table=streaming_table,
    )
