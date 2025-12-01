from sin_model import generate_data, SinPredictor
from torch.utils.data import DataLoader, TensorDataset
import torch
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor


def collate_fn(batch):
    # When iterating over TensorDataset, it returns Tuple(x, y). For quantization, only x is needed, not label y.
    batch = batch[0].to(DEVICE)
    return batch


if __name__ == "__main__":

    ESPDL_MODEL_PATH = "sin_model.espdl"
    INPUT_SHAPE = [1, 1]  # 1 input feature
    TARGET = (
        "esp32s3"  # Quantization target type, options: 'c', 'esp32s3', or 'esp32p4'
    )
    NUM_OF_BITS = 8  # Number of quantization bits
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available

    x_test, y_test = generate_data()
    # Dataloader shuffle must be set to False.
    # Because when calculating quantization error, the dataset will be traversed multiple times. If shuffle is True, the quantization error will be incorrect.
    dataset = TensorDataset(x_test, y_test)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = SinPredictor()
    model.load_state_dict(torch.load("sin_model.pth"))
    model.eval()

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # Number of calibration steps
        input_shape=INPUT_SHAPE,  # Input shape, batch size is 1
        inputs=None,
        target=TARGET,  # Quantization target type
        num_of_bits=NUM_OF_BITS,  # Number of quantization bits
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # Output detailed log information
    )

    criterion = torch.nn.MSELoss()
    # Test accuracy of the original model on the test set
    loss = 0
    for batch_x, batch_y in dataloader:
        y_pred = model(batch_x)
        loss += criterion(y_pred, batch_y)
    loss /= len(dataloader)
    print(f"origin model loss: {loss.item():.5f}")

    # Test accuracy of the quantized model on the test set
    executor = TorchExecutor(graph=quant_ppq_graph, device=DEVICE)
    loss = 0
    for batch_x, batch_y in dataloader:
        y_pred = executor(batch_x)
        loss += criterion(y_pred[0], batch_y)
    loss /= len(dataloader)
    print(f"quant model loss: {loss.item():.5f}")
