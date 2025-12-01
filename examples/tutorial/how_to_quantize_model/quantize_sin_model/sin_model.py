import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxsim


class SinPredictor(nn.Module):
    def __init__(self):
        super(SinPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # Input layer, 1 feature (x)
        self.fc2 = nn.Linear(64, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 1)  # Output layer, 1 feature (y)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_data(num_samples=1000, noise=0.05):
    x = torch.linspace(0, 2 * np.pi, num_samples)
    y = torch.sin(x) + noise * torch.randn(num_samples)  # Sine function with noise
    return x.unsqueeze(1), y.unsqueeze(1)


if __name__ == "__main__":
    x_train, y_train = generate_data()

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SinPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()

        for batch_x, batch_y in dataloader:
            # Forward pass
            y_pred = model(batch_x)

            # Compute loss
            loss = criterion(y_pred, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Switch to evaluation mode
    model.eval()
    x_test, y_test = generate_data()
    with torch.no_grad():
        y_pred = model(x_test)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_test.flatten().numpy(), y_test.flatten().numpy(), label="Noisy sine wave"
    )  # Original noisy sine wave
    plt.plot(
        x_test.flatten().numpy(),
        y_pred.flatten().numpy(),
        label="Predicted sine wave",
        color="r",
    )  # Model prediction
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Original vs Predicted Sine Wave")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save pytorch model weights
    torch.save(model.state_dict(), "sin_model.pth")

    # Export onnx model
    # esp-dl currently only supports batch size=1
    dummy_input = torch.randn([1, 1], dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        "sin_model.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )
    onnx_model = onnx.load_model("sin_model.onnx")
    onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, "sin_model.onnx")
