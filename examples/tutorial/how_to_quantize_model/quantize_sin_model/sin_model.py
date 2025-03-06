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
        self.fc1 = nn.Linear(1, 64)  # 输入层，1 个特征（x）
        self.fc2 = nn.Linear(64, 64)  # 隐藏层
        self.fc3 = nn.Linear(64, 1)  # 输出层，1 个特征（y）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_data(num_samples=1000, noise=0.05):
    x = torch.linspace(0, 2 * np.pi, num_samples)
    y = torch.sin(x) + noise * torch.randn(num_samples)  # 加噪声的正弦函数
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
            # 前向传播
            y_pred = model(batch_x)

            # 计算损失
            loss = criterion(y_pred, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印训练进度
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 切换到评估模式
    model.eval()
    x_test, y_test = generate_data()
    with torch.no_grad():
        y_pred = model(x_test)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_test.flatten().numpy(), y_test.flatten().numpy(), label="Noisy sine wave"
    )  # 原始带噪声正弦波
    plt.plot(
        x_test.flatten().numpy(),
        y_pred.flatten().numpy(),
        label="Predicted sine wave",
        color="r",
    )  # 模型预测结果
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Original vs Predicted Sine Wave")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存pytorch模型权重
    torch.save(model.state_dict(), "sin_model.pth")

    # 导出onnx模型
    # esp-dl当前只支持 batch size=1
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
