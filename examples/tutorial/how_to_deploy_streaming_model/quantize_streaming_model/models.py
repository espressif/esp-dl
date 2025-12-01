import sys
import os
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Tuple


class TestModel_0(nn.Module):
    def __init__(
        self,
        in_channels,
        expand_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.expand_channels = expand_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.prev_conv = nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1)

        self.layer = nn.ModuleList()
        self.layer.append(
            nn.Sequential(
                # conv_0
                nn.Conv1d(self.in_channels, self.expand_channels, kernel_size=1),
                nn.SiLU(),
            )
        )

        self.layer.append(
            nn.Sequential(
                # conv_1
                nn.Conv1d(
                    self.expand_channels,
                    self.expand_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=0,
                    dilation=self.dilation,
                    groups=self.expand_channels,
                ),
                nn.SiLU(),
            )
        )

        self.layer.append(
            nn.Sequential(
                # Squeeze-and-Excitation
                nn.Conv1d(
                    self.expand_channels, self.expand_channels // 4, kernel_size=1
                ),
                nn.SiLU(),
                nn.Conv1d(
                    self.expand_channels // 4, self.expand_channels, kernel_size=1
                ),
                nn.Sigmoid(),
            )
        )

        self.layer.append(
            # conv_2
            nn.Conv1d(self.expand_channels, self.out_channels, kernel_size=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        # input [B, C, T] -> output [B, C, T]
        input = self.prev_conv(input)
        out1 = self.layer[0](input)
        out1 = F.pad(out1, (self.padding, 0), "constant", 0)
        out1 = self.layer[1](out1)
        out2 = self.layer[2](out1)
        output = self.layer[3](out1 * out2) + input
        return output


class TestModel_1(nn.Module):
    def __init__(self, in_channels, expand_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.expand_channels = expand_channels
        self.out_channels = out_channels

        self.layer = nn.ModuleList()
        self.layer.append(
            nn.Sequential(
                # conv_0
                nn.Conv1d(self.in_channels, self.expand_channels, kernel_size=1),
                nn.ReLU(),
            )
        )

        self.layer.append(
            # conv_2
            nn.Conv1d(self.expand_channels, self.out_channels, kernel_size=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        # input [B, C, T] -> output [B, C, T]
        output = self.layer[0](input)
        output = self.layer[1](output)
        return output


class TestModel_2(nn.Module):
    def __init__(
        self,
        in_channels,
        expand_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.expand_channels = expand_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.prev_conv = nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1)

        self.layer = nn.ModuleList()
        self.layer.append(
            nn.Sequential(
                # conv_0
                nn.Conv1d(self.in_channels, self.expand_channels, kernel_size=1),
                nn.SiLU(),
            )
        )

        self.layer.append(
            nn.Sequential(
                # conv_1
                nn.Conv1d(
                    self.expand_channels,
                    self.expand_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=0,
                    dilation=self.dilation,
                    groups=self.expand_channels,
                ),
                nn.SiLU(),
            )
        )

        self.layer.append(
            # conv_2
            nn.Conv1d(self.expand_channels, self.out_channels, kernel_size=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        # input [B, C, T] -> output [B, C, T]
        input = self.prev_conv(input)
        out1 = self.layer[0](input)
        out1 = F.pad(out1, (self.padding, 0), "constant", 0)
        out1 = self.layer[1](out1)
        output = self.layer[2](out1) + input
        return output


class CausalDSConv1d(nn.Module):
    """因果一维卷积封装：手动左 padding"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size,
            stride=stride,
            padding=0,  # 我们自己 pad
            dilation=dilation,
            groups=in_ch,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )
        self.batchnorm = nn.BatchNorm1d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.pad, 0))
        x = self.depthwise(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return x


class CausalAvgPool1d(nn.Module):
    """因果一维平均池化：左 padding 后取平均"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.k = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.pad, 0))  # 左边补 pad
        return F.avg_pool1d(x, self.k, stride=1)


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expand_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        # 1×1 前置卷积
        self.prev_conv = nn.Conv1d(in_channels, expand_channels, 1)

        # 主干三层因果卷积
        self.layer = nn.Sequential(
            CausalDSConv1d(
                expand_channels, expand_channels, kernel_size, stride=stride, dilation=1
            ),
            nn.SiLU(),
            CausalDSConv1d(
                expand_channels,
                expand_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            nn.SiLU(),
            CausalDSConv1d(
                expand_channels, expand_channels, kernel_size, stride=stride, dilation=1
            ),
            nn.SiLU(),
        )

        # 因果 MaxPool
        self.avgpool = CausalAvgPool1d(kernel_size)

        # 输出 1×1
        self.out_conv = nn.Conv1d(expand_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.prev_conv(x)
        x1 = self.layer(x)
        x = x + x1
        x = self.avgpool(x)
        x = self.out_conv(x)
        print("output shape:", x.shape)
        return x


# ---------------- 测试 ----------------
if __name__ == "__main__":
    B, C, T = 2, 16, 128
    model = TCN(
        in_channels=16,
        expand_channels=32,
        out_channels=24,
        kernel_size=5,
        stride=1,
        dilation=2,
    )
    inp = torch.randn(B, C, T)
    out = model(inp)
    print("input:", inp.shape, "-> output:", out.shape)
