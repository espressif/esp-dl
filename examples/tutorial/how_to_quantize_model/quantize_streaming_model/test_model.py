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

    def streaming_forward(self, input: Tensor, cache: Tensor) -> Tuple[Tensor, Tensor]:
        # input [B, C, T] -> output [B, C, T]
        input = self.prev_conv(input)
        out1 = self.layer[0](input)
        # 1D Depthwise Conv
        assert cache.shape == (out1.size(0), out1.size(1), self.padding), (
            cache.shape,
            (out1.size(0), out1.size(1), self.padding),
        )
        out1 = torch.cat([cache, out1], dim=2)
        # Update cache
        cache = out1[:, :, -self.padding :]

        out1 = self.layer[1](out1)
        out2 = self.layer[2](out1)
        output = self.layer[3](out1 * out2) + input
        return output, cache


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

    def streaming_forward(self, input: Tensor) -> Tensor:
        return self.forward(input)


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

    def streaming_forward(self, input: Tensor, cache: Tensor) -> Tuple[Tensor, Tensor]:
        # input [B, C, T] -> output [B, C, T]
        input = self.prev_conv(input)
        out1 = self.layer[0](input)
        # 1D Depthwise Conv
        assert cache.shape == (out1.size(0), out1.size(1), self.padding), (
            cache.shape,
            (out1.size(0), out1.size(1), self.padding),
        )
        out1 = torch.cat([cache, out1], dim=2)
        # Update cache
        cache = out1[:, :, -self.padding :]

        out1 = self.layer[1](out1)
        output = self.layer[2](out1) + input
        return output, cache
