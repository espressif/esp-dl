# -*- coding: utf-8 -*-

import argparse
import os
import sys

import toml
import torch
import torch.nn as nn


class CONV2D_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()

        op_list = [
            nn.Conv2d(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                padding=config["padding"],
                dilation=config["dilation"],
                groups=config["groups"],
                bias=config["bias"],
            )
        ]
        if config["activation_func"] == "ReLU":
            op_list.append(nn.ReLU())
        self.ops = nn.Sequential(*op_list)
        self.config = config

    def forward(self, inputs):
        output = self.ops(inputs)
        return output


class LINEAR_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()

        op_list = [
            nn.Linear(
                in_features=config["in_features"],
                out_features=config["out_features"],
                bias=config["bias"],
            )
        ]
        if config["activation_func"] == "ReLU":
            op_list.append(nn.ReLU())
        self.ops = nn.Sequential(*op_list)
        self.config = config

    def forward(self, inputs):
        output = self.ops(inputs)
        return output


class ADD2D_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["activation_func"] == "ReLU":
            self.act = nn.ReLU()

    def forward(self, input1, input2):
        output = input1 + input2
        if hasattr(self, "act"):
            output = self.act(output)
        return output


class MUL2D_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["activation_func"] == "ReLU":
            self.act = nn.ReLU()

    def forward(self, input1, input2):
        output = input1 * input2
        if hasattr(self, "act"):
            output = self.act(output)
        return output


class GLOBAL_AVERAGE_POOLING_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        return self.global_avg_pool(input)


class AVERAGE_POOLING_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.avg_pool = nn.AvgPool2d(
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            padding=config["padding"],
        )

    def forward(self, input):
        return self.avg_pool(input)


class RESIZE2D_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["conv"]:
            op_list = [
                nn.Conv2d(
                    in_channels=config["in_features"],
                    out_channels=config["out_features"],
                    kernel_size=[1, 1],
                    stride=[1, 1],
                    padding=[0, 0],
                    dilation=[1, 1],
                    groups=1,
                    bias=True,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
            ]
        else:
            op_list = [nn.Upsample(scale_factor=2, mode="nearest")]
        self.ops = nn.Sequential(*op_list)

    def forward(self, input):
        return self.ops(input)


class SIGMOID_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.sigmoid(input)


class CONCAT_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input1, input2):
        return torch.cat([input1, input2], dim=self.config["axis"])


class CLIP_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, inputs):
        output = torch.clip(inputs, min=self.config["min"], max=self.config["max"])
        return output


class FLATTEN_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.flatten = nn.Flatten(config["start_dim"], config["end_dim"])
        self.config = config

    def forward(self, inputs):
        output = self.flatten(inputs)
        return output


class RESHAPE_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, inputs):
        output = torch.reshape(inputs, self.config["shape"])
        return output


class TRANSPOSE_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input):
        return torch.permute(input, dims=self.config["perm"])


if __name__ == "__main__":
    print(f"Test {os.path.basename(sys.argv[0])} Module Start...")

    parser = argparse.ArgumentParser(description="Module Test")
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config (*.toml)."
    )
    args = parser.parse_args()

    # get config
    config = toml.load(args.config)

    # get model
    conv2d_cfg = config["ops_test"]["conv2d"]["cfg"][0]
    add2d_cfg = config["ops_test"]["add2d"]["cfg"][0]
    add2d_relu_cfg = config["ops_test"]["add2d"]["cfg"][1]
    mul2d_cfg = config["ops_test"]["mul2d"]["cfg"][0]
    mul2d_relu_cfg = config["ops_test"]["mul2d"]["cfg"][1]
    global_average_pooling_cfg = config["ops_test"]["global_average_pooling"]["cfg"][0]
    average_pooling_cfg = config["ops_test"]["average_pooling"]["cfg"][0]
    resize2d_cfg = config["ops_test"]["resize2d"]["cfg"][0]
    conv2d = CONV2D_TEST(conv2d_cfg)
    add2d = ADD2D_TEST(add2d_cfg)
    add2d_relu = ADD2D_TEST(add2d_relu_cfg)
    mul2d = MUL2D_TEST(mul2d_cfg)
    mul2d_relu = ADD2D_TEST(mul2d_relu_cfg)
    global_average_pooling = GLOBAL_AVERAGE_POOLING_TEST(global_average_pooling_cfg)
    average_pooling = AVERAGE_POOLING_TEST(average_pooling_cfg)
    resize2d = RESIZE2D_TEST(resize2d_cfg)

    # get inputs
    conv2d_inputs = torch.randn(conv2d_cfg["input_shape"])
    add2d_inputs = [
        torch.randn(add2d_cfg["input_shape"][0]),
        torch.randn(add2d_cfg["input_shape"][1]),
    ]
    mul2d_inputs = [
        torch.randn(mul2d_cfg["input_shape"][0]),
        torch.randn(mul2d_cfg["input_shape"][1]),
    ]
    global_average_pooling_inputs = torch.randn(
        global_average_pooling_cfg["input_shape"]
    )
    average_pooling_inputs = torch.randn(average_pooling_cfg["input_shape"])
    resize2d_inputs = torch.randn(resize2d_cfg["input_shape"])
    # print network
    # summary(conv2d, input_data=conv2d_inputs, col_names=("input_size", "output_size", "num_params"), device=torch.device('cpu'))
    # forward
    conv2d_outputs = conv2d(conv2d_inputs)
    add2d_outputs = add2d(*add2d_inputs)
    add2d_relu_outputs = add2d_relu(*add2d_inputs)
    mul2d_outputs = mul2d(*mul2d_inputs)
    mul2d_relu_outputs = mul2d_relu(*mul2d_inputs)
    global_average_pooling_outputs = global_average_pooling(
        global_average_pooling_inputs
    )
    average_pooling_outputs = average_pooling(average_pooling_inputs)
    resize2d_outputs = resize2d(resize2d_inputs)

    print(f"Test {os.path.basename(sys.argv[0])} Module End...")
    pass