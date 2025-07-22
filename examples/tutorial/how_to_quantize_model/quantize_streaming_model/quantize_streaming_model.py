# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
from typing import Iterable, List, Tuple, Union, Any, Optional

import onnx
import torch
import toml
import torch.nn.functional as F
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import (
    espdl_quantize_onnx,
    espdl_quantize_torch,
    get_target_platform,
    generate_test_value,
)
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor, nn
from test_model import TestModel_0, TestModel_1, TestModel_2

torch.manual_seed(42)


class ModelStreamingWrapper(nn.Module):
    """A wrapper for model"""

    def __init__(self, model: nn.Module):
        """
        Args:
          model: A pytorch model.
        """
        super().__init__()
        self.model = model

    def forward(
        self, input: Tensor, cache: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Please see the help information of TestModel_0.streaming_forward"""

        if cache is not None:
            output, new_cache = self.model.streaming_forward(input, cache)
            return output, new_cache
        else:
            output = self.model.streaming_forward(input)
            return output


def generate_dataset(input_shape: List[Any], batch_size: int):
    if not isinstance(input_shape[0], list):
        return [torch.randn(size=input_shape[1:]) for _ in range(batch_size)]
    else:
        return [
            [torch.randn(size=i[1:]) for i in input_shape] for _ in range(batch_size)
        ]


class QuantModel:
    def __init__(self, model, args, meta_config, model_config, dataset):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        self.export_path = args.output_path

        self.model = model
        self.meta_config = meta_config
        self.model_config = model_config
        self.dataset = dataset
        self.onnx_file = None

        if isinstance(self.model, onnx.ModelProto):
            self.onnx_file = os.path.join(
                self.export_path, self.model_config["export_name_prefix"] + ".onnx"
            )
            onnx.save(self.model, self.onnx_file)

        # config device
        self.device_str = args.device
        self.calib_steps = (
            self.meta_config["calib_steps"] if self.meta_config is not None else 32
        )
        self.batch_size = (
            self.meta_config["batch_size"] if self.meta_config is not None else 1
        )
        self.streaming = args.streaming
        self.streaming_window_size = 0
        self.streaming_cache_size = 0
        if self.streaming:
            if self.model_config.get("streaming_input_shape", []):
                self.streaming_window_size = self.model_config["streaming_input_shape"][
                    2
                ]
            if self.model_config.get("streaming_cache_shape", []):
                self.streaming_cache_size = self.model_config["streaming_cache_shape"][
                    2
                ]

        self.offline_input_shape = self.model_config["offline_input_shape"]
        if not isinstance(self.offline_input_shape[0], list):
            self.multi_input = False
        else:
            self.multi_input = True

        # get calibration dataset.
        calibration_dataset = self.load_calibration_dataset()
        self.calib_dataloader = DataLoader(
            dataset=calibration_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # get quantization config.
        self.num_of_bits = args.bits
        self.model_version = args.version
        self.target = args.target
        self.input_dtype = torch.float

    def load_calibration_dataset(self) -> Iterable:
        if self.streaming:
            data_total = []
            if self.model_config.get("streaming_cache_shape", []):
                caches = []
                caches.append(
                    torch.zeros(size=self.model_config["streaming_cache_shape"][1:])
                )
                if not self.multi_input:
                    for data in self.dataset:
                        # Ensure that the size of the W dimension is divisible by self.streaming_window_size.
                        # Split the input and collect cache data.
                        split_tensors = torch.split(
                            data[0] if isinstance(data, tuple) else data,
                            self.streaming_window_size,
                            dim=1,
                        )
                        for index, split_tensor in enumerate(split_tensors):
                            _, cache = self.model(
                                split_tensor.unsqueeze(0), caches[index].unsqueeze(0)
                            )
                            caches.append(cache.squeeze(0))

                        data_total += [
                            list(pair) for pair in zip(list(split_tensors), caches)
                        ]
                else:
                    # It depends on which inputs of the model require streaming, so multiple inputs have not been added.
                    pass

                return data_total
            else:
                if not self.multi_input:
                    for data in self.dataset:
                        # Ensure that the size of the W dimension is divisible by self.streaming_window_size.
                        # Split the input and collect cache data.
                        split_tensors = torch.split(
                            data[0] if isinstance(data, tuple) else data,
                            self.streaming_window_size,
                            dim=1,
                        )
                        data_total += list(split_tensors)
                else:
                    pass

                return data_total
        else:
            return self.dataset

    def __call__(self):
        # get the export files path
        name_prefix = (
            self.model_config["export_name_prefix"] + "_s" + str(self.num_of_bits)
        )
        if self.streaming:
            name_prefix += "_streaming"
        espdl_export_file = os.path.join(self.export_path, name_prefix + ".espdl")

        collate_fn = lambda x: (
            x.type(self.input_dtype).to(self.device_str)
            if not isinstance(x, list)
            else [xx.type(self.input_dtype).to(self.device_str) for xx in x]
        )

        print("start PTQ")
        # create a setting for quantizing your network with ESPDL.
        quant_setting = QuantizationSettingFactory.espdl_setting()
        if self.model_config.get("dispatch_table", {}):
            for op_name, num_bit in self.model_config.get("dispatch_table", {}).items():
                quant_setting.dispatching_table.append(
                    op_name, get_target_platform(self.target, num_bit)
                )
        # quant_setting.dispatcher = "allin"

        # Batch size must be 1.
        export_input_shape = self.offline_input_shape
        if self.streaming:
            if self.model_config.get("streaming_cache_shape", []):
                export_input_shape = [
                    [1] + self.model_config["streaming_input_shape"][1:],
                    [1] + self.model_config["streaming_cache_shape"][1:],
                ]
            else:
                export_input_shape = [1] + self.model_config["streaming_input_shape"][
                    1:
                ]

        ppq_graph = None
        if self.onnx_file is None:
            ppq_graph = espdl_quantize_torch(
                model=self.model,
                espdl_export_file=espdl_export_file,
                calib_dataloader=self.calib_dataloader,
                calib_steps=self.calib_steps,
                input_shape=export_input_shape,
                target=self.target,
                num_of_bits=self.num_of_bits,
                collate_fn=collate_fn,
                setting=quant_setting,
                device=self.device_str,
                error_report=False,
                skip_export=False,
                export_test_values=False,
                export_config=True,
                verbose=1,
                int16_lut_step=1,
            )

        else:
            ppq_graph = espdl_quantize_onnx(
                onnx_import_file=self.onnx_file,
                espdl_export_file=espdl_export_file,
                calib_dataloader=self.calib_dataloader,
                calib_steps=self.calib_steps,
                input_shape=export_input_shape,
                target=self.target,
                num_of_bits=self.num_of_bits,
                collate_fn=collate_fn,
                setting=quant_setting,
                device=self.device_str,
                error_report=False,
                skip_export=False,
                export_test_values=False,
                export_config=True,
                verbose=1,
                int16_lut_step=1,
            )
        return ppq_graph


def generate_c_array(
    input, output, target, bits, input_channels, streaming_window_size, streaming_number
):
    test_data = """
#pragma once

#include <stdint.h>

#define TEST_INPUT_CHANNELS {input_channels}
#define STREAMING_WINDOW_SIZE  {streaming_window_size}
#define STREAMING_NUMBER  {streaming_number}
#define TIME_SERIES_LENGTH  (STREAMING_NUMBER * STREAMING_WINDOW_SIZE)

// NWC layout
const {type_str} test_inputs[TIME_SERIES_LENGTH][TEST_INPUT_CHANNELS] = {{
{input_str}
}};

const {type_str} test_outputs[] = {{
{output_str}
}};
"""

    input_str = ""
    for sub_input in input:
        sub_input = sub_input.cpu()
        sub_input = sub_input.flatten()
        sub_input = (
            sub_input.type(torch.int8) if bits == 8 else sub_input.type(torch.int16)
        )
        elements = sub_input.tolist()
        element_strs = [str(int(x)) for x in elements]
        array_str = ", ".join(element_strs)
        input_str += f"{{\n\t{array_str}\n}},\n"

    output = output.cpu()
    output = output.type(torch.int8) if bits == 8 else output.type(torch.int16)
    output_elements = output.tolist()
    output_element_strs = [str(int(x)) for x in output_elements]
    output_array_str = ", ".join(output_element_strs)
    output_str = f"\t{output_array_str}"

    test_data_context_base = dict(
        input_channels=input_channels,
        streaming_window_size=streaming_window_size,
        streaming_number=streaming_number,
        type_str="int8_t" if bits == 8 else "int16_t",
        input_str=input_str,
        output_str=output_str,
    )
    test_data = test_data.format(**test_data_context_base)

    # 生成hpp数据文件
    with open(f"test_data_{target}.hpp", "w") as f:
        f.write(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config (*.toml)."
    )
    parser.add_argument(
        "-o", "--output-path", type=str, default="./", help="Output Path."
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="esp32p4",
        help="c, esp32s3 or esp32p4, (defaults: esp32p4).",
    )
    parser.add_argument(
        "-b",
        "--bits",
        type=int,
        default=8,
        help="the number of bits, support 8 or 16, (defaults: 8).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="execution device, (defaults: cpu).",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="1.0",
        help="the version of the test case, (defaults: 1.0)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Whether to export the model in a streaming manner.",
    )
    parser.add_argument(
        "--export_test_values",
        action="store_true",
        help="Whether to export the test inputs and outputs.",
    )
    args = parser.parse_args()

    config = toml.load(args.config)
    meta_config = config["meta"]
    model_0_config = config["model_0_config"]
    model_1_config = config["model_1_config"]
    model_2_config = config["model_2_config"]

    # generate test models
    if args.bits == 8 or args.bits == 16:
        input_dataset = generate_dataset(
            model_0_config["offline_input_shape"], meta_config["batch_size"]
        )

        model_0 = TestModel_0(
            in_channels=model_0_config["in_channels"],
            expand_channels=model_0_config["expand_channels"],
            out_channels=model_0_config["out_channels"],
            kernel_size=model_0_config["kernel_size"],
            stride=model_0_config["stride"],
            padding=model_0_config["padding"],
            dilation=model_0_config["dilation"],
        )

        model_1 = TestModel_1(
            in_channels=model_1_config["in_channels"],
            expand_channels=model_1_config["expand_channels"],
            out_channels=model_1_config["out_channels"],
        )

        model_2 = TestModel_2(
            in_channels=model_2_config["in_channels"],
            expand_channels=model_2_config["expand_channels"],
            out_channels=model_2_config["out_channels"],
            kernel_size=model_2_config["kernel_size"],
            stride=model_2_config["stride"],
            padding=model_2_config["padding"],
            dilation=model_2_config["dilation"],
        )

        # Collect the dataset.
        model_0_output = model_0(torch.stack(input_dataset))
        model_1_output = model_1(model_0_output)

        if args.streaming:
            model_0 = ModelStreamingWrapper(model_0)
            model_1 = ModelStreamingWrapper(model_1)
            model_2 = ModelStreamingWrapper(model_2)

        ppq_graph_0 = QuantModel(
            model_0, args, meta_config, model_0_config, input_dataset
        )()
        ppq_graph_1 = QuantModel(
            model_1, args, meta_config, model_1_config, TensorDataset(model_0_output)
        )()
        ppq_graph_2 = QuantModel(
            model_2, args, meta_config, model_2_config, TensorDataset(model_1_output)
        )()

        # It's for debug.
        if args.export_test_values:
            torch.set_printoptions(threshold=torch.inf)
            in_exponent = 5
            out_exponent = 8
            Q_MAX = 2**args.bits - 1
            Q_MIN = -(2**args.bits)
            test_input = torch.stack(
                generate_dataset(model_0_config["offline_input_shape"], 1)
            )
            values_for_test = generate_test_value(ppq_graph_0, args.device, test_input)
            values_for_test = generate_test_value(
                ppq_graph_1, args.device, list(values_for_test["outputs"].values())
            )
            values_for_test = generate_test_value(
                ppq_graph_2, args.device, list(values_for_test["outputs"].values())
            )
            test_input_q = None
            test_output_q = None
            if args.target == "esp32p4":
                test_input_q = torch.clamp(
                    (test_input * (2**in_exponent)).round(), Q_MIN, Q_MAX
                )
                test_output_q = torch.clamp(
                    (values_for_test["outputs"]["41"] * (2**out_exponent)).round(),
                    Q_MIN,
                    Q_MAX,
                )
            elif args.target == "esp32s3":
                test_input_q = torch.clamp(
                    torch.floor(test_input * (2**in_exponent) + 0.5), Q_MIN, Q_MAX
                )
                test_output_q = torch.clamp(
                    torch.floor(
                        values_for_test["outputs"]["41"] * (2**out_exponent) + 0.5
                    ),
                    Q_MIN,
                    Q_MAX,
                )
            else:
                print(f"Don't support target: {args.target}")

            # get test input value on esp-dl
            # from NCW to NWC
            test_input_q = test_input_q.permute(0, 2, 1)
            split_tensors = torch.split(test_input_q, 1, dim=1)

            # get test output value on esp-dl
            offset = 1
            for size in model_2_config["streaming_input_shape"]:
                offset *= size
            test_output_q = test_output_q.permute(0, 2, 1).flatten()[:-offset]

            generate_c_array(
                input=split_tensors,
                output=test_output_q,
                target=args.target,
                bits=args.bits,
                input_channels=model_0_config["offline_input_shape"][1],
                streaming_window_size=model_0_config["streaming_input_shape"][2],
                streaming_number=(
                    model_0_config["offline_input_shape"][2]
                    // model_0_config["streaming_input_shape"][2]
                ),
            )

    else:
        print(f"Don't support quantization with {args.bits} bits.")
