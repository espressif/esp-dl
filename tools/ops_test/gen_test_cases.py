# -*- coding: utf-8 -*-

import argparse
import importlib
import math
import os
import re
import sys
from typing import (
    Iterable,
)

import onnx
import toml
import torch
from ppq import QuantizationSettingFactory
from ppq.api import espdl_quantize_onnx, espdl_quantize_torch, get_target_platform
from ppq.quantization.optim import *
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
torch.manual_seed(42)


class BaseInferencer:
    def __init__(
        self,
        model,
        export_path,
        model_cfg,
        target="esp32p4",
        num_of_bits=8,
        model_version="1.0",
        meta_cfg=None,
    ):
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        self.export_path = export_path

        self.model = model
        self.model_cfg = model_cfg
        self.onnx_file = None

        if isinstance(model, onnx.ModelProto):
            self.model_cfg = model_cfg
            self.onnx_file = os.path.join(
                export_path, self.model_cfg["export_name_prefix"] + ".onnx"
            )
            onnx.save(self.model, self.onnx_file)

        # config device
        self.device_str = "cpu"
        self.calib_steps = meta_cfg["calib_steps"] if meta_cfg is not None else 32
        self.batch_size = meta_cfg["batch_size"] if meta_cfg is not None else 1

        self.input_shape = self.model_cfg["input_shape"]
        if not isinstance(self.input_shape[0], list):
            self.input_shape = [self.input_shape]
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
        self.num_of_bits = num_of_bits
        self.model_version = model_version
        self.target = target
        self.input_dtype = torch.float

    def __call__(self):
        # get the export files path
        name_prefix = (
            self.model_cfg["export_name_prefix"] + "_s" + str(self.num_of_bits)
        )
        espdl_export_file = os.path.join(self.export_path, name_prefix + ".espdl")

        collate_fn = (
            (lambda x: x.type(self.input_dtype).to(self.device_str))
            if not self.multi_input
            else (lambda x: [xx.type(self.input_dtype).to(self.device_str) for xx in x])
        )

        print("start PTQ")
        # create a setting for quantizing your network with ESPDL.
        quant_setting = QuantizationSettingFactory.espdl_setting()
        quant_setting.dispatcher = "allin"
        if self.model_cfg.get("dispatch_table", {}):
            for op_name, num_bit in self.model_cfg.get("dispatch_table", {}).items():
                quant_setting.dispatching_table.append(
                    op_name, get_target_platform(self.target, num_bit)
                )

        if self.onnx_file is None:
            espdl_quantize_torch(
                model=self.model,
                espdl_export_file=espdl_export_file,
                calib_dataloader=self.calib_dataloader,
                calib_steps=self.calib_steps,
                input_shape=self.input_shape,
                target=self.target,
                num_of_bits=self.num_of_bits,
                collate_fn=collate_fn,
                setting=quant_setting,
                device=self.device_str,
                error_report=False,
                skip_export=False,
                export_test_values=True,
                export_config=True,
                verbose=1,
                int16_lut_step=1,
                metadata_props={"target": self.target},
            )

        else:
            espdl_quantize_onnx(
                onnx_import_file=self.onnx_file,
                espdl_export_file=espdl_export_file,
                calib_dataloader=self.calib_dataloader,
                calib_steps=self.calib_steps,
                input_shape=self.input_shape,
                target=self.target,
                num_of_bits=self.num_of_bits,
                collate_fn=collate_fn,
                setting=quant_setting,
                device=self.device_str,
                error_report=False,
                skip_export=False,
                export_test_values=True,
                export_config=True,
                verbose=1,
                int16_lut_step=1,
                metadata_props={"target": self.target},
            )

    def load_calibration_dataset(self) -> Iterable:
        if not self.multi_input:
            return [
                torch.randn(size=self.input_shape[0][1:])
                for _ in range(self.batch_size)
            ]
        else:
            return [
                [torch.randn(size=i[1:]) for i in self.input_shape]
                for _ in range(self.batch_size)
            ]


def get_op_set(op_test_config, ops):
    op_set = []
    for op_type in op_test_config:
        op_set.append(op_type)

    if "opset_" in ops[0]:
        set_name = ops[0]
        pattern = r"opset_(\d+)_(\d+)"
        match = re.match(pattern, set_name)
        if match:
            number1 = int(match.group(1))
            number2 = int(match.group(2))
            left_num = len(op_set)
            step = int(math.ceil(left_num / number1))
            for i in range(number2):
                left_num = left_num - step
                step = int(math.ceil(left_num / (number1 - i - 1)))
            start = len(op_set) - left_num
            end = start + step
            if end > len(op_set):
                end = len(op_set)
            op_set[:] = op_set[start:end]
            return op_set
    elif ops[0] == "ALL" or ops[0] == "all":
        return op_set
    else:
        return ops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config (*.toml)."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        type=str,
        default=None,
        help="Output Path.",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="esp32p4",
        help="esp32p4, esp32s3 or c, (defaults: esp32p4).",
    )
    parser.add_argument(
        "-b",
        "--bits",
        type=int,
        default=8,
        help="the number of bits, support 8 or 16, (defaults: 8).",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="v1.0",
        help="the version of the test case, (defaults: v1.0)",
    )
    parser.add_argument("--ops", nargs="+", type=str, help="An array of ops")
    parser.add_argument(
        "--models", type=str, default=None, help="Specify the model to test."
    )
    args = parser.parse_args()

    # load config
    config = toml.load(args.config)

    if not args.models:
        op_test_config = config["ops_test"]

        # generate test cases
        op_set = get_op_set(op_test_config, args.ops)

        for op_type in op_set:
            pkg = importlib.import_module(op_test_config[op_type]["package"])
            op_configs = op_test_config[op_type]["cfg"]
            op_test_func = op_test_config[op_type]["test_func"]
            quant_bits = op_test_config[op_type].get("quant_bits", [])

            if (args.bits == 8 and "int8" in quant_bits) or (
                args.bits == 16 and "int16" in quant_bits
            ):
                export_path = os.path.join(args.output_path, op_type)
                for cfg in op_configs:
                    print(
                        "Op Test Function: ",
                        op_test_func,
                        "Configs: ",
                        cfg,
                        "Package: ",
                        pkg.__name__,
                        "Output Path: ",
                        export_path,
                    )
                    op = getattr(pkg, op_test_func)(cfg)
                    BaseInferencer(
                        op,
                        export_path=export_path,
                        model_cfg=cfg,
                        target=args.target,
                        num_of_bits=args.bits,
                        model_version=args.version,
                        meta_cfg=config["meta"],
                    )()
            else:
                print(
                    f"Skip op: {op_type}, do not support quantization with {args.bits} bits."
                )
    else:
        model_config = config["models_test"][args.models]
        if args.bits == 8 or args.bits == 16:
            model = onnx.load(model_config["onnx_model_path"])
            BaseInferencer(
                model,
                export_path=args.output_path,
                model_cfg=model_config,
                target=args.target,
                num_of_bits=args.bits,
                model_version=args.version,
                meta_cfg=config["meta"],
            )()
        else:
            print(f"Do not support quantization with {args.bits} bits.")
