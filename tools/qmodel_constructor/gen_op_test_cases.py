# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import sys
from typing import (
    Dict,
    Iterable,
    List,
    Union,
)

import numpy as np
import onnx
import ppq.lib as PFL
import toml
import torch
import torch.nn as nn
from onnxsim import simplify
from ppq import graphwise_error_analyse
from ppq.quantization.analyse.layerwise import layerwise_error_analyse
from ppq.quantization.analyse.graphwise import statistical_analyse
from ppq.api import load_onnx_graph
from ppq.core import TargetPlatform, QuantizationVisibility
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import BaseGraph
from ppq.IR.search import SearchableGraph
from ppq.quantization.optim import *
from ppq.quantization.quantizer import EspdlInt16Quantizer, EspdlQuantizer
from torch.utils.data import DataLoader
from module.interface import quantize_model_wrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fbs_construct.helper as helper


def prepare_device(index=0, keep_reproducibility=False):
    """
    Choose to use CPU or GPU depend on the value of "n_gpu".

    Args:
        n_gpu(int): the number of GPUs used in the experiment. if n_gpu == 0, use CPU; if n_gpu >= 1, use GPU.
        keep_reproducibility (bool): if we need to consider the repeatability of experiment, set keep_reproducibility to True.

    See Also
        Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
        # possibly at the cost of reduced performance
        if keep_reproducibility:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.benchmark = False  # ensures that CUDA selects the same convolution algorithm each time
            # torch.set_deterministic(True)  # configures PyTorch only to use deterministic implementation
            torch.backends.cudnn.deterministic = True
        else:
            # causes cuDNN to benchmark multiple convolution algorithms and select the fastest
            torch.backends.cudnn.benchmark = True

        device = torch.device("cuda", index)

    return device


def generate_test_value(
    graph: BaseGraph,
    inputs: Union[dict, list, torch.Tensor],
    executor: BaseGraphExecutor,
    output_names: List[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    test_inputs_value = {}
    test_outputs_value = {}

    outputs = executor.forward(inputs=inputs, output_names=output_names)
    # get test_inputs_value
    if isinstance(inputs, dict):
        for name, value in inputs.items():
            if name in graph.inputs:
                test_inputs_value[name] = value.clone().detach().cpu()
            else:
                print(f"Can not find input {name} in your graph inputs, please check.")
    else:
        inputs_tmp = executor.prepare_input(inputs=inputs)
        test_inputs_value = {
            name: value.clone().detach().cpu() for name, value in inputs_tmp.items()
        }

    # get test_outputs_value
    if output_names is None:
        outputs_dictionary = graph.outputs
        test_outputs_value = {
            key: outputs[idx].clone().detach().cpu()
            for idx, key in enumerate(outputs_dictionary)
        }
    else:
        test_outputs_value = {
            output_name: output.clone().detach().cpu()
            for output_name, output in zip(output_names, outputs)
        }

    return {"inputs": test_inputs_value, "outputs": test_outputs_value}


class BaseInferencer:
    def __init__(self, meta_cfg, model, model_cfg=None, export_path=None):
        self.meta_cfg = meta_cfg
        self.model_cfg = model_cfg if model_cfg is not None else model.config
        # get model
        self.model = model
        if export_path:
            self.export_path = export_path
        else:
            self.export_path = os.getcwd()
        # config device
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device_str: {self.device_str}")
        # self.device = prepare_device()

        self.input_shape = self.model_cfg["input_shape"]
        if not isinstance(self.input_shape[0], list):
            self.input_shape = [self.input_shape]
            self.multi_input = False
        else:
            self.multi_input = True

        # get calibration dataset.
        calibration_dataset = self.load_calibration_dataset()
        self.cali_iter = DataLoader(
            dataset=calibration_dataset,
            batch_size=self.meta_cfg["batch_size"],
            shuffle=False,
        )
        self.calib_steps = self.meta_cfg["calib_steps"]

        # get fbs config.
        self.quant_bits = self.meta_cfg["quant_bits"]
        self.model_version = self.meta_cfg["model_version"]
        self.target = self.meta_cfg["target"]
        self.input_dtype = torch.float if self.quant_bits == 8 else torch.float64
        self.platform = (
            TargetPlatform.ESPDL_INT8
            if self.quant_bits == 8
            else TargetPlatform.ESPDL_INT16
        )

    def __call__(self):
        # get the export files path
        name_prefix = self.model_cfg["export_name_prefix"]
        os.makedirs(self.export_path, exist_ok=True)
        export_onnx_path = os.path.join(self.export_path, name_prefix + ".espdl")
        export_config_path = os.path.join(self.export_path, name_prefix + ".json")

        orig_onnx_path = None
        if isinstance(self.model, nn.Module):
            orig_onnx_path = name_prefix + "_orig.onnx"
            orig_onnx_path = os.path.join(self.export_path, orig_onnx_path)
            # to device
            model = self.model.eval()
            model = model.to(self.device_str)
            if self.quant_bits == 16:
                model = model.double()
            torch.onnx.export(
                model=model,
                args=tuple(
                    [
                        torch.zeros(
                            size=[1] + input_shape[1:],
                            device=self.device_str,
                            dtype=self.input_dtype,
                        )
                        for input_shape in self.input_shape
                    ]
                ),
                f=orig_onnx_path,
                opset_version=11,
                do_constant_folding=True,
            )

            model = onnx.load(orig_onnx_path)
            model_sim, check = simplify(model)
            onnx.save(model_sim, orig_onnx_path)
            assert check, "Simplified ONNX model could not be validated"
        else:
            orig_onnx_path = self.model

        print("start PTQ")
        collate_fn = (
            (lambda x: x.type(self.input_dtype).to(self.device_str))
            if not self.multi_input
            else (lambda x: [xx.type(self.input_dtype).to(self.device_str) for xx in x])
        )

        ppq_graph, executor = quantize_model_wrapper(
            onnx_import_file=orig_onnx_path,
            calib_dataloader=self.cali_iter,
            calib_steps=self.calib_steps,
            input_shape=self.input_shape,
            platform=self.platform,
            input_dtype=self.input_dtype,
            dispatching_override=None,
            dispatching_method="allin",
            collate_fn=collate_fn,
            device=self.device_str,
            verbose=1,
        )

        print(
            "==================== start graphwise_error_analyse: ===================="
        )
        reports = graphwise_error_analyse(
            graph=ppq_graph,
            running_device=self.device_str,
            collate_fn=collate_fn,
            dataloader=self.cali_iter,
        )

        print(
            "==================== start layerwise_error_analyse: ===================="
        )
        reports = layerwise_error_analyse(
            graph=ppq_graph,
            running_device=self.device_str,
            collate_fn=collate_fn,
            dataloader=self.cali_iter,
        )

        # report = statistical_analyse(
        #     graph=ppq_graph, running_device=self.device_str,
        #     collate_fn=collate_fn, dataloader=self.cali_iter)

        # from pandas import DataFrame

        # report = DataFrame(report)
        # report.to_csv('1.csv')

        # The invocation of generate_test_value must be placed before the PFL.Exporter(platform=TargetPlatform.ESPDL_INT8).export() function
        # because the export() function will modifies the graph.
        print("Start to generate test value.")
        valuesForTest = generate_test_value(
            graph=ppq_graph,
            inputs=[
                torch.randn(
                    size=[1] + input_shape[1:],
                    device=self.device_str,
                    dtype=self.input_dtype,
                )
                for input_shape in self.input_shape
            ],
            executor=executor,
        )

        print("Start to export onnx.")
        executor.tracing_operation_meta(
            inputs=[
                torch.zeros(
                    size=[1] + input_shape[1:],
                    device=self.device_str,
                    dtype=self.input_dtype,
                )
                for input_shape in self.input_shape
            ]
        )
        PFL.Exporter(platform=self.platform).export(
            file_path=export_onnx_path,
            graph=ppq_graph,
            config_path=export_config_path,
            modelVersion=self.model_version,
            valuesForTest=valuesForTest,
        )

    def load_calibration_dataset(self) -> Iterable:
        if not self.multi_input:
            return [
                torch.randn(size=self.input_shape[0][1:])
                for _ in range(self.meta_cfg["batch_size"])
            ]
        else:
            return [
                [torch.randn(size=i[1:]) for i in self.input_shape]
                for _ in range(self.meta_cfg["batch_size"])
            ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config (*.toml)."
    )
    parser.add_argument(
        "-o", "--output-path", type=str, default=None, help="Output Path."
    )
    args = parser.parse_args()

    # load config
    config = toml.load(args.config)

    # generate test cases
    pkg = importlib.import_module(config["ops_test"]["class_package"])
    op_set = [
        "conv2d",
        "add2d",
        "mul2d",
        "sigmoid",
        "average_pooling",
        "global_average_pooling",
        "linear",
        "concat",
        "resize2d",
        "clip",
        "flatten",
        "reshape",
        "transpose",
    ]

    for op_type in op_set:
        op_configs = config["ops_test"][op_type]["cfg"]
        op_class_name = config["ops_test"][op_type]["class_name"]
        if args.output_path:
            export_path = os.path.join(args.output_path, op_type)
        for cfg in op_configs:
            op = getattr(pkg, op_class_name)(cfg)
            BaseInferencer(config["meta"], op, export_path=export_path)()
