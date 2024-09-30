import os
from typing import Any, Callable, List, Dict, Tuple

import torch
from torch.utils.data import DataLoader

import ppq.lib as PFL
from ppq.api import load_onnx_graph, dispatch_graph
from ppq.core import TargetPlatform, empty_ppq_cache, QuantizationVisibility
from ppq.executor import TorchExecutor
from ppq.IR import BaseGraph
from ppq.quantization.optim import *


@empty_ppq_cache
def quantize_model_wrapper(
    onnx_import_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[Any],
    platform: TargetPlatform,
    input_dtype: torch.dtype = torch.float,
    dispatching_override: Dict[str, TargetPlatform] = None,
    dispatching_method: str = "conservative",
    collate_fn: Callable = None,
    device: str = "cuda",
    verbose: int = 0,
) -> Tuple[BaseGraph, TorchExecutor]:
    """量化一个 onnx 原生的模型 输入一个 onnx 模型的文件路径 返回一个量化后的 PPQ.IR.BaseGraph quantize
    onnx model, input onnx model and return quantized ppq IR graph.

    Args:
        onnx_import_file (str): 被量化的 onnx 模型文件路径 onnx model location

        calib_dataloader (DataLoader): 校准数据集 calibration data loader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.

        dispatching_override: override dispatching result.

        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息
                   The quantized IR, containing all information needed for backend execution
    """
    if calib_dataloader is None or calib_steps is None:
        raise TypeError(
            "Quantization needs a valid calib_dataloader and calib_steps setting."
        )

    ppq_graph = load_onnx_graph(onnx_import_file=onnx_import_file)
    quantizer = PFL.Quantizer(
        platform=platform, graph=ppq_graph
    )  # 初始化一个 quantizer 没有很大代价...
    dispatching_table = PFL.Dispatcher(
        graph=ppq_graph, method=dispatching_method
    ).dispatch(quantizer.quant_operation_types)
    # override dispatching result
    if dispatching_override is not None:
        for opname, platform in dispatching_override.items():
            if opname not in ppq_graph.operations:
                continue
            assert isinstance(platform, int) or isinstance(platform, TargetPlatform), (
                f"Your dispatching_override table contains a invalid setting of operation {opname}, "
                "All platform setting given in dispatching_override table is expected given as int or TargetPlatform, "
                f"however {type(platform)} was given."
            )
            dispatching_table[opname] = TargetPlatform(platform)

    for opname, platform in dispatching_table.items():
        if platform == TargetPlatform.UNSPECIFIED:
            dispatching_table[opname] = TargetPlatform(quantizer.target_platform)

    if not isinstance(input_shape[0], list):
        input_shape = [input_shape]
    dummy_inputs = [
        torch.zeros(size=shape, device=device, dtype=input_dtype)
        for shape in input_shape
    ]

    # 为算子初始化量化信息
    for op in ppq_graph.operations.values():
        quantizer.quantize_operation(
            op_name=op.name, platform=dispatching_table[op.name]
        )
    executor = TorchExecutor(graph=ppq_graph, device=device)
    executor.tracing_operation_meta(inputs=dummy_inputs)

    # with ENABLE_CUDA_KERNEL():
    # ------------------------------------------------------------
    # 创建优化管线，由于后续还要继续训练我们的模型，我们不能在此处调用
    # ParameterBakingPass()，一旦模型权重完成烘焙，则它们不能被进一步调整
    # ------------------------------------------------------------
    pipeline = PFL.Pipeline(
        [
            QuantizeSimplifyPass(),
            QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
            ParameterQuantizePass(),
            RuntimeCalibrationPass(method="kl"),
            PassiveParameterQuantizePass(
                clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE
            ),
            QuantAlignmentPass(elementwise_alignment="Align to Output"),
            # LearnedStepSizePass(steps=500, block_size=5)
        ]
    )
    print(
        f"Calibrate audio number: {len(calib_dataloader.dataset)}, len(Calibrate iter): {len(calib_dataloader)}"
    )
    pipeline.optimize(
        calib_steps=calib_steps,
        collate_fn=collate_fn,
        graph=ppq_graph,
        dataloader=calib_dataloader,
        executor=executor,
    )
    if verbose:
        print(quantizer.report(), end="")
        print("Network Quantization Finished.")

    return ppq_graph, executor
