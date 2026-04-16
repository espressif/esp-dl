"""ESP-PPQ quantization wrapper.

Bug fixes vs. the original:
    - Don't overwrite the user's ONNX file with the simplified version
      (B5). Simplified result is saved next to the original as
      `<base>_simplified.onnx`.
    - `next_layers` no longer leaks out of its for loop (A3).
"""

import os
from collections import defaultdict
from typing import List

import onnx
from onnxsim import simplify

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from esp_ppq.parser import NativeExporter
from esp_ppq.quantization.analyse import layerwise_error_analyse


def simplify_onnx(onnx_path: str) -> str:
    """Simplify `onnx_path` and save the result next to the original as
    `<base>_simplified.onnx`. Returns the simplified file path.

    Skips work if the simplified file is already up-to-date. The user's
    original ONNX file is never touched.
    """
    base, _ext = os.path.splitext(onnx_path)
    out_path = f"{base}_simplified.onnx"

    if not os.path.isfile(out_path) or os.path.getmtime(out_path) < os.path.getmtime(
        onnx_path
    ):
        print(f"Simplifying ONNX {onnx_path} -> {out_path}")
        model = onnx.load(onnx_path)
        model, ok = simplify(model)
        if not ok:
            raise RuntimeError(f"onnxsim failed on {onnx_path}")
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, out_path)

    return out_path


def build_quant_setting(
    strategy, sampled_param, target, layerwise_error, runtime_config
):
    """Pure mapping: strategy + sampled_param -> ESP-PPQ QuantizationSetting."""
    num_of_bits = sampled_param["num_of_bits"]["value"]
    calib_algorithm = sampled_param["calib_algorithm"]["method"]

    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.quantize_activation_setting.calib_algorithm = calib_algorithm
    quant_setting.equalization = strategy["weight_equalization"]["value"]
    quant_setting.weight_split = strategy["horizontal_layer_split"]["value"]
    quant_setting.bias_correct = strategy["bias_correction"]["value"]
    quant_setting.tqt_optimization = strategy["tqt"]["value"]

    if strategy["mixed_precision"]["value"]:
        mp_cfg = sampled_param.get("mixed_precision")
        if mp_cfg is None:
            raise ValueError("Missing sampled mixed_precision config")
        interested_layers = select_topk_layers(layerwise_error, mp_cfg.get("topk", 1))

        # Promote both the picked layers AND their direct successors to int16.
        # (Use the simplified ONNX, NOT the user's original file.)
        simplified_onnx_path = simplify_onnx(runtime_config["onnx_path"])
        mp_layers = []
        for layer_name in interested_layers:
            mp_layers.append(layer_name)
            for nxt in get_next_node(simplified_onnx_path, layer_name):
                if nxt not in mp_layers:
                    mp_layers.append(nxt)
        for name in mp_layers:
            quant_setting.dispatching_table.append(
                name, get_target_platform(target, 16)
            )
        print(f"Mixed precision int16 layers: {mp_layers}")

    if strategy["bias_correction"]["value"]:
        bc_cfg = sampled_param["bias_correction"]
        quant_setting.bias_correct_setting.interested_layers = select_topk_layers(
            layerwise_error, bc_cfg.get("topk", 1)
        )
        quant_setting.bias_correct_setting.block_size = bc_cfg["block_size"]
        quant_setting.bias_correct_setting.steps = bc_cfg["steps"]

    if strategy["horizontal_layer_split"]["value"]:
        hls_cfg = sampled_param["horizontal_layer_split"]
        quant_setting.weight_split_setting.interested_layers = select_topk_layers(
            layerwise_error, hls_cfg.get("topk", 1)
        )
        quant_setting.weight_split_setting.value_threshold = hls_cfg["value_threshold"]

    if strategy["weight_equalization"]["value"]:
        wq_cfg = sampled_param["weight_equalization"]
        quant_setting.equalization_setting.iterations = wq_cfg["iterations"]
        quant_setting.equalization_setting.value_threshold = wq_cfg["value_threshold"]
        quant_setting.equalization_setting.opt_level = wq_cfg["opt_level"]

    if strategy["tqt"]["value"]:
        tqt_cfg = sampled_param["tqt"]
        quant_setting.tqt_optimization_setting.lr = tqt_cfg["lr"]
        quant_setting.tqt_optimization_setting.steps = tqt_cfg["steps"]
        quant_setting.tqt_optimization_setting.block_size = tqt_cfg["block_size"]

    return quant_setting, num_of_bits


def quant_model(runtime_config, quant_setting, num_of_bits):
    """Run ESP-PPQ quantization for one strategy."""
    onnx_path = runtime_config["onnx_path"]
    export_path = runtime_config["export_path"]
    input_shape = runtime_config["input_shape"]
    calib_dataloader = runtime_config["calib_dataloader"]
    device = runtime_config.get("device", "cpu")
    target = runtime_config.get("target", "esp32p4")

    # Use the simplified ONNX; never touch the user's original file.
    simplified_onnx_path = simplify_onnx(onnx_path)

    def collate_fn(batch):
        return batch.to(device)

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=simplified_onnx_path,
        espdl_export_file=export_path,
        calib_dataloader=calib_dataloader,
        calib_steps=32,
        input_shape=[1] + list(input_shape),
        target=target,
        num_of_bits=num_of_bits,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=device,
        error_report=False,
        skip_export=False,
        export_test_values=False,
        verbose=0,
        inputs=None,
    )

    # save .native for later testing
    exporter = NativeExporter()
    native_path = export_path.replace(".espdl", ".native")
    exporter.export(file_path=native_path, graph=quant_graph)

    return quant_graph


def get_layerwise_error_from_default_quant(runtime_config):
    num_of_bits = 8
    calib_dataloader = runtime_config["calib_dataloader"]
    device = runtime_config.get("device", "cpu")

    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_graph = quant_model(runtime_config, quant_setting, num_of_bits)

    def collate_fn(batch):
        return batch.to(device)

    layerwise_error = layerwise_error_analyse(
        graph=quant_graph,
        running_device=device,
        collate_fn=collate_fn,
        dataloader=calib_dataloader,
    )
    return layerwise_error


def select_topk_layers(layerwise_error: dict, topk: int) -> List[str]:
    """Select top-k most sensitive layers (highest error)."""
    return [
        name
        for name, _ in sorted(
            layerwise_error.items(), key=lambda x: x[1], reverse=True
        )[:topk]
    ]


def get_next_node(onnx_path, layer_name):
    model = onnx.load(onnx_path)
    consumers = defaultdict(list)
    target = None
    for n in model.graph.node:
        for inp in n.input:
            consumers[inp].append(n.name)
        if n.name == layer_name:
            target = n
    if target is None:
        raise ValueError(f"Node not found: {layer_name}")

    next_names = set()
    for out in target.output:
        for name in consumers.get(out, []):
            next_names.add(name)
    return list(next_names)
