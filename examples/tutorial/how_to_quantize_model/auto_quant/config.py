"""User-facing AutoQuant configuration.

Note:
`main.py` builds `calib_dataloader` via `create_calib_dataloader()` and injects it into
`runtime_config` at runtime.
"""

runtime_config = {
    "onnx_path": "./quantize_mobilenetv2/models/torch/mobilenet_v2_relu.onnx",  # path to the ONNX model
    "export_path": "temp/model.espdl",  # path to the ESP-DL model
    "input_shape": [3, 224, 224],
    "device": "cpu",  # "cpu" or "cuda"
    "batchsz": 32,
    "target": "esp32p4",  # "esp32p4" or "esp32s3"
    "test_app_path": "../../../test_apps/autoquant_latency",  # path to the test app
    "port": "/dev/ttyUSB0",  # "/dev/ttyUSB0" for ESP32P4, "/dev/ttyACM0" for ESP32S3
    "num_of_candidates": 5,  # number of candidates to keep
    "primary_metric": "top1",  # name of the metric used to rank candidates;
    # must equal one of the keys returned by evaluation() in eval.py
    # "calib_dataloader": injected at runtime by main.py
}


# ---------------------------------------------------------------------------
# Search space.
#
# strategy_space[k] = list of "switch values" for module k.
#   - [True]            -> module is always enabled (e.g. quantization itself)
#   - [True, False]     -> search will try with AND without
#   - [False]           -> module is disabled (still listed for visibility)
#
# When the switch is True, AutoQuant pulls the per-module knobs from
# param_space[k] and enumerates their cartesian product.
# ---------------------------------------------------------------------------

strategy_space = {
    "num_of_bits": [True],  # always quantize
    "calib_algorithm": [True],  # always pick a calib algo
    "weight_equalization": [True, False],
    "horizontal_layer_split": [False],  # rarely used, off by default
    "bias_correction": [False],  # rarely used, off by default
    "mixed_precision": [True, False],
    "tqt": [True, False],
}

# param_space[k][p] = list of values OR (start, end, step) tuple for grid sampling.
param_space = {
    "num_of_bits": {
        "value": [8],  # use [8, 16] if int16 is supported
    },
    "calib_algorithm": {
        "method": ["kl", "percentile"],  # add "minmax" when you need
    },
    "weight_equalization": {
        "iterations": [10, 50, 100],
        "value_threshold": [0.1, 0.15, 0.2],  # list -> enumerate; tuple (a,b,s) -> grid
        "opt_level": [2],
    },
    "horizontal_layer_split": {
        "topk": [1],  # K most error-sensitive layers
        "value_threshold": [0.1, 0.2],
    },
    "bias_correction": {
        "block_size": [2, 4],
        "steps": [32],
        "topk": [1, 2, 3],
    },
    "mixed_precision": {
        "topk": [1],  # K layers and their next K layers promoted to int16
    },
    "tqt": {
        "lr": [1e-5, 1e-4],
        "steps": [500, 1000],
        "block_size": [1, 2, 4],
    },
}
