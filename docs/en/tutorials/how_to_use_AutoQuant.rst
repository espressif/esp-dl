AutoQuant
=========

:link_to_translation:`zh_CN:[中文]`

.. contents::
  :local:
  :depth: 1

This document describes how to use the AutoQuant toolchain to quantize
and deploy deep learning models. AutoQuant is designed for Espressif
chips (ESP32-P4 and ESP32-S3). You only need to define the quantization
strategy and parameter search space. AutoQuant then runs quantization,
accuracy evaluation, and on-device latency testing, and provides Top-K
candidate models for selection.
Compared with repeated manual tuning, AutoQuant can significantly reduce
trial-and-error cost.


Preparation
-----------

1. :ref:`Install ESP_IDF <requirements_esp_idf>`
2. :ref:`Install ESP_PPQ <requirements_esp_ppq>`

.. _how_to_use_AutoQuant:


Quick start
-----------

By default, AutoQuant uses MobileNetV2 + ImageNet for quantization and
deployment. It is recommended to run this example first, get familiar
with the full AutoQuant flow, and then switch to your own model.

Custom model quantization and deployment
----------------------------------------

To use a custom model, modify the following three files:
- ``config.py`` — runtime parameters and search space
- ``calib_dataloader.py`` — calibration data
- ``eval.py`` — accuracy evaluation

.. note::

   No change is needed in ``main.py``. You only need to implement
   ``create_calib_dataloader()`` and ``evaluation()``. AutoQuant will call
   them automatically.

1. ``config.py`` 
~~~~~~~~~~~~~~~~

First, edit ``runtime_config``::

    runtime_config = {
        "onnx_path": "./quantize_mobilenetv2/models/torch/mobilenet_v2_relu.onnx",
        "export_path": "temp/model.espdl",
        "input_shape": [3, 224, 224],
        "device": "cpu",            # "cpu" or "cuda"
        "batchsz": 32,
        "target": "esp32p4",         # "esp32p4" or "esp32s3"
        "test_app_path": "../../../test_apps/autoquant_latency",
        "port": "/dev/ttyUSB0",      # use "/dev/ttyACM0" on ESP32-S3
        "num_of_candidates": 5,      # number of Top-K models to keep
        "primary_metric": "top1",    # see eval.py section below
    }

``calib_dataloader`` is not in ``runtime_config``. It is built and
injected by ``main.py`` at runtime.

The same file also defines ``strategy_space`` and ``param_space``.
Each entry in the strategy space defines whether a quantization module
(e.g., mixed precision, weight equalization, TQT) is always enabled,
always disabled, or left optional for automatic selection during search.
``param_space`` provides the parameter values that AutoQuant should try
when a module is enabled.

2. ``calib_dataloader.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the body of ``create_calib_dataloader()`` with your own
implementation. The function must return a PyTorch ``DataLoader``. The
shape of each batch must match ``runtime_config["input_shape"]``.
Pre/post-processing (resize, normalize, etc.) is model-dependent and
must stay consistent with your training pipeline. The default
implementation downloads an ImageNet calibration subset and applies the
standard MobileNetV2 transforms. Replace it as a whole when switching
to another model.

3. ``eval.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the body of ``evaluation(quant_graph)`` with your own
implementation. The function takes the PPQ-quantized graph as the only
input and must return a ``dict`` where keys are metric names and values
are scalar numbers. The default implementation returns
``{"top1": ..., "top5": ...}`` for ImageNet classification.

If evaluation needs GPU/CPU, read from ``runtime_config["device"]``.

AutoQuant uses ``runtime_config["primary_metric"]`` to rank candidates.
So the dict returned by ``eval.py`` must contain that key, and
``primary_metric`` in ``config.py`` must match it. Otherwise, the
program fails immediately.
For example:

.. list-table::
   :header-rows: 1

   * - File
     - Change
   * - ``eval.py``
     - Implement ``evaluation(quant_graph)``, for example return
       ``{"map": map_score}``
   * - ``config.py``
     - Set ``"primary_metric": "map"``

Stage 1: search and accuracy evaluation
---------------------------------------

Run Stage 1 on any machine with calibration data and GPU (or CPU). No
ESP board is required. AutoQuant enumerates all (strategy, parameter)
combinations, quantizes each one, and evaluates accuracy. Then it keeps
K results with a balanced rule (``K = runtime_config["num_of_candidates"]``):
take ``K//2`` from ``low_latency_pool`` (``mixed_precision=False`` and
``horizontal_layer_split=False``), picking the highest by
``runtime_config["primary_metric"]``; fill the remaining slots from the
other candidates using the same metric. If the other candidates are not
enough, the remaining slots are filled from ``low_latency_pool``. If one
side is empty, the selection degenerates to the regular Top-K ranked by
``primary_metric``.

The following commands are expected to run in
``esp-dl/examples/tutorial/how_to_quantize_model``. If you are not in
this directory, ``cd`` into it first.

::

    python -m auto_quant.main             # fresh run; resets summary/candidates
    python -m auto_quant.main --resume    # continue from interruption

The ``--resume`` flag reuses ``outputs/run/summary.json`` and skips
already finished (strategy, parameter) combinations. Use it after a
crash, OOM, or manual interruption.

Stage 1 generates the following under ``outputs/run/``:

- ``<index>/`` (e.g. ``0000/``, ``0001/`` ...) — one directory per
  experiment, containing ``config.json`` and quantized model files
  (``.espdl``, ``.info``, ``.json``, ``.native``).
- ``summary.json`` — append log for each experiment (accuracy,
  parameters, hash).
- ``candidates.json`` — current Top-K list, rewritten after each
  experiment, maintained by the Stage 1 balanced rule so different
  strategy families are represented, and partial results stay usable.


Stage 2: on-device latency test
-------------------------------

Stage 2 must run on a local machine with a physically connected ESP
board (remote servers cannot access USB devices). Stage 2 flashes each
Top-K candidate from Stage 1, measures on-device inference latency, and
finally writes a Markdown report ``topk_report.md`` so the final model
can be chosen based on accuracy/latency trade-off.

Before running
~~~~~~~~~~~~~~

1. Make sure Stage 1 has finished and
   ``outputs/run/candidates.json`` exists. If Stage 1 ran on a remote
   machine, copy the full ``outputs/run/`` directory to local (for
   example using ``rsync``).
2. Connect an ESP32-P4 or ESP32-S3 board, and verify
   ``runtime_config["target"]`` and ``runtime_config["port"]`` in
   ``config.py`` match the connected hardware.
3. Activate the ESP-IDF environment.

Run
~~~

::

    python -m auto_quant.test_candidates
    python -m auto_quant.test_candidates --port /dev/ttyUSB1
    python -m auto_quant.test_candidates --target esp32s3

Optional ``--port`` and ``--target`` temporarily override corresponding
fields in ``runtime_config`` for a single run. This is useful when
switching boards without editing ``config.py``.

Output files
~~~~~~~~~~~~

- ``outputs/run/candidates.json`` — same Top-K list as Stage 1, with an
  extra ``latency_ms`` field. It is updated after each candidate, so
  measured results are preserved even on crash or Ctrl-C.
- ``outputs/run/topk_report.md`` — Markdown report. ``primary_metric``
  and ``latency_ms`` of all Top-K candidates are organized into two
  four-column tables (``index | folder | primary_metric | latency_ms``):
  one sorted by ``primary_metric`` descending, one sorted by
  ``latency_ms`` ascending, so accuracy/latency trade-off is easy to
  compare. Full quantization strategy and parameters are not duplicated
  in the report. Check ``candidates.json`` (full list) or
  ``<folder>/config.json`` (single experiment snapshot). Failed
  candidates are listed in the "Failed measurements" section at the end.
  After reading the report, flash the ``model.espdl`` in the selected
  row's ``folder`` directory into your firmware.

Switch target chip
~~~~~~~~~~~~~~~~~~

When ``runtime_config["target"]`` changes between runs (for example from
``esp32p4`` to ``esp32s3``), Stage 2 automatically cleans
``sdkconfig``, ``build/``, ``managed_components/``, and
``dependencies.lock`` in the test app. No manual cleanup is needed.


Output layout
-------------

::

    outputs/run/
        summary.json          # all Stage 1 experiment records (append)
        candidates.json       # Top-K; Stage 2 adds latency_ms field
        topk_report.md        # Stage 2 Markdown report for manual selection
        0000/
            config.json       # quantization config used by this experiment
            model.espdl       # quantized model (can be deployed directly)
            model.info
            model.json
            model.native
        0001/
            ...


Troubleshooting
---------------

``AssertionError: primary_metric=... not in metrics=...``
    The key in ``runtime_config["primary_metric"]`` does not match the
    key returned by ``evaluation()``. Align them and re-run.

No latency printed for a candidate
    - Check ``runtime_config["port"]`` (``/dev/ttyUSB0`` for ESP32-P4,
      ``/dev/ttyACM0`` for ESP32-S3).
    - Make sure the serial port is not occupied by another process
      (``idf.py monitor``, ``minicom``, etc.).
    - Confirm ``runtime_config["target"]`` matches the connected board.

``idf.py flash`` failed
    - Wrong chip type — set ``runtime_config["target"]`` correctly and
      run again; Stage 2 automatically cleans stale build outputs.
    - Serial port busy — close all open serial monitors.

Model file not found
    - Check ``runtime_config["export_path"]`` in ``config.py``.
    - Confirm Stage 1 has finished and corresponding
      ``outputs/run/<index>/model.espdl`` exists.
