AutoQuant
===========

:link_to_translation:`en:[English]`

.. contents::
  :local:
  :depth: 1

本文档介绍如何使用 AutoQuant 工具链量化并部署深度学习模型。AutoQuant 面向乐鑫芯片（ESP32-P4 和 ESP32-S3）设计，你只需要定义量化策略和参数搜索空间，AutoQuant 会自动完成从量化、精度评估到端侧延时测试的整个流程，并给出Top-K候选模型供你选择。
相比手动反复调参，AutoQuant 可以显著降低试错成本。


准备工作
---------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`

.. _how_to_use_AutoQuant:


快速开始
--------

AutoQuant 默认基于 MobileNetV2 + ImageNet 进行量化部署。建议先运行该示例，熟悉 AutoQuant 的整体流程，再替换为自定义模型。

自定义模型量化部署
------------------

接入自定义模型时，需要修改以下三个文件：
- ``config.py`` —— 运行参数和搜索空间
- ``calib_dataloader.py`` —— 校准数据
- ``eval.py`` —— 精度评估

.. note::

   不需要修改``main.py`` ，只需要实现 ``create_calib_dataloader()`` 和 ``evaluation()`` ，AutoQuant会自动调用它们。

1. ``config.py``
~~~~~~~~~~~~~~~~~~~~~~

首先修改 ``runtime_config``::

    runtime_config = {
        "onnx_path": "./quantize_mobilenetv2/models/torch/mobilenet_v2_relu.onnx",
        "export_path": "temp/model.espdl",
        "input_shape": [3, 224, 224],
        "device": "cpu",            # "cpu" 或 "cuda"
        "batchsz": 32,
        "target": "esp32p4",         # "esp32p4" 或 "esp32s3"
        "test_app_path": "../../../test_apps/autoquant_latency",
        "port": "/dev/ttyUSB0",      # ESP32-S3 使用 "/dev/ttyACM0"
        "num_of_candidates": 5,      # 保留多少个 Top-K 模型
        "primary_metric": "top1",    # 见下文 eval.py 章节
    }

``calib_dataloader`` 不在runtime_config里，它由 ``main.py`` 在运行时构建并注入。

同一文件中还定义了 ``strategy_space`` 和 ``param_space``。
``strategy_space`` 中的每一项配置定义某个量化模块（如混合精度、权重均衡、
TQT）是始终启用、始终禁用，还是在搜索时作为可选项自动决策。
``param_space`` 则给出模块启用时 AutoQuant 需要尝试的参数取值。

2. ``calib_dataloader.py`` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将 ``create_calib_dataloader()`` 的函数体替换为你自己的实现。该函数必须
返回一个 PyTorch ``DataLoader``，其每个 batch 的形状要与
``runtime_config["input_shape"]`` 匹配。前/后处理（resize、normalize 等）
是模型相关的，必须与训练阶段保持一致。默认实现会下载一份 ImageNet
校准子集并应用标准的 MobileNetV2 transforms；切换到其他模型时请整体替换。

3. ``eval.py`` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将 ``evaluation(quant_graph)`` 的函数体替换为你自己的实现。该函数接收
PPQ 量化后的图作为唯一参数，必须返回一个 ``dict``，键为指标名，
值为标量。默认实现针对 ImageNet 分类任务返回
``{"top1": ..., "top5": ...}``。

如果评估需要使用GPU/CPU，请从 ``runtime_config["device"]`` 读取。

AutoQuant 使用 ``runtime_config["primary_metric"]`` 来排序候选模型，因此需要保证 ``eval.py`` 返回的字典中包含该key，且 ``config.py`` 中的 ``primary_metric`` 与该key一致，否则程序会直接报错。
例如：

.. list-table::
   :header-rows: 1

   * - 文件
     - 修改内容
   * - ``eval.py``
     - 实现 ``evaluation(quant_graph)``，例如返回 ``{"map": map_score}``
   * - ``config.py``
     - 设置 ``"primary_metric": "map"``

Stage 1：搜索与精度评估
-----------------------

在任何拥有校准数据和 GPU（或 CPU）的主机上运行 Stage 1。无需 ESP
开发板。 AutoQuant 会枚举所有（策略，参数）组合，对每个组合进行量化
和精度评估，随后按均衡规则保留 K 个结果（``K =
runtime_config["num_of_candidates"]``）：先在 ``low_latency_pool``
（``mixed_precision=False`` 且 ``horizontal_layer_split=False``）
里按 ``runtime_config["primary_metric"]`` 从高到低取 ``K//2`` 个，其余名额从
其他候选按同一指标补齐。若其他候选不足，则由 ``low_latency_pool`` 中的候选
补齐。若其中一侧候选为空，则退化为按 ``primary_metric`` 排序的普通 Top-K。

以下命令默认在 ``esp-dl/examples/tutorial/how_to_quantize_model`` 目录执行。
若当前不在该目录，请先 ``cd`` 到该目录。

::

    python -m auto_quant.main             # 全新运行；会重置 summary/candidates
    python -m auto_quant.main --resume    # 从中断处继续

``--resume`` 标志会复用已有的 ``outputs/run/summary.json``，跳过其中
已经完成的（策略，参数）组合。在程序崩溃、OOM 或人为中断后使用。

Stage 1 在 ``outputs/run/`` 下生成：

- ``<index>/``（如 ``0000/``、``0001/`` …）—— 每个实验一个子目录，
  内含 ``config.json`` 和量化后的模型文件
  （``.espdl``、``.info``、``.json``、``.native``）。
- ``summary.json`` —— 每个实验的追加日志（精度、参数、hash）。
- ``candidates.json`` —— 当前的 Top-K 列表，每个实验后都会重写一次，
  按 Stage 1 的均衡规则维护，从而保证不同策略族都有代表项，且部分结果
  随时可用。


Stage 2：板上延时测试
---------------------

Stage 2 必须在物理连接 ESP 开发板的本地机器上运行（远程服务器无法访问
USB 设备）。Stage 2 会依次烧录 Stage 1 产出的每个 Top-K 候选，测量
片上推理延时，最后输出一份 Markdown 报告 ``topk_report.md``，由用户
根据精度/延时折中自行选择最终部署哪个模型。

运行前
~~~~~~

1. 确认 Stage 1 已完成、``outputs/run/candidates.json`` 已存在。
   如果 Stage 1 是在远程机器上运行的，把整个 ``outputs/run/`` 目录拷贝
   到本地（例如使用 ``rsync``）。
2. 接入 ESP32-P4 或 ESP32-S3 开发板，并仔细确认 ``config.py`` 中
   的 ``runtime_config["target"]`` 和 ``runtime_config["port"]`` 与所连
   硬件一致。
3. 激活 ESP-IDF 环境。

运行
~~~~

::

    python -m auto_quant.test_candidates
    python -m auto_quant.test_candidates --port /dev/ttyUSB1
    python -m auto_quant.test_candidates --target esp32s3

可选的 ``--port`` 和 ``--target`` 参数会在单次调用里临时覆盖
``runtime_config`` 里的对应字段，方便在多个开发板之间切换而不改
``config.py``。

输出文件
~~~~~~~~

- ``outputs/run/candidates.json`` —— 与 Stage 1 相同的 Top-K 列表，
  额外增加了 ``latency_ms`` 字段。每完成一个候选就更新一次，因此即使
  中途崩溃或 Ctrl-C 也能保留已测的部分结果。
- ``outputs/run/topk_report.md`` —— Markdown 报告，所有 Top-K 候选的
  ``primary_metric`` 和 ``latency_ms`` 整理成两张四列表
  （``index | folder | primary_metric | latency_ms``）：一张按
  ``primary_metric`` 降序，一张按 ``latency_ms`` 升序，方便对照查看精度
  /延时折中。报告里不再重复展开每个候选的完整量化策略和参数，请直接查
  ``candidates.json``（全量列表）或 ``<folder>/config.json``（单次实验
  快照）。失败的候选会单独列在报告末尾的 "Failed measurements" 小节。
  读完报告后，把所选行 ``folder`` 对应实验目录里的 ``model.espdl``
  烧录到你的固件中。

切换目标芯片
~~~~~~~~~~~~

当 ``runtime_config["target"]`` 在两次运行之间发生变化（例如从
``esp32p4`` 切换到 ``esp32s3``），Stage 2 会自动清理 test app 中的
``sdkconfig``、``build/``、``managed_components/`` 以及
``dependencies.lock``，不用手动清理。


输出目录结构
------------

::

    outputs/run/
        summary.json          # Stage 1 的全部实验记录（追加写入）
        candidates.json       # Top-K；Stage 2 之后会附加 latency_ms 字段
        topk_report.md        # Stage 2 之后供人工选型的 Markdown 报告
        0000/
            config.json       # 该实验使用的量化配置
            model.espdl       # 量化后的模型（可直接部署）
            model.info
            model.json
            model.native
        0001/
            ...


故障排查
--------

``AssertionError: primary_metric=... not in metrics=...``
    ``runtime_config["primary_metric"]`` 设置的 key 与
    ``evaluation()`` 实际返回的 key 不一致。对齐两者后重新运行即可。

某个候选没有打印出延时
    - 检查 ``runtime_config["port"]``（ESP32-P4 用 ``/dev/ttyUSB0``，
      ESP32-S3 用 ``/dev/ttyACM0``）。
    - 确认串口没有被其他进程占用（``idf.py monitor``、``minicom`` 等）。
    - 确认 ``runtime_config["target"]`` 与所连开发板一致。

``idf.py flash`` 失败
    - 芯片类型设错 —— 正确设置 ``runtime_config["target"]`` 后重新运行；
      Stage 2 会自动清理过期的构建产物。
    - 串口被占用 —— 关闭所有打开的串口监视器。

找不到模型文件
    - 检查 ``config.py`` 中的 ``runtime_config["export_path"]``。
    - 确认 Stage 1 已完成、对应的 ``outputs/run/<index>/model.espdl``
      存在。
