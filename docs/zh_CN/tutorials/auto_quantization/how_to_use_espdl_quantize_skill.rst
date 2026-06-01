使用 espdl-quantize Skill 自动调优量化精度
==============================================

:link_to_translation:`en:[English]`

.. contents::
  :local:
  :depth: 2

本文档介绍如何使用 **espdl-quantize skill** 对深度学习模型进行自动量化调优。espdl-quantize 是一个**在线迭代、分布感知**的 Agent skill：它会先跑 baseline，再逐轮分析 layerwise 误差分布，并据此自动选择最适合的 esp-ppq 量化方法（如 TQT、混合精度、权重均衡、偏置校正等），反复迭代直至精度收敛或达到上限。

你只需提供校准数据加载和模型评估逻辑；skill 负责驱动 ``QuantizationSettingFactory.espdl_setting()`` 的迭代搜索。


什么是 espdl-quantize skill
----------------------------

esp-ppq 暴露了十余种可调量化 passes（校准算法、层间权重均衡、偏置校正、水平权重拆分、混合精度、TQT、LSQ、块重建等），每种又有 2-6 个参数。手动尝试不仅耗时，还容易因参数冲突导致 silent degradation。

espdl-quantize skill 将"看误差报告 → 猜测参数 → 重跑"的人工循环改造为**结构化、分布感知的自动搜索**，核心能力包括：

1. **知识库** —— ``references/ppq_methods.md`` 中编码了每种 esp-ppq 方法的原理、参数、适用场景与反模式。
2. **决策手册** —— ``references/decision_playbook.md`` 根据 Top-K 最差层的输入/权重/输出分布，将观察到的模式映射到候选方法。
3. **固定 harness** —— ``scripts/run_iteration.py`` 接收你的契约模块（``user_quant.py``）和一个 JSON setting，输出结构化产物（metrics、layerwise error、per-layer stats），Agent 只需读取 JSON 即可做出下一轮决策。
4. **搜索状态机** —— ``scripts/compare_iterations.py`` 检查已尝试过的配置，告诉 Agent 下一轮该跑什么。
5. **目标感知安全网** —— 自动检测与目标芯片量化策略冲突的 passes：

   - 针对 **POWER_OF_2 目标** （``esp32p4 / esp32s3 / c``），skill 会自动跳过 LSQ：LSQ 学习任意 scale，与 POWER_OF_2 的 2 的幂次约束冲突，因此改用 TQT（它直接训练 ``log2_scale``，天然适配 POWER_OF_2）。
   - **esp32p4** 上启用逐层均衡（Layerwise Equalization）时，skill 会发出警告：该机制与 Conv/Gemm 的 per-channel 量化策略在功能上重叠，esp-ppq 官方标记为 "Not recommend"，但实验表明部分 MobileNet 族/深度可分离网络仍可从中受益，因此 skill 仍会选择性尝试。


依赖要求
--------

1. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`
2. 安装 skill 额外依赖：

   在 skill 目录（包含 ``SKILL.md`` 的目录）下执行：

   .. code-block:: bash

      pip install -r assets/extra_requirements.txt

   额外依赖包括：``pandas``、``scipy``、``tqdm``。

   .. note::

      skill 直接运行在当前的 Python 解释器中，**无需 Docker**。


skill 仓库路径
--------------

espdl-quantize skill 在 esp-dl 仓库中的路径为：

.. code-block:: text

   esp-dl/tools/agents/skills/espdl-quantize/

目录结构：

.. code-block:: text

   esp-dl/tools/agents/skills/espdl-quantize/
   ├── SKILL.md                          # skill 入口与使用说明
   ├── assets/
   │   ├── extra_requirements.txt        # 额外 Python 依赖
   │   ├── user_quant_torch_example.py   # Torch 契约模板
   │   ├── user_quant_onnx_example.py    # ONNX 契约模板
   │   ├── example_quantize_mobilenetv2/ # MobileNet-V2 完整示例
   │   └── example_quantize_yolo11n/     # YOLO11n 完整示例
   ├── references/
   │   ├── contract.md                   # 用户契约完整规范
   │   ├── decision_playbook.md          # 分布→方法决策手册
   │   └── ppq_methods.md                # esp-ppq 各方法详解
   └── scripts/
       ├── run_iteration.py              # 单轮迭代 harness
       └── compare_iterations.py         # 迭代比较与状态机


安装 espdl-quantize skill
-------------------------

espdl-quantize skill 是 Agent 目录无关的，可以安装在任何 Agent 的 skills 文件夹下。常见安装路径如下。

方法一：使用 npx（推荐 - 最简单）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为 OpenCode、Cursor、Claude Code 及其他兼容工具安装 skill 最简单的方法是使用 npx：

.. code-block:: bash

    npx skills add espressif/esp-dl --skill espdl-quantize

.. note::

    npx 是 Node.js（通过 npm）自带的包执行器。如果你尚未安装 Node.js，请参考 `Node.js 安装指南 <https://nodejs.org/en/download/>`__ 进行安装。

执行命令后，skill 将被自动安装并在你的 Coding Agent 工具中可用。

方法二：手动安装
^^^^^^^^^^^^^^^^^

如果你偏好手动安装，或者电脑未安装 Node.js，请根据你使用的具体工具按照以下说明操作：

Cursor 用户
~~~~~~~~~~~

将 skill 目录复制到 Cursor 的 skills 文件夹(以用户级 skills 目录为例)：

.. code-block:: bash

   # Linux / macOS
   cp -r esp-dl/tools/agents/skills/espdl-quantize \
         ~/.cursor/skills/espdl-quantize

   # Windows (PowerShell)
   Copy-Item -Recurse esp-dl\tools\agents\skills\espdl-quantize \
             $env:USERPROFILE\.cursor\skills\espdl-quantize

OpenCode 用户
~~~~~~~~~~~~~

将 skill 目录复制到 OpenCode 的 skills 文件夹：

.. code-block:: bash

   # 复制到项目的 skills 目录
   cp -r esp-dl/tools/agents/skills/espdl-quantize \
         .opencode/skills/espdl-quantize

   # 或复制到用户级 skills 目录
   cp -r esp-dl/tools/agents/skills/espdl-quantize \
         ~/.agents/skills/espdl-quantize


如何使用 espdl-quantize skill
-----------------------------

使用本 skill 前，你需要准备一份 **用户契约文件** （通常命名为 ``user_quant.py``），其中包含：

- ``QUANT_CONFIG``：量化配置字典（模型类型、输入形状、目标芯片、primary_metric 等）。
- ``create_calib_dataloader()``：返回 PyTorch ``DataLoader``，用于校准。
- ``get_torch_model()`` 或提供 ``onnx_path``：返回 eval 模式的 ``nn.Module`` （torch 模型）或 ONNX 路径。
- ``evaluate(quant_graph) -> dict``：接收 PPQ 量化后的图，返回指标字典（必须包含 ``QUANT_CONFIG["primary_metric"]`` 对应的 key）。
- （可选）``evaluate_fast(quant_graph) -> dict``：用于快速中间迭代评估。

完整契约规范请参考 skill 目录下的 ``references/contract.md``。


快速开始示例
^^^^^^^^^^^^

以下命令展示了典型的使用方式。你可以直接复制到 Agent 对话中使用。

**示例 1：基本的量化调优请求**

.. code-block:: text

   // 量化 mobilenet_v2 模型
   我想用 espdl-quantize skill 量化 example_quantize_mobilenetv2/user_quant.py 这个 mobilenet_v2 模型。先跑 baseline，然后根据 layerwise 误差迭代 13 轮，目标 top1 越接近 71.878 越好，能超越，那就更好了。

   // 量化 yolo11n 模型
   我想用 espdl-quantize skill 量化 example_quantize_yolo11n/user_quant.py 这个 yolo11n 模型。先跑 baseline，然后根据 layerwise 误差迭代 13 轮，目标 map5095 越接近 39.0 越好，能超越，那就更好了。

**示例 2：追加迭代**

.. code-block:: text

   // 量化 mobilenet_v2 模型
   我想用 espdl-quantize skill 量化 example_quantize_mobilenetv2/user_quant.py 这个 mobilenet_v2 模型。之前已经迭代了13轮，我还想再迭代10轮，目标 top1 越接近 71.878 越好，能超越，那就更好了。

   // 量化 yolo11n 模型
   我想用 espdl-quantize skill 量化 example_quantize_yolo11n/user_quant.py 这个 yolo11n 模型。之前已经迭代了13轮，我还想再迭代10轮，目标 map5095 越接近 39.0 越好，能超越，那就更好了。

skill 接收到指令后会自动执行以下流程：

1. **Phase 0 — 基线评估**：使用默认 ``QuantizationSettingFactory.espdl_setting()`` 跑一次量化，得到 baseline 指标。
2. **Phase 1 — 误差分析**：读取 ``layerwise_error.json``、``layer_stats.json``、``non_computing_hot_ops.json``、``graphwise_jumps.json``，识别 Top-K 最差层。
3. **Phase 2 — 决策与配置生成**：根据 decision playbook 选择候选方法，生成下一轮 ``setting.json``。
4. **Phase 3 — 迭代执行**：运行 ``scripts/run_iteration.py`` 进行量化，评估指标。
5. **Phase 4 — 比较与收敛判断**：使用 ``scripts/compare_iterations.py`` 比较历史迭代，若达到 ``target_metric`` 或迭代上限则停止，否则回到 Phase 1。

每轮迭代的结果保存在 ``outputs/iter_<N>/`` 目录下，包含：

- ``metrics.json`` —— 当前迭代的评估指标。
- ``layerwise_error.json`` —— 各层的孤立误差。
- ``layer_stats.json`` / ``layer_stats_full.json`` —— 各层张量的分布统计。
- ``setting.json`` —— 本轮使用的完整量化配置。
- ``model.espdl``、``model.info``、``model.json``、``model.native`` —— 量化后的模型文件。

.. note::

   - Agent 指令中的目标指标（如 top1、map5095）必须同时满足两个一致性：与 QUANT_CONFIG["primary_metric"] 一致，且与 user_quant.py 中 evaluate() 函数返回的指标 key 一致。
   - Agent 需支持上下文压缩或自动摘要功能；上下文窗口更大的模型稳定性更佳，例如 deepseek_v4_pro。
   - 在自动搜索过程中，skill 会根据收敛规则（连续 3 轮结果相对 best 变化均小于 0.1%）自动判断迭代是否结束。但 AI Agent 可能因上下文丢失或误判而在到达用户指定迭代轮次前提前终止。此时你只需追加迭代指令（如"再迭代 N 轮"），skill 会从 ``outputs/`` 下已有产物恢复搜索，无需重新开始。


完整示例
--------

skill 目录下提供了两个可直接运行的完整示例，供你参考契约实现和项目结构：

1. **MobileNet-V2 图像分类（Torch）**

   相对路径：``assets/example_quantize_mobilenetv2/``

   包含：

   - ``user_quant.py`` —— 完整的 Torch 模型契约实现（ImageNet 校准与评估）。
   - ``datasets/`` —— 模型 evaluate 函数实现。

2. **YOLO11n 目标检测（ONNX）**

   相对路径：``assets/example_quantize_yolo11n/``

   包含：

   - ``user_quant.py`` —— 完整的 ONNX 模型契约实现（COCO 校准与评估）。
   - ``yolo11n_eval.py`` —— 检测任务 evaluate 辅助脚本。

从 esp-dl 仓库根目录出发的绝对路径示例：

.. code-block:: bash

   # MobileNet-V2 示例
   cd esp-dl/tools/agents/skills/espdl-quantize/assets/example_quantize_mobilenetv2

   # YOLO11n 示例
   cd esp-dl/tools/agents/skills/espdl-quantize/assets/example_quantize_yolo11n


查看最终产出结果
----------------

当 skill 完成所有迭代（或达到收敛条件）后，会在 ``outputs/`` 目录下生成以下关键产物，便于你查阅和部署最优模型：

final_report.md
^^^^^^^^^^^^^^^

``outputs/final_report.md`` 是每次量化调优的**最终总结报告**，包含：

- **Summary** —— 最优迭代轮次、最佳指标值、与目标的差距。
- **Iteration history** —— 所有迭代轮次的完整历史表，包括每轮改动的方法、指标变化（delta）和效果评估（improvement / regression / new best）。
- **Best setting** —— 最优迭代的完整量化配置（JSON 格式），可直接复用。
- **Python snippet** —— 可直接复制到代码中使用的 ``QuantizationSettingFactory.espdl_setting()`` 配置代码片段。
- **Key findings** —— Skill 对搜索过程的总结：最有效的方法、回归的方法、剩余误差分布热点。
- **Remaining gap** —— 若未达到目标指标，报告会分析剩余差距并提供后续优化建议（如增大数据量、混合精度、blockwise reconstruction 等）。

以下是一份 MobileNet-V2 on ESP32-S3 示例中的 ``final_report.md`` 片段：

.. code-block:: text

   ## Summary
   - Best iteration: iter_8
   - top1: 71.5000 (target_metric=71.8780)
   - Other metrics: top5=89.4500

   ## Iteration history

   | iter | method changed | top1 | delta | outcome | rank | affects inference speed |
   |---|---|---|---|---|---|---|
   | 0 | default espdl_setting() baseline | 60.5000 | +0.0000 = | baseline | 11 | No |
   | 1 | Phase 2: calib=kl × TQT(default) | 71.2250 | +10.7250 ↑ | improvement | 2 | No |
   | 2 | Phase 2: calib=mse × TQT(default) | 70.5250 | +10.0250 ↑ | improvement | 10 | No |
   | 3 | Phase 2: calib=percentile × TQT(default) | 70.9000 | +10.4000 ↑ | improvement | 9 | No |
   | 8 | Phase 3 lever 3c: Fusion alignment | 71.5000 | +11.0000 ↑ | **best** | **1** | No |

   ## Best setting
   ```json
   {
     "iteration_id": 8,
     "rationale": "Phase 3 lever 3c: Fusion alignment for elementwise ops...",
     "calib_algorithm": "kl",
     "tqt_optimization": { "enabled": true, "lr": 1e-05, "steps": 500, ... },
     "fusion_alignment": { "align_elementwise_to": "Align to Large" }
   }
   ```

   ## Key findings
   - **Best vs baseline**: +11.0000 (from 60.5000 to 71.5000).
   - **Most effective lever**: Phase 3 lever 3c (Fusion alignment) ...
   - **Largest regression**: iter_12 (Blockwise reconstruction) ...

best 目录
^^^^^^^^^

``outputs/best/`` 目录下保存了**最优迭代**的完整产物，便于直接部署或进一步分析：

- ``model.espdl`` —— 最优量化模型文件，可直接用于 esp-dl 部署。
- ``metrics.json`` —— 最优迭代的评估指标。
- ``setting.json`` —— 最优迭代的完整量化配置（与 ``final_report.md`` 中的 Best setting 一致）。
- ``iteration_index.json`` —— 指向最优迭代的元数据。

以下是最优迭代的 ``metrics.json`` 示例：

.. code-block:: json

   {
     "_primary_key": "top1",
     "_primary_value": 71.5,
     "metric_direction": "max",
     "top1": 71.5,
     "top5": 89.45
   }

以及 ``setting.json`` 示例：

.. code-block:: text

   {
     "iteration_id": 8,
     "rationale": "Phase 3 lever 3c: Fusion alignment for elementwise ops...",
     "calib_algorithm": "kl",
     "tqt_optimization": { "enabled": true, "lr": 1e-05, "steps": 500, ... },
     "fusion_alignment": { "align_elementwise_to": "Align to Large" }
   }

.. tip::

   你可以直接复制 ``final_report.md`` 中的 Python snippet 到生产代码中复现最优配置，无需手动拼凑参数。


故障排查
--------

``ModuleNotFoundError: No module named 'pandas'`` （或 pandas / scipy / tqdm）
   未安装 skill 额外依赖。在 skill 目录下运行：

   .. code-block:: bash

      pip install -r assets/extra_requirements.txt

``KeyError: 'primary_metric'`` 或 ``AssertionError: primary_metric=... not in metrics=...``
   ``QUANT_CONFIG["primary_metric"]`` 设置的 key 与 ``evaluate()`` 实际返回的 dict key 不一致。请对齐两者后重新运行。

``OSError: [Errno 2] No such file or directory: '.../user_quant.py'``
   skill 找不到契约文件。确认：
   - 你当前的工作目录包含 ``user_quant.py``；或
   - 在指令中显式指定了正确的相对/绝对路径。

某轮迭代耗时极长（TQT / LSQ / blockwise reconstruction）
   这些方法需要训练，PC 时间较高。若急需结果，可以在指令中要求 skill 跳过 TQT/LSQ 等高耗时 passes，优先尝试 layerwise equalization、bias correction 等零推理开销的方法。

``LSQ pass was skipped because target policy is POWER_OF_2``
   **正常现象**。esp-ppq 在 POWER_OF_2 目标（所有 esp-dl 芯片）上 LSQ 会静默退化，skill 自动跳过并以 TQT 替代。无需处理。

``Layer-wise equalization is not recommended for per-channel weight targets``
   **warn-only**。esp-ppq 使用逐层均衡量化时会发出警告，这是正常现象。

SKILL 没有反应 / 没有执行完所有迭代
   可能是 Agent 没有正确触发 skill，或中途出现错误但未正确反馈。尝试：

   - 使用显式触发词："使用 espdl-quantize skill..."。
   - 确认 skill 目录结构完整，且已安装额外依赖。
   - 使用指令遵循能力更强的模型。
   - 换用上下文窗口更大的模型，并开启 Agent 的上下文压缩/自动摘要功能。

迭代多轮后精度不再提升
   检查 ``outputs/iter_<N>/layer_stats.json``，若所有层的 ``Noise:Signal Power Ratio`` 均低于 0.05（5%），说明量化误差已接近噪声下限，继续调参收益有限。此时可考虑：

   - 增加校准数据量（``calib_steps``）。
   - 尝试混合精度（mixed precision）为敏感层保留更高位宽。
   - 检查输入预处理是否与训练时完全一致。
