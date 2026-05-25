Using the espdl-quantize Skill for Automatic Quantization Tuning
=================================================================

:link_to_translation:`zh_CN:[中文]`

.. contents::
  :local:
  :depth: 2

This document explains how to use the **espdl-quantize skill** to automatically tune the quantization accuracy of deep learning models. Unlike the offline exhaustive search in `AutoQuant <how_to_use_AutoQuant.html>`__, espdl-quantize is an **online, distribution-aware** Agent skill: it first runs a baseline, then iteratively analyzes layer-wise error distributions and automatically selects the most suitable esp-ppq quantization methods (such as TQT, mixed precision, weight equalization, bias correction, etc.), repeating until accuracy converges or the limit is reached.

You only need to provide the calibration data loading and model evaluation logic; the skill drives the iterative search of ``QuantizationSettingFactory.espdl_setting()``.


What is the espdl-quantize Skill
---------------------------------

esp-ppq exposes more than a dozen tunable quantization passes (calibration algorithms, layer-wise weight equalization, bias correction, horizontal weight splitting, mixed precision, TQT, LSQ, blockwise reconstruction, etc.), each with 2-6 parameters. Manual trial-and-error is not only time-consuming but also prone to silent degradation due to parameter conflicts.

The espdl-quantize skill transforms the manual loop of "read error report → guess parameters → rerun" into a **structured, distribution-aware automatic search**. Its core capabilities include:

1. **Knowledge Base** — ``references/ppq_methods.md`` encodes the principles, parameters, applicable scenarios, and anti-patterns of each esp-ppq method.
2. **Decision Playbook** — ``references/decision_playbook.md`` maps observed patterns to candidate methods based on the input/weight/output distributions of the Top-K worst layers.
3. **Fixed Harness** — ``scripts/run_iteration.py`` receives your contract module (``user_quant.py``) and a JSON setting, and outputs structured artifacts (metrics, layer-wise error, per-layer stats). The Agent only needs to read the JSON to make the next decision.
4. **Search State Machine** — ``scripts/compare_iterations.py`` checks tried configurations and tells the Agent what to run next.
5. **Target-Aware Safety Net** — Automatically detects passes that conflict with the target chip's quantization policy:

   - For **POWER_OF_2 targets** (``esp32p4 / esp32s3 / c``), the skill automatically skips LSQ: LSQ learns arbitrary scales, which conflicts with the power-of-2 constraint of POWER_OF_2, so TQT is used instead (it directly trains ``log2_scale``, naturally fitting POWER_OF_2).
   - When enabling Layerwise Equalization on **esp32p4**, the skill issues a warning: this mechanism functionally overlaps with the per-channel quantization policy of Conv/Gemm, and esp-ppq officially marks it as "Not recommend". However, experiments show that some MobileNet-family/depthwise-separable networks can still benefit from it, so the skill will still selectively try it.


Dependencies
------------

1. :ref:`Install ESP_PPQ <requirements_esp_ppq>`
2. Install the skill's additional dependencies:

   In the skill directory (the directory containing ``SKILL.md``), run:

   .. code-block:: bash

      pip install -r assets/extra_requirements.txt

   Additional dependencies include: ``pandas``, ``scipy``, ``tqdm``.

   .. note::

      The skill runs directly in the current Python interpreter, **no Docker required**.


Skill Repository Path
---------------------

The espdl-quantize skill is located in the esp-dl repository at:

.. code-block:: text

   esp-dl/tools/agents/skills/espdl-quantize/

Directory structure:

.. code-block:: text

   esp-dl/tools/agents/skills/espdl-quantize/
   ├── SKILL.md                          # Skill entry and usage instructions
   ├── assets/
   │   ├── extra_requirements.txt        # Additional Python dependencies
   │   ├── user_quant_torch_example.py   # Torch contract template
   │   ├── user_quant_onnx_example.py    # ONNX contract template
   │   ├── example_quantize_mobilenetv2/ # MobileNet-V2 complete example
   │   └── example_quantize_yolo11n/     # YOLO11n complete example
   ├── references/
   │   ├── contract.md                   # Full user contract specification
   │   ├── decision_playbook.md          # Distribution→method decision playbook
   │   └── ppq_methods.md                # Detailed esp-ppq methods
   └── scripts/
       ├── run_iteration.py              # Single-round iteration harness
       └── compare_iterations.py         # Iteration comparison and state machine


Installing the espdl-quantize Skill
------------------------------------

The espdl-quantize skill is agent-directory-agnostic and can be installed under any agent's skills folder. Common installation paths are as follows.

Method 1: Using npx (Recommended - Easiest)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install the skill for OpenCode, Cursor, Claude Code, and other compatible tools is to use npx:

.. code-block:: bash

    npx skills add espressif/esp-dl --skill espdl-quantize

.. note::

    npx is a package runner bundled with Node.js (via npm). If you haven't installed Node.js yet, please refer to the `Node.js Installation Guide <https://nodejs.org/en/download/>`__.

After executing the command, the skill will be automatically installed and available in your Coding Agent tool.

Method 2: Manual Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer manual installation, or if Node.js is not installed on your computer, follow the instructions below based on your specific tool:

Cursor Users
~~~~~~~~~~~~

Copy the skill directory to Cursor's skills folder (using the user-level skills directory as an example):

.. code-block:: bash

   # Linux / macOS
   cp -r esp-dl/tools/agents/skills/espdl-quantize \
         ~/.cursor/skills/espdl-quantize

   # Windows (PowerShell)
   Copy-Item -Recurse esp-dl\tools\agents\skills\espdl-quantize \
             $env:USERPROFILE\.cursor\skills\espdl-quantize

OpenCode Users
~~~~~~~~~~~~~~

Copy the skill directory to OpenCode's skills folder:

.. code-block:: bash

   # Copy to the project's skills directory
   cp -r esp-dl/tools/agents/skills/espdl-quantize \
         .opencode/skills/espdl-quantize

   # Or copy to the user-level skills directory
   cp -r esp-dl/tools/agents/skills/espdl-quantize \
         ~/.agents/skills/espdl-quantize


How to Use the espdl-quantize Skill
------------------------------------

Before using this skill, you need to prepare a **user contract file** (usually named ``user_quant.py``), which includes:

- ``QUANT_CONFIG``: Quantization configuration dictionary (model type, input shape, target chip, primary_metric, etc.).
- ``create_calib_dataloader()``: Returns a PyTorch ``DataLoader`` for calibration.
- ``get_torch_model()`` or provide ``onnx_path``: Returns an eval-mode ``nn.Module`` (torch model) or ONNX path.
- ``evaluate(quant_graph) -> dict``: Receives the PPQ quantized graph and returns a metrics dictionary (must contain the key corresponding to ``QUANT_CONFIG["primary_metric"]``).
- (Optional) ``evaluate_fast(quant_graph) -> dict``: Used for fast intermediate iteration evaluation.

For the full contract specification, please refer to ``references/contract.md`` in the skill directory.


Quick Start Examples
^^^^^^^^^^^^^^^^^^^^

The following commands show typical usage. You can copy them directly into your Agent conversation.

**Example 1: Basic Quantization Tuning Request**

.. code-block:: text

   // Quantize mobilenet_v2 model
   I want to use the espdl-quantize skill to quantize the mobilenet_v2 model in example_quantize_mobilenetv2/user_quant.py. First run a baseline, then iterate 13 rounds based on layer-wise error, with the goal of getting top1 as close to 71.878 as possible, and even better if it can exceed it.

   // Quantize yolo11n model
   I want to use the espdl-quantize skill to quantize the yolo11n model in example_quantize_yolo11n/user_quant.py. First run a baseline, then iterate 13 rounds based on layer-wise error, with the goal of getting map5095 as close to 39.0 as possible, and even better if it can exceed it.

**Example 2: Additional Iterations**

.. code-block:: text

   // Quantize mobilenet_v2 model
   I want to use the espdl-quantize skill to quantize the mobilenet_v2 model in example_quantize_mobilenetv2/user_quant.py. 13 rounds have already been iterated, and I want to iterate another 10 rounds, with the goal of getting top1 as close to 71.878 as possible, and even better if it can exceed it.

   // Quantize yolo11n model
   I want to use the espdl-quantize skill to quantize the yolo11n model in example_quantize_yolo11n/user_quant.py. 13 rounds have already been iterated, and I want to iterate another 10 rounds, with the goal of getting map5095 as close to 39.0 as possible, and even better if it can exceed it.

After receiving the instruction, the skill automatically executes the following workflow:

1. **Phase 0 — Baseline Evaluation**: Run quantization once using the default ``QuantizationSettingFactory.espdl_setting()`` to get baseline metrics.
2. **Phase 1 — Error Analysis**: Read ``layerwise_error.json``, ``layer_stats.json``, ``non_computing_hot_ops.json``, ``graphwise_jumps.json``, and identify the Top-K worst layers.
3. **Phase 2 — Decision and Configuration Generation**: Select candidate methods according to the decision playbook and generate the next ``setting.json``.
4. **Phase 3 — Iteration Execution**: Run ``scripts/run_iteration.py`` for quantization and evaluate metrics.
5. **Phase 4 — Comparison and Convergence Check**: Use ``scripts/compare_iterations.py`` to compare historical iterations. If ``target_metric`` is reached or the iteration limit is hit, stop; otherwise, return to Phase 1.

The results of each iteration are saved in the ``outputs/iter_<N>/`` directory, containing:

- ``metrics.json`` — Evaluation metrics of the current iteration.
- ``layerwise_error.json`` — Isolated error of each layer.
- ``layer_stats.json`` / ``layer_stats_full.json`` — Distribution statistics of each layer's tensors.
- ``setting.json`` — The complete quantization configuration used in this round.
- ``model.espdl``, ``model.info``, ``model.json``, ``model.native`` — Quantized model files.

.. note::

   - The target metric in the Agent instruction (e.g., top1, map5095) must satisfy two consistency requirements: it must match ``QUANT_CONFIG["primary_metric"]``, and it must match the key returned by the ``evaluate()`` function in user_quant.py.
   - The Agent must support context compression or auto-summarization; models with larger context windows provide more stable results, such as deepseek_v4_pro.
   - During automated search, the skill automatically determines convergence based on the convergence rule (3 consecutive iterations all within 0.1% relative to best). However, the AI Agent may terminate early before reaching the user-specified iteration rounds due to context loss or misjudgment. In this case, simply issue an additional iteration command (e.g., "iterate N more rounds"), and the skill will resume the search from the existing artifacts under ``outputs/`` without starting over.


Complete Examples
-----------------

The skill directory provides two ready-to-run complete examples for you to reference contract implementation and project structure:

1. **MobileNet-V2 Image Classification (Torch)**

   Relative path: ``assets/example_quantize_mobilenetv2/``

   Contains:

   - ``user_quant.py`` — Complete Torch model contract implementation (ImageNet calibration and evaluation).
   - ``datasets/`` — Model evaluate function implementation.

2. **YOLO11n Object Detection (ONNX)**

   Relative path: ``assets/example_quantize_yolo11n/``

   Contains:

   - ``user_quant.py`` — Complete ONNX model contract implementation (COCO calibration and evaluation).
   - ``yolo11n_eval.py`` — Detection task evaluate helper script.

Absolute path examples from the esp-dl repository root:

.. code-block:: bash

   # MobileNet-V2 example
   cd esp-dl/tools/agents/skills/espdl-quantize/assets/example_quantize_mobilenetv2

   # YOLO11n example
   cd esp-dl/tools/agents/skills/espdl-quantize/assets/example_quantize_yolo11n


Viewing the Final Output
------------------------

When the skill completes all iterations (or reaches the convergence condition), it generates the following key artifacts in the ``outputs/`` directory for your review and deployment of the optimal model:

final_report.md
^^^^^^^^^^^^^^^

``outputs/final_report.md`` is the **final summary report** of each quantization tuning session, containing:

- **Summary** — The best iteration round, the best metric value, and the gap to the target.
- **Iteration history** — Complete history table of all iteration rounds, including the method changed in each round, metric changes (delta), outcome (improvement/regression/neutral), and rank.
- **Best setting** — The complete quantization configuration (JSON format) of the best iteration, which can be directly reused.
- **Python snippet** — A ``QuantizationSettingFactory.espdl_setting()`` configuration code snippet that can be directly copied into your code.
- **Key findings** — The skill's summary of the search process: most effective methods, regressed methods, and remaining error distribution hotspots.
- **Remaining gap** — If the target metric is not reached, the report analyzes the remaining gap and provides follow-up optimization suggestions (e.g., increase data volume, mixed precision, blockwise reconstruction, etc.).

Below is a snippet from a ``final_report.md`` example of MobileNet-V2 on ESP32-S3:

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

best Directory
^^^^^^^^^^^^^^

The ``outputs/best/`` directory saves the **complete artifacts of the best iteration**, facilitating direct deployment or further analysis:

- ``model.espdl`` — The best quantized model file, ready for esp-dl deployment.
- ``metrics.json`` — Evaluation metrics of the best iteration.
- ``setting.json`` — The complete quantization configuration of the best iteration (consistent with the Best setting in ``final_report.md``).
- ``iteration_index.json`` — Metadata pointing to the best iteration.

Below is an example of ``metrics.json`` for the best iteration:

.. code-block:: json

   {
     "_primary_key": "top1",
     "_primary_value": 71.5,
     "metric_direction": "max",
     "top1": 71.5,
     "top5": 89.45
   }

And an example of ``setting.json``:

.. code-block:: text

   {
     "iteration_id": 8,
     "rationale": "Phase 3 lever 3c: Fusion alignment for elementwise ops...",
     "calib_algorithm": "kl",
     "tqt_optimization": { "enabled": true, "lr": 1e-05, "steps": 500, ... },
     "fusion_alignment": { "align_elementwise_to": "Align to Large" }
   }

.. tip::

   You can directly copy the Python snippet from ``final_report.md`` into your production code to reproduce the optimal configuration, without manually assembling parameters.


Troubleshooting
---------------

``ModuleNotFoundError: No module named 'pandas'`` (or pandas / scipy / tqdm)
   The skill's additional dependencies are not installed. Run in the skill directory:

   .. code-block:: bash

      pip install -r assets/extra_requirements.txt

``KeyError: 'primary_metric'`` or ``AssertionError: primary_metric=... not in metrics=...``
   The key set in ``QUANT_CONFIG["primary_metric"]`` does not match the dict key actually returned by ``evaluate()``. Please align both and rerun.

``OSError: [Errno 2] No such file or directory: '.../user_quant.py'``
   The skill cannot find the contract file. Confirm that:
   - Your current working directory contains ``user_quant.py``; or
   - You explicitly specified the correct relative/absolute path in the instruction.

A certain iteration takes extremely long (TQT / LSQ / blockwise reconstruction)
   These methods require training, which takes considerable PC time. If you need results urgently, you can ask the skill in the instruction to skip time-consuming passes like TQT/LSQ and prioritize zero-inference-overhead methods like layerwise equalization and bias correction.

``LSQ pass was skipped because target policy is POWER_OF_2``
   **Normal behavior**. On POWER_OF_2 targets (all esp-dl chips), LSQ silently degrades, so the skill automatically skips it and uses TQT instead. No action needed.

``Layer-wise equalization is not recommended for per-channel weight targets``
   **Warn-only**. esp-ppq issues a warning when using layer-wise equalization quantization; this is normal behavior.

SKILL has no response / does not complete all iterations
   The Agent may not have triggered the skill correctly, or an error occurred midway without proper feedback. Try:

   - Use explicit trigger words: "Use the espdl-quantize skill...".
   - Confirm the skill directory structure is complete and additional dependencies are installed.
   - Use a model with stronger instruction-following capability.
   - Switch to a model with a larger context window and enable context compression / auto-summarization in your Agent tool.

Accuracy no longer improves after multiple iterations
   Check ``outputs/iter_<N>/layer_stats.json``. If the ``Noise:Signal Power Ratio`` of all layers is below 0.05 (5%), it means the quantization error is approaching the noise floor, and the benefit of continuing to tune parameters is limited. At this point, consider:

   - Increasing the calibration data volume (``calib_steps``).
   - Trying mixed precision to preserve higher bit-width for sensitive layers.
   - Checking whether input preprocessing is completely consistent with training.
