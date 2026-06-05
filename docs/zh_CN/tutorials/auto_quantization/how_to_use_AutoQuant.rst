使用AutoQuant自动量化模型
=========================

:link_to_translation:`en:[English]`

.. contents::
  :local:
  :depth: 1

在面向 ESP32 芯片部署模型时，我们通常会使用 ``espdl_quantize_onnx`` 这类接口完成模型量化。只要提供 ONNX 模型、校准数据以及一组量化设置，就可以导出用于部署的 ``.espdl`` 模型。

但对于自定义模型而言，真正困难的部分往往并不在于量化流程本身，而在于如何选择合适的量化配置。例如，量化位宽应如何设置，应选择哪种校准算法，是否需要启用额外的优化策略，以及这些优化策略的参数应该如何调整等。这些因素都会影响量化误差，并进一步影响模型量化后的精度表现以及在 ESP32 芯片上的推理延迟。

因此，在实际开发过程中，寻找一组合适的量化配置通常需要一定经验积累和反复实验。开发者往往需要不断调整量化参数，重新执行量化与测试，再根据结果决定下一步的尝试方向。AutoQuant 想解决的，正是这类重复、繁琐且依赖经验的调参过程，将其自动化完成。

下面我们来看如何使用AutoQuant来自动找到合适的量化配置，并获取量化后的模型。


准备工作
---------

1. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`

.. _how_to_use_AutoQuant:


怎么使用AutoQuant
------------------

AutoQuant的使用方式和 ``espdl_quantize_onnx`` 类似，同样通过接口调用完成模型量化，只不过这里使用的是 ``espdl_auto_quantize_onnx`` 来启动自动搜索流程。使用时依然需要准备 ONNX 模型、校准数据以及输入 shape 等。

两者的主要区别在于： ``espdl_quantize_onnx`` 用于执行一次固定配置的量化，因此需要用户明确给出量化设置；而 ``espdl_auto_quantize_onnx`` 用于在多组量化设置中自动进行搜索、评测与筛选，用户只需提供搜索相关设置即可。

下面以这段代码为例进行说明：

.. code-block:: python

   from esp_ppq.api import AutoQuantSearchSetting, espdl_auto_quantize_onnx

   def evaluate_fn(graph):
      # Replace this with your validation logic.
      top1, top5 = ...
      return top1, {"top1": top1, "top5": top5}

   setting = AutoQuantSearchSetting(
      search_mode="fast",
      num_of_candidates=5,
      score_direction="maximize",
      run_dir="outputs/auto_quant_mobilenetv2",
   )

   espdl_auto_quantize_onnx(
      onnx_import_file="model.onnx",
      espdl_export_file="outputs/model.espdl",
      calib_dataloader=calib_loader,
      calib_steps=32,
      input_shape=[3, 224, 224],
      evaluate_fn=evaluate_fn,
      target="esp32p4",
      setting=setting,
      device="cuda",
   )

在该示例中，我们给 ``espdl_auto_quantize_onnx`` 传入了 9 个参数：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 参数
     - 说明
   * - ``onnx_import_file``
     - ONNX 模型路径
   * - ``espdl_export_file``
     - ``.espdl`` 模型导出路径
   * - ``calib_dataloader``
     - 校准数据
   * - ``calib_steps``
     - 校准时使用的 batch 数量
   * - ``input_shape``
     - 模型输入 shape，不包含 batch 维度。例如 ``[3, 224, 224]``
   * - ``evaluate_fn``
     - 可选的评测函数
   * - ``target``
     - 部署平台
   * - ``setting``
     - AutoQuant 的搜索设置
   * - ``device``
     - 运行设备，例如 ``"cpu"`` 或 ``"cuda"``

其中，我们重点关注 ``evaluate_fn`` 和 ``setting``，其他参数的使用方法和 ``espdl_quantize_onnx`` 基本一致。

**evaluate_fn**

``evaluate_fn`` 是 AutoQuant 用来比较不同量化配置结果的评测函数，是一个可选参数。

- **提供该函数时**：AutoQuant 会按照 ``evaluate_fn`` 返回的指标排序，
  例如分类模型的 top1 / top5、或检测模型的 mAP。
- **未提供该函数时**：AutoQuant 会使用下方的 **默认评测方式** 进行排序。

默认评测方式：首先调用 ``graphwise_error_analyse``，在校准数据上计算模型量化的 graphwise SNR
误差，然后取误差最大的 3 层求平均，将该平均值作为评测结果。


如果需要自定义 ``evaluate_fn``，则必须满足以下要求：

- 返回值必须是 ``(score, extras)`` 二元组。
- ``score`` 必须是有限数字，不能是 ``nan``、 ``inf`` 或 ``bool``。
- ``extras`` 必须是 ``dict``。
- ``extras`` 中不能使用 AutoQuant 的保留字段： ``score``、 ``hash``、 ``index``、 ``folder``、 ``files``、 ``strategy``、 ``params``

**setting**

``setting`` 用来控制 AutoQuant 的搜索行为。例如搜索模式、保留多少个候选结果、实验结果保存路径，以及是否从已有结果继续搜索等，都由 ``AutoQuantSearchSetting`` 配置。

``AutoQuantSearchSetting`` 的常用构造参数如下：

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - 参数
     - 默认值
     - 说明
   * - ``search_mode``
     - ``"exhaustive"``
     - 搜索模式。``"exhaustive"`` 会枚举所有候选配置；``"fast"`` 会先快速筛选一轮，再继续搜索更有希望的配置。
   * - ``num_of_candidates``
     - ``5``
     - 最终保留的候选模型数量，也就是 Top-K 的 K。
   * - ``score_direction``
     - ``"maximize"``
     - score 的排序方向。准确率、mAP 等指标越大越好，使用 ``"maximize"``；误差、loss 等指标越小越好，使用 ``"minimize"``。例如未传入 ``evaluate_fn`` 时，应使用 ``"minimize"``
   * - ``candidate_filter``
     - ``low_latency_candidate_filter``
     - Top-K 候选过滤器。默认更偏向保留部署成本较低的结果；如果只想严格按 score 排序，可以设置为 ``None``
   * - ``run_dir``
     - ``"outputs/auto_quant"``
     - 实验结果保存目录
   * - ``resume``
     - ``False``
     - 是否从已有 ``run_dir`` 继续搜索

如果使用 ``"fast"`` 搜索模式，还可以进一步设置：

.. code-block:: python

   setting.top_strategy = 10
   setting.early_stop_patience = 3

对应属性如下：

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - 属性
     - 默认值
     - 说明
   * - ``top_strategy``
     - ``10``
     - 第一轮筛选后保留多少组策略继续搜索
   * - ``early_stop_patience``
     - ``3``
     - 当前策略连续多少次没有提升后停止搜索


真正决定“要尝试哪些量化策略和参数”的，是 ``setting.strategy_space`` 和 ``setting.param_space``：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 说明
   * - ``strategy_space``
     - 控制每个量化策略是否参与搜索，例如是否尝试 mixed_precision
   * - ``param_space``
     - 控制每个策略对应的参数候选，例如校准算法候选、优化步数、学习率等

通常不需要调整这两个字段，如果你有需要，可以修改需要调整的部分：

.. code-block:: python

   setting.strategy_space["mixed_precision"] = [True, False]
   setting.param_space["calib_algorithm"]["method"] = ["kl", "mse"]


完整默认值可以查看 ``esp_ppq/autoquant/setting.py``。

结果获取
------------------

AutoQuant 每完成一个候选配置的搜索，都会将结果写入 ``run_dir``。目录结构如下：

.. code-block:: text

   <run_dir>/
       summary.json
       candidates.json
       0000/
           config.json
           model.espdl
           model.info
           model.json
           model.native
       0001/
           ...
 
其中， ``summary.json`` 记录所有已经完成的实验。 ``candidates.json`` 记录当前 Top-K 候选。每个编号目录保存一次实验的配置和导出的模型文件。

开启 ``resume=True`` 后，AutoQuant 会读取 ``summary.json`` ，跳过已经完成的 ``(strategy, params)`` 组合。这个功能主要用于搜索中断后的继续执行，以及细化搜索。


示例
------------------
ESP-PPQ 提供了两个可以直接运行的AutoQuant示例：

.. code-block:: python

   python -m esp_ppq.samples.AutoQuant.mobilenetv2
   python -m esp_ppq.samples.AutoQuant.yolo11n 

