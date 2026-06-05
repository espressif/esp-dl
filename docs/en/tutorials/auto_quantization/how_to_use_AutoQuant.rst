Quantize Models Automatically with AutoQuant
============================================

:link_to_translation:`zh_CN:[中文]`

.. contents::
  :local:
  :depth: 1

When deploying models on ESP32 chips, we typically use interfaces such as ``espdl_quantize_onnx`` to perform model quantization. By providing an ONNX model, calibration data, and a set of quantization settings, you can export a ``.espdl`` model for deployment.

For custom models, however, the truly difficult part is often not the quantization process itself, but how to choose an appropriate quantization configuration. For example, what bit width to use, which calibration algorithm to choose, whether to enable additional optimization strategies, and how to tune the parameters of those optimization strategies. All of these factors affect the quantization error, and in turn the post-quantization accuracy of the model as well as its inference latency on ESP32 chips.

Therefore, in actual development, finding an appropriate quantization configuration usually requires accumulated experience and repeated experimentation. Developers often need to continually adjust quantization parameters, re-run quantization and testing, and then decide on the next direction to try based on the results. What AutoQuant aims to solve is exactly this kind of repetitive, tedious and experience-dependent tuning process, by automating it.

Below we'll look at how to use AutoQuant to automatically find an appropriate quantization configuration and obtain the quantized model.


Prerequisites
-------------

1. :ref:`Install ESP_PPQ <requirements_esp_ppq>`

.. _how_to_use_AutoQuant:


How to Use AutoQuant
--------------------

AutoQuant is used in a way similar to ``espdl_quantize_onnx``: model quantization is also done through an interface call, except that here we use ``espdl_auto_quantize_onnx`` to launch the automatic search process. As before, you still need to prepare an ONNX model, calibration data, the input shape, and so on.

The main difference between the two is that ``espdl_quantize_onnx`` performs a single quantization run with a fixed configuration and therefore requires the user to explicitly provide the quantization settings; whereas ``espdl_auto_quantize_onnx`` is used to automatically search, evaluate and filter across multiple quantization settings — the user only needs to provide search-related settings.

The following code is used as an example for explanation:

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

In this example, we pass 9 parameters to ``espdl_auto_quantize_onnx``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``onnx_import_file``
     - Path to the ONNX model
   * - ``espdl_export_file``
     - Export path of the ``.espdl`` model
   * - ``calib_dataloader``
     - Calibration data
   * - ``calib_steps``
     - Number of batches used during calibration
   * - ``input_shape``
     - Model input shape, excluding the batch dimension. For example ``[3, 224, 224]``
   * - ``evaluate_fn``
     - Optional evaluation function
   * - ``target``
     - Deployment platform
   * - ``setting``
     - AutoQuant search settings
   * - ``device``
     - Runtime device, e.g. ``"cpu"`` or ``"cuda"``

Among these, we focus mainly on ``evaluate_fn`` and ``setting``; the other parameters are used in essentially the same way as in ``espdl_quantize_onnx``.

**evaluate_fn**

``evaluate_fn`` is the evaluation function that AutoQuant uses to compare the results of different quantization configurations. It is an optional parameter.

- **When this function is provided**: AutoQuant ranks results by the metric returned by ``evaluate_fn``,
  for example top1 / top5 for classification models, or mAP for detection models.
- **When this function is not provided**: AutoQuant uses the **default evaluation method** described below to rank results.

Default evaluation method: first, ``graphwise_error_analyse`` is called to compute the graphwise SNR
error of model quantization on the calibration data; then the average of the 3 layers with the largest errors is taken as the evaluation result.


If you want to customize ``evaluate_fn``, it must meet the following requirements:

- The return value must be a ``(score, extras)`` tuple.
- ``score`` must be a finite number; it cannot be ``nan``, ``inf`` or ``bool``.
- ``extras`` must be a ``dict``.
- ``extras`` must not use the following reserved fields of AutoQuant: ``score``, ``hash``, ``index``, ``folder``, ``files``, ``strategy``, ``params``

**setting**

``setting`` controls AutoQuant's search behavior. Things such as the search mode, how many candidate results to keep, the path to save experiment results, and whether to continue searching from existing results, are all configured via ``AutoQuantSearchSetting``.

The commonly used constructor parameters of ``AutoQuantSearchSetting`` are as follows:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Parameter
     - Default
     - Description
   * - ``search_mode``
     - ``"exhaustive"``
     - Search mode. ``"exhaustive"`` enumerates all candidate configurations; ``"fast"`` performs a quick filtering round first, and then continues to search the more promising configurations.
   * - ``num_of_candidates``
     - ``5``
     - The number of candidate models to keep in the end, i.e. the K in Top-K.
   * - ``score_direction``
     - ``"maximize"``
     - The direction in which scores are ranked. For metrics like accuracy or mAP, which are better when higher, use ``"maximize"``; for metrics like error or loss, which are better when lower, use ``"minimize"``. For example, when ``evaluate_fn`` is not provided, ``"minimize"`` should be used.
   * - ``candidate_filter``
     - ``low_latency_candidate_filter``
     - Top-K candidate filter. By default it favors keeping results with lower deployment cost; if you want to rank strictly by score, set it to ``None``.
   * - ``run_dir``
     - ``"outputs/auto_quant"``
     - Directory where experiment results are saved.
   * - ``resume``
     - ``False``
     - Whether to continue searching from an existing ``run_dir``.

If you use the ``"fast"`` search mode, you can further configure:

.. code-block:: python

   setting.top_strategy = 10
   setting.early_stop_patience = 3

The corresponding attributes are as follows:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Default
     - Description
   * - ``top_strategy``
     - ``10``
     - How many strategy groups to keep for further searching after the first filtering round.
   * - ``early_stop_patience``
     - ``3``
     - Stop searching after the current strategy has failed to improve for this many consecutive iterations.


What really decides "which quantization strategies and parameters to try" is ``setting.strategy_space`` and ``setting.param_space``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``strategy_space``
     - Controls whether each quantization strategy participates in the search, e.g. whether to try mixed_precision.
   * - ``param_space``
     - Controls the parameter candidates for each strategy, e.g. calibration algorithm candidates, number of optimization steps, learning rate, etc.

Usually you don't need to adjust these two fields; if you do need to, you can modify only the parts that need adjusting:

.. code-block:: python

   setting.strategy_space["mixed_precision"] = [True, False]
   setting.param_space["calib_algorithm"]["method"] = ["kl", "mse"]


The complete default values can be found in ``esp_ppq/autoquant/setting.py``.

Obtaining Results
-----------------

Each time AutoQuant finishes searching one candidate configuration, it writes the result to ``run_dir``. The directory structure is as follows:

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
 
Here, ``summary.json`` records all completed experiments, ``candidates.json`` records the current Top-K candidates, and each numbered subdirectory stores the configuration and exported model files of one experiment.

With ``resume=True`` enabled, AutoQuant will read ``summary.json`` and skip the ``(strategy, params)`` combinations that have already been completed. This feature is mainly used for resuming after an interrupted search, and for refining the search.


Examples
--------
ESP-PPQ provides two ready-to-run AutoQuant examples:

.. code-block:: python

   python -m esp_ppq.samples.AutoQuant.mobilenetv2
   python -m esp_ppq.samples.AutoQuant.yolo11n 