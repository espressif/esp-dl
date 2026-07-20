# PP-OCRv6 Models

Two-stage OCR based on [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR). Recognizes Chinese, English and 46 Latin-script languages on-device on ESP32-P4.

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     | DET_S8                 | REC_S8                 | REC_S16                | REC_S16_W640           |
|----------|------------------------|------------------------|------------------------|------------------------|
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] | ![alt text][supported] | ![alt text][supported] |

## Model Latency

Measured on ESP32-P4, single crop for recognition (detection once per image):

| name | input (h\*w\*c) | preprocess(ms) | model(ms) | postprocess(ms) |
|------|-----------------|-----------------|-----------|------------------|
| `pp_ocr_v6_det_s8_p4` | 736 \* 736 \* 3 | 225.7 | 2334.0 | 146.9 |
| `pp_ocr_v6_rec_s8_p4` | 48 \* 320 \* 3 | 16.1 | 168.9 | 22.1 |
| `pp_ocr_v6_rec_s16_p4` | 48 \* 320 \* 3 | 16.6 | 383.3 | 19.1 |
| `pp_ocr_v6_rec_s16_w640_p4` | 48 \* 640 \* 3 | 38.6 | 801.9 | 48.1 |

### OCR Configurations

The component supports three end-to-end OCR configurations. Every configuration
runs `pp_ocr_v6_det_s8` once per image, then recognizes each detected text
crop. In the Dual configuration, only crops with W/H > 6.67 use the wide model
to avoid horizontal squeezing.

| Configuration | Latency characteristics | Recommended scenario |
|---------------|-------------------------|----------------------|
| `pp_ocr_v6_det_s8` + `pp_ocr_v6_rec_s8` | Fastest. | Flash or latency is tightly constrained, and text lines are normally short. |
| `pp_ocr_v6_det_s8` + `pp_ocr_v6_rec_s16` (default) | Medium. | Default choice for short text lines when recognition accuracy matters. Long lines with W/H > 6.67 are horizontally squeezed. |
| `pp_ocr_v6_det_s8` + `pp_ocr_v6_rec_s16` + `pp_ocr_v6_rec_s16_w640` | Variable: same as the default for short crops; potentially slowest when long crops use the 640-wide model. | Images mixing short and long lines, such as addresses, sentences, and dense documents. It avoids long-line squeezing without reducing recognition precision. |

Select `CONFIG_PP_OCR_V6_REC_MODE_SHORT` with
`CONFIG_PP_OCR_V6_REC_S8` or `CONFIG_PP_OCR_V6_REC_S16` for the first two
configurations. For the third, select `CONFIG_PP_OCR_V6_REC_MODE_DUAL`,
enable `CONFIG_FLASH_PP_OCR_V6_REC_S16_W640`, and normally select
`CONFIG_PP_OCR_V6_REC_S16`. The total OCR time is the one-time detection time
plus the recognition time for every detected crop.

## Model Usage

``PPOCRV6`` is configured by menuconfig ([default rec model](#default-rec-model), [rec width mode](#rec-width-mode); see [OCR Configurations](#ocr-configurations) for how to choose). You can also pass explicit ``RecMode`` / ``RecModel`` to the constructor.

### How to New `PPOCRV6`

#### Default (menuconfig)

```cpp
pp_ocr_v6::PPOCRV6 ocr;
```

#### Explicit rec variant / mode

```cpp
// short 48x320, INT16 rec
pp_ocr_v6::PPOCRV6 ocr(pp_ocr_v6::RecMode::Short, pp_ocr_v6::RecModel::PP_OCR_V6_REC_S16);

// dual: short 48x320 (S8 here, S16 also fine) + long 48x640 (S16, fixed);
// requires FLASH_PP_OCR_V6_REC_S16_W640 or sdcard
// pp_ocr_v6::PPOCRV6 ocr(pp_ocr_v6::RecMode::Dual, pp_ocr_v6::RecModel::PP_OCR_V6_REC_S8);
```

### How to Run OCR

```cpp
dl::image::img_t img = {.data = DATA, .width = WIDTH, .height = HEIGHT, .pix_type = PIX_TYPE};
std::vector<pp_ocr_v6::OCRResult> results = ocr.run(img);
```

Each ``OCRResult`` contains:

- ``box`` — detected quad (``TextBox``: 8 ints TL/TR/BR/BL + ``det_score``)
- ``text`` — recognized string
- ``score`` — CTC mean confidence (results with ``score < drop_score`` are dropped; default ``0.5``)

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`pp_ocr_v6.hpp`](pp_ocr_v6.hpp).

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_PP_OCR_V6_REC_S8
- CONFIG_FLASH_PP_OCR_V6_REC_S16
- CONFIG_FLASH_PP_OCR_V6_REC_S16_W640

Recognition models to include when model location is set to FLASH rodata or
FLASH partition. The required `pp_ocr_v6_det_s8` detection model is always
included.

## Default Rec Model

- CONFIG_PP_OCR_V6_REC_S8
- CONFIG_PP_OCR_V6_REC_S16

Default short-line recognition model if no parameter is passed to ``PPOCRV6``.

## Rec Width Mode

- CONFIG_PP_OCR_V6_REC_MODE_SHORT
- CONFIG_PP_OCR_V6_REC_MODE_DUAL

``short``: only 48x320. ``dual``: also load 48x640 (``REC_S16_W640``) and route
crops wider (relative to height) than the short model's native 320/48 ≈ 6.67
to it instead of squeezing them.

## Model Location

- CONFIG_PP_OCR_V6_MODEL_IN_FLASH_RODATA
- CONFIG_PP_OCR_V6_MODEL_IN_FLASH_PARTITION
- CONFIG_PP_OCR_V6_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE]
> If model location is set to FLASH partition, the custom partition table must
> contain a `pp_ocr_v6` partition large enough for the packed model file. The
> [pp_ocr_v6 example](../../examples/pp_ocr_v6) provides `partitions2.csv`
> (6 MB model partition); select it in menuconfig. The default `partitions.csv`
> is for FLASH rodata and does not contain this partition.

## SDCard Directory

- CONFIG_PP_OCR_V6_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.

The default value of this option is `models/p4` for ESP32P4.
When using the default value, copy [models](models) to the SD-card root directory.

> [!NOTE]
> Do not change the model name when copy the models to sdcard.
