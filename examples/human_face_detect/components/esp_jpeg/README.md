# JPEG Decoder: TJpgDec - Tiny JPEG Decompressor

[![Component Registry](https://components.espressif.com/components/espressif/esp_jpeg/badge.svg)](https://components.espressif.com/components/espressif/esp_jpeg)

TJpgDec is a generic JPEG image decompressor that is highly optimized for small embedded systems. It works with very low memory consumption.

Some microcontrollers have TJpg decoder in ROM, it is used, if there is. [^1] Using ROM code can be disabled in menuconfig.

[^1]: **_NOTE:_** When the ROM decoder is used, the configuration can't be changed. The configuration is fixed.

## Features

**Compilation configuration:**
- Size of stream input buffer (default 512)
- Output pixel format (default RGB888): RGB888/RGB565
- Switches output descaling feature (default enabled)
- Use table conversion for saturation arithmetic (default enabled)
- Three optimization levels (default basic): 8/16-bit MCUs, 32-bit MCUs, Table conversion for huffman decoding 

**Runtime configuration:**
- Pixel Format: RGB888, RGB565
- Scaling Ratio: 1/1, 1/2, 1/4 or 1/8 Selectable on Decompression
- Allow swap the first and the last byte of the color

## TJpgDec in ROM

Some microcontrollers have TJpg decoder in ROM. It is used as default, but it can be disabled in menuconfig. Then there will be used code saved in this component. 

### List of MCUs, which have TJpgDec in ROM
- ESP32
- ESP32-S3
- ESP32-C3
- ESP32-C6

### Fixed compilation configuration of the ROM code
- Stream input buffer: 512
- Output pixel format: RGB888
- Descaling feature for output: Enabled
- Table for saturation: Enabled
- Optimization level: Basic (JD_FASTDECODE = 0)

### Pros and cons using ROM code

**Advantages:**
- Save 5k of the flash memory (in the same configuration)

**Disadvantages:**
- Cannot be changed compilation configuration
- Some configuration can be faster in some cases

## Speed comparison

In this table are examples of speed decoding JPEG image with this configuration:
* Image size: 320 x 180 px
* Output format: RGB565
* CPU: ESP32-S3
* CPU frequency: 240 MHz
* SPI mode: DIO
* Internal RAM used
* Measured in 1000 retries

| ROM used | JD_SZBUF | JD_FORMAT | JD_USE_SCALE | JD_TBLCLIP | JD_FASTDECODE | RAM buffer | Flash size | Approx. time |
| :------: | :------: | :-------: | :----------: | :--------: | :-----------: | :--------: | :--------: | :----------: |
|   YES    |    512   |   RGB888  |      1       |      1     |       0       |    3.1 kB  |    0 kB    |     52 ms    |    
|   NO     |    512   |   RGB888  |      1       |      1     |       0       |    3.1 kB  |    5 kB    |     50 ms    |    
|   NO     |    512   |   RGB888  |      1       |      0     |       0       |    3.1 kB  |    4 kB    |     68 ms    |     
|   NO     |    512   |   RGB888  |      1       |      1     |       1       |    3.1 kB  |    5 kB    |     50 ms    |      
|   NO     |    512   |   RGB888  |      1       |      0     |       1       |    3.1 kB  |    4 kB    |     62 ms    |   
|   NO     |    512   |   RGB888  |      1       |      1     |       2       |   65.5 kB  |   5.5 kB   |     46 ms    |  
|   NO     |    512   |   RGB888  |      1       |      0     |       2       |   65.5 kB  |   4.5 kB   |     59 ms    |  
|   NO     |    512   |   RGB565  |      1       |      1     |       0       |    5 kB    |    5 kB    |     60 ms    |     
|   NO     |    512   |   RGB565  |      1       |      1     |       1       |    5 kB    |    5 kB    |     59 ms    |     
|   NO     |    512   |   RGB565  |      1       |      1     |       2       |   65.5 kB  |   5.5 kB   |     56 ms    |     

## Add to project

Packages from this repository are uploaded to [Espressif's component service](https://components.espressif.com/).
You can add them to your project via `idf.py add-dependancy`, e.g. 
```
    idf.py add-dependency esp_jpeg==1.0.0
```

Alternatively, you can create `idf_component.yml`. More is in [Espressif's documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/tools/idf-component-manager.html).

## Example use

Here is example of usage. This calling is **blocking**.

```
esp_jpeg_image_cfg_t jpeg_cfg = {
    .indata = (uint8_t *)jpeg_img_buf,
    .indata_size = jpeg_img_buf_size,
    .outbuf = out_img_buf,
    .outbuf_size = out_img_buf_size,
    .out_format = JPEG_IMAGE_OUT_FORMAT_RGB565,
    .out_scale = JPEG_IMAGE_SCALE_0,
    .flags = {
        .swap_color_bytes = 1,
    }
};
esp_jpeg_image_output_t outimg;

esp_jpeg_decode(&jpeg_cfg, &outimg);
```
