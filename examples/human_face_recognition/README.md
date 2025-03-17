[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# Human Face Recognition Example

A simple image inference example. See full example in [esp-who](https://github.com/espressif/esp-who/tree/master/examples/human_face_recognition).

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
id: 3, sim: 0.750027
```
## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options. This example includes two models, one for human face detection and another for human face feature extraction. See 

- [Human Face Detect Model](https://github.com/espressif/esp-dl/blob/master/models/human_face_detect/README.md)
- [Human Face Recognition Model](https://github.com/espressif/esp-dl/blob/master/models/human_face_recognition/README.md)

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`

- CONFIG_DB_FATFS_FLASH
- CONFIG_DB_FATFS_SDCARD
- CONFIG_DB_SPIFFS

We extracts human face feature and saves them to a database. Three diffrent types of file system for database is supported. 

> [!NOTE]  
> - fatfs_flash and spiffs save features to a 1MB flash partition named `storage`. It's defined in `partitions.csv` and `partitions2.csv`.
> - fatfs_sdcard save features to sdcard. 
> - Each feature cosumes 2050 bytes, including 2 bytes for id and 2048 bytes for feature data. 