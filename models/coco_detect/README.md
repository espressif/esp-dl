| Supported Targets | ESP32-S3 | ESP32-P4 |
| ----------------- | -------- | -------- |

# List of YOLO11 Detect Models

COCO Detect now support yolo11-series detectors. Up to now, only yolo11n is tested.

input_shape : h * w * c  
yolo11n : 640 * 640 * 3  

|                                     | preprocess(us) | model(us)   | postprocess(us) |
| ----------------------------------- | -------------- | ----------- | --------------- |
| yolo11n_s8_v1_s3                    | 207953         | 33610434    | 58566           |
| yolo11n_s8_v1_p4                    | 105772         | 3297940     | 16400           |
| yolo11n_s8_v2_p4                    | 105758         | 4214450     | 16398           |

Note that when running yolo11n model on ESP32-S3, the param_copy item is set to false for the model, which will increase the inference time.