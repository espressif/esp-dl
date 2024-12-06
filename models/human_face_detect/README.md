| Supported Targets | ESP32-S3 | ESP32-P4 |
| ----------------- | -------- | -------- |

# List of Human Face Detect Models

Human Face Detect now support one model msr+mnp. It's a two stage model.  
First stage model msr predicts some candidates, then every candidate go through the next stage model mnp.

input_shape : h * w * c  
msr : 120 * 160 * 3  
mnp : 48 * 48 * 3  

|                  | preprocess(us) | model(us) | postprocess(us) |
| ---------------- | -------------- | --------- | --------------- |
| msr_s8_v1_s3     | 10691          | 30922     | 679             |
| msr_s8_v1_p4     | 5509           | 13308     | 352             |
| mnp_s8_v1_s3     | 5101           | 5263      | 105             |
| mnp_s8_v1_p4     | 5021           | 2465      | 41              |
