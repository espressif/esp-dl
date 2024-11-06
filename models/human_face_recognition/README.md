| Supported Targets | ESP32-S3 | ESP32-P4 |
| ----------------- | -------- | -------- |

# List of Human Face Recognition Models

Up to now, we support two different versions of human face recognition models. The performance of human face recognition models is as follows:

| Model                    | Params(M) | GFLOPs | TAR@FAR=1E-4 on IJB-C(%) | Time Cost on ESP32-S3(ms) | Time Cost on ESP32-P4(ms) |
| ---------------------------- | --------- | ------ | ------------------------ | ------------ | ------------ |
| human_face_feat_mfn_s8_v1 | 1.2       | 0.46   | 90.03                  | 239                     | 86                        |
| human_face_feat_mbf_s8_v1 | 3.4       | 0.45   | 93.94                    | 1122      | 191          |
