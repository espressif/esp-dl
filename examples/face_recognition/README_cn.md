# 人脸识别 [[English]](./README.md)

本项目为人脸识别接口的示例。人脸识别接口的输入图片为一张带有人脸的静态图片，我们的人脸识别器提供了录入人脸， 识别人脸， 删除人脸等功能。 每一个功能的运行结果可显示在终端中。我们还提供了16位量化与8位量化两个版本的模型。16位量化的模型相比于8位量化的模型， 精度更高， 但是占用内存更多， 运行速度也更慢。 使用者可以根据实际使用场景挑选合适的模型。

项目所在文件夹结构如下：

```shell
human_face_detect/
├── CMakeLists.txt
├── image.jpg
├── main
│   ├── app_main.cpp
│   ├── CMakeLists.txt
│   └── image.hpp
├── partitions.csv
└── README.md
```



## 运行示例

1. 打开终端，进入人脸检测示例所在文件夹 esp-dl/examples/human_face_detect：

    ```shell
    cd ~/esp-dl/examples/face_recognition
    ```

2. 设定目标芯片：

    ```shell
    idf.py set-target [SoC]
    ```
    将 [SoC] 替换为您的目标芯片，如 esp32、 esp32s2、esp32s3。
    我们更推荐使用esp32s3芯片， 在AI应用上，它的运行速度会远快于其他芯片。

3. 烧录固件，打印每一阶段的运行结果：

   ```shell
   idf.py flash monitor
   
   ... ...
   
   enroll id ...
   name: Sandra, id: 1
   name: Jiong, id: 2

   recognize face ...
   [recognition result] id: 1, name: Sandra, similarity: 0.728666
   [recognition result] id: 2, name: Jiong, similarity: 0.827225

   recognizer information ...
   recognizer threshold: 0.55
   input shape: 112, 112, 3

   face id information ...
   number of enrolled ids: 2
   id: 1, name: Sandra
   id: 2, name: Jiong

   delete id ...
   number of remaining ids: 1
   [recognition result] id: -1, name: unknown, similarity: 0.124767

   enroll id ...
   name: Jiong, id: 2

   recognize face ...
   [recognition result] id: 1, name: Sandra, similarity: 0.758815
   [recognition result] id: 2, name: Jiong, similarity: 0.722041

   ```

## 其他设置
1. [./main/app_main.cpp](./main/app_main.cpp) 开头处的宏定义 `QUANT_S16`，可定义模型的量化类型。具体区别如下：

- `QUANT_S16` = 1：识别器为16位量化模型，识别精度更高，但速度更慢，更占内存。
- `QUANT_S16` = 0：识别器为8位量化模型，识别精度低于16位模型，但速度更快，内存占用更少。

  您可实际使用场景挑选合适的模型。

2. [./main/app_main.cpp](./main/app_main.cpp) 开头处的宏定义 `USE_FACE_DETECTOR`，可以定义人脸landmark坐标的获得方式：

- `USE_FACE_DETECTOR` = 1：使用我们的人脸检测模型获得landmark坐标。
- `USE_FACE_DETECTOR` = 0：使用存放在image.hpp中的landmark坐标。

   请注意landmark的坐标顺序为：
   
    ```
    left_eye_x, left_eye_y, 
    mouth_left_x, mouth_left_y,
    nose_x, nose_y,
    right_eye_x, right_eye_y, 
    mouth_right_x, mouth_right_y
    ```

