# 使用 TVM 部署模型 [[English]](./README.md)

## 运行脚本
```
. script.sh
```
运行完脚本后，在当前目录下会生成可运行的示例项目 new_project。

## 运行项目

```
cd new_project
idf.py set-target esp32s3
idf.py flash monitor
```

更多可查看 [使用 TVM 部署模型](https://docs.espressif.com/projects/esp-dl/zh_CN/latest/esp32/tutorials/deploying-models-through-tvm.html).