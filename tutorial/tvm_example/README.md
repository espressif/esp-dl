# Deploying Models through TVM [[中文]](./README_cn.md)

## Run script
```
. script.sh
```
After running this script, an executable sample project named "new_project" will be generated in the current directory.

## Run inference project

```
cd new_project
idf.py set-target esp32s3
idf.py flash monitor
```

For more details please read [Deploying Models through TVM](https://docs.espressif.com/projects/esp-dl/en/latest/esp32/tutorials/deploying-models-through-tvm.html).