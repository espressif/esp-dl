# Usage of convert.py

The [convert.py](../../tool/convert.py) converts coefficients from float-point format in npy file to bit-quantize format in C/C++ file. It also converts the element sequence of coefficient for boosting some operation. It should be noted that convert.py runs according to config.json. So, the config.json of a model is necessary. [Specification of config.json](./specification_of_config_json.md) tells how to write a config.json file.



## Arguments Description

When run convert.py, some arguments are needed to be filled. Here is the description.

| Argument            | Value                                        |
| ------------------- | -------------------------------------------- |
| -t \| --target_chip | esp32 \| esp32s2 \|esp32s3 \| esp32c3        |
| -i \| --input_root  | npy files and config.json directory          |
| -n \| --name        | generated C/C++ coefficients filename        |
| -o \| --output_root | generated C/C++ coefficients files directory |



## Example

Let's assume that

- The relative directory of convert.py is **../../tool/convert.py**.
- My target_chip is **esp32s3**.
- My npy files and config.json are in directory **./my_input_directory**.
- I hope the generated file named **my_coefficient**.
- I want the generated file located in **./my_output_directory**.

So, I run convert.py like this.

```sh
python ../../tool/convert.py -t esp32s3 -i ./my_input_directory -n my_coefficient -o ./my_output_directory
```

Then, I'll get `my_coefficient.cpp` and `my_coefficient.hpp` in `./my_output_directory`.
