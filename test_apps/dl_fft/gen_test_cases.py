import os

import numpy as np

np.random.seed(42)  # 固定随机种子保证可重复性


def numpy_to_c_array(data, array_name, file):
    """
    将numpy数组转换为C数组定义并写入文件
    :param data: numpy数组
    :param array_name: C数组名称
    :param file: 文件对象
    """
    if np.iscomplexobj(data):
        # 复数数组处理
        file.write(f"static const float {array_name}[{len(data)*2}] = {{\n")
        for i, val in enumerate(data):
            file.write(f"    {val.real}f, {val.imag}f")
            if i != len(data) - 1:
                file.write(",")
            file.write("\n")
        file.write("};\n\n")
    else:
        # 实数数组处理
        file.write(f"static const float {array_name}[{len(data)}] = {{\n")
        for i, val in enumerate(data):
            file.write(f"    {val}f")
            if i != len(data) - 1:
                file.write(",")
            file.write("\n")
        file.write("};\n\n")


def numpy_to_c_array_s16(data, array_name, file):
    """
    将numpy数组转换为C数组定义并写入文件
    :param data: numpy数组
    :param array_name: C数组名称
    :param file: 文件对象
    """

    # 实数数组处理
    if np.iscomplexobj(data):
        # 复数数组处理
        file.write(f"static const short {array_name}[{len(data)*2}] = {{\n")
        for i, val in enumerate(data):
            file.write(f"    {int(np.round(val.real))}, {int(np.round(val.imag))}")
            if i != len(data) - 1:
                file.write(",")
            file.write("\n")
        file.write("};\n\n")
    else:
        # 实数数组处理
        file.write(f"static const short {array_name}[{len(data)}] = {{\n")
        for i, val in enumerate(data):
            file.write(f"    {int(np.round(val))}")
            if i != len(data) - 1:
                file.write(",")
            file.write("\n")
        file.write("};\n\n")


def generate_random_input(size):
    # 生成随机复数输入数据（使用float32）
    input_data = np.random.randint(-(2**14), high=2**14 - 1, size=size, dtype=int)
    input_data = input_data / 2**15
    input_data = input_data.astype(np.float32)

    return input_data


def generate_fft_test_case(n, output_dir):
    """
    生成单个FFT测试用例
    :param n: FFT点数
    :param output_dir: 输出目录
    """

    input_data = (generate_random_input(n) + 1j * generate_random_input(n)).astype(
        np.complex64
    )
    input_data_s16 = input_data * (2**15)

    # 计算FFT
    fft_result = np.fft.fft(input_data)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成C头文件
    with open(f"{output_dir}/fft_test_{n}.h", "w") as f:
        # 写入头文件保护
        f.write(f"#ifndef FFT_TEST_{n}_H\n")
        f.write(f"#define FFT_TEST_{n}_H\n\n")

        # 写入输入数据
        numpy_to_c_array(input_data, f"fft_input_{n}", f)
        numpy_to_c_array_s16(input_data_s16, f"fft_input_s16_{n}", f)

        # 写入预期输出数据
        numpy_to_c_array(fft_result, f"fft_output_{n}", f)
        numpy_to_c_array_s16(fft_result, f"fft_output_s16_{n}", f)

        # 结束头文件保护
        f.write(f"#endif // FFT_TEST_{n}_H\n")

    print(f"Generated test case for N={n} in fft_test_{n}.h")


def generate_rfft_test_case(n, output_dir):
    """
    生成单个FFT测试用例
    :param n: FFT点数
    :param output_dir: 输出目录
    """
    # 生成随机复数输入数据（使用float32）
    input_data = generate_random_input(n)
    input_data_s16 = input_data * (2**15)

    # 计算FFT
    fft_result = np.fft.rfft(input_data)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成C头文件
    with open(f"{output_dir}/rfft_test_{n}.h", "w") as f:
        # 写入头文件保护
        f.write(f"#ifndef FFTR_TEST_{n}_H\n")
        f.write(f"#define FFTR_TEST_{n}_H\n\n")

        # 写入输入数据
        numpy_to_c_array(input_data, f"rfft_input_{n}", f)
        numpy_to_c_array_s16(input_data_s16, f"rfft_input_s16_{n}", f)

        # 写入预期输出数据
        numpy_to_c_array(fft_result, f"rfft_output_{n}", f)

        # 结束头文件保护
        f.write(f"#endif // FFTR_TEST_{n}_H\n")

    print(f"Generated test case for N={n} in rfft_test_{n}.h")


def generate_all_test_cases():
    """生成所有测试用例"""
    fft_points = [128, 256, 512, 1024, 2048]
    output_dir = "main/test_data"

    for n in fft_points:
        generate_fft_test_case(n, output_dir)
        generate_rfft_test_case(n, output_dir)


if __name__ == "__main__":
    generate_all_test_cases()
