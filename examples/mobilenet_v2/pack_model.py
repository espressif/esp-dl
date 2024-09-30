import argparse
import shutil
import struct
from pathlib import Path


def struct_pack_string(string, max_len=None):
    """
    pack string to binary data.
    if max_len is None, max_len = len(string) + 1
    else len(string) < max_len, the left will be padded by struct.pack('x')

    string: input python string
    max_len: output
    """

    if max_len == None:
        max_len = len(string)
    else:
        assert len(string) <= max_len

    left_num = max_len - len(string)
    out_bytes = None
    for char in string:
        if out_bytes == None:
            out_bytes = struct.pack("b", ord(char))
        else:
            out_bytes += struct.pack("b", ord(char))
    for i in range(left_num):
        out_bytes += struct.pack("x")
    return out_bytes


def read_data(filename):
    """
    Read binary data, like index and mndata
    """
    data = None
    with open(filename, "rb") as f:
        data = f.read()
    return data


def pack_models(model_file_or_path, out_file="models.espdl"):
    """
    Pack all models into one binary file by the following format:
    {
        "PDL1": char[4]
        model_num: uint32
        model1_data_offset: uint32
        model1_name_offset: uint32
        model1_name_length: uint32
        model2_data_offset: uint32
        model2_name_offset: uint32
        model2_name_length: uint32
        ...
        model1_name,
        model2_name,
        ...
        model1_data,
        model2_data,
        ...
    }model_pack_t


    model_path: the path of models
    out_file: the ouput binary filename
    """
    model_path = None
    model_file_or_path = Path(model_file_or_path)
    if model_file_or_path.is_file():
        shutil.copyfile(model_file_or_path, out_file)
        return
    else:
        model_path = model_file_or_path

    model_names = []
    model_bins = []
    name_length = 0
    for model_file in model_path.glob("*.espdl"):
        model_names.append(model_file.name)
        model_bins.append(read_data(model_file))
        name_length += len(model_file.name)
        print(model_file.name)

    model_num = len(model_names)
    header_bin = struct_pack_string("PDL1", 4)
    header_bin += struct.pack("I", model_num)
    name_offset = 4 + 4 + model_num * 12
    data_offset = name_offset + name_length
    name_bin = None
    data_bin = None
    for idx, name in enumerate(model_names):
        if not name_bin:
            name_bin = struct_pack_string(name, len(name))  # + model name
        else:
            name_bin += struct_pack_string(name, len(name))
            name_offset += len(model_names[idx - 1])

        if not data_bin:
            data_bin = model_bins[idx]
        else:
            data_bin += model_bins[idx]
            data_offset += len(model_bins[idx - 1])

        header_bin += struct.pack("I", data_offset)
        header_bin += struct.pack("I", name_offset)
        header_bin += struct.pack("I", len(name))
    out_bin = header_bin + name_bin + data_bin
    with open(out_file, "wb") as f:
        f.write(out_bin)


if __name__ == "__main__":
    # input parameter
    parser = argparse.ArgumentParser(description="esp-dl model package tool")
    parser.add_argument("-m", "--model_path", type=str, help="the path of model files")
    parser.add_argument(
        "-o",
        "--out_file",
        type=str,
        default="models.espdl",
        help="the path of binary file",
    )
    args = parser.parse_args()

    pack_models(args.model_path, out_file=args.out_file)
