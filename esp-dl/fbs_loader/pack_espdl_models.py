import argparse
import hashlib
import shutil
import struct
from pathlib import Path

# PDL3 header layout (in bytes):
#   magic          : char[4]    offset 0
#   package_version: char[16]   offset 4
#   package_size   : uint32     offset 20
#   package_sha256 : uint8[32]  offset 24
#   model_num      : uint32     offset 56
# -> header size = 60 bytes, model entries start at offset 60.
PDL3_VERSION_MAX_LEN = 15  # not counting the terminating '\0'
PDL3_VERSION_FIELD_SIZE = 16
PDL3_SHA256_OFFSET = 24
PDL3_SHA256_SIZE = 32
PDL3_HEADER_SIZE = 60


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


def get_model_format(filename):
    """
    Get model format, EDL1 or EDL2
    """
    with open(filename, "rb") as f:
        data = f.read(4)
        format = data.decode("utf-8")
        if format != "EDL1" and format != "EDL2":
            raise RuntimeError("Wrong model format.")
        return format


def read_data(filename, format):
    """
    Read binary data, like index and mndata
    """
    data = None
    with open(filename, "rb") as f:
        data = f.read()
    if format == "EDL2" and len(data) % 16 != 0:
        padding = 16 - len(data) % 16
        data += struct.pack("x") * padding
    return data


def collect_model_files(model_path_or_dir):
    """
    Resolve the input model path(s) into a sorted list of model file Paths.

    Returns (model_files, single_file):
      - model_files: the list of model files to pack. A single input file is
        included here as well, so it can be packed (e.g. into PDL3) for SHA256
        integrity verification instead of being skipped.
    """
    if len(model_path_or_dir) == 1:
        single = Path(model_path_or_dir[0])
        if single.is_file():
            return [single]
        return sorted(list(single.glob("*.espdl")))

    model_files = []
    for model_path in sorted(model_path_or_dir):
        model_path = Path(model_path)
        assert model_path.is_file(), f"invalid model_path.{str(model_path)}"
        model_files.append(model_path)
    return model_files


def pack_models_pdl3(model_files, out_file="models.espdl", package_version="v1.0"):
    """
    Pack all models into one binary file with the PDL3 format:
    {
        magic          : char[4] = "PDL3"
        package_version : char[16]    # ASCII, '\0' terminated, max 15 chars
        package_size    : uint32      # valid byte count of the whole package
        package_sha256  : uint8[32]   # integrity digest, see rule below
        model_num       : uint32
        model1_data_offset: uint32    # 16-byte aligned
        model1_name_offset: uint32
        model1_name_length: uint32
        ...
        model1_name,
        model2_name,
        ...
        zero padding
        model1_data(format:FBS_FILE_FORMAT_EDL2)
        model2_data(format:FBS_FILE_FORMAT_EDL2)
        ...
    }

    package_sha256 = SHA256(package[0:package_size]) where the 32 bytes of the
    package_sha256 field itself are treated as all zeros.
    """
    assert (
        len(str(package_version)) <= PDL3_VERSION_MAX_LEN
    ), f"package_version must be <= {PDL3_VERSION_MAX_LEN} ASCII characters."

    # PDL3 sub-models use the EDL2 (16-byte aligned) layout.
    model_formats = [get_model_format(file) for file in model_files]
    for fmt in model_formats:
        if fmt != "EDL2":
            raise RuntimeError("PDL3 only supports EDL2 sub-models.")

    model_names = []
    model_bins = []
    name_length = 0
    for model_file in model_files:
        model_names.append(model_file.name)
        model_bins.append(read_data(model_file, "EDL2"))
        name_length += len(model_file.name)

    model_num = len(model_names)
    name_offset = PDL3_HEADER_SIZE + model_num * 12
    data_offset = (name_offset + name_length + 15) & ~15
    padding_bin = struct.pack("x") * (data_offset - name_offset - name_length)

    entry_bin = b""
    name_bin = b""
    data_bin = b""
    cur_name_offset = name_offset
    cur_data_offset = data_offset
    for idx, name in enumerate(model_names):
        entry_bin += struct.pack("I", cur_data_offset)
        entry_bin += struct.pack("I", cur_name_offset)
        entry_bin += struct.pack("I", len(name))
        name_bin += struct_pack_string(name, len(name))
        data_bin += model_bins[idx]
        cur_name_offset += len(name)
        cur_data_offset += len(model_bins[idx])

    package_size = data_offset + sum(len(b) for b in model_bins)

    header_bin = struct_pack_string("PDL3", 4)
    header_bin += struct_pack_string(str(package_version), PDL3_VERSION_FIELD_SIZE)
    header_bin += struct.pack("I", package_size)
    header_bin += struct.pack("x") * PDL3_SHA256_SIZE  # sha256 placeholder (zeros)
    header_bin += struct.pack("I", model_num)

    out_bin = bytearray(header_bin + entry_bin + name_bin + padding_bin + data_bin)
    assert (
        len(out_bin) == package_size
    ), f"package layout mismatch: {len(out_bin)} != {package_size}"

    # SHA256 is computed over the whole package with the digest field zeroed,
    # which is already the case in out_bin.
    package_sha256 = hashlib.sha256(bytes(out_bin)).digest()
    out_bin[PDL3_SHA256_OFFSET : PDL3_SHA256_OFFSET + PDL3_SHA256_SIZE] = package_sha256

    output_path = Path(out_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(out_bin)

    print("PDL3 package generated:")
    print(f"  package_version: {package_version}")
    print(f"  package_size   : {package_size}")
    print(f"  package_sha256 : {package_sha256.hex()}")
    print(f"  model_num      : {model_num}")
    for name in model_names:
        print(f"  model_name     : {name}")


def pack_models(
    model_path_or_dir, out_file="models.espdl", pack_format=None, package_version="v1.0"
):
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

    or

    {
        "PDL2": char[4]
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
        zero padding
        model1_data
        zero padding
        model2_data
        zero padding
    }

    When pack_format is "PDL3", the PDL3 format (with package_version / package_size
    / package_sha256 integrity fields) is generated instead. See pack_models_pdl3.

    model_path: the path of models
    out_file: the output binary filename
    pack_format: None for the legacy PDL1/PDL2 behavior, or "PDL3".
    package_version: package version string for PDL3 (max 15 ASCII chars).
    """

    model_files = collect_model_files(model_path_or_dir)
    if pack_format == "PDL3":
        # A single file is also packed (with header + SHA256) for integrity verification.
        pack_models_pdl3(
            model_files, out_file=out_file, package_version=package_version
        )
        return

    model_formats = [get_model_format(file) for file in model_files]
    format = model_formats[0]
    for i in range(1, len(model_formats)):
        if format != model_formats[i]:
            raise RuntimeError("All packed model format should be same.")

    model_names = []
    model_bins = []
    name_length = 0
    for model_file in model_files:
        model_names.append(model_file.name)
        model_bins.append(read_data(model_file, format))
        name_length += len(model_file.name)
        print(model_file.name)

    model_num = len(model_names)
    if format == "EDL1":
        header_bin = struct_pack_string("PDL1", 4)
    else:
        header_bin = struct_pack_string("PDL2", 4)
    header_bin += struct.pack("I", model_num)
    name_offset = 4 + 4 + model_num * 12
    if format == "EDL1":
        data_offset = name_offset + name_length
        padding_bin = b""
    else:
        data_offset = (name_offset + name_length + 15) & ~15
        padding_bin = struct.pack("x") * (data_offset - name_offset - name_length)
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
    out_bin = header_bin + name_bin + padding_bin + data_bin
    output_path = Path(out_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(out_bin)


if __name__ == "__main__":
    # input parameter
    parser = argparse.ArgumentParser(description="esp-dl model package tool")
    parser.add_argument(
        "-m", "--model_path", type=str, nargs="+", help="the path of model files"
    )
    parser.add_argument(
        "-o",
        "--out_file",
        type=str,
        default="models.espdl",
        help="the path of binary file",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="PDL3",
        choices=["PDL3", "PDL2"],
        help="package format. Default is PDL3; "
        "use PDL3 to generate an integrity-verifiable package.",
    )
    parser.add_argument(
        "--package_version",
        type=str,
        default="v1.0",
        help="package version string for PDL3 (max 15 ASCII characters).",
    )
    args = parser.parse_args()

    pack_models(
        args.model_path,
        out_file=args.out_file,
        pack_format=args.format,
        package_version=args.package_version,
    )
