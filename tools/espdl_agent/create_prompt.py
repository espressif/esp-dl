import argparse
import os
from pathlib import Path
import re

from onnx_op import extract_operator_content, return_operator_info

# ---------- Configuration ----------

DEFAULT_ESP_DL = Path(__file__).resolve().parent / "../../esp-dl"
ONNX_OP_URL_TMPL = "https://onnx.ai/onnx/operators/onnx__{op}.html"
HEADERS_SUBPATH = Path("dl") / "module" / "include"
# ------------------------------------


def esp_dl_root() -> Path:
    """Returns the esp-dl root directory"""
    if "ESP_DL_ROOT" in os.environ:
        return Path(os.environ["ESP_DL_ROOT"]).resolve()

    return DEFAULT_ESP_DL


def camel_to_snake(name):
    if name == "PRelu":
        return "prelu"
    if name == "MatMul":
        return "matmul"

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def gen_op_create_prompt(
    op_type: str,
    op_info: str,
    ref_ops: list[str],
    support_dtype="int8,int16,float",
    target="esp32p4",
) -> str:
    # 1. Use the provided ONNX operator definition
    onnx_definition = op_info

    quantized_bits = []
    for dtype in support_dtype.split(","):
        if dtype == "int8":
            quantized_bits.append("8")
        elif dtype == "int16":
            quantized_bits.append("16")
        elif dtype == "float":
            quantized_bits.append("32")
        else:
            raise ValueError(f"Unsupported data type: {dtype}")
    quantized_bits = ",".join(quantized_bits)

    # 2. Read reference operator header files
    ref_content = ""
    ref_files = [f"dl_module_{camel_to_snake(ref_op)}.hpp" for ref_op in ref_ops]
    root_path = esp_dl_root()
    for ref_op in ref_ops:
        file_name = f"dl_module_{camel_to_snake(ref_op)}.hpp"
        file_path = root_path / HEADERS_SUBPATH / file_name
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                ref_content += f"// Reference implementation for {ref_op}:\n"
                ref_content += f.read() + "\n\n"
        except Exception as e:
            ref_content += f"Error reading reference operator {ref_op}: {str(e)}\n\n"

    # 3. Compose the prompt
    prompt = f"""
**Objective**: Implement the {op_type} operator in the ESP-DL project, ensuring compatibility with ESP32-S3 and ESP32-P4 hardware.

**Step 1: Research Implementation Approaches**
- Study how {op_type} operator is implemented in ncnn, tflite-micro, and Tengine.
- Focus on CPU-based implementations and optimization strategies.
- Note: ESP32-S3 and ESP32-P4 are CPU-only with custom SIMD instructions. Ignore NPU/GPU optimizations.

**Step 2: Design C++ Interface**
- Inherit from `Module` class in `dl_module_base.hpp`.
- Create C++ interface functions for {op_type} based on ONNX specification:

**ONNX Operator Definition**:
{onnx_definition}

**Reference Implementations**:
{ref_files}

**Requirements**:
- Analyze ONNX specification thoroughly (inputs, outputs, attributes).
- Design C++ class interface following ESP-DL conventions.
- Include proper constructors, destructors, and member functions.
- Consider memory allocation needs and computational complexity.

**Step 3: Implement Core Operator Logic**
- Choose implementation pattern based on complexity:
  - Complex operators: Follow `dl_module_conv.hpp` pattern
  - Simple operators: Follow `dl_module_sqrt.hpp` pattern
- Must support: {support_dtype}
- Implement core computation logic
- Handle edge cases and error conditions
- Optimize for ESP32 hardware capabilities
- Include proper memory management

**Step 4: Register Operator**
- Add operator to appropriate registry in `dl_module_creator.py`
- Ensure proper instantiation and configuration
- Maintain compatibility with existing registration system
- Follow existing naming conventions

**Step 5: Generate Test Cases**
1. Add test script for {op_type} in `tools/ops_test/`
    - If torch has a {op_type} interface, modify `tools/ops_test/torch_ops_test.py` to add {op_type} tests, Use torch {op_type} interface for implementation.
    Otherwise, modify `tools/ops_test/onnx_ops_test.py` and use ONNX {op_type} interface for implementation.
    - Modify `tools/ops_test/config/op_cfg.toml` with test configurations
    - Create 3-5 test cases covering various configuration and Edge cases
    - Follow existing configuration formats
    - Validate across different input sizes and data types
2. Generate test cases using `generate_espdl_op_test_cases` tool by espdl_mcp
   - op_type: {op_type}, target: {target}, bits: {quantized_bits}

**Step 6: Testing Cycle**
esp-dl operator testing uses the esp-dl/test_apps/esp-dl project. For convenience, two tools have been defined using esp-mcp for project compilation, flashing, and testing.
Please use the tools built with espdl-mcp directly instead of the original esp-idf commands.

1. Build test cases using `build_espdl_op`
   input parameters: "op_type": {op_type}, "target": {target}

2. Run tests using `test_espdl_op`
   input parameters: "op_type": {op_type}, "target": {target}

**Success Criteria**: All test cases must pass.

**Iteration Process**: If tests fail, return to code modifications and repeat Steps 6 until all tests pass.

"""

    return prompt


def gen_op_optimize_prompt(op_type: str) -> str:

    prompt = f"""
**Objective**: Optimize the `{op_type}` operator in the ESP-DL project, ensuring compatibility with ESP32-S3 and ESP32-P4 hardware.

---

**Step 1: Research Implementation Approaches**
- Study how the `{op_type}` operator is implemented in **ncnn**, **tflite-micro**, and **Tengine**.
- Focus on **CPU-based implementations** and related optimization strategies.
- Note: ESP32-S3 and ESP32-P4 are **CPU-only** platforms with custom SIMD instructions. Ignore NPU/GPU-specific optimizations.

---

**Step 2: Optimize the `{op_type}` Operator**
- Review the current implementation of `{op_type}` in **esp-dl**.
- If `{op_type}` can be optimized using **SIMD**, follow the optimization approach used in esp-dl operators like **Add** or **Conv**, and write corresponding assembly functions.
- If **SIMD is not applicable**, optimize and refactor the operator using other suitable methods.
- Ensure proper **memory management** throughout the implementation.

---

**Step 3: Testing Cycle**
1. Generate test cases using the `generate_espdl_op_test_cases` tool.
   - Must support all original quantized data types
2. Build test cases using the `build_espdl_op` tool.
3. Run tests using the `test_espdl_op` tool.

**Success Criteria**: All test cases must pass.

**Iteration Process**:
If any test fails, return to **Step 2** for code modifications, then repeat **Step 3** until all tests pass.

---

Let me know if you'd like this structured as a numbered checklist or formatted for a specific agent workflow.
"""

    return prompt


# 示例用法
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="从ONNX Markdown文档中提取算子定义和测试用例"
    )
    parser.add_argument("operator", help="要查找的算子名称")
    parser.add_argument("--create", action="store_true", help="是否创建算子定义")
    parser.add_argument(
        "--markdown", "-m", help="Markdown文件路径", default="Operators.md"
    )

    args = parser.parse_args()

    ref_ops = ["Base", "Add", "Conv", "Sqrt"]

    if args.create:
        operator_data = extract_operator_content(args.markdown, args.operator)
        op_info = return_operator_info(operator_data)
        prompt = gen_op_create_prompt(args.operator, op_info, ref_ops)
    else:
        prompt = gen_op_optimize_prompt(args.operator)

    print(prompt)
