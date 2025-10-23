import argparse
import re


def extract_operator_content(markdown_file, operator_name):
    """
    从Markdown文件中提取指定算子的定义和测试用例内容

    Args:
        markdown_file (str): Markdown文件路径
        operator_name (str): 要查找的算子名称

    Returns:
        dict: 包含算子定义和测试用例的字典
    """
    with open(markdown_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 构建正则表达式模式来匹配算子部分
    # 匹配格式：### <a name="算子名"></a><a name="小写算子名">**算子名**</a>
    operator_pattern = rf'### <a name="{operator_name}"></a><a name="{operator_name.lower()}">\*\*{operator_name}\*\*</a>(.*?)(?=### <a name=|$)'

    match = re.search(operator_pattern, content, re.DOTALL)

    if not match:
        return {"error": f"未找到算子 '{operator_name}' 的定义"}

    operator_content = match.group(1).strip()

    # 提取各个部分
    result = {"operator_name": operator_name, "full_content": operator_content}

    # Summary通常在算子标题后，Version信息前
    summary_match = re.search(
        r"^\s*(.*?)(?=#### Version|\Z)", operator_content, re.DOTALL
    )
    if summary_match:
        summary_text = summary_match.group(1).strip()
        # 清理HTML标签
        summary_text = re.sub(r"<[^>]+>", "", summary_text)
        # 清理多余的空行和空格
        summary_text = re.sub(r"\n\s*\n", "\n\n", summary_text)
        result["summary"] = summary_text.strip()

    # 提取版本信息
    version_match = re.search(
        r"#### Version\s*(.*?)(?=####|$)", operator_content, re.DOTALL
    )
    if version_match:
        result["version_info"] = version_match.group(1).strip()

    # 提取输入信息
    inputs_match = re.search(
        r"#### Inputs\s*<dl>(.*?)</dl>", operator_content, re.DOTALL
    )
    if inputs_match:
        result["inputs"] = inputs_match.group(1).strip()

    # 提取输出信息
    outputs_match = re.search(
        r"#### Outputs\s*<dl>(.*?)</dl>", operator_content, re.DOTALL
    )
    if outputs_match:
        result["outputs"] = outputs_match.group(1).strip()

    # 提取类型约束
    type_constraints_match = re.search(
        r"#### Type Constraints\s*<dl>(.*?)</dl>", operator_content, re.DOTALL
    )
    if type_constraints_match:
        result["type_constraints"] = type_constraints_match.group(1).strip()

    # 提取测试用例
    examples_match = re.search(
        r"#### Examples\s*(.*?)(?=####|$)", operator_content, re.DOTALL
    )
    if examples_match:
        examples_content = examples_match.group(1).strip()

        # 提取各个测试用例
        test_cases = re.findall(
            r"<details>\s*<summary>(.*?)</summary>\s*```python(.*?)```\s*</details>",
            examples_content,
            re.DOTALL,
        )

        result["test_cases"] = []
        for test_name, test_code in test_cases:
            result["test_cases"].append(
                {"name": test_name.strip(), "code": test_code.strip()}
            )

    return result


def return_operator_info(operator_data):
    """Format operator information as a string and return it in English"""
    output_lines = []

    if "error" in operator_data:
        output_lines.append(f"Error: {operator_data['error']}")
        return "\n".join(output_lines)

    output_lines.append("=" * 80)
    output_lines.append(f"Operator Name: {operator_data['operator_name']}")
    output_lines.append("=" * 80)

    if "summary" in operator_data:
        output_lines.append("\nSummary:")
        output_lines.append("-" * 40)
        output_lines.append(operator_data["summary"])

    # Commented out version information section
    # if "version_info" in operator_data:
    #     output_lines.append("\nVersion Information:")
    #     output_lines.append("-" * 40)
    #     output_lines.append(operator_data["version_info"])

    if "inputs" in operator_data:
        output_lines.append("\nInput Parameters:")
        output_lines.append("-" * 40)
        output_lines.append(operator_data["inputs"])

    if "outputs" in operator_data:
        output_lines.append("\nOutput Parameters:")
        output_lines.append("-" * 40)
        output_lines.append(operator_data["outputs"])

    # Commented out type constraints section
    # if "type_constraints" in operator_data:
    #     output_lines.append("\nType Constraints:")
    #     output_lines.append("-" * 40)
    #     output_lines.append(operator_data["type_constraints"])

    if "test_cases" in operator_data:
        output_lines.append(f"\nTest Cases ({len(operator_data['test_cases'])}):")
        output_lines.append("-" * 40)
        for i, test_case in enumerate(operator_data["test_cases"], 1):
            output_lines.append(f"\n{i}. {test_case['name']}:")
            output_lines.append(f"Code:\n{test_case['code']}\n")

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="从ONNX Markdown文档中提取算子定义和测试用例"
    )
    parser.add_argument("markdown_file", help="Markdown文件路径")
    parser.add_argument("operator_name", help="要查找的算子名称")

    args = parser.parse_args()

    try:
        operator_data = extract_operator_content(args.markdown_file, args.operator_name)
        # print(operator_data["full_content"])
        print(return_operator_info(operator_data))
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{args.markdown_file}'")
    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == "__main__":
    # 如果不使用命令行参数，可以直接在这里修改文件路径和算子名
    # markdown_file = "your_file.md"
    # operator_name = "Add"
    # operator_data = extract_operator_content(markdown_file, operator_name)
    # print_operator_info(operator_data)

    main()
