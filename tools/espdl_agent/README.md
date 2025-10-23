# ESP-DL Agent Tool

## Overview

The ESP-DL Agent tool is a utility designed to assist developers in implementing new operators for the ESP-DL framework. It extracts operator definitions from ONNX documentation and generates detailed prompts to guide the creation of ESP-DL operator implementations that are compatible with ESP32 chips.

The tool consists of two main components:
- `get_onnx_def.py`: Extracts operator definitions and test cases from ONNX Markdown documentation
- `create_prompt.py`: Generates detailed implementation prompts based on the extracted ONNX specifications

This tool streamlines the process of adding new operators to ESP-DL by providing structured guidance for implementing operators that follow ESP-DL conventions and are optimized for ESP32 hardware.

## Installation

### Prerequisites
- Python 3.7 or higher
- ESP-DL framework

### Dependencies
First, install the required dependencies:

```bash
pip install -r requirements.txt
```

If there's no requirements.txt file in this directory, the tool requires the following basic Python packages:
- argparse
- pathlib
- re (regular expressions, part of Python standard library)
- Standard Python libraries (os, sys)

### Setup

1. Ensure you have a copy of the ESP-DL repository
2. Set the `ESP_DL_ROOT` environment variable to point to your esp-dl repository:
   ```bash
   export ESP_DL_ROOT=/path/to/your/esp-dl
   ```
   Or the tool will default to using `../../esp-dl` relative to this directory

3. Make sure the `Operators.md` file is available (it contains the ONNX operator definitions)

## Usage

### Command Line Usage

The tool can be used from the command line in two ways:

#### 1. Extract ONNX Operator Definition

```bash
python get_onnx_def.py <path_to_operators_md> <operator_name>
```

Example:
```bash
python get_onnx_def.py Operators.md Add
```

This will extract and format the definition for the Add operator from the Operators.md file.

#### 2. Generate Implementation Prompt

```bash
python create_prompt.py <operator_name> <path_to_operators_md>
```

Example:
```bash
python create_prompt.py Conv Operators.md
```

This will generate a detailed prompt for implementing the Conv operator in ESP-DL, including reference implementations and requirements.

### Programmatic Usage

You can also import and use the functions in your own scripts:

```python
from get_onnx_def import extract_operator_content, return_operator_info
from create_prompt import gen_op_create_prompt

# Extract operator information
operator_data = extract_operator_content("Operators.md", "Add")
op_info = return_operator_info(operator_data)

# Generate implementation prompt
reference_ops = ["Base", "Add", "Conv", "Sqrt"]
prompt = gen_op_create_prompt("Add", op_info, reference_ops)
print(prompt)
```

## Key Features

- **Operator Extraction**: Automatically parses ONNX operator specifications from Markdown documentation
- **Detailed Implementation Prompts**: Generates comprehensive instructions for implementing operators in ESP-DL
- **Reference Implementation Analysis**: Incorporates patterns from existing operators (Base, Add, Conv, Sqrt)
- **Hardware Optimization Guidance**: Includes specific instructions for ESP32 optimization
- **Test Case Generation**: Provides guidance for creating test cases for new operators
- **Multi-step Process**: Breaks down the implementation into 4 clear phases:
  1. C++ interface design based on ONNX specification
  2. Core operator implementation
  3. Operator registration
  4. Test case development

## Output Format

The generated prompts contain:

1. **ONNX specification** for the target operator
2. **Reference implementations** from similar operators
3. **Step-by-step instructions** for:
   - Inheriting from the Module base class
   - Implementing the core operator logic
   - Registering the operator
   - Adding test cases
4. **Coding standards** and ESP32-specific requirements
5. **Memory management** and optimization guidelines

## Integration with ESP-DL Development

The generated prompts are designed to help developers follow the standard process for adding new operators to ESP-DL:

1. Implement the C++ interface in the `esp-dl/dl/module` directory
2. Register the operator in `dl_module_creator.py`
3. Add tests in `tools/ops_test/torch_ops_test.py`
4. Configure tests in `tools/ops_test/config/op_cfg.toml`

## Example Output

Running `python create_prompt.py Add` will generate a detailed prompt with instructions for implementing the Add operator, including requirements for data type support (int8, int16, float), memory optimization for ESP32, and test case specifications.