# ESP-DL Agent Tool

## Overview

The ESP-DL Agent Tool is a specialized system designed to assist with implementing and optimizing neural network operators for the ESP-DL framework. This toolset provides automation and guidance for developers to efficiently create, test, and validate operators in the ESP-DL framework.

The project combines several utilities to streamline the operator development process:
- **Prompt Generation**: Creates detailed development instructions for implementing or optimizing operators
- **MCP Tools**: Provides automated testing, building, and validation capabilities via a Model Context Protocol
- **ONNX Integration**: Extracts operator definitions and test cases from ONNX documentation

## Prerequisites

Before using the ESP-DL Agent Tool, ensure you have:
- Docker installed on your system
- Python 3.8+ with pip
- ESP development board (ESP32-S3, ESP32-P4, or ESP32) connected via USB for hardware testing
- Appropriate permissions to access serial ports (may require adding user to `dialout` group on Linux)

## 1. Building ESP-DL Docker

The ESP-DL agent uses Docker as its development and testing environment. All necessary dependencies have already been installed, including esp-idf, PyTorch, esp-ppq, and other tools. To build the Docker image:

```bash
# Navigate to the docker directory
cd docker

# Build the Docker image
docker build -t espdl/idf-ppq:latest .

# This process may take 20-30 minutes to complete
```

## 2. Setting up ESPDL-MCP in Agent

The ESPDL-MCP server provides three essential tools for operator development

### Available Tools

1. **generate_espdl_op_test_cases**
   - Generates test cases for the specified operator
   - Parameters:
     - `op_type`: The operator type to test (default: "Gemm")
     - `target`: The target chip (default: "esp32p4", options: "esp32", "esp32s3", "esp32p4")
     - `bits`: Quantization type (default: 8, options: 8, 16, 32)

2. **build_espdl_op**
   - Compiles the specified ESP-DL operator test
   - Parameters:
     - `op_type`: The operator type to compile
     - `target`: The target chip (default: "esp32p4")

3. **test_espdl_op**
   - Flashes and runs pytest tests for the specified operator on actual hardware
   - Parameters:
     - `op_type`: The operator type to test (default: "Gemm")
     - `target`: The target chip (default: "esp32p4")
     - `ports`: List of serial ports to use (automatically detected if not provided)

### Setting the MCP Server

The JSON snippet below is an example of how you might configure this espdl-mcp server within an agent system that supports the Model Context Protocol. The exact configuration steps and format may vary depending on the specific agent system you are using. 

```json
  "mcpServers": {
    "espdl-mcp": {
      "command": "python",
      "args": [
        "/your-esp-dl-project-path/tools/espdl_agent/espdl_mcp.py"
      ],
      "env": {
        "ESP_DL_TOOR": "your-esp-dl-project-path",
        "ESP_DL_IMAGE": "espdl/idf-ppq:latest"
      },
      "timeout": 500000
    }
  }
```


## 3. QwenCode Examples

[QwenCode CLI](https://github.com/QwenLM/qwen-code) is a powerful and free command-line AI coding agent. Let's use QwenCode to create a ScatterND operator:


### Create a prompt

```bash
python create_prompt.py ScatterND --create
```

### Set espdl-mcp

Open `~/.qwen/settings.json` and add the following content to it.
```json
  "mcpServers": {
    "espdl-mcp": {
      "command": "python",
      "args": [
        "/your-esp-dl-project-path/tools/espdl_agent/espdl_mcp.py"
      ],
      "env": {
        "ESP_DL_TOOR": "your-esp-dl-project-path",
        "ESP_DL_IMAGE": "espdl/idf-ppq:latest"
      },
      "timeout": 500000
    }
  }
```

### Start coding

Copy the prompt into QwenCode and [ScatterND_QwenCode.md](./ScatterND_QwenCode.md) is the full log of Qwen Code’s generation process.



## 4. Troubleshooting

1. **Serial Port Access Issues**: Ensure your user has permissions to access serial ports. On Linux, you might need to add your user to the `dialout` group:
   ```bash
   sudo usermod -a -G dialout $USER
   ```
   Then log out and log back in.

2. **Docker Permission Issues**: Make sure your user is in the `docker` group:
   ```bash
   sudo usermod -aG docker $USER
   ```

3. **Missing ESP-DL Directory**: Ensure the ESP-DL repository is properly cloned and the `ESP_DL_ROOT` environment variable points to the correct location.

