#!/usr/bin/env python3
"""
ESP-DL-MCP
Provides two tools: build_espdl_op and test_espdl_op
"""
import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------- Initialization ----------------
mcp = FastMCP("espdl_mcp")

# If the ESP-DL root directory is not in the current working directory, modify here
ESP_DL_ROOT = Path(os.environ.get("ESP_DL_ROOT", ".")).resolve()

# ---------------- Tool 1 ----------------

@mcp.tool()
def generate_espdl_op_test_cases(op_type: str = "Gemm", target: str = "esp32p4", bits: int = 8) -> str:
    """
    Generate test cases for the specified operator.
    :param op_type: The operator type to test, default is "Gemm".
    :param target: The target chip, default is "esp32p4".
    :param bits: The quantization type of the operator, default is 8, options are 8(int8), 16(int16) or 32(float)
    :return: A string indicating the test case generation result.
    """
    tool_path = ESP_DL_ROOT / "tools" / "ops_test"
    gen_script = tool_path / "gen_test_cases.py"
    config_file = tool_path / "config" / "op_cfg.toml"
    if not gen_script.exists():
        return f"❌ Cannot find test case generation script {gen_script}"

    if not config_file.exists():
        return f"❌ Cannot find test case config file {config_file}"

    output_path = ESP_DL_ROOT / "test_apps" / "esp-dl" / "models" / target
    
    cmd = [
        "python",
        str(gen_script),
        "--config",
        str(config_file),
        "--op",
        op_type,
        "--target",
        target,
        "--output-path",
        str(output_path),
        "--bits",
        str(bits),
    ]
    try:
        completed = subprocess.run(
            cmd, cwd=ESP_DL_ROOT, capture_output=True, text=True, check=True
        )
        return f"✅ Test case generation ucceeded\n\n{completed.stdout[-1500:]}"  # Only show the last 1500 characters to avoid excessive output
    except subprocess.CalledProcessError as e:
        return f"❌ Test case generation failed returncode={e.returncode}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"

# ---------------- Tool 2 ----------------
@mcp.tool()
def build_espdl_op(
    op_type: str,
    target: str = "esp32s3",
) -> str:
    """
    Compile the specified ESP-DL example and model.
    :param op_type: The operator type to compile.
    :param target: The target chip, default is "esp32s3".
    :return: A string indicating the compilation result.
    """
    test_path = ESP_DL_ROOT / "test_apps"
    build_script = test_path / "build_apps.py"
    project_root = test_path / "esp-dl"
    if not build_script.exists():
        return f"❌ Cannot find build script {build_script}"

    if not project_root.exists():
        return f"❌ Cannot find project root directory {project_root}"

    cmd = [
        "python",
        str(build_script),
        str(project_root),
        "-op",
        op_type,
        "-t",
        target,
        "-vv",
    ]
    try:
        completed = subprocess.run(
            cmd, cwd=ESP_DL_ROOT, capture_output=True, text=True, check=True
        )
        return f"✅ Compilation succeeded\n\n{completed.stdout[-1500:]}"  # Only show the last 1500 characters to avoid excessive output
    except subprocess.CalledProcessError as e:
        return f"❌ Compilation failed returncode={e.returncode}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"


# ---------------- Tool 3 ----------------
@mcp.tool()
def test_espdl_op(
    op_type: str = "Gemm",
    target: str = "esp32p4",
) -> str:
    """
    Flash and run pytest tests for the specified operator.
    :param op_type: The operator type to test, default is "Gemm".
    :param target: The target chip, default is "esp32p4".
    :return: A string indicating the test result.
    """
    test_path = ESP_DL_ROOT / "test_apps"
    pytest_file = test_path / "esp-dl" / "pytest_espdl_op.py"
    gen_op_test_script = test_path / "esp-dl" / "gen_op_test.py"
    if not gen_op_test_script.exists():
        return f"❌ Cannot find test generation script {gen_op_test_script}"

    # Generate pytest test script
    cmd = [
        "pytest",
        str(gen_op_test_script),
        "--target",
        target,
        "--env",
        target,
        "--op_type",
        op_type,
        "-v",  # Verbose output
    ]

    if not pytest_file.exists():
        return f"❌ Cannot find test script {pytest_file}"

    cmd = [
        "pytest",
        str(pytest_file),
        "--target",
        target,
        "--env",
        target,
        "--op_type",
        op_type,
        "-v",  # Verbose output
    ]
    try:
        completed = subprocess.run(
            cmd, cwd=ESP_DL_ROOT, capture_output=True, text=True, check=True
        )
        return f"✅ All tests passed\n\n{completed.stdout[-1500:]}"
    except subprocess.CalledProcessError as e:
        return f"❌ Test failed returncode={e.returncode}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"


# ---------------- Entry Point ----------------
if __name__ == "__main__":
    mcp.run()
