#!/usr/bin/env python3
"""
ESPDL-MCP
Provides three tools: generate_espdl_op_test_cases, build_espdl_op and test_espdl_op
"""
import os, glob, stat
import subprocess
from pathlib import Path
import shlex
from mcp.server.fastmcp import FastMCP
from typing import List, Optional

# ---------------- Initialization ----------------
TIMEOUT = 1200  # 20 minutes
mcp = FastMCP("espdl_mcp")

# If the ESP-DL root directory is not in the current working directory, modify here
ESP_DL_ROOT = Path(os.environ.get("ESP_DL_ROOT", ".")).resolve()
IMAGE = os.environ.get("ESP_DL_IMAGE", "espdl/idf-ppq:latest")
HOST_DL_ROOT = ESP_DL_ROOT  # host path
CTN_DL_ROOT = Path("/esp-dl")  # espdl docker path

# ---------------- Tool 1 ----------------


def available_ports():
    """Return ['/dev/ttyUSB0', '/dev/ttyACM0', ...] after filtering by permissions."""
    ports = []
    for p in glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"):
        try:
            # Exists + character device + current user has read/write permissions
            if (
                os.path.exists(p)
                and stat.S_ISCHR(os.stat(p).st_mode)
                and os.access(p, os.R_OK | os.W_OK)
            ):
                ports.append(p)
        except OSError:
            continue
    return ports


def docker_run(
    cmd: List[str],
    ports: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """
    Execute a command inside the espdl/idf5.4_ppq container.
    cmd  : list of commands to execute (paths inside the container)
    ports: list of serial ports to map, e.g. ["/dev/ttyUSB0", "/dev/ttyACM0"]
    cwd  : working directory inside the container, defaults to CTN_DL_ROOT
    capture: whether to capture output
    """
    cwd = cwd or CTN_DL_ROOT

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "-v",
        f"{HOST_DL_ROOT}:{CTN_DL_ROOT}",
        "-e",
        f"ESP_DL_ROOT={CTN_DL_ROOT}",
        "-w",
        str(cwd),
    ]

    # map serial ports and add dialout group
    if ports:
        for p in ports:
            docker_cmd.extend(["--device", p])
        docker_cmd.extend(["--group-add", "dialout"])

    # 直接列表追加，避免 bash -c 二次解析
    docker_cmd.append(IMAGE)
    docker_cmd.extend(cmd)
    print("[DOCKER]", shlex.join(docker_cmd))
    return subprocess.run(docker_cmd, capture_output=capture, text=True, check=False)


@mcp.tool()
def generate_espdl_op_test_cases(
    op_type: str = "Gemm", target: str = "esp32p4", bits: int = 8
) -> str:
    """
    Generate test cases for the specified operator.
    :param op_type: The operator type to test, default is "Gemm".
    :param target: The target chip, default is "esp32p4".
    :param bits: The quantization type of the operator, default is 8, options are 8(int8), 16(int16) or 32(float)
    :return: A string indicating the test case generation result.
    """
    tool_path = CTN_DL_ROOT / "tools" / "ops_test"
    gen_script = tool_path / "gen_test_cases.py"
    config_file = tool_path / "config" / "op_cfg.toml"
    output_path = CTN_DL_ROOT / "test_apps" / "esp-dl" / "models" / target

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
        completed = docker_run(cmd)
        return f"✅ Test case generation ucceeded\n\n{completed.stdout[-1500:]}"  # Only show the last 1500 characters to avoid excessive output
    except subprocess.CalledProcessError as e:
        return f"❌ Test case generation failed returncode={e.returncode}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"


# ---------------- Tool 2 ----------------
@mcp.tool()
def build_espdl_op(
    op_type: str,
    target: str = "esp32p4",
) -> str:
    """
    Compile the specified ESP-DL example and model.
    :param op_type: The operator type to compile.
    :param target: The target chip, default is "esp32p4".
    :return: A string indicating the compilation result.
    """
    test_path = CTN_DL_ROOT / "test_apps"
    build_script = test_path / "build_apps.py"
    project_root = test_path / "esp-dl"

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
        completed = docker_run(cmd)  # 10分钟超时
        return f"✅ Compilation succeeded\n\n{completed.stdout[-1500:]}"  # Only show the last 1500 characters to avoid excessive output
    except subprocess.CalledProcessError as e:
        return f"❌ Compilation failed returncode={e.returncode}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"


# ---------------- Tool 3 ----------------
@mcp.tool()
def test_espdl_op(
    op_type: str = "Gemm",
    target: str = "esp32p4",
    ports: list[str] = None,
) -> str:
    """
    Flash and run pytest tests for the specified operator.
    :param op_type: The operator type to test, default is "Gemm".
    :param target: The target chip, default is "esp32p4".
    :return: A string indicating the test result.
    """
    test_path = CTN_DL_ROOT / "test_apps"
    pytest_file = test_path / "esp-dl" / "pytest_espdl_op.py"
    gen_op_test_script = test_path / "esp-dl" / "gen_op_test.py"

    # Generate pytest test script
    cmd = [
        "python",
        str(gen_op_test_script),
        "--target",
        target,
        "--env",
        target,
        "--op_type",
        op_type,
        "--pytest_file",
        str(pytest_file),
    ]

    try:
        completed = docker_run(cmd)
    except subprocess.CalledProcessError as e:
        return f"❌ Test failed returncode={e.returncode}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"

    if not ports:
        available = available_ports()
        if not available:
            return "❌ No available serial ports found. Please connect the device and try again."
        ports = available

    cmd = [
        "pytest",
        str(pytest_file),
        "--target",
        target,
        "--env",
        target,
        "--model",
        op_type,
        "-v",  # Verbose output
    ]
    try:
        completed = docker_run(cmd, ports=ports)
        return f"✅ All tests passed\n\n{completed.stdout[-1500:]}"
    except subprocess.CalledProcessError as e:
        return f"❌ Test failed returncode={e.returncode}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"


# ---------------- Entry Point ----------------
if __name__ == "__main__":
    mcp.run()
