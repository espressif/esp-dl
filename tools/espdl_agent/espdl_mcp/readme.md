# ESP-DL-MCP 配置指南

## 简介
`espdl_mcp.py` 是一个基于 `FastMCP` 的工具，提供了两个主要功能：
1. **`build_espdl_op`**: 编译指定的 ESP-DL 示例和模型。
2. **`test_espdl_op`**: 运行针对指定操作符的 pytest 测试。

## 前置条件
1. 确保已安装 Python 3.x。
2. 确保已安装 `FastMCP` 库。
3. 确保已正确设置 `ESP_DL_ROOT` 环境变量（指向 ESP-DL 的根目录）。

