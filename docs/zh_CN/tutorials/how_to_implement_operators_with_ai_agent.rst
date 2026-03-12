使用 AI Agent 自动实现算子
==========================

:link_to_translation:`en:[English]`

本文档介绍如何在各种 Coding Agent 工具（如 Claude、Cursor、OpenCode 等）中安装和使用 ``espdl-operator`` skill，以便在 ESP-DL 框架中完成对神经网络算子的自动化实现。**以下说明以 Linux 环境为例。**

.. contents::
   :local:
   :depth: 1

什么是 ``espdl-operator`` skill
---------------------------------

``espdl-operator`` 是一个供 Coding Agent 使用的自动化开发 skill，用于在 ESP-DL 框架中实现、测试和优化神经网络算子。当你向 Coding Agent（如 Claude、Cursor、OpenCode 等）提出算子相关请求时，该 skill 会指导 AI 自动完成以下工作：

**Coding Agent 将自动为你：**

1. **分析算子需求** - 解析 ONNX 算子规范，确定算子类型和数据类型支持
2. **生成 esp-dl C++ 代码** - 自动创建 Module 层和 Base 层的头文件/实现文件
3. **修改 esp-ppq 量化配置** - 在量化工具中注册算子支持，配置 layout pattern
4. **创建测试用例** - 生成 PyTorch/ONNX 测试模型，配置测试参数
5. **执行构建与测试** - 运行 Docker 构建，生成测试数据，执行硬件测试
6. **验证结果对齐** - 确保 esp-dl 和 esp-ppq 的推理结果一致

**Skill 的核心价值：**

- **端到端自动化** - 从需求到可运行代码，Coding Agent 自动完成所有步骤
- **跨仓库协调** - 同时修改 esp-dl（C++）和 esp-ppq（Python）两个代码库
- **遵循最佳实践** - 自动应用 ESP-DL 的代码规范、目录结构和测试流程
- **增量式开发** - 支持新算子实现、数据类型扩展等多种场景

**适用场景：**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 场景
     - 示例请求
   * - 实现新算子
     - "实现 HardSwish 算子，支持 int8 和 float32"
   * - 添加数据类型支持
     - "给 Tanh 算子添加 int16 支持"
   * - 量化支持
     - "为 Mod 添加量化支持"
   * - 结果对齐
     - "验证 LogSoftmax 在 esp-dl 和 esp-ppq 的结果是否一致"

依赖要求
---------

在使用 ``espdl-operator`` skill 之前，你需要提前安装以下依赖：

必需提前安装的依赖
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 依赖
     - 用途
     - 安装命令
   * - **Docker**
     - 用于构建和测试环境
     - `官方安装指南 <https://docs.docker.com/get-docker/>`__
   * - **uv**
     - Python 包管理器
     - ``curl -LsSf https://astral.sh/uv/install.sh | sh``
   * - **Git**
     - 版本控制
     - ``apt install git`` (Ubuntu/Debian)

由 Skill 自动处理的依赖
^^^^^^^^^^^^^^^^^^^^^^^^

以下依赖**无需手动安装**, ``espdl-operator`` skill 会在执行过程中自动处理：

- **esp-ppq**: Python 量化工具包 - skill 会自动在 Docker 容器中以源码的形式安装
- **文档生成脚本**: ``gen_ops_markdown.py`` 等工具 - skill 会自动运行
- **Docker 镜像**: ``espdl/idf-ppq`` 镜像 - skill 会自动构建（如不存在）
- **ESP-IDF**: 开发框架 - 已包含在 Docker 镜像中

验证依赖安装
^^^^^^^^^^^^^^

安装完成后，请验证以下命令可以正常执行：

.. code-block:: bash

   # 检查 Docker
   docker --version

   # 检查 uv
   uv --version

   # 检查 Git
   git --version

如果以上命令都能正常输出版本信息，说明环境已就绪，可以开始使用 skill。

安装/放置 Skill
----------------

项目结构
^^^^^^^^^

首先，确认你的项目目录结构如下（``esp_dl_project_1`` 是项目根目录，根目录目录名称可以任意）。以 opencode 为例，目录结构全景图如下：

.. code-block::

   esp_dl_project_1/                    <-- 项目根目录（所有命令在此执行）
   ├── esp-dl/                          # ESP-DL 主代码库
   │   ├── esp-dl/                      # 核心库源代码（dl/, vision/, audio/ 等）
   │   ├── examples/                    # 示例程序
   │   ├── test_apps/                   # 测试应用
   │   ├── tools/                       # 工具脚本
   |   └── ...
   ├── esp-ppq/                         # 量化工具（与 esp-dl 同级）
   │   ├── esp_ppq/                     # 主包源代码
   │   ├── pyproject.toml               # 项目配置文件
   |   └── ...
   └── .opencode/                       # OpenCode 配置（需要创建）
       └── skills/espdl-operator/       # skill 安装位置（指向 esp-dl/tools/agents/skills/espdl-operator/）

**注意**: skill 的源代码位于 ``esp-dl/tools/agents/skills/espdl-operator/``，你需要将其复制或链接到 ``.opencode/skills/espdl-operator/`` （或其他 Agent 对应的目录）。skill 文件包括 ``SKILL.md`` （主文件）和 ``references/`` （参考模板和检查清单）。

重要：执行命令的位置
^^^^^^^^^^^^^^^^^^^^^

**所有以下命令都必须在项目根目录 ``esp_dl_project_1/`` 下执行。**

如果你不确定当前位置，请先执行：

.. code-block:: bash

   # 查看当前目录
   pwd

   # 应该输出类似: /home/username/workspace/esp_dl_project_1
   # 或 /path/to/esp_dl_project_1

   # 如果不在项目根目录，请先跳转
   cd /path/to/esp_dl_project_1

方法一：使用 npx（推荐 - 最简单）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为 OpenCode、Cursor、Claude Code 及其他兼容工具安装 skill 最简单的方法是使用 npx：

.. code-block:: bash

    npx skills add https://github.com/espressif/esp-dl/tree/master/tools/agents/skills/espdl-operator

.. note::

    npx 是 Node.js（通过 npm）自带的包执行器。如果你尚未安装 Node.js，请参考 `Node.js 安装指南 <https://nodejs.org/en/download/>`__ 进行安装。

执行命令后，skill 将被自动安装并在你的 Coding Agent 工具中可用。

方法二：手动安装
^^^^^^^^^^^^^^^^^

如果你偏好手动安装，或者你的工具不支持 npx，请根据你使用的具体工具按照以下说明操作：

.. note::

   不同 Agent 工具的 skill 安装目录可能有所不同，请根据实际使用的工具选择对应的安装路径。

OpenCode
^^^^^^^^^^

**方法一：复制文件**

.. code-block:: bash

   # 确保你在项目根目录 esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # 创建 .opencode/skills 目录
   mkdir -p .opencode/skills/espdl-operator

   # 从 esp-dl/tools/agents/skills/espdl-operator 复制到 .opencode/skills/espdl-operator
   cp -r esp-dl/tools/agents/skills/espdl-operator/* .opencode/skills/espdl-operator/

**方法二：使用符号链接（推荐用于开发，保持同步更新）**

.. code-block:: bash

   # 确保你在项目根目录 esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # 创建 .opencode/skills 目录
   mkdir -p .opencode/skills

   # 创建符号链接（使用相对路径）
   # 注意：从 .opencode/skills/espdl-operator 指向 esp-dl/tools/agents/skills/espdl-operator
   ln -s ../../esp-dl/tools/agents/skills/espdl-operator .opencode/skills/espdl-operator

   # 验证链接是否成功
   ls -la .opencode/skills/espdl-operator
   # 应该显示 SKILL.md 和 references/ 目录

启动 OpenCode 后，系统会自动加载该 skill。

Cursor
^^^^^^^^

**方法一：复制文件**

.. code-block:: bash

   # 确保你在项目根目录 esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # 创建 Cursor skills 目录
   mkdir -p .cursor/skills/espdl-operator

   # 复制 skill 文件
   cp -r esp-dl/tools/agents/skills/espdl-operator/* .cursor/skills/espdl-operator/

**方法二：使用符号链接**

.. code-block:: bash

   # 确保你在项目根目录 esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # 创建 .cursor/skills 目录
   mkdir -p .cursor/skills

   # 创建符号链接
   ln -s ../../esp-dl/tools/agents/skills/espdl-operator .cursor/skills/espdl-operator

Claude Desktop (Claude Code)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**方法一：复制文件**

.. code-block:: bash

   # 确保你在项目根目录 esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # 创建 Claude skills 目录
   mkdir -p .claude/skills/espdl-operator

   # 复制 skill 文件
   cp -r esp-dl/tools/agents/skills/espdl-operator/* .claude/skills/espdl-operator/

**方法二：使用符号链接**

.. code-block:: bash

   # 确保你在项目根目录 esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # 创建 .claude/skills 目录
   mkdir -p .claude/skills

   # 创建符号链接
   ln -s ../../esp-dl/tools/agents/skills/espdl-operator .claude/skills/espdl-operator

快速开始示例
-------------

假设你要实现一个新的算子 ``MyOp``：

1. **确保 skill 已安装**

   .. code-block:: bash

      ls -la .opencode/skills/espdl-operator/SKILL.md

2. **在 Agent 中提问**

   .. code-block::

      "帮我实现一个 MyOp 算子，支持 int8、int16 和 float32"

3. **Agent 会自动**

   - 加载 skill
   - 按照 9 个阶段指导 Coding Agent 工作
   - 生成必要的代码文件
   - 运行 Docker 测试

Skill 触发使用
---------------

安装完成后，你可以通过以下方式触发 ``espdl-operator`` skill：

自然语言触发
^^^^^^^^^^^^^^

直接在对话中使用以下关键词：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - 中文触发词
     - 英文触发词
   * - "实现算子"
     - "implement operator"
   * - "添加算子"
     - "add operator"
   * - "量化支持"
     - "quantization support"
   * - "算子对齐"
     - "operator alignment"
   * - "添加新的算子"
     - "add a new op"

示例对话
^^^^^^^^^

.. code-block::

   用户: "帮我实现一个 Mod 算子"
   Agent: [自动加载 espdl-operator skill 并开始指导]

   用户: "添加 LogSoftmax 算子到 esp-dl"
   Agent: [自动加载 skill 并提供实现步骤]

显式调用
^^^^^^^^^

如果自动触发未生效，可以显式要求 Agent 使用该 skill：

.. code-block::

   "使用 espdl-operator skill 帮我实现 Softmax 算子"
   "根据 espdl-operator skill 的指导，给 LogSoftmax 添加 int16 支持"

主要功能流程
-------------

该 skill 指导 Coding Agent 完成以下主要阶段：

Phase 1: 研究与分类
^^^^^^^^^^^^^^^^^^^^

- 理解 ONNX 算子规范
- 确定算子类型（Elementwise、Convolution、Pooling 等）
- 确定支持的数据类型（int8、int16、float32）

Phase 2: 实现 esp-dl Module 层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 创建算子模块头文件 (``dl_module_<op>.hpp``)
- 实现 ``get_output_shape()`` 和 ``forward()`` 方法
- 在 ``dl_module_creator.hpp`` 中注册算子

Phase 3: 实现 esp-dl Base 层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 创建 C 参考实现 (``dl_base_<op>.hpp/cpp``)

Phase 4: esp-ppq 集成
^^^^^^^^^^^^^^^^^^^^^^

- 在 ``EspdlQuantizer.py`` 中注册量化支持
- 在 ``espdl_typedef.py`` 中配置 layout pattern

Phase 5: 配置测试用例
^^^^^^^^^^^^^^^^^^^^^^

- 添加 PyTorch/ONNX 测试模型构建器
- 在 ``op_cfg.toml`` 中配置测试参数

Phase 6: Docker 构建与测试
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 生成测试用例（int8、int16、float32）
- 构建测试应用
- 在硬件上运行测试

Phase 7: SIMD 优化（可选）
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 这一部分暂未支持，后续会迭代优化

Phase 8: 算子对齐验证
^^^^^^^^^^^^^^^^^^^^^^

- 确保 esp-dl 和 esp-ppq 推理结果一致

Phase 9: 更新文档
^^^^^^^^^^^^^^^^^^

- 运行 ``gen_ops_markdown.py`` 更新算子支持状态文档

故障排除
---------

Skill 未触发
^^^^^^^^^^^^^

- 确认 skill 目录位于正确的目录（以 opencode 为例：``.opencode/skills/espdl-operator``）
- 尝试使用显式触发词："使用 espdl-operator skill..."
- 确认 Agent 工具支持 skill

Skill 流程未全部执行
^^^^^^^^^^^^^^^^^^^^^

- 中间有流程未执行，显式调用 ``espdl-operator`` skill 执行，例如：

  .. code-block:: bash

     # 未执行 docker 命令进行硬件烧录/测试，则在对话中下达命令:
     基于 espdl-operator skill 的说明，对硬件进行烧录测试，硬件已连接

算子实现效果不理想
^^^^^^^^^^^^^^^^^^^

该 skill 的作用是指导 AI 自动完成算子代码的编写。所以最终效果会受到两个因素影响：

1. **你使用的 AI 工具** (如 OpenCode、Cursor、Claude 等)
2. **AI 模型本身的代码能力** (不同模型写代码的水平，理解能力，调用 tool 的能力有差异)

根据我们的测试经验，以下组合在实现 C 语言版本的算子时，效果较好（因资源有限，许多组合还未覆盖，可自行尝试）：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Coding Agent 工具
     - 使用的模型
   * - Cursor
     - Claude Opus 4.6
   * - OpenCode
     - Kimi 2.5 + GLM 5 或 Claude Opus 4.6

**如果发现生成的代码质量不好，可以尝试：**

- 换用更强的 AI 模型（如 Claude Opus、GPT-4 等）
- 尝试下效果更好的 Coding Agent 工具
- 参考 SKILL.md 中的详细指导手动实现

Docker 问题
^^^^^^^^^^^^

.. code-block:: bash

   # 检查 Docker 是否运行
   docker ps

   # 重新构建镜像
   cd esp-dl/tools/agents/skills/espdl-operator/assets/docker
   docker build -t espdl/idf-ppq:latest .

权限问题
^^^^^^^^^

.. code-block:: bash

   # 确保设备访问权限（Linux）
   sudo usermod -a -G dialout $USER
   # 重新登录以生效

相关资源
---------

- **SKILL.md**: ``esp-dl/tools/agents/skills/espdl-operator/SKILL.md`` - 完整的开发指南
- **模板**: ``esp-dl/tools/agents/skills/espdl-operator/references/esp-dl-templates.md``
- **检查清单**: ``esp-dl/tools/agents/skills/espdl-operator/references/esp-ppq-checklist.md``
- **esp-dl**: ``esp-dl/`` - `esp-dl <https://github.com/espressif/esp-dl>`__ 主代码库
- **esp-ppq**: ``esp-ppq/`` - `esp-ppq <https://github.com/espressif/esp-ppq>`__ 量化工具（与 esp-dl 同级目录）
