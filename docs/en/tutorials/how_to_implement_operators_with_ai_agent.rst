Implement Operators Automatically with AI Agent
================================================

:link_to_translation:`zh_CN:[中文]`

This document describes how to install and use the ``espdl-operator`` skill in various Coding Agent tools (such as Claude, Cursor, OpenCode, etc.) for automated neural network operator implementation in the ESP-DL framework. **The following instructions use Linux environment as an example.**

.. contents::
   :local:
   :depth: 1

What is ``espdl-operator`` skill
---------------------------------

``espdl-operator`` is an automated development skill for Coding Agents, used to implement, test, and optimize neural network operators in the ESP-DL framework. When you make operator-related requests to a Coding Agent (such as Claude, Cursor, OpenCode, etc.), this skill guides the AI to automatically complete the following tasks:

**The Coding Agent will automatically:**

1. **Analyze operator requirements** - Parse ONNX operator specifications, determine operator type and data type support
2. **Generate esp-dl C++ code** - Automatically create Module layer and Base layer header/implementation files
3. **Modify esp-ppq quantization config** - Register operator support in the quantization tool, configure layout patterns
4. **Create test cases** - Generate PyTorch/ONNX test models, configure test parameters
5. **Execute build and testing** - Run Docker builds, generate test data, execute hardware tests
6. **Verify result alignment** - Ensure inference results are consistent between esp-dl and esp-ppq

**Core Value of the Skill:**

- **End-to-end automation** - From requirements to runnable code, the Coding Agent automatically completes all steps
- **Cross-repository coordination** - Simultaneously modifies both esp-dl (C++) and esp-ppq (Python) codebases
- **Follows best practices** - Automatically applies ESP-DL code standards, directory structure, and testing workflows
- **Incremental development** - Supports new operator implementation, data type extension, and other scenarios

**Applicable Scenarios:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Scenario
     - Example Request
   * - Implement new operator
     - "Implement HardSwish operator with int8 and float32 support"
   * - Add data type support
     - "Add int16 support for Tanh operator"
   * - Quantization support
     - "Add quantization support for Mod"
   * - Result alignment
     - "Verify if LogSoftmax results are consistent between esp-dl and esp-ppq"

Dependencies
------------

Before using the ``espdl-operator`` skill, you need to install the following dependencies in advance:

Required Pre-installed Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Dependency
     - Purpose
     - Installation Command
   * - **Docker**
     - For build and test environment
     - `Official Installation Guide <https://docs.docker.com/get-docker/>`__
   * - **uv**
     - Python package manager
     - ``curl -LsSf https://astral.sh/uv/install.sh | sh``
   * - **Git**
     - Version control
     - ``apt install git`` (Ubuntu/Debian)

Dependencies Handled Automatically by Skill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following dependencies **do not need manual installation**; the ``espdl-operator`` skill will handle them automatically during execution:

- **esp-ppq**: Python quantization toolkit - skill automatically installs from source in Docker container
- **Documentation generation scripts**: ``gen_ops_markdown.py`` and other tools - skill runs automatically
- **Docker image**: ``espdl/idf-ppq`` image - skill builds automatically (if not exists)
- **ESP-IDF**: Development framework - included in Docker image

Verify Dependency Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installation, verify that the following commands work:

.. code-block:: bash

   # Check Docker
   docker --version

   # Check uv
   uv --version

   # Check Git
   git --version

If all commands output version information normally, the environment is ready to use the skill.

Installing/Placing Skill
-------------------------

Project Structure
^^^^^^^^^^^^^^^^^

First, confirm your project directory structure as follows (``esp_dl_project_1`` is the project root directory, the root directory name can be arbitrary). Using opencode as an example, the full directory structure is shown below:

.. code-block::

   esp_dl_project_1/                    <-- Project root directory (all commands executed here)
   ├── esp-dl/                          # ESP-DL main codebase
   │   ├── esp-dl/                      # Core library source code (dl/, vision/, audio/, etc.)
   │   ├── examples/                    # Example programs
   │   ├── test_apps/                   # Test applications
   │   ├── tools/                       # Tool scripts
   |   └── ...
   ├── esp-ppq/                         # Quantization tool (same level as esp-dl)
   │   ├── esp_ppq/                     # Main package source code
   │   ├── pyproject.toml               # Project configuration file
   |   └── ...
   └── .opencode/                       # OpenCode configuration (needs to be created)
       └── skills/espdl-operator/       # skill installation location (points to esp-dl/tools/agents/skills/espdl-operator/)

**Note**: The skill source code is located at ``esp-dl/tools/agents/skills/espdl-operator/``. You need to copy or link it to ``.opencode/skills/espdl-operator/`` (or the corresponding directory for other Agents). The skill files include ``SKILL.md`` (main file) and ``references/`` (reference templates and checklists).

Important: Command Execution Location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**All commands below must be executed in the project root directory ``esp_dl_project_1/``.**

If you are unsure of your current location, first execute:

.. code-block:: bash

   # Check current directory
   pwd

   # Should output something like: /home/username/workspace/esp_dl_project_1
   # or /path/to/esp_dl_project_1

   # If not in project root, navigate there first
   cd /path/to/esp_dl_project_1

Method 1: Use npx (Recommended - Simplest)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install the skill for OpenCode, Cursor, Claude Code, and other compatible tools is using npx:

.. code-block:: bash

    npx skills add https://github.com/espressif/esp-dl/tree/master/tools/agents/skills/espdl-operator

.. note::

    npx is the package runner that comes with Node.js (via npm). If you haven't installed Node.js yet, please refer to `Node.js Installation Guide <https://nodejs.org/en/download/>`__ to install it first.

After running the command, the skill will be automatically installed and ready to use in your Coding Agent tool.

Method 2: Manual Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer manual installation or your tool doesn't support npx, follow the instructions below for your specific tool:

.. note::

   The skill installation directory varies by Agent tool. Please choose the appropriate path based on the tool you are using.

OpenCode
^^^^^^^^^

**Method 1: Copy Files**

.. code-block:: bash

   # Ensure you are in the project root directory esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # Create .opencode/skills directory
   mkdir -p .opencode/skills/espdl-operator

   # Copy from esp-dl/tools/agents/skills/espdl-operator to .opencode/skills/espdl-operator
   cp -r esp-dl/tools/agents/skills/espdl-operator/* .opencode/skills/espdl-operator/

**Method 2: Use Symbolic Link (Recommended for Development, Keeps in Sync)**

.. code-block:: bash

   # Ensure you are in the project root directory esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # Create .opencode/skills directory
   mkdir -p .opencode/skills

   # Create symbolic link (using relative path)
   # Note: from .opencode/skills/espdl-operator pointing to esp-dl/tools/agents/skills/espdl-operator
   ln -s ../../esp-dl/tools/agents/skills/espdl-operator .opencode/skills/espdl-operator

   # Verify link is successful
   ls -la .opencode/skills/espdl-operator
   # Should show SKILL.md and references/ directory

After starting OpenCode, the system will automatically load this skill.

Cursor
^^^^^^^

**Method 1: Copy Files**

.. code-block:: bash

   # Ensure you are in the project root directory esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # Create Cursor skills directory
   mkdir -p .cursor/skills/espdl-operator

   # Copy skill files
   cp -r esp-dl/tools/agents/skills/espdl-operator/* .cursor/skills/espdl-operator/

**Method 2: Use Symbolic Link**

.. code-block:: bash

   # Ensure you are in the project root directory esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # Create .cursor/skills directory
   mkdir -p .cursor/skills

   # Create symbolic link
   ln -s ../../esp-dl/tools/agents/skills/espdl-operator .cursor/skills/espdl-operator

Claude Desktop (Claude Code)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Method 1: Copy Files**

.. code-block:: bash

   # Ensure you are in the project root directory esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # Create Claude skills directory
   mkdir -p .claude/skills/espdl-operator

   # Copy skill files
   cp -r esp-dl/tools/agents/skills/espdl-operator/* .claude/skills/espdl-operator/

**Method 2: Use Symbolic Link**

.. code-block:: bash

   # Ensure you are in the project root directory esp_dl_project_1/
   cd /path/to/esp_dl_project_1

   # Create .claude/skills directory
   mkdir -p .claude/skills

   # Create symbolic link
   ln -s ../../esp-dl/tools/agents/skills/espdl-operator .claude/skills/espdl-operator

Quick Start Example
-------------------

Suppose you want to implement a new operator ``MyOp``:

1. **Ensure skill is installed**

   .. code-block:: bash

      ls -la .opencode/skills/espdl-operator/SKILL.md

2. **Ask in Agent**

   .. code-block::

      "Help me implement a MyOp operator with int8, int16, and float32 support"

3. **Agent will automatically**

   - Load the skill
   - Guide the Coding Agent through 9 phases
   - Generate necessary code files
   - Run Docker tests

Skill Trigger Usage
--------------------

After installation, you can trigger the ``espdl-operator`` skill in the following ways:

Natural Language Trigger
^^^^^^^^^^^^^^^^^^^^^^^^^

Use the following keywords directly in conversation:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Chinese Trigger
     - English Trigger
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

Example Conversations
^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

   User: "Help me implement a Mod operator"
   Agent: [Automatically loads espdl-operator skill and starts guiding]

   User: "Add LogSoftmax operator to esp-dl"
   Agent: [Automatically loads skill and provides implementation steps]

Explicit Invocation
^^^^^^^^^^^^^^^^^^^^

If automatic triggering doesn't work, you can explicitly ask the Agent to use this skill:

.. code-block::

   "Use espdl-operator skill to help me implement Softmax operator"
   "Following espdl-operator skill guidance, add int16 support for LogSoftmax"

Main Functionality Workflow
----------------------------

This skill guides the Coding Agent through the following main phases:

Phase 1: Research and Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Understand ONNX operator specifications
- Determine operator type (Elementwise, Convolution, Pooling, etc.)
- Determine supported data types (int8, int16, float32)

Phase 2: Implement esp-dl Module Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Create operator module header file (``dl_module_<op>.hpp``)
- Implement ``get_output_shape()`` and ``forward()`` methods
- Register operator in ``dl_module_creator.hpp``

Phase 3: Implement esp-dl Base Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Create C reference implementation (``dl_base_<op>.hpp/cpp``)

Phase 4: esp-ppq Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Register quantization support in ``EspdlQuantizer.py``
- Configure layout pattern in ``espdl_typedef.py``

Phase 5: Configure Test Cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Add PyTorch/ONNX test model builders
- Configure test parameters in ``op_cfg.toml``

Phase 6: Docker Build and Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Generate test cases (int8, int16, float32)
- Build test applications
- Run tests on hardware

Phase 7: SIMD Optimization (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- This part is not yet supported and will be iteratively improved in the future

Phase 8: Operator Alignment Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Ensure esp-dl and esp-ppq inference results are consistent

Phase 9: Update Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Run ``gen_ops_markdown.py`` to update operator support status documentation

Troubleshooting
---------------

Skill Not Triggered
^^^^^^^^^^^^^^^^^^^^

- Confirm skill directory is in the correct location (using opencode as example: ``.opencode/skills/espdl-operator``)
- Try using explicit trigger words: "Use espdl-operator skill..."
- Confirm the Agent tool supports skills

Skill Workflow Not Fully Executed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- If some steps were not executed, explicitly invoke the ``espdl-operator`` skill, for example:

  .. code-block:: bash

     # If Docker commands for hardware flashing/testing were not executed, issue the command in conversation:
     Based on espdl-operator skill instructions, perform hardware flashing and testing, hardware is connected

Operator Implementation Quality Not Ideal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The role of this skill is to guide the AI to automatically complete operator code writing. Therefore, the final result is affected by two factors:

1. **The AI tool you use** (such as OpenCode, Cursor, Claude, etc.)
2. **The AI model's own coding capabilities** (different models have varying coding skills, comprehension abilities, and tool calling capabilities)

Based on our testing experience, the following combinations work well when implementing C language version operators (due to limited resources, many combinations have not been covered; you can try them yourself):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Coding Agent Tool
     - Model Used
   * - Cursor
     - Claude Opus 4.6
   * - OpenCode
     - Kimi 2.5 + GLM 5 or Claude Opus 4.6

**If the generated code quality is poor, you can try:**

- Switch to a more powerful AI model (such as Claude Opus, GPT-4, etc.)
- Try a Coding Agent tool with better results
- Manually implement following the detailed guidance in SKILL.md

Docker Issues
^^^^^^^^^^^^^

.. code-block:: bash

   # Check if Docker is running
   docker ps

   # Rebuild image
   cd esp-dl/tools/agents/skills/espdl-operator/assets/docker
   docker build -t espdl/idf-ppq:latest .

Permission Issues
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Ensure device access permissions (Linux)
   sudo usermod -a -G dialout $USER
   # Log out and back in for changes to take effect

Related Resources
-----------------

- **SKILL.md**: ``esp-dl/tools/agents/skills/espdl-operator/SKILL.md`` - Complete development guide
- **Templates**: ``esp-dl/tools/agents/skills/espdl-operator/references/esp-dl-templates.md``
- **Checklist**: ``esp-dl/tools/agents/skills/espdl-operator/references/esp-ppq-checklist.md``
- **esp-dl**: ``esp-dl/`` - `esp-dl <https://github.com/espressif/esp-dl>`__ main codebase
- **esp-ppq**: ``esp-ppq/`` - `esp-ppq <https://github.com/espressif/esp-ppq>`__ quantization tool (same level directory as esp-dl)
