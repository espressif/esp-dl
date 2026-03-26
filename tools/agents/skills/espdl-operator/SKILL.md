---
name: espdl-operator
description: >
  End-to-end guide for implementing, testing, and optimizing neural network operators in the ESP-DL framework.
  Covers C++ module implementation, C reference kernels, SIMD assembly optimization, esp-ppq quantization
  strategy integration, Docker-based build/test, and inference result alignment between esp-dl and esp-ppq.
  Use this skill whenever the user wants to add a new operator, implement an operator, optimize an existing
  operator with SIMD, add quantization support for an operator, or test/validate operator correctness.
  Also triggers for "算子实现", "添加算子", "SIMD优化", "量化支持", "算子对齐" and similar phrases.
---

# ESP-DL Operator Development Skill

This skill guides you through the complete lifecycle of implementing a neural network operator in ESP-DL:
from C++ module code, through quantization support in esp-ppq, to Docker-based validation that ensures
inference results align between the quantization tool and the on-device runtime.

## Workflow Continuity — Read This First

This skill describes a multi-phase pipeline: research → implement → test → optimize → document.
The most critical transition is **from code modification (Phases 2–5) to testing (Phase 6)**.

**After completing ANY code change — whether it's a new module, a base layer fix, an esp-ppq
tweak, or a test config update — immediately proceed to Phase 6 (Docker Build & Test) without
stopping to ask the user.** The user expects the full implement-then-test cycle to happen as
one continuous flow. Pausing after code changes to ask "should I run tests now?" breaks the
workflow and forces unnecessary back-and-forth.

The only reasons to pause before testing are:
- You need information the user hasn't provided (e.g., target chip, Docker image location)
- A build/compilation error requires the user's input to resolve
- The user explicitly asked you to stop at a certain phase

When multiple code files need modification, complete ALL code changes first (Phases 2–5 as
applicable), then run the full test pipeline once. Don't test after each individual file change.

---

## Project Layout

```
esp-dl/esp-dl/
├── dl/module/include/dl_module_<op>.hpp   # Module layer: interface, shape inference, forward dispatch
├── dl/module/include/dl_module_creator.hpp # Operator registry
├── dl/base/dl_base_<op>.hpp/.cpp          # Base layer: C reference impl + ISA dispatch
├── dl/base/isa/tie728/dl_tie728_*.S       # TIE728 SIMD (ESP32-S3)
├── dl/base/isa/esp32p4/dl_esp32p4_*.S     # ESP32-P4 SIMD (RISC-V PIE)
esp-dl/tools/ops_test/
├── config/op_cfg.toml                      # Test configurations per operator
├── torch_ops_test.py                       # PyTorch-based test model builders
├── onnx_ops_test.py                        # ONNX-based test model builders
├── gen_test_cases.py                       # Generates quantized .espdl test models
esp-dl/test_apps/esp-dl/                    # Test application (builds + runs on hardware)

esp-ppq/esp_ppq/
├── quantization/quantizer/EspdlQuantizer.py  # Quantization config per op type
├── parser/espdl/espdl_typedef.py              # Op set classifications
├── parser/espdl/export_patterns.py            # Export pattern rules (LUT, layout, fusion)
├── IR/base/opdef.py                           # OpSocket definitions for dispatch
├── executor/op/torch/espdl.py                 # LUT computation backend
```

---

## Phase 1: Research & Classify the Operator

Before writing any code, understand what you're building.

### 1.1 Read the ONNX Specification

Look up the operator at `https://onnx.ai/onnx/operators/onnx__<OpName>.html`.
Understand its inputs, outputs, attributes, broadcasting rules, and edge cases.

### 1.2 Classify the Operator

The classification determines which templates and patterns to follow:

| Category | Examples | Module Pattern | Base Pattern |
|----------|----------|---------------|--------------|
| **Elementwise binary** | Add, Sub, Mul, Div, Mod, Pow | `dl_module_add.hpp` | `dl_base_add.hpp/cpp` (elemwiseArgsType) |
| **Elementwise unary** | Relu, Sigmoid, Exp, Neg, Sqrt | `dl_module_relu.hpp` | `dl_base_relu.hpp/cpp` (ArgsType) |
| **Convolution-like** | Conv, ConvTranspose, DepthwiseConv | `dl_module_conv.hpp` | `dl_base_conv2d.hpp/cpp` |
| **Pooling** | AveragePool, MaxPool, GlobalAveragePool | `dl_module_average_pool.hpp` | `dl_base_avg_pool2d.hpp/cpp` |
| **Reduce** | ReduceSum, ReduceMean, ReduceMax | `dl_module_reduce_sum.hpp` | `dl_base_reduce.hpp/cpp` |
| **Shape manipulation** | Reshape, Transpose, Flatten, Slice | `dl_module_reshape.hpp` | Typically no base layer needed |
| **Sequence/RNN** | GRU, LSTM | `dl_module_gru.hpp` | Complex multi-step base |
| **Activation (LUT)** | HardSwish, HardSigmoid, Tanh | `dl_module_lut.hpp` | LUT-based implementation |

Read 2-3 reference implementations from the same category. The references directory at
`references/esp-dl-templates.md` has annotated templates for each category.

### 1.3 Determine Scope

Decide which data types to support: `int8`, `int16`, `float32`.

**Default rule: ALL operators MUST implement float32 unless technically impossible.**
Float32 serves as the high-precision inference path — it preserves full model accuracy without
quantization loss and is the baseline for correctness validation. Every operator that can
accept float inputs and produce float outputs should support float32, regardless of whether
it is "compute-heavy" or "typically run quantized". Conv, ConvTranspose, MatMul, Linear, and
all other operators should include float32 support.

**When float32 is appropriate (the vast majority of operators):**
- Elementwise ops (Add, Sub, Mul, Relu, Sigmoid, etc.) — always
- Reduce ops (ReduceSum, ReduceMean, etc.) — always
- Shape ops (Reshape, Transpose, etc.) — always (dtype-agnostic)
- Activation/LUT ops (HardSwish, Tanh, etc.) — always (float uses direct computation, not LUT)
- Conv, ConvTranspose, MatMul, Linear — always (float32 preserves high-precision inference)
- Pooling ops (AveragePool, MaxPool, etc.) — always
- Sequence/RNN ops (GRU, LSTM, etc.) — always

**The only exceptions where float32 may be omitted:**
- Comparison ops (Greater, Equal, Less) — output is boolean, not a numeric tensor
- Ops whose output dtype is inherently non-float (e.g., ArgMax returns indices)

If you are unsure whether an operator should support float32, the answer is **yes, it should**.
Only omit float32 when the operator's semantics make float input/output meaningless.

Float32 implementation is generally simpler than quantized: no scale/rescale, no truncation,
no exponent handling, and no SIMD optimization needed.

---

## Phase 2: Implement esp-dl Module Layer

Create `esp-dl/dl/module/include/dl_module_<op_snake>.hpp` where `<op_snake>` is the
snake_case version of the ONNX operator name (e.g., `HardSwish` → `hard_swish`).

### 2.1 Module Class Structure

Every operator module must:

1. **Inherit from `Module`** (defined in `dl_module_base.hpp`)
2. **Implement `get_output_shape()`** — compute output shape from input shapes
3. **Implement `forward()`** — dispatch to the correct typed `forward_template<T>()`
4. **Implement `forward_template<T>()`** — get tensors from context, prepare args, call base layer
5. **Implement static `deserialize()`** — reconstruct from FlatBuffers model
6. **Optionally implement `forward_args()`** — for dual-core dispatch support
7. **Optionally implement `print()`** — debug info

Key conventions:
- Use `#pragma once` as header guard
- Namespace: `dl::module`
- Constructor takes `name`, `inplace`, `quant_type` at minimum
- Additional ONNX attributes become constructor parameters and class members
- `quant_type` dispatches to `QUANT_TYPE_SYMM_8BIT`, `QUANT_TYPE_SYMM_16BIT`, `QUANT_TYPE_FLOAT32`

See `references/esp-dl-templates.md` for full annotated templates.

### 2.2 Deserialization

The `deserialize()` static method reads attributes from FlatBuffers:

```cpp
static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
{
    Module *op = nullptr;
    quant_type_t quant_type;
    fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

    // Read operator-specific attributes
    int some_attr;
    fbs_model->get_operation_attribute(node_name, "some_attr", some_attr);

    op = new MyOp(node_name.c_str(), MODULE_NON_INPLACE, quant_type, some_attr);
    return op;
}
```

### 2.3 Register in Creator

Add the operator to `dl_module_creator.hpp` in the `register_dl_modules()` method:

```cpp
this->register_module("MyOp", MyOp::deserialize);
```

Also add the `#include "dl_module_<op_snake>.hpp"` at the top of the creator header.

**→ Continue to Phase 3 if a base layer is needed, or skip to Phase 4 (esp-ppq) if this is a
shape-only op. After all code phases are done, proceed directly to Phase 6 for testing.**

---

## Phase 3: Implement esp-dl Base Layer (C Reference)

The base layer provides the actual computation kernel. Create:
- `esp-dl/dl/base/dl_base_<op_snake>.hpp` — declarations
- `esp-dl/dl/base/dl_base_<op_snake>.cpp` — C reference implementation

### 3.1 Architecture

```
Module::forward_template<T>()
    → prepares ArgsType / elemwiseArgsType
    → calls base::<op_function>(args)
        → selects ISA-optimized or C reference impl
        → executes the kernel
```

For elementwise binary ops, use `elemwiseArgsType<T>` and the `elemwise_loop_*d()` helpers.
For unary ops, use `ArgsType<T>` and the `activation_shell()` helper.
For other ops, define a custom args struct.

### 3.2 ISA Dispatch Pattern

In the `.cpp` file, the implementation selection follows this pattern:

```cpp
#if CONFIG_ESP32P4_BOOST
    impl_func = dl_esp32p4_s8_<op>_11c;  // P4 SIMD
#elif CONFIG_TIE728_BOOST
    impl_func = dl_tie728_s8_<op>_11c;   // S3 SIMD
#else
    impl_func = c_impl_<op>;             // C reference (always present)
#endif
```

The C reference implementation is the fallback and must always exist. SIMD implementations
are added as a later optimization step.

### 3.3 Float32 Implementation Differences

Float32 kernels are fundamentally simpler than int8/int16 quantized kernels because there is
no quantization overhead. Here are the key differences:

| Aspect | int8 / int16 (quantized) | float32 |
|--------|--------------------------|---------|
| **Arithmetic** | `tool::truncate<int32_t>(result)` — clamp to type range | Direct arithmetic, no truncation |
| **Scale/Rescale** | Uses `args->mul_shift`, `input_scale`, `output_rescale` | Ignores these fields (exponent=0, scale=1.0) |
| **SIMD dispatch** | ISA-specific implementations (TIE728, ESP32-P4) | C reference only — no SIMD needed |
| **Template specialization** | Generic template handles quantization math | Explicit `template<>` specialization for `float` |

**Two patterns for float implementation:**

**Pattern A — Base layer with float specialization (recommended for binary/complex ops):**
The base layer `.cpp` provides a `template<> ... <float>` specialization that does direct
arithmetic. The module calls the same `base::op(args)` function for all types. This pattern
keeps the module layer clean and uniform. See `dl_base_add.cpp` for example.

**Pattern B — Module-level inline implementation (acceptable for simple unary ops):**
Some simple unary ops (like ReLU) implement float32 directly in the module's `forward()`
method without calling the base layer. This avoids creating a base-layer float overload for
trivial operations. The float path is a simple loop over elements.

```cpp
// Pattern B example: float implemented directly in forward()
void forward(ModelContext *context, runtime_mode_t mode)
{
    if (quant_type == QUANT_TYPE_SYMM_8BIT) {
        forward_template<int8_t>(context, mode);
    } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
        forward_template<int16_t>(context, mode);
    } else if (quant_type == QUANT_TYPE_FLOAT32) {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        float *input_ptr = (float *)input->get_element_ptr();
        float *output_ptr = (float *)output->get_element_ptr();
        for (size_t i = 0; i < input->size; i++) {
            output_ptr[i] = /* direct float operation on input_ptr[i] */;
        }
    }
}
```

Use Pattern A when the operation has multiple broadcast variants, multi-dimensional looping,
or dual-core dispatch. Use Pattern B only for straightforward element-by-element operations
with a single input.

See `references/esp-dl-templates.md` for complete float32 template examples.

**→ Continue to Phase 4 (esp-ppq checks). Do not stop here — esp-ppq modifications and test
configuration (Phases 4–5) are prerequisites for testing.**

---

## Phase 4: Determine esp-ppq Modifications

Every new operator needs at least TWO checks in esp-ppq, because the export pipeline has
two independent systems that must both recognize the operator:

1. **Quantization system** — `quant_operation_types` determines if an op gets quantized
2. **Layout system** — `layout_patterns` in `layout_patterns.py` handles NCHW→NHWC
   transformation. Every operator in the graph MUST be in one of the layout pattern op sets,
   otherwise `reset_graph_layout()` will error with "Can not reset {op_type} layout"

**Important: Float32 and esp-ppq.** When `float=True` is passed to the quantization API,
the entire graph uses `TargetPlatform.FP32` and skips quantization entirely — the model is
loaded from ONNX and exported directly without calling `EspdlQuantizer`. This means:
- The two checks below are only relevant for **int8/int16 quantized** exports
- Float32 export still runs the layout patterns and export patterns, but `InsertQuantTypePattern`
  sets `quant_type = EspQuantType.F32` for all ops, and patterns like `ResetParamLayoutPattern`
  and `AddLUTPattern` short-circuit when they see `quant_type == F32`
- You do NOT need to modify esp-ppq specifically for float32 support — it's handled automatically
- However, the operator still needs to be in a layout op set, because `reset_graph_layout()`
  runs for both quantized and float32 exports

### 4.1 Check #1: Quantization Registration

Check `EspdlQuantizer.quant_operation_types` in
`esp-ppq/esp_ppq/quantization/quantizer/EspdlQuantizer.py`.

If the operator is NOT listed → add it.

### 4.2 Check #2: Layout Pattern Op Set (ALWAYS required)

Check `esp-ppq/esp_ppq/parser/espdl/espdl_typedef.py` — the operator MUST be in one of
these op sets, which map to layout transformation patterns in `layout_patterns.py`:

| Op Set in `espdl_typedef.py` | Layout Pattern | When to Use |
|------------------------------|---------------|-------------|
| `CONV_LAYOUT_OP_SET` | `ResetConvLayoutPattern` | Conv, Pool, DepthToSpace — ops with spatial layout |
| `PASSIVE_LAYOUT_OP_SET` | `BypassPassiveLayoutPattern` | Activations (Relu, Sigmoid...) + Math (Exp, Log...) — pass through layout |
| `ADD_LIKE_OP_SET` | `BypassAddLikePattern` | Binary elementwise (Add, Sub, Mul, Div, **Mod**, Pow...) — handles shape broadcasting between two inputs |
| `AXIS_TRANSFORM_OP_SET` | `AxisTransformPattern` | Softmax, Split, Reduce ops — transforms axis attributes |
| `OTHER_OP_SET` | `RestoreOriginLayoutPattern` | Reshape, Transpose, Gather, GRU... — restores to original layout |

The `BypassAddLikePattern` is particularly important for binary elementwise ops: it ensures
that when the two inputs have different permutations (due to upstream layout changes), the
pattern either propagates the permutation consistently or inserts a transpose to fix the
mismatch. Without this, binary ops will produce incorrect results after layout transformation.

**If the operator is NOT in any op set → add it to the correct one.** Even if the operator
is already in `quant_operation_types`, a missing op set entry will cause export failure.

### 4.3 Does it need special quantization rules?

Most operators use the default quantization config. Special rules are needed when:
- **Bias input** exists (like Conv/Gemm) → set bias to 32-bit, PASSIVE_INIT
- **Output should stay FP32** (like Softmax) → set output state to FP32
- **Multiple hidden states** (like GRU/LSTM) → custom per-input config
- **LUT-based activation** → ensure it's in `ACTIVATION_OP_SET` in `espdl_typedef.py`
  and `AddLUTPattern` in `export_patterns.py` handles it

### 4.4 Does it need a custom OpSocket?

Most operators use `DEFAULT_SOCKET_CREATOR`. Custom sockets are needed when inputs
have different platform requirements (e.g., Gather's index input stays FP32).
Check `DEFAULT_SOCKET_TABLE` in `esp-ppq/esp_ppq/IR/base/opdef.py`.

### 4.5 Does it need additional export pattern changes?

Beyond the layout patterns above, check `export_patterns.py` for:
- Weight repacking (Conv-like ops)
- LUT generation (activation ops)
- Node fusion (e.g., Conv+Relu via `FuseReluLikePattern`)

### Summary: What to Check for Every New Operator

| Check | File | Action |
|-------|------|--------|
| In `quant_operation_types`? | `EspdlQuantizer.py` | Add if missing |
| In a layout op set? | `espdl_typedef.py` | **Always verify** — add to correct op set |
| Special quant config? | `EspdlQuantizer.py` | Add rules in `create_espdl_quant_config()` if needed |
| Custom OpSocket? | `IR/base/opdef.py` | Add if inputs have heterogeneous platform needs |
| Export patterns? | `export_patterns.py` | Add if LUT/fusion/weight-layout needed |

### Quick Category → Op Set Mapping

| Operator Category | Add to Op Set | Why |
|-------------------|--------------|-----|
| Elementwise binary (Add-like) | `ADD_LIKE_OP_SET` | BypassAddLikePattern handles input shape broadcasting |
| Elementwise unary (activation) | `ACTIVATION_OP_SET` | BypassPassiveLayoutPattern passes through layout |
| Elementwise unary (math) | `MATH_OP_SET` | Also covered by PASSIVE_LAYOUT_OP_SET |
| Convolution-like | `CONV_LAYOUT_OP_SET` | ResetConvLayoutPattern transforms spatial layout |
| Reduce / Softmax-like | `REDUCE_OP_SET` or `SOFTMAX_LIKE_OP_SET` | AxisTransformPattern adjusts axis attrs |
| Shape manipulation | `OTHER_OP_SET` | RestoreOriginLayoutPattern restores original |

**→ Continue to Phase 5 to configure test cases. Test configuration is the last step before
the actual build & test pipeline.**

---

## Phase 5: Configure Test Cases

### 5.1 Add Test Model Builder

If PyTorch has the operator, add a test class in `tools/ops_test/torch_ops_test.py`:

```python
class MYOP_TEST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize PyTorch op from config params

    def forward(self, *inputs):
        # Compute forward pass
        return output
```

If PyTorch doesn't have it (or ONNX-only), add a function in `tools/ops_test/onnx_ops_test.py`:

```python
def MYOP_TEST(config) -> onnx.ModelProto:
    # Build ONNX graph using onnx.helper
    return model
```

### 5.2 Add Test Configuration

Add to `tools/ops_test/config/op_cfg.toml`:

```toml
[ops_test.MyOp]
test_func = "MYOP_TEST"
quant_bits = ["int8", "int16", "float32"]
package = "torch_ops_test"           # or "onnx_ops_test"
targets = ["esp32s3", "esp32p4"]

    [[ops_test.MyOp.cfg]]
    input_shape = [1, 16, 32, 32]
    export_name_prefix = "myop_basic_test"
    # operator-specific parameters...

    [[ops_test.MyOp.cfg]]
    input_shape = [1, 3, 8, 8]
    export_name_prefix = "myop_edge_case"
    # different parameters for edge cases...
```

**`quant_bits` field**: Controls which quantization types to generate test cases for.
- `"int8"` → generates `*_s8.espdl` test models (quantized to 8-bit)
- `"int16"` → generates `*_s16.espdl` test models (quantized to 16-bit)
- `"float32"` → generates `*_f32.espdl` test models (no quantization, direct float)

All operators should include all three: `["int8", "int16", "float32"]`.
Only omit `"float32"` for the rare ops where float output is meaningless (e.g., comparison ops that output boolean).

Create 3-5 test configurations covering:
- Aligned dimensions (multiples of 16 for SIMD)
- Unaligned dimensions
- Small and large tensors
- Different attribute combinations
- Edge cases specific to the operator

**→ All code changes are now complete. Proceed IMMEDIATELY to Phase 6 to build and test.
Do not stop to ask the user — go straight into the Docker build & test pipeline.**

---

## Phase 6: Docker Build & Test

**This phase should run automatically after any code modification — do not wait for user
confirmation to start.** If you just completed Phases 2–5 (or any subset of them), execute
the steps below immediately. The user expects code changes to be validated, not left untested.

All build and test commands run inside a Docker container. The Docker image
`espdl/idf-ppq:latest` contains ESP-IDF v5.4.3, PyTorch, and esp-ppq.

### 6.1 Docker Run Template

Every Docker command uses the same base template. Define these variables first,
then use the `DOCKER_RUN` function for all operations:

```bash
# ========== Configuration (set these once) ==========
OP_TYPE="MyOp"                          # ONNX operator name (PascalCase)
TARGET="esp32p4"                        # Target chip: esp32p4, esp32s3, esp32
ESP_DL_ROOT="/path/to/esp-dl"           # esp-dl project root
ESP_PPQ_ROOT="/path/to/esp-ppq"         # esp-ppq root (optional, for editable mode)
ESP_DL_IMAGE="espdl/idf-ppq:latest"     # Docker image name
SKILL_DIR="/path/to/skills/espdl-operator" # Skill base directory (the directory containing SKILL.md).
                                          # The agent should resolve this from its skill load path.

# ========== Auto-build Docker image if missing ==========
# The Dockerfile lives at assets/docker/Dockerfile inside this skill's directory.
# Building takes 20-30 min (downloads ESP-IDF + PyTorch). Only runs once.
if ! docker image inspect "${ESP_DL_IMAGE}" > /dev/null 2>&1; then
  echo "Docker image ${ESP_DL_IMAGE} not found. Building (this may take 20-30 minutes)..."
  docker build -t "${ESP_DL_IMAGE}" "${SKILL_DIR}/assets/docker"
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build Docker image ${ESP_DL_IMAGE}. Fix Dockerfile issues and retry."
    return 1 2>/dev/null || exit 1
  fi
  echo "Docker image ${ESP_DL_IMAGE} built successfully."
fi

# ========== Docker base command builder ==========
DOCKER_BASE="docker run --rm -i -v ${ESP_DL_ROOT}:/esp-dl -w /esp-dl"
if [ -n "${ESP_PPQ_ROOT}" ] && [ -d "${ESP_PPQ_ROOT}" ]; then
  DOCKER_BASE="${DOCKER_BASE} -v ${ESP_PPQ_ROOT}:/esp-ppq"
  PPQ_INSTALL="pip install -e /esp-ppq[cpu] > /dev/null 2>&1"
else
  PPQ_INSTALL="pip install esp-ppq > /dev/null 2>&1"
fi
DOCKER_PREAMBLE=". \$IDF_PATH/export.sh && ${PPQ_INSTALL}"
```

**Important:** The auto-build block above MUST be included whenever Phase 6 commands are
executed. It is idempotent — if the image already exists, `docker image inspect` succeeds
instantly and the build is skipped. On first run, the build pulls `espressif/idf:v5.4.3` as
the base image and installs PyTorch + other dependencies, which takes 20-30 minutes.
If the build fails (e.g., network issues), the script exits early with an error message
so subsequent Docker commands don't fail with a confusing "image not found" error.

### 6.2 Step 1: Generate Test Cases

Generates `.espdl` model files with embedded test values:

```bash
# Generate int8 test cases (quantized, produces *_s8.espdl)
${DOCKER_BASE} ${ESP_DL_IMAGE} bash -c "${DOCKER_PREAMBLE} && \
  python tools/ops_test/gen_test_cases.py \
    --config tools/ops_test/config/op_cfg.toml \
    --ops ${OP_TYPE} \
    --output-path test_apps/esp-dl/models/${TARGET} \
    --target ${TARGET} \
    --bits 8"

# Generate int16 test cases (quantized, produces *_s16.espdl)
${DOCKER_BASE} ${ESP_DL_IMAGE} bash -c "${DOCKER_PREAMBLE} && \
  python tools/ops_test/gen_test_cases.py \
    --config tools/ops_test/config/op_cfg.toml \
    --ops ${OP_TYPE} \
    --output-path test_apps/esp-dl/models/${TARGET} \
    --target ${TARGET} \
    --bits 16"

# Generate float32 test cases (no quantization, produces *_f32.espdl)
${DOCKER_BASE} ${ESP_DL_IMAGE} bash -c "${DOCKER_PREAMBLE} && \
  python tools/ops_test/gen_test_cases.py \
    --config tools/ops_test/config/op_cfg.toml \
    --ops ${OP_TYPE} \
    --output-path test_apps/esp-dl/models/${TARGET} \
    --target ${TARGET} \
    --float"
```

Note: `--bits 8` and `--bits 16` control quantized test generation. The `--float` flag
(not `--bits 32`) triggers float32 test generation. Float32 test cases are only generated
when `"float32"` is present in the operator's `quant_bits` in `op_cfg.toml`.

### 6.3 Step 2: Build Test Application

Compiles the esp-dl test app with the operator's model data:

```bash
${DOCKER_BASE} ${ESP_DL_IMAGE} bash -c "${DOCKER_PREAMBLE} && \
  python test_apps/build_apps.py test_apps/esp-dl \
    -op ${OP_TYPE} -t ${TARGET} -vv"
```

### 6.4 Step 3: Generate Pytest Script

Creates the pytest file for the specific operator:

```bash
${DOCKER_BASE} ${ESP_DL_IMAGE} bash -c "${DOCKER_PREAMBLE} && \
  python test_apps/esp-dl/gen_op_test.py \
    --target ${TARGET} --env ${TARGET} \
    --op_type ${OP_TYPE} \
    --pytest_file test_apps/esp-dl/pytest_espdl_op.py"
```

### 6.5 Step 4: Flash & Run Tests on Hardware

**Before flashing, always detect the serial port programmatically — never assume the device
is disconnected or ask the user without checking first.** Run the detection command below
and inspect the output. If it returns one or more device paths (e.g. `/dev/ttyUSB0`), the
device IS connected — proceed directly to flashing. Only if the command returns empty output
should you inform the user that no device was found.

```bash
# Step A: Detect serial port (ALWAYS run this first, don't skip)
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

# Step B: Set the port and flash (only after Step A confirms a device exists)
SERIAL_PORT=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1)

${DOCKER_BASE} --device ${SERIAL_PORT} --group-add dialout \
  ${ESP_DL_IMAGE} bash -c "${DOCKER_PREAMBLE} && \
  pytest test_apps/esp-dl/pytest_espdl_op.py \
    --target ${TARGET} --env ${TARGET} \
    --model ${OP_TYPE} -v"
```

Common device paths on Linux: `/dev/ttyUSB0`, `/dev/ttyUSB1`, `/dev/ttyACM0`.
If multiple ports exist, the first one (`head -1`) is usually correct for single-board setups.
For JTAG+UART dual-port setups (common on ESP32-P4 devkits), both `/dev/ttyUSB0` and
`/dev/ttyUSB1` may appear — the higher-numbered port is typically the UART console.

### 6.6 Quick One-Liner (All Steps)

For convenience, you can chain Steps 1-3 in a single Docker run (excluding
hardware test which needs device access):

```bash
${DOCKER_BASE} ${ESP_DL_IMAGE} bash -c "${DOCKER_PREAMBLE} && \
  python tools/ops_test/gen_test_cases.py \
    --config tools/ops_test/config/op_cfg.toml \
    --ops ${OP_TYPE} \
    --output-path test_apps/esp-dl/models/${TARGET} \
    --target ${TARGET} --bits 8 && \
  python tools/ops_test/gen_test_cases.py \
    --config tools/ops_test/config/op_cfg.toml \
    --ops ${OP_TYPE} \
    --output-path test_apps/esp-dl/models/${TARGET} \
    --target ${TARGET} --bits 16 && \
  python tools/ops_test/gen_test_cases.py \
    --config tools/ops_test/config/op_cfg.toml \
    --ops ${OP_TYPE} \
    --output-path test_apps/esp-dl/models/${TARGET} \
    --target ${TARGET} --float && \
  python test_apps/build_apps.py test_apps/esp-dl \
    -op ${OP_TYPE} -t ${TARGET} -vv"
```

### 6.7 Interpreting Test Results

The test framework works by:
1. esp-ppq generates `.espdl` models with `export_test_values=True`
2. Test inputs and expected outputs (from esp-ppq's forward pass) are embedded in `.espdl`
3. esp-dl loads the model, runs inference with test inputs, compares against expected outputs
4. Comparison uses `equal(output, expected, tolerance=2e-5)`, int16 allows ±1 error

**Differences by quantization type:**

| Type | Model suffix | Tolerance | Common failure causes |
|------|-------------|-----------|----------------------|
| int8 | `*_s8.espdl` | Strict (2e-5) | Quantization config mismatch, rounding, exponent calculation |
| int16 | `*_s16.espdl` | ±1 allowed | Similar to int8, but wider range means fewer edge cases |
| float32 | `*_f32.espdl` | 2e-5 | Usually data layout (NCHW vs NHWC), or missing float specialization |

Float32 tests are the easiest to debug because there's no quantization involved — if a float32
test fails, the issue is almost certainly in the computation logic itself or the data layout,
not in scale/exponent handling. Start debugging with float32 tests first.

**When all tests pass on all targets: proceed to Phase 9 to update `operator_support_state.md`.**
This is not optional — the operator documentation must stay in sync with the test config.
Do not consider the task complete until Phase 9 is done.

If tests fail, check:
- Quantization config alignment (scale, zero-point, exponents) — int8/int16 only
- Data layout (NCHW vs NHWC) handling — all types
- Rounding behavior differences — int8/int16 only
- Off-by-one in dimension calculations — all types
- Missing `float` template specialization — float32 only

---

## Phase 7: SIMD Optimization (Optional)

After the C reference implementation passes all tests, you can add SIMD-optimized kernels.
This is optional and should only be done when performance matters.

### 7.1 When to Add SIMD

SIMD optimization is worthwhile for:
- Operators called frequently in inference (Conv, Add, Mul, Relu)
- Operators with data-parallel computation patterns
- Operators processing large tensors

Skip SIMD for:
- Shape manipulation ops (Reshape, Transpose) — no computation
- Operators that are rarely the bottleneck
- Operators with complex control flow

### 7.2 SIMD Architecture Overview

- **ESP32-S3 (TIE728)**: Xtensa SIMD with 128-bit SIMD registers (Q registers),
  instructions like `EE.VLD.128.IP`, `EE.VRELU.S8`, `EE.VSMULAS.S8.QACC`
- **ESP32-P4**: RISC-V with PIE vector extension, similar 128-bit operations

Assembly files go in:
- `dl/base/isa/tie728/dl_tie728_<dtype>_<op>.S`
- `dl/base/isa/esp32p4/dl_esp32p4_<dtype>_<op>.S`

### 7.3 SIMD Implementation Steps

1. Study a similar operator's SIMD code from the same category
2. Write the assembly function with the naming convention:
   - `dl_tie728_s8_<op>_11c` — aligned, int8, TIE728
   - `dl_tie728_s8_unaligned_<op>_11c` — unaligned variant
   - `dl_esp32p4_s8_<op>_11c` — aligned, int8, ESP32-P4
3. Declare the function in the base layer header with `extern "C"`
4. Update the ISA dispatch in `dl_base_<op>.cpp` to use the SIMD function
5. Rerun all tests to ensure correctness

### 7.4 Important SIMD Conventions

- Do NOT use `.section .iram1` — this forces the function into IRAM which is a scarce
  resource on ESP chips. IRAM is needed for interrupt handlers and critical system code.
  Let the linker place functions in flash by default; use `.text` section only.
  (You may see `.section .iram1` in some older files, but it's being phased out.)
- Register conventions: `a2`=output_ptr, `a3`=input_ptr, `a4`=args_struct
- Args struct offsets must match the C struct definition exactly
- Always handle both aligned and unaligned cases
- Include loop unrolling for performance-critical paths

See `references/esp-dl-templates.md` for SIMD template examples.

---

## Phase 8: Alignment Verification

The alignment between esp-dl and esp-ppq is verified through the test framework:

1. **esp-ppq side**: `export_test_values=True` in `gen_test_cases.py` causes the quantized
   forward pass results to be embedded in the `.espdl` model file
2. **esp-dl side**: `Model::test()` in `dl_model_base.cpp` loads these values, runs inference,
   and compares outputs

If alignment fails after all individual steps pass:
- Check that `espdl_typedef.py` op set classification matches esp-dl's behavior
- Verify quantization exponent calculation matches esp-dl's requantize logic
- For LUT ops, ensure the LUT computation in `executor/op/torch/espdl.py` matches
- Check layout annotations in the export process

---

## Phase 9: Update Operator Support State (REQUIRED)

**This phase is mandatory — execute it immediately after all tests pass.** The operator is
not considered fully delivered until `operator_support_state.md` reflects the new operator.
Skipping this step leaves the public documentation out of sync with the actual capabilities.

The script `tools/ops_test/gen_ops_markdown.py` reads `op_cfg.toml` and produces a markdown
table listing each operator with its supported quantization types and restrictions. The generated
file `operator_support_state.md` lives in the esp-dl root directory and serves as the public
reference for which operators are available.

Run from the esp-dl project root (this can run outside Docker — it only reads `op_cfg.toml`):

```bash
cd ${ESP_DL_ROOT}
uv run --with toml --with tabulate \
  python tools/ops_test/gen_ops_markdown.py \
    -c tools/ops_test/config/op_cfg.toml \
    -o .
```

After running, verify the diff looks correct — the new operator should appear in the table
with the right quantization type checkmarks and any restrictions you configured in `op_cfg.toml`.
Show the user the relevant diff so they can confirm the documentation update.

---

## Quick Reference: Complete Checklist

For a new operator `MyOp`:

### esp-dl files to create/modify:
- [ ] `esp-dl/dl/module/include/dl_module_<op>.hpp` — Module class (NEW)
- [ ] `esp-dl/dl/base/dl_base_<op>.hpp` — Base layer header (NEW, if computation needed)
- [ ] `esp-dl/dl/base/dl_base_<op>.cpp` — Base layer impl (NEW, if computation needed)
- [ ] `esp-dl/dl/module/include/dl_module_creator.hpp` — Register deserialize (MODIFY)
- [ ] `tools/ops_test/torch_ops_test.py` or `onnx_ops_test.py` — Test builder (MODIFY)
- [ ] `tools/ops_test/config/op_cfg.toml` — Test config (MODIFY)

### esp-ppq files to verify/modify (Phase 4 checks — ALWAYS do both):
- [ ] `esp_ppq/quantization/quantizer/EspdlQuantizer.py` — Verify op is in `quant_operation_types` (add if missing)
- [ ] `esp_ppq/parser/espdl/espdl_typedef.py` — **Verify op is in correct layout op set** (add if missing — export WILL fail otherwise)
- [ ] `esp_ppq/quantization/quantizer/EspdlQuantizer.py` — Special quant rules in `create_espdl_quant_config()` (if needed)
- [ ] `esp_ppq/parser/espdl/export_patterns.py` — Export patterns: LUT, fusion, weight layout (if needed)
- [ ] `esp_ppq/IR/base/opdef.py` — Custom OpSocket (if needed)

### SIMD files (optional optimization):
- [ ] `esp-dl/dl/base/isa/tie728/dl_tie728_<dtype>_<op>.S` — TIE728 assembly
- [ ] `esp-dl/dl/base/isa/esp32p4/dl_esp32p4_<dtype>_<op>.S` — P4 assembly
- [ ] `esp-dl/dl/base/dl_base_<op>.cpp` — Update ISA dispatch

### Validation:
- [ ] Docker: generate int8 test cases (`--bits 8`)
- [ ] Docker: generate int16 test cases (`--bits 16`)
- [ ] Docker: generate float32 test cases (`--float`)
- [ ] Docker: build test app (build_apps.py)
- [ ] Docker: flash and run tests (pytest)
- [ ] All test cases pass for all target platforms and all quantization types (int8, int16, float32)

### Documentation (REQUIRED — do not skip):
- [ ] Run `gen_ops_markdown.py` to regenerate `operator_support_state.md` (`uv run --with toml --with tabulate python tools/ops_test/gen_ops_markdown.py -c tools/ops_test/config/op_cfg.toml -o .`)
- [ ] Verify the diff shows the new operator with correct quantization type checkmarks
