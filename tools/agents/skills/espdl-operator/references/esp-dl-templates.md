# ESP-DL Operator Implementation Templates

This document provides annotated templates for each operator category. Choose the template
that matches your operator's classification from Phase 1.

## Table of Contents

1. [Elementwise Binary Operator](#elementwise-binary)
2. [Elementwise Unary Operator](#elementwise-unary)
3. [Activation with LUT](#activation-lut)
4. [Shape Manipulation Operator](#shape-manipulation)
5. [Reduce Operator](#reduce)
6. [Naming Conventions](#naming-conventions)
7. [SIMD Assembly Template](#simd-template)

---

## 1. Elementwise Binary Operator {#elementwise-binary}

Reference: `dl_module_add.hpp`, `dl_base_add.hpp/cpp`

These operators take two inputs, apply a binary operation, and produce one output.
They support Numpy-style broadcasting.

### Module Header Template

```cpp
#pragma once

#include "dl_base_<op>.hpp"
#include "dl_base_shape.hpp"
#include "dl_module_base.hpp"

namespace dl {
namespace module {

class MyOp : public Module {
public:
    MyOp(const char *name = NULL,
         module_inplace_t inplace = MODULE_NON_INPLACE,
         quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    ~MyOp() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<int> output_shape =
            base::get_multidirectional_broadcasting_shape(input_shapes[0], input_shapes[1]);
        return std::vector<std::vector<int>>(1, output_shape);
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_template<float>(context, mode);
        }
    }

    void forward_args(void *args)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            base::elemwise_myop((base::elemwiseArgsType<int8_t> *)args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::elemwise_myop((base::elemwiseArgsType<int16_t> *)args);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            base::elemwise_myop((base::elemwiseArgsType<float> *)args);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input0 = context->get_tensor(m_inputs_index[0]);
        TensorBase *input1 = context->get_tensor(m_inputs_index[1]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        std::vector<base::elemwiseArgsType<T>> m_args =
            base::get_elemwise_operation_args<T>(output, input0, input1, mode);
        int task_size = m_args.size();
        if (task_size == 1) {
            forward_args((void *)&m_args[0]);
        } else if (task_size == 2) {
            module_forward_dual_core(this, (void *)&m_args[0], (void *)&m_args[1]);
        } else {
            ESP_LOGE("MyOp", "Only support task size 1 or 2, got %d", task_size);
        }
    }

    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        op = new MyOp(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }
};

} // namespace module
} // namespace dl
```

### Base Layer Template

**Header** (`dl_base_<op>.hpp`):

```cpp
#pragma once

#include "dl_base_elemwise.hpp"

namespace dl {
namespace base {

void elemwise_myop(elemwiseArgsType<int8_t> *args);
void elemwise_myop(elemwiseArgsType<int16_t> *args);
void elemwise_myop(elemwiseArgsType<float> *args);

} // namespace base
} // namespace dl
```

**Implementation** (`dl_base_<op>.cpp`):

```cpp
#include "dl_base_<op>.hpp"
#include "dl_base_isa.hpp"

namespace dl {
namespace base {

// ====== Quantized (int8/int16) C reference ======
// Generic template handles quantization: uses tool::truncate and rescaling.
template <typename T>
inline void c_impl_myop_n_n(T *output_ptr, T *input0_ptr, T *input1_ptr, void *args)
{
    elemwiseArgsType<T> *myargs = (elemwiseArgsType<T> *)args;
    int length = myargs->output_stride[0] * myargs->output_shape[0];
    for (int i = 0; i < length; i++) {
        int32_t result = /* your quantized operation using input0_ptr[i], input1_ptr[i] */;
        output_ptr[i] = tool::truncate<int32_t>(result);
    }
}

// ====== Float32 C reference ======
// Float specialization: direct arithmetic, no truncation, no rescaling.
template <>
inline void c_impl_myop_n_n<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *myargs = (elemwiseArgsType<float> *)args;
    int length = myargs->output_stride[0] * myargs->output_shape[0];
    for (int i = 0; i < length; i++) {
        output_ptr[i] = /* direct float operation: input0_ptr[i] OP input1_ptr[i] */;
    }
}

// Quantized int8 with ISA dispatch
void elemwise_myop(elemwiseArgsType<int8_t> *args)
{
    int ilen = 16 / sizeof(int8_t);  // SIMD vector length
    ImplFunc_t<int8_t, int8_t, int8_t> elemwise_func = c_impl_myop_n_n<int8_t>;

    // ISA dispatch (SIMD implementations added later in Phase 7)
#if CONFIG_IDF_TARGET_ESP32P4
    // if (aligned) { elemwise_func = dl_esp32p4_s8_myop_11c; }
#elif CONFIG_TIE728_BOOST
    // if (aligned) { elemwise_func = dl_tie728_s8_myop_11c; }
#endif

    switch (args->dims) {
    case 1: elemwise_loop_1d(args, elemwise_func); break;
    case 2: elemwise_loop_2d(args, elemwise_func); break;
    case 3: elemwise_loop_3d(args, elemwise_func); break;
    case 4: elemwise_loop_4d(args, elemwise_func); break;
    }
}

// Quantized int16 with ISA dispatch
void elemwise_myop(elemwiseArgsType<int16_t> *args)
{
    int ilen = 16 / sizeof(int16_t);
    ImplFunc_t<int16_t, int16_t, int16_t> elemwise_func = c_impl_myop_n_n<int16_t>;

#if CONFIG_IDF_TARGET_ESP32P4
    // if (aligned) { elemwise_func = dl_esp32p4_s16_myop_11c; }
#elif CONFIG_TIE728_BOOST
    // if (aligned) { elemwise_func = dl_tie728_s16_myop_11c; }
#endif

    switch (args->dims) {
    case 1: elemwise_loop_1d(args, elemwise_func); break;
    case 2: elemwise_loop_2d(args, elemwise_func); break;
    case 3: elemwise_loop_3d(args, elemwise_func); break;
    case 4: elemwise_loop_4d(args, elemwise_func); break;
    }
}

// Float32: C reference only, no SIMD dispatch needed
void elemwise_myop(elemwiseArgsType<float> *args)
{
    ImplFunc_t<float, float, float> elemwise_func = c_impl_myop_n_n<float>;

    switch (args->dims) {
    case 1: elemwise_loop_1d(args, elemwise_func); break;
    case 2: elemwise_loop_2d(args, elemwise_func); break;
    case 3: elemwise_loop_3d(args, elemwise_func); break;
    case 4: elemwise_loop_4d(args, elemwise_func); break;
    }
}

} // namespace base
} // namespace dl
```

Key differences between quantized and float overloads:
- **int8/int16**: Use `tool::truncate<int32_t>()` to clamp results to type range, may use
  `args->mul_shift` / `args->input0_scale` / `args->output_rescale` for requantization
- **float**: Direct arithmetic, no truncation, ignores scale/rescale fields
- **ISA dispatch**: Only for int8/int16 — float has no SIMD and always uses C reference

---

## 2. Elementwise Unary Operator {#elementwise-unary}

Reference: `dl_module_relu.hpp`, `dl_module_sqrt.hpp`, `dl_base_relu.hpp/cpp`

Single input, single output, applied element-wise.

### Module Header Template (Pattern A — all types via base layer)

Use this when the float operation is complex enough to warrant a base-layer function:

```cpp
#pragma once

#include "dl_base_<op>.hpp"
#include "dl_module_base.hpp"

namespace dl {
namespace module {

class MyOp : public Module {
public:
    MyOp(const char *name = NULL,
         module_inplace_t inplace = MODULE_NON_INPLACE,
         quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        return std::vector<std::vector<int>>(1, input_shapes[0]);
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_template<float>(context, mode);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        ArgsType<T> args = base::get_activation_args<T>(output, input, mode);
        base::myop(&args);
    }

    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        op = new MyOp(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }
};

} // namespace module
} // namespace dl
```

### Module Header Template (Pattern B — float inline in module)

For simple operations (like ReLU, Neg, Abs), float32 can be implemented directly in the
module's `forward()` method. This avoids needing a float overload in the base layer:

```cpp
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
                output_ptr[i] = /* direct float operation, e.g., std::abs(input_ptr[i]) */;
            }
        }
    }
```

Choose Pattern A when the operation needs broadcast handling or multi-dimensional looping.
Choose Pattern B when the operation is a simple per-element transform with no broadcasting.

---

## 3. Activation with LUT {#activation-lut}

Reference: `dl_module_lut.hpp`, `dl_module_hard_swish.hpp`

For non-linear activations (Sigmoid, Tanh, HardSwish, etc.), esp-dl uses look-up tables
for int8/int16 quantized inference. The LUT is pre-computed by esp-ppq during export.

The module typically delegates to `LUT::deserialize()` which handles LUT loading and
applies the table during inference. Check `dl_module_lut.hpp` for the pattern.

---

## 4. Shape Manipulation Operator {#shape-manipulation}

Reference: `dl_module_reshape.hpp`, `dl_module_transpose.hpp`

These operators change tensor shape/layout without computation. They typically:
- Use `MODULE_INPLACE_UNCHANGED_BUFFER` (no data copy needed)
- Have no base layer (just metadata changes)
- Forward is very lightweight

---

## 5. Reduce Operator {#reduce}

Reference: `dl_module_reduce_sum.hpp`, `dl_base_reduce.hpp/cpp`

Reduce operators collapse one or more axes. They use `reduceArgsType` and `reduce_loop_*d()`.

---

## 6. Naming Conventions {#naming-conventions}

| ONNX Name | File Name | Class Name | Base Function |
|-----------|-----------|------------|---------------|
| `HardSwish` | `dl_module_hard_swish.hpp` | `HardSwish` | `base::hard_swish()` |
| `MatMul` | `dl_module_matmul.hpp` | `MatMul` | `base::matmul()` |
| `PRelu` | `dl_module_prelu.hpp` | `PRelu` | `base::prelu()` |
| `Add` | `dl_module_add.hpp` | `Add` | `base::elemwise_add()` |
| `ReduceSum` | `dl_module_reduce_sum.hpp` | `ReduceSum` | `base::reduce_sum()` |

Snake case conversion rules:
- `HardSwish` → `hard_swish`
- `MatMul` → `matmul`
- `PRelu` → `prelu`
- `ConvTranspose` → `conv_transpose`
- `GlobalAveragePool` → `global_average_pool`

---

## 7. SIMD Assembly Template {#simd-template}

### TIE728 (ESP32-S3) Assembly Structure

```asm
#include "dl_tie728_s8.S"

    .align 4
    .text
    .global dl_tie728_s8_myop_11c
    .type   dl_tie728_s8_myop_11c, @function
    # Do NOT use .section .iram1 — IRAM is scarce, let linker place in flash

dl_tie728_s8_myop_11c:
    entry sp, 16

    # Register conventions:
    # a2: output_ptr
    # a3: input_ptr (or input0_ptr for binary ops)
    # a4: args struct pointer
    # Read additional args from struct offsets

    # Load loop count from args
    l32i a5, a4, <offset>    # loop count

    # Main SIMD loop
    loopgtz a5, .loop_end
        EE.VLD.128.IP q0, a3, 16    # Load 16 bytes
        # ... your SIMD operation ...
        EE.VST.128.IP q0, a2, 16    # Store 16 bytes
    .loop_end:

    # Handle remaining elements (< 16 bytes)
    # ...

    retw.n
```

### ESP32-P4 RISC-V PIE Assembly Structure

```asm
#include "dl_esp32p4_s8.S"

    .align 4
    .text
    .global dl_esp32p4_s8_myop_11c
    .type   dl_esp32p4_s8_myop_11c, @function
    # Do NOT use .section .iram1 — IRAM is scarce, let linker place in flash

dl_esp32p4_s8_myop_11c:
    # ESP32-P4 uses RISC-V calling convention
    # a0: output_ptr, a1: input_ptr, a2: args_ptr

    # Load loop count
    lw t0, <offset>(a2)

    # Main loop
    .loop:
        esp.vld.128.ip q0, a1, 16
        # ... your SIMD operation ...
        esp.vst.128.ip q0, a0, 16
        addi t0, t0, -1
        bnez t0, .loop

    ret
```

### Declaring SIMD Functions in C++

In the base layer header, declare with `extern "C"`:

```cpp
// In dl_base_<op>.hpp or via dl_base_isa.hpp includes
extern "C" {
void dl_tie728_s8_myop_11c(int8_t *output, int8_t *input, void *args);
void dl_tie728_s8_unaligned_myop_11c(int8_t *output, int8_t *input, void *args);
void dl_esp32p4_s8_myop_11c(int8_t *output, int8_t *input, void *args);
}
```
