#pragma once
#include "dl_base.hpp"
#include "dl_define.hpp"
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"
#include "dl_tool_cache.hpp"
#include "fbs_model.hpp"
#include <functional>
#include <iostream>
namespace dl {
// Define the enum type for module in-place operation mode
typedef enum {
    MODULE_NON_INPLACE = 0,              ///< Non inplace operation. the output will store to a separate memory
    MODULE_INPLACE_UNCHANGED_BUFFER = 1, ///< Inplace operation which don't change the buffer data.
    MODULE_INPLACE_CHANGED_BUFFER = 2    ///< Inplace operation which will change the buffer data.
} module_inplace_t;

namespace module {
/**
 * @brief Base class for module.
 */
class Module {
public:
    char *name;                       ///< Name of module
    module_inplace_t inplace;         ///< Inplace type
    quant_type_t quant_type;          ///< Quantization type
    std::vector<int> m_inputs_index;  ///< Tensor index of model's tensors that used for inputs
    std::vector<int> m_outputs_index; ///< Tensor index of model's tensors that used for outputs

    /**
     * @brief Construct a new Module object.
     *
     * @param name Name of module.
     * @param inplace   Inplace operation mode
     * @param quant_type Quantization type
     */
    Module(const char *name = NULL,
           module_inplace_t inplace = MODULE_NON_INPLACE,
           quant_type_t quant_type = QUANT_TYPE_NONE);

    /**
     * @brief Destroy the Module object. Return resource.
     *
     */
    virtual ~Module();

    /**
     * @brief Retrieve the shape of this module's inputs
     *
     * @param input_shapes The feature_map shape of this module's inputs.
     * @param inputs If the module has constant inputs, the order and quantity
     *              of the parameters passed must be consistent with those defined
     *              in ONNX. If it is a constant input, pass in its TensorBase pointer;
     *              if not, pass in nullptr.
     *
     * @return std::vector<std::vector<int>> Input shapes
     */
    virtual std::vector<std::vector<int>> retrieve_inputs_shape(std::vector<std::vector<int>> &input_shapes,
                                                                std::vector<dl::TensorBase *> inputs = {});

    /**
     * @brief Retrieve the module's inputs
     *
     * @param tensors All inputs and outputs from MemoryManager
     * @param inputs If the module has constant inputs, the order and quantity
     *              of the parameters passed must be consistent with those defined
     *              in ONNX. If it is a constant input, pass in its TensorBase pointer;
     *              if not, pass in nullptr.
     * @return std::vector<TensorBase *> The final inputs.
     */
    virtual std::vector<TensorBase *> retrieve_inputs(std::vector<TensorBase *> &tensors,
                                                      std::vector<dl::TensorBase *> inputs = {});

    /**
     * @brief Get the tensor index of this module's outputs
     *
     * @return Tensor index of model's tensors
     */
    virtual std::vector<int> get_outputs_index() { return m_outputs_index; }

    /**
     * @brief Calculate output shape by input shape
     *
     * @param input_shapes   Input shapes
     *
     * @return outputs shapes
     */
    virtual std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) = 0;

    /**
     * @brief Run the module, high-level inferface for model layer
     *
     * @param tensors       All inputs and outputs from MemoryManager
     * @param mode    Runtime mode, default is RUNTIME_MODE_AUTO
     *
     */
    virtual void forward(std::vector<dl::TensorBase *> &tensors, runtime_mode_t mode = RUNTIME_MODE_AUTO) = 0;

    /**
     * @brief Run the module, Low-level interface for base layer and multi-core processing
     *
     * @param args      ArgsType, arithArgsType, resizeArgsType and so on
     */
    virtual void forward_args(void *args) {};

    /**
     * @brief create module instance by node serialization information
     *
     * @param fbs_model  Flatbuffer's model
     * @param node_name  The node name in model's graph
     *
     * @return The pointer of module instance
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name) { return nullptr; }

    /**
     * @brief print module information
     */
    virtual void print() {}

    /**
     * @brief set preload RAM pointer
     *
     * @param addr Internal RAM address, should be aligned to 16 bytes
     * @param size The size of RAM address
     */
    virtual void set_preload_addr(void *addr, size_t size) {}

    /**
     * @brief Perform a preload operation
     *
     * @warning Not implemented
     */
    virtual void preload() {}

    /**
     * @brief reset all state of module, include inputs， outputs and preload cache setting
     */
    virtual void reset()
    {
        this->m_inputs_index.clear();
        this->m_outputs_index.clear();
    }

    /**
     * @brief Run the module with single input and single output
     *
     * @param input   Input tensor
     * @param output  Output tensor
     * @param mode    Runtime mode
     */
    virtual void run(TensorBase *input, TensorBase *output, runtime_mode_t mode = RUNTIME_MODE_AUTO);

    /**
     * @brief Run the module by inputs and outputs
     *
     * @param inputs   Input tensors
     * @param outputs  Output tensors
     * @param mode    Runtime mode
     */
    virtual void run(std::vector<dl::TensorBase *> inputs,
                     std::vector<dl::TensorBase *> outputs,
                     runtime_mode_t mode = RUNTIME_MODE_AUTO);

    /**
     * @brief Get the memory size of module parameters
     *
     * @param param         Module parameter tensor
     * @param in_fbs        Memory info of parameter inside Flatbuffers model
     * @param out_fbs       Memory info of parameter outside Flatbuffers model
     * @param fbs_model     Flatbuffers model
     */
    static void get_param_memory_size(dl::TensorBase *param,
                                      mem_info *in_fbs,
                                      mem_info *out_fbs,
                                      fbs::FbsModel *fbs_model)
    {
        *in_fbs = {};
        *out_fbs = {};
        if (param) {
            memory_addr_type_t mem_type = dl::tool::memory_addr_type(param->data);
            switch (mem_type) {
            case MEMORY_ADDR_INTERNAL:
                if (fbs_model->memory_addr_in_model(param->data)) {
                    in_fbs->internal = param->get_bytes();
                } else {
                    out_fbs->internal = param->get_bytes();
                }
                break;
            case MEMORY_ADDR_PSRAM:
                if (fbs_model->memory_addr_in_model(param->data)) {
                    in_fbs->psram = param->get_bytes();
                } else {
                    out_fbs->psram = param->get_bytes();
                }
                break;
            case MEMORY_ADDR_FLASH:
                if (fbs_model->memory_addr_in_model(param->data)) {
                    in_fbs->flash = param->get_bytes();
                } else {
                    out_fbs->flash = param->get_bytes();
                }
                break;
            default:
                ESP_LOGE("module", "Wrong memory addr type.");
            }
        }
    }

    /**
     * @brief Get the memory size of module parameters
     *
     * @param params        Module parameter tensors
     * @param in_fbs        Memory info of parameters inside Flatbuffers model
     * @param out_fbs       Memory info of parameters outside Flatbuffers model
     * @param fbs_model     Flatbuffers model
     */
    static void get_param_memory_size(const std::vector<dl::TensorBase *> &params,
                                      mem_info *in_fbs,
                                      mem_info *out_fbs,
                                      fbs::FbsModel *fbs_model)
    {
        *in_fbs = {};
        *out_fbs = {};
        for (const auto &param : params) {
            if (param) {
                memory_addr_type_t mem_type = dl::tool::memory_addr_type(param->data);
                switch (mem_type) {
                case MEMORY_ADDR_INTERNAL:
                    if (fbs_model->memory_addr_in_model(param->data)) {
                        in_fbs->internal += param->get_bytes();
                    } else {
                        out_fbs->internal += param->get_bytes();
                    }
                    break;
                case MEMORY_ADDR_PSRAM:
                    if (fbs_model->memory_addr_in_model(param->data)) {
                        in_fbs->psram += param->get_bytes();
                    } else {
                        out_fbs->psram += param->get_bytes();
                    }
                    break;
                case MEMORY_ADDR_FLASH:
                    if (fbs_model->memory_addr_in_model(param->data)) {
                        in_fbs->flash += param->get_bytes();
                    } else {
                        out_fbs->flash += param->get_bytes();
                    }
                    break;
                default:
                    ESP_LOGE("module", "Wrong memory addr type.");
                }
            }
        }
    }

    /**
     * @brief Get the memory size of module parameters
     *
     * @param in_fbs        Memory info of module parameters inside Flatbuffers model
     * @param out_fbs       Memory info of module parameters outside Flatbuffers model
     * @param fbs_model     Flatbuffers model
     */
    virtual void get_param_memory_size(mem_info *in_fbs, mem_info *out_fbs, fbs::FbsModel *fbs_model)
    {
        *in_fbs = {};
        *out_fbs = {};
    };
};

/**
 * @brief The data struct of module task. Pack all necessary information as the input for module task.
 */
typedef struct {
    Module *op;                   ///< Module instance pointer
    void *args;                   ///< ArgsType, arithArgsType, resizeArgsType and so on
    SemaphoreHandle_t &semaphore; ///< recommend xSemaphoreCreateCounting
} module_task_data_t;

/**
 * @brief The function of module task.
 * @param args The data of module task.
 */
static void module_forward_task(void *args)
{
    module_task_data_t *task = (module_task_data_t *)args;
    task->op->forward_args(task->args);
    xSemaphoreGive(task->semaphore);
    vTaskSuspend(NULL);
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
/**
 * @brief Run the module with dual core and use semaphores to keep tasks in sync
 *
 * @param op            Module instance
 * @param args1         Task1 args: ArgsType, arithArgsType, resizeArgsType and so on
 * @param args2         Task2 args: ArgsType, arithArgsType, resizeArgsType and so on
 */
static void module_forward_dual_core(Module *op, void *args1, void *args2)
{
    BaseType_t current_core_id = xPortGetCoreID();
    UBaseType_t current_priority = uxTaskPriorityGet(xTaskGetCurrentTaskHandle());
    SemaphoreHandle_t semaphore = xSemaphoreCreateCounting(2, 0);
    TaskHandle_t xHandleTask1, xHandleTask2;

    module_task_data_t task_data1 = {
        .op = op,
        .args = args1,
        .semaphore = semaphore,
    };
    xTaskCreatePinnedToCore(
        module_forward_task, NULL, 2048, &task_data1, current_priority, &xHandleTask1, (current_core_id + 1) % 2);

    module_task_data_t task_data2 = {
        .op = op,
        .args = args2,
        .semaphore = semaphore,
    };
    xTaskCreatePinnedToCore(
        module_forward_task, NULL, 2048, &task_data2, current_priority, &xHandleTask2, current_core_id);

    xSemaphoreTake(semaphore, portMAX_DELAY);
    xSemaphoreTake(semaphore, portMAX_DELAY);
    vSemaphoreDelete(semaphore);
    vTaskDelete(xHandleTask1);
    vTaskDelete(xHandleTask2);
}
#pragma GCC diagnostic pop

} // namespace module
} // namespace dl
