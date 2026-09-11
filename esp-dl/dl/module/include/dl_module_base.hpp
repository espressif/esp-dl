#pragma once
#include "dl_base.hpp"
#include "dl_define.hpp"
#include "dl_model_context.hpp"
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"
#include "dl_tool_cache.hpp"
#include "fbs_model.hpp"
#include <functional>
#include <iostream>

namespace dl {
// Define the enum type for module in-place operation mode
typedef enum {
    MODULE_NON_INPLACE = 0, ///< Non inplace operation. the output will store to a separate memory
    MODULE_INPLACE_UNCHANGED_BUFFER =
        1,                            ///< Inplace operation which don't change the buffer data, like Reshape, Squeeze
    MODULE_INPLACE_CHANGED_BUFFER = 2 ///< Inplace operation which will change the buffer data, like Add, Sub
} module_inplace_t;

namespace module {
/**
 * @brief One scratch buffer a module asks the memory planner for.
 */
struct WorkspaceRequest {
    size_t size = 0;         ///< Size in bytes
    bool psram_only = false; ///< Never takes the model's internal RAM budget; ignored without PSRAM
};

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
    // Native workspace binding IDs assigned by the memory planner.
    std::vector<WorkspaceBindingId> m_workspace_binding_ids;

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
     */
    virtual ~Module();

    /**
     * @brief Allocate the module object itself.
     *
     * MALLOC_CAP_DEFAULT prefers PSRAM when one is present and falls back to internal RAM otherwise,
     * so a model's modules do not spend internal RAM on the objects themselves.
     *
     * @param size Size of the object in bytes
     *
     * @return Pointer to the allocated memory
     */
    void *operator new(size_t size) { return tool::malloc_aligned(size, MALLOC_CAP_DEFAULT); }

    /**
     * @brief Deallocate the module object.
     *
     * @param ptr Pointer previously returned by operator new
     */
    void operator delete(void *ptr) { heap_caps_free(ptr); }

    /**
     * @brief Get the tensor index of this module's outputs
     *
     * @return Tensor index of model's tensors
     */
    virtual std::vector<int> get_outputs_index() { return m_outputs_index; }

    /**
     * @brief Whether this module uses a lookup table (LUT) as an extra input.
     *
     * LUT modules load the table from the model initializer and append it to `m_inputs_index`.
     *
     * @return true if the module is a LUT module, false otherwise
     */
    virtual bool is_lut_module() const { return false; }

    /**
     * @brief Calculate output shape by input shape
     *
     * @param input_shapes   Input shapes
     *
     * @return outputs shapes
     */
    virtual std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) = 0;

    /**
     * @brief Scratch memory needed while this module's forward runs
     *
     * The memory planner places each buffer with a lifetime of this node only and records its native workspace
     * binding ID in `m_workspace_binding_ids`. Workspace bindings are separate from model tensors. The contents
     * are undefined on entry and are not kept between calls. A module run without the planner (e.g. through
     * `run()`) has no workspace bound in its temporary context and must provide its own memory.
     *
     * When internal RAM is planned, a buffer competes for it like a tensor unless it is `psram_only`.
     * A zero-size buffer gets a binding ID but no memory.
     *
     * @param input_shapes   Input shapes
     *
     * @return One request per workspace buffer; empty when none is needed
     */
    virtual std::vector<WorkspaceRequest> get_workspace_requests(std::vector<std::vector<int>> &input_shapes)
    {
        return {};
    }

    /**
     * @brief Build the module, high-level interface for Module layer
     *
     * @param context   Model context including  all inputs and outputs and other runtime information
     * @param mode    Runtime mode, default is RUNTIME_MODE_AUTO
     */
    virtual void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO) = 0;

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
    virtual void reset() {}

    /**
     * @brief Run the module with single input and single output
     *
     * @param input   Input tensor
     * @param output  Output tensor
     * @param mode    Runtime mode
     */
    virtual void run(TensorBase *input, TensorBase *output, runtime_mode_t mode = RUNTIME_MODE_SINGLE_CORE);

    /**
     * @brief Run the module by inputs and outputs
     *
     * @param inputs   Input tensors
     * @param outputs  Output tensors
     * @param mode    Runtime mode
     */
    virtual void run(std::vector<dl::TensorBase *> inputs,
                     std::vector<dl::TensorBase *> outputs,
                     runtime_mode_t mode = RUNTIME_MODE_SINGLE_CORE);
};

/**
 * @brief Run the module with dual core and use semaphores to keep tasks in sync
 *
 * @param op            Module instance
 * @param args1         Task1 args: ArgsType, arithArgsType, resizeArgsType and so on
 * @param args2         Task2 args: ArgsType, arithArgsType, resizeArgsType and so on
 */
void module_forward_dual_core(Module *op, void *args1, void *args2);

} // namespace module
} // namespace dl
