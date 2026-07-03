#include "dl_module_base.hpp"

#include <assert.h>
#include <freertos/idf_additions.h>
#include <freertos/semphr.h>
#include <string.h>

using namespace dl;

namespace dl {
namespace module {

namespace {

#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1

constexpr size_t kWorkerCount = 2;
constexpr uint32_t kWorkerStackBytes = 2048;
constexpr UBaseType_t kWorkerDefaultPriority = tskIDLE_PRIORITY + 1;

struct DualCoreWorkerRuntime;

struct DualCoreWorkerTaskArgs {
    DualCoreWorkerRuntime *runtime = nullptr;
    size_t slot = 0;
};

struct DualCoreWorkerRuntime {
    StaticSemaphore_t dispatch_mutex_buffer = {};
    SemaphoreHandle_t dispatch_mutex = nullptr;
    StaticSemaphore_t completion_buffer = {};
    SemaphoreHandle_t completion = nullptr;
    StaticTask_t task_buffers[kWorkerCount] = {};
    StackType_t task_stacks[kWorkerCount][kWorkerStackBytes / sizeof(StackType_t)] = {};
    TaskHandle_t task_handles[kWorkerCount] = {nullptr, nullptr};
    DualCoreWorkerTaskArgs task_args[kWorkerCount] = {};
    Module *op = nullptr;
    void *args[kWorkerCount] = {nullptr, nullptr};
};

void DualCoreWorkerTask(void *arg)
{
    auto *task_arg = static_cast<DualCoreWorkerTaskArgs *>(arg);
    auto *runtime = task_arg->runtime;
    const size_t slot = task_arg->slot;
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        __sync_synchronize();
        Module *op = runtime->op;
        void *task_args = runtime->args[slot];
        if (op != nullptr) {
            op->forward_args(task_args);
        }
        xSemaphoreGive(runtime->completion);
    }
}

bool InitDualCoreWorkerRuntime(DualCoreWorkerRuntime &runtime)
{
    runtime.dispatch_mutex = xSemaphoreCreateMutexStatic(&runtime.dispatch_mutex_buffer);
    runtime.completion = xSemaphoreCreateCountingStatic(kWorkerCount, 0, &runtime.completion_buffer);
    if (runtime.dispatch_mutex == nullptr || runtime.completion == nullptr) {
        return false;
    }

    for (size_t i = 0; i < kWorkerCount; ++i) {
        runtime.task_args[i].runtime = &runtime;
        runtime.task_args[i].slot = i;
        runtime.task_handles[i] = xTaskCreateStaticPinnedToCore(DualCoreWorkerTask,
                                                                i == 0 ? "dl_mc0" : "dl_mc1",
                                                                kWorkerStackBytes,
                                                                &runtime.task_args[i],
                                                                kWorkerDefaultPriority,
                                                                runtime.task_stacks[i],
                                                                &runtime.task_buffers[i],
                                                                static_cast<BaseType_t>(i));
        if (runtime.task_handles[i] == nullptr) {
            return false;
        }
    }
    return true;
}

DualCoreWorkerRuntime &GetDualCoreWorkerRuntime()
{
    static DualCoreWorkerRuntime runtime;
    static const bool initialized = InitDualCoreWorkerRuntime(runtime);
    assert(initialized);
    (void)initialized;
    return runtime;
}

#endif

} // namespace

Module::Module(const char *name, module_inplace_t inplace, quant_type_t quant_type) :
    inplace(inplace), quant_type(quant_type)
{
#if DL_LOG_MODULE_NAME
    if (name) {
        int length = strlen(name) + 1;
        this->name = (char *)malloc(sizeof(char) * length);
        memcpy(this->name, name, length);
    } else {
        this->name = NULL;
    }
#else
    this->name = NULL;
#endif
}

Module::~Module()
{
    if (this->name) {
        free((void *)this->name);
    }
}

void Module::run(TensorBase *input, TensorBase *output, runtime_mode_t mode)
{
    ModelContext context;
    m_inputs_index.push_back(context.push_back_tensor(input));
    m_outputs_index.push_back(context.push_back_tensor(output));
    forward(&context, mode);
}

void Module::run(std::vector<dl::TensorBase *> inputs, std::vector<dl::TensorBase *> outputs, runtime_mode_t mode)
{
    ModelContext context;
    for (int i = 0; i < inputs.size(); i++) {
        m_inputs_index.push_back(context.push_back_tensor(inputs[i]));
    }

    for (int i = 0; i < outputs.size(); i++) {
        m_outputs_index.push_back(context.push_back_tensor(outputs[i]));
    }

    forward(&context, mode);
}

void module_forward_dual_core(Module *op, void *args1, void *args2)
{
#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
    auto &runtime = GetDualCoreWorkerRuntime();
    xSemaphoreTake(runtime.dispatch_mutex, portMAX_DELAY);
    while (xSemaphoreTake(runtime.completion, 0) == pdTRUE) {
    }

    const UBaseType_t current_priority = uxTaskPriorityGet(xTaskGetCurrentTaskHandle());
    for (size_t i = 0; i < kWorkerCount; ++i) {
        vTaskPrioritySet(runtime.task_handles[i], current_priority);
    }

    runtime.op = op;
    runtime.args[0] = args1;
    runtime.args[1] = args2;
    __sync_synchronize();

    for (size_t i = 0; i < kWorkerCount; ++i) {
        xTaskNotifyGive(runtime.task_handles[i]);
    }
    for (size_t i = 0; i < kWorkerCount; ++i) {
        xSemaphoreTake(runtime.completion, portMAX_DELAY);
    }

    runtime.op = nullptr;
    runtime.args[0] = nullptr;
    runtime.args[1] = nullptr;
    xSemaphoreGive(runtime.dispatch_mutex);
#else
    op->forward_args(args1);
    op->forward_args(args2);
#endif
}

} // namespace module
} // namespace dl
