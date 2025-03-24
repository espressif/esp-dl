#include "dl_module_base.hpp"
#include <string.h>

using namespace dl;

namespace dl {
namespace module {
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

} // namespace module
} // namespace dl
