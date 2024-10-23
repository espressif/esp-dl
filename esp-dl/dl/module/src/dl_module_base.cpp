#include "dl_module_base.hpp"
#include <string.h>

using namespace dl;

namespace dl {
namespace module {
Module::Module(const char *name, module_inplace_t inplace, quant_type_t quant_type) :
    inplace(inplace), quant_type(quant_type)
{
    if (name) {
        int length = strlen(name) + 1;
        this->name = (char *)malloc(sizeof(char) * length);
        memcpy(this->name, name, length);
    } else {
        this->name = NULL;
    }
}

Module::~Module()
{
    if (this->name) {
        free((void *)this->name);
    }
}

void Module::run(TensorBase *input, TensorBase *output, runtime_mode_t mode)
{
    std::vector<dl::TensorBase *> tensors = {input, output};
    m_inputs_index.push_back(0);
    m_outputs_index.push_back(1);
    forward(tensors, mode);
}

void Module::run(std::vector<dl::TensorBase *> inputs, std::vector<dl::TensorBase *> outputs, runtime_mode_t mode)
{
    std::vector<dl::TensorBase *> tensors;
    for (int i = 0; i < inputs.size(); i++) {
        tensors.push_back(inputs[i]);
        m_inputs_index.push_back(tensors.size() - 1);
    }

    for (int i = 0; i < outputs.size(); i++) {
        tensors.push_back(outputs[i]);
        m_outputs_index.push_back(tensors.size() - 1);
    }

    forward(tensors, mode);
}

} // namespace module
} // namespace dl
