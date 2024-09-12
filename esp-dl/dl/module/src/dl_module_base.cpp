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
} // namespace module
} // namespace dl
