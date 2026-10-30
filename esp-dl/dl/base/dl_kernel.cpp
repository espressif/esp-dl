#include "dl_kernel.hpp"
#include "dl_base_isa.hpp"
#include "dl_compile_config.h"
#include <algorithm>
#include <cstring>

struct dl_kernel_entry {
    const char *name;
    dl_kernel_erased_t fn;
};

static const dl_kernel_entry dl_kernel_table[] = {
#include "dl_kernel.inc"
    {nullptr, nullptr},
};

extern "C" dl_kernel_erased_t dl_kernel_lookup(const char *name)
{
    if (!name) {
        return nullptr;
    }
    const dl_kernel_entry *first = dl_kernel_table;
    const dl_kernel_entry *last = dl_kernel_table + (sizeof(dl_kernel_table) / sizeof(dl_kernel_table[0]) - 1);
    const dl_kernel_entry *it = std::lower_bound(
        first, last, name, [](const dl_kernel_entry &e, const char *n) { return std::strcmp(e.name, n) < 0; });
    if (it == last || std::strcmp(it->name, name) != 0) {
        return nullptr;
    }
    return it->fn;
}
