#include "dl_base_conv_select.hpp"
#include <algorithm>

namespace dl {
namespace base {

uint16_t dl_conv_pack_key(int dtype, int group_dw, int kshape, int bias, int act, int per_ch, int c_align)
{
    return (uint16_t)((dtype & 3) | ((group_dw & 1) << 2) | ((kshape & 3) << 3) | ((bias & 1) << 5) | ((act & 3) << 6) |
                      ((per_ch & 1) << 8) | ((c_align & 1) << 9));
}

struct dl_conv_sel_row {
    uint16_t key;
    const char *slot0;
    const char *slot1;
    const char *slot2;
};

static const dl_conv_sel_row dl_conv_sel[] = {
#include "dl_conv_select.inc"
    {0, nullptr, nullptr, nullptr},
};

bool dl_conv_select(uint16_t key, const char **slot0, const char **slot1, const char **slot2)
{
    const dl_conv_sel_row *first = dl_conv_sel;
    const dl_conv_sel_row *last = dl_conv_sel + (sizeof(dl_conv_sel) / sizeof(dl_conv_sel[0]) - 1);
    const dl_conv_sel_row *it =
        std::lower_bound(first, last, key, [](const dl_conv_sel_row &e, uint16_t k) { return e.key < k; });
    if (it == last || it->key != key) {
        return false;
    }
    *slot0 = it->slot0;
    *slot1 = it->slot1;
    *slot2 = it->slot2;
    return true;
}

} // namespace base
} // namespace dl
