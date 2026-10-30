#pragma once

// Conv, depthwise, and tail names are constexpr pointers in dl_c_conv.hpp,
// dl_c_dwconv.hpp, and dl_c_conv_tail.hpp. This header is on the public include path (via
// dl_base_isa.hpp on plain-C targets) and must not pull in the generated
// dl_compile_config.h or the heavy dl_base_conv_args.hpp. Each kernel header
// forward-declares ConvArgsType; implementations (dl_c_*.cpp) include the
// full definition themselves.

#include "dl_c_conv.hpp"
#include "dl_c_conv_tail.hpp"
#include "dl_c_dwconv.hpp"
