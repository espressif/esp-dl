#include "dl_module_creator.hpp"
#include "dl_compile_config.h"

// Gated #include of each module header, generated from spec/ops.yml.
#include "dl_module_includes.inc"

namespace dl {
namespace module {

/**
 * @brief Pre-register the already implemented modules.
 *        The gated register_module() rows are generated from spec/ops.yml
 *        into dl_module_register.inc. This file is a fixed shell: adding an
 *        op means editing spec/ops.yml, not this code.
 */
void ModuleCreator::register_dl_modules()
{
    if (creators.empty()) {
#include "dl_module_register.inc"
    }
}

} // namespace module
} // namespace dl
