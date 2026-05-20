#pragma once

extern "C" {
#if CONFIG_XTENSA_BOOST
#include "dl_base_xtensa.h"
#endif

#if CONFIG_PIE_V1_BOOST
#include "dl_base_tie728.h"
#include "dl_base_xtensa.h"
#endif

#if CONFIG_PIE_V2_BOOST
#include "dl_base_esp32p4.h"
#endif // CONFIG_IDF_TARGET_ESP32P4
}
