#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*dl_kernel_erased_t)();

/** Look up an intern kernel name. Returns NULL if the symbol was stripped or unknown. */
dl_kernel_erased_t dl_kernel_lookup(const char *name);

#ifdef __cplusplus
}
#endif
