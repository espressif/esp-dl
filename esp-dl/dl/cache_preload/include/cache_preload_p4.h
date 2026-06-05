/*
 * MIT License
 *
 * Copyright (c) 2025 Boumedine Billal
 *
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Error Codes for Cache Preload
 */
/**
 * @brief Error Codes / Result Types for Cache Preload
 */
typedef enum {
    CACHE_PRELOAD_OK = 0,
    CACHE_PRELOAD_ERR_INVALID_SIZE = 100,
    CACHE_PRELOAD_ERR_BUSY = 101,
    CACHE_PRELOAD_ERR_INVALID_ARG = 102,
    
    /* Hardware reported errors */
    CACHE_PRELOAD_HW_ERR_SIZE = 2,
    CACHE_PRELOAD_HW_ERR_UNKNOWN = -1
} cache_preload_result_t;

/* ==================================================================================
 *                                L1 DATA CACHE PRELOAD
 * ================================================================================== */

/**
 * @brief Start a non-blocking preload operation for L1 Data Cache.
 * 
 * Triggers the hardware to fetch data from L2 into L1.
 * Returns immediately. You must check 'is_done' to know when it finishes.
 * 
 * @param addr Physical address to start fetching from.
 * @param size Number of bytes to fetch (Max 16384 bytes).
 * @return cache_preload_result_t CACHE_PRELOAD_OK if started, or error code.
 */
cache_preload_result_t cache_preload_l1_data_start(void *addr, uint32_t size);

/**
 * @brief Check if the L1 Data preload engine is finished/idle.
 * 
 * @return true if the engine is Idle (Job Done).
 * @return false if the engine is Busy (Working).
 */
bool cache_preload_l1_data_is_done(void);

/**
 * @brief Check if the last L1 preload caused an exception.
 * 
 * @return int HW error code (0 if success, 2 if size error).
 */
int cache_preload_l1_data_get_error(void);


/* ==================================================================================
 *                                   L2 CACHE PRELOAD
 * ================================================================================== */

/**
 * @brief Start a non-blocking preload operation for L2 Cache.
 * 
 * Triggers the hardware to fetch data from External Memory (PSRAM/Flash) into L2.
 * 
 * @param addr Physical address to start fetching from.
 * @param size Number of bytes to fetch (Max 65535 bytes).
 * @return cache_preload_result_t CACHE_PRELOAD_OK if started, or error code.
 */
cache_preload_result_t cache_preload_l2_start(void *addr, uint32_t size);

/**
 * @brief Check if the L2 preload engine is finished/idle.
 * 
 * @return true if the engine is Idle (Job Done).
 * @return false if the engine is Busy (Working).
 */
bool cache_preload_l2_is_done(void);

/**
 * @brief Check if the last L2 preload caused an exception.
 * 
 * @return int HW error code (0 if success, 2 if size error).
 */
int cache_preload_l2_get_error(void);

/* ==================================================================================
 *                                   UTILITIES
 * ================================================================================== */

/**
 * @brief Flush a range of addresses from the data cache.
 * 
 * @param addr Start address
 * @param size Size in bytes
 */
void cache_flush_range(void *addr, size_t size);

#ifdef __cplusplus
}
#endif
