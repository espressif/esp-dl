/*
 * MIT License
 *
 * Copyright (c) 2025 Boumedine Billal
 *
 */

#include "cache_preload_p4.h"
#include "soc/soc.h"
#include "soc/cache_struct.h"
#include "soc/cache_reg.h"
#include "esp_cache.h"

/* Hardware Base Address */
static volatile cache_dev_t *HW_CACHE = (volatile cache_dev_t *)DR_REG_CACHE_BASE;

/* Register Constraints defined by Hardware */
#define L1_PRELOAD_MAX_SIZE  0x3FFF  /* 14 bits -> 16383 bytes */
#define L2_PRELOAD_MAX_SIZE  0xFFFF  /* 16 bits -> 65535 bytes */

/* ==================================================================================
 *                            L1 DATA CACHE IMPLEMENTATION
 * ================================================================================== */

cache_preload_result_t cache_preload_l1_data_start(void *addr, uint32_t size)
{
    /* 1. Validate Size Limit */
    if (size > L1_PRELOAD_MAX_SIZE) {
        return CACHE_PRELOAD_ERR_INVALID_SIZE;
    }

    /* 2. Check if Busy */
    /* Bit 1: preload_done. 0 = Busy, 1 = Done/Idle */
    if (HW_CACHE->l1_dcache_preload_ctrl.l1_dcache_preload_done == 0) {
        return CACHE_PRELOAD_ERR_BUSY;
    }

    /* 3. Configure Engine */
    /* Order matters: Addr -> Size -> Order -> Enable */
    HW_CACHE->l1_dcache_preload_addr.l1_dcache_preload_addr = (uint32_t)addr;
    HW_CACHE->l1_dcache_preload_size.l1_dcache_preload_size = size;
    HW_CACHE->l1_dcache_preload_ctrl.l1_dcache_preload_order = 0; // 0 = Ascending

    /* 4. Trigger Start */
    /* Bit 0: preload_ena. Write 1 to start. HW clears it automatically. */
    HW_CACHE->l1_dcache_preload_ctrl.l1_dcache_preload_ena = 1;

    return CACHE_PRELOAD_OK;
}

bool cache_preload_l1_data_is_done(void)
{
    /* Return true if Done bit is 1 */
    return (HW_CACHE->l1_dcache_preload_ctrl.l1_dcache_preload_done == 1);
}

int cache_preload_l1_data_get_error(void)
{
    /* Read Exception Register specific to L1 Data */
    uint32_t err_code = HW_CACHE->sync_l1_cache_preload_exception.l1_dcache_pld_err_code;

    if (err_code == 0) return CACHE_PRELOAD_OK;
    if (err_code == 2) return CACHE_PRELOAD_HW_ERR_SIZE;
    
    return CACHE_PRELOAD_HW_ERR_UNKNOWN;
}


/* ==================================================================================
 *                               L2 CACHE IMPLEMENTATION
 * ================================================================================== */

cache_preload_result_t cache_preload_l2_start(void *addr, uint32_t size)
{
    /* 1. Validate Size Limit */
    if (size > L2_PRELOAD_MAX_SIZE) {
        return CACHE_PRELOAD_ERR_INVALID_SIZE;
    }

    /* 2. Check if Busy */
    if (HW_CACHE->l2_cache_preload_ctrl.l2_cache_preload_done == 0) {
        return CACHE_PRELOAD_ERR_BUSY;
    }

    /* 3. Configure Engine */
    HW_CACHE->l2_cache_preload_addr.l2_cache_preload_addr = (uint32_t)addr;
    HW_CACHE->l2_cache_preload_size.l2_cache_preload_size = size;
    HW_CACHE->l2_cache_preload_ctrl.l2_cache_preload_order = 0; // 0 = Ascending

    /* 4. Trigger Start */
    HW_CACHE->l2_cache_preload_ctrl.l2_cache_preload_ena = 1;

    return CACHE_PRELOAD_OK;
}

bool cache_preload_l2_is_done(void)
{
    return (HW_CACHE->l2_cache_preload_ctrl.l2_cache_preload_done == 1);
}

int cache_preload_l2_get_error(void)
{
    /* Read Exception Register specific to L2 */
    uint32_t err_code = HW_CACHE->l2_cache_sync_preload_exception.l2_cache_pld_err_code;

    if (err_code == 0) return CACHE_PRELOAD_OK;
    if (err_code == 2) return CACHE_PRELOAD_HW_ERR_SIZE;

    return CACHE_PRELOAD_HW_ERR_UNKNOWN;
}

/* ==================================================================================
 *                                   UTILITIES
 * ================================================================================== */

void cache_flush_range(void *addr, size_t size)
{
    // M2C (Memory to Cache) direction invalidation requires strict alignment
    // Cache Line Size for L2 is 128 Bytes.
    // We must align the start address DOWN and the size UP to cover the full range.
    
    uint32_t start_addr = (uint32_t)addr;
    uint32_t end_addr = start_addr + size;
    
    // Align Start Address DOWN to 128-byte boundary
    uint32_t aligned_start = start_addr & ~(128 - 1);
    
    // Align End Address UP to 128-byte boundary
    uint32_t aligned_end = (end_addr + (128 - 1)) & ~(128 - 1);
    
    uint32_t aligned_size = aligned_end - aligned_start;

    // Flush (Invalidate) with M2C direction, NO unaligned flag
    esp_cache_msync((void*)aligned_start, aligned_size, ESP_CACHE_MSYNC_FLAG_DIR_M2C);
}
