/*
 *  Copyright (c) 2003-2004, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kfc.h"

typedef struct cached_fft *kfc_cfg;

struct cached_fft {
    int nfft;
    int inverse;
    kiss_fft_cfg cfg;
    kfc_cfg next;
};

static kfc_cfg cache_root = NULL;
static int ncached = 0;

static kiss_fft_cfg find_cached_fft(int nfft, int inverse)
{
    size_t len;
    kfc_cfg cur = cache_root;
    kfc_cfg prev = NULL;
    while (cur) {
        if (cur->nfft == nfft && inverse == cur->inverse)
            break; /*found the right node*/
        prev = cur;
        cur = prev->next;
    }
    if (cur == NULL) {
        /* no cached node found, need to create a new one*/
        kiss_fft_alloc(nfft, inverse, 0, &len);
#ifdef USE_SIMD
        int padding = (16 - sizeof(struct cached_fft)) & 15;
        // make sure the cfg aligns on a 16 byte boundary
        len += padding;
#endif
        cur = (kfc_cfg)KISS_FFT_MALLOC((sizeof(struct cached_fft) + len));
        if (cur == NULL)
            return NULL;
        cur->cfg = (kiss_fft_cfg)(cur + 1);
#ifdef USE_SIMD
        cur->cfg = (kiss_fft_cfg)((char *)(cur + 1) + padding);
#endif
        kiss_fft_alloc(nfft, inverse, cur->cfg, &len);
        cur->nfft = nfft;
        cur->inverse = inverse;
        cur->next = NULL;
        if (prev)
            prev->next = cur;
        else
            cache_root = cur;
        ++ncached;
    }
    return cur->cfg;
}

void kfc_cleanup(void)
{
    kfc_cfg cur = cache_root;
    kfc_cfg next = NULL;
    while (cur) {
        next = cur->next;
        free(cur);
        cur = next;
    }
    ncached = 0;
    cache_root = NULL;
}
void kfc_fft(int nfft, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
    kiss_fft(find_cached_fft(nfft, 0), fin, fout);
}

void kfc_ifft(int nfft, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
    kiss_fft(find_cached_fft(nfft, 1), fin, fout);
}

#ifdef KFC_TEST
static void check(int nc)
{
    if (ncached != nc) {
        fprintf(stderr, "ncached should be %d,but it is %d\n", nc, ncached);
        exit(1);
    }
}

int main(void)
{
    kiss_fft_cpx buf1[1024], buf2[1024];
    memset(buf1, 0, sizeof(buf1));
    check(0);
    kfc_fft(512, buf1, buf2);
    check(1);
    kfc_fft(512, buf1, buf2);
    check(1);
    kfc_ifft(512, buf1, buf2);
    check(2);
    kfc_cleanup();
    check(0);
    return 0;
}
#endif
