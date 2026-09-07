#include "dl_fft_base.h"
#include "sys/lock.h"

typedef struct dl_fft_table_node {
    dl_fft_table_kind_t kind;
    int fft_point;
    uint32_t caps;
    void *data;
    int extra;
    int refcount;
    struct dl_fft_table_node *next;
} dl_fft_table_node_t;

static _lock_t s_lock;
static dl_fft_table_node_t *s_list = NULL;

// newlib _lock_t is lazily turned into a FreeRTOS mutex on first _lock_acquire
// (xQueueCreateMutex). That heap object lives for the process lifetime, so the
// first FFT after boot looks like a leak. Create it at load time instead.
static void __attribute__((constructor)) dl_fft_table_lock_init(void)
{
    _lock_init(&s_lock);
}

static void *generate_table(dl_fft_table_kind_t kind, int fft_point, uint32_t caps, int *extra)
{
    switch (kind) {
    case DL_FFT_TBL_F32_FFT2R:
        return dl_gen_fft2r_table_f32(fft_point, caps);
    case DL_FFT_TBL_F32_FFT4R:
        return dl_gen_fft4r_table_f32(fft_point, caps);
    case DL_FFT_TBL_F32_RFFT:
        return dl_gen_rfft_table_f32(fft_point, caps);
    case DL_FFT_TBL_F32_BITREV2R:
        return dl_gen_bitrev2r_table(fft_point, caps, extra);
    case DL_FFT_TBL_F32_BITREV4R:
        return dl_gen_bitrev4r_table(fft_point, caps, extra);
    case DL_FFT_TBL_S16_DIF_FFT:
        return dl_gen_dif_fft_table(fft_point, caps);
    case DL_FFT_TBL_S16_DIF_RFFT:
        return dl_gen_dif_rfft_table(fft_point, caps);
    default:
        return NULL;
    }
}

void *dl_fft_table_acquire(dl_fft_table_kind_t kind, int fft_point, uint32_t caps, int *extra)
{
    _lock_acquire(&s_lock);

    for (dl_fft_table_node_t *node = s_list; node; node = node->next) {
        if (node->kind == kind && node->fft_point == fft_point && node->caps == caps) {
            node->refcount++;
            if (extra) {
                *extra = node->extra;
            }
            _lock_release(&s_lock);
            return node->data;
        }
    }

    int local_extra = 0;
    void *data = generate_table(kind, fft_point, caps, &local_extra);
    if (extra) {
        *extra = local_extra;
    }
    if (!data) {
        _lock_release(&s_lock);
        return NULL;
    }

    dl_fft_table_node_t *node = (dl_fft_table_node_t *)heap_caps_malloc(sizeof(dl_fft_table_node_t), MALLOC_CAP_8BIT);
    if (!node) {
        heap_caps_free(data);
        _lock_release(&s_lock);
        return NULL;
    }

    node->kind = kind;
    node->fft_point = fft_point;
    node->caps = caps;
    node->data = data;
    node->extra = local_extra;
    node->refcount = 1;
    node->next = s_list;
    s_list = node;

    _lock_release(&s_lock);
    return data;
}

void dl_fft_table_release(void *data)
{
    if (!data) {
        return;
    }

    _lock_acquire(&s_lock);

    dl_fft_table_node_t **pp = &s_list;
    while (*pp) {
        if ((*pp)->data == data) {
            (*pp)->refcount--;
            if ((*pp)->refcount <= 0) {
                dl_fft_table_node_t *victim = *pp;
                *pp = victim->next;
                heap_caps_free(victim->data);
                heap_caps_free(victim);
            }
            _lock_release(&s_lock);
            return;
        }
        pp = &(*pp)->next;
    }

    _lock_release(&s_lock);
}
