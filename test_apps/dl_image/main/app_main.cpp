#include "esp_heap_caps.h"
#include "unity.h"
#include "unity_test_runner.h"

#define TEST_INTERNAL_MEMORY_LEAK_THRESHOLD (-800)
#define TEST_SPIRAM_MEMORY_LEAK_THRESHOLD (-100)

static int internal_leak_threshold = TEST_INTERNAL_MEMORY_LEAK_THRESHOLD;
static int spiram_leak_threshold = TEST_SPIRAM_MEMORY_LEAK_THRESHOLD;

void set_internal_leak_threshold(int threshold)
{
    internal_leak_threshold = threshold;
}
void set_spiram_leak_threshold(int threshold)
{
    spiram_leak_threshold = threshold;
}

static size_t before_free_internal;
static size_t before_free_spiram;

static void check_leak(size_t before_free, size_t after_free, const char *type, int leak_threshold)
{
    ssize_t delta = after_free - before_free;
    printf(
        "MALLOC_CAP_%s: Before %u bytes free, After %u bytes free (delta %d)\n", type, before_free, after_free, delta);
    TEST_ASSERT_MESSAGE(delta >= leak_threshold, "memory leak");
}

void setUp(void)
{
    before_free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    before_free_spiram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

void tearDown(void)
{
    size_t after_free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t after_free_spiram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    check_leak(before_free_internal, after_free_internal, "INTERNAL", internal_leak_threshold);
    check_leak(before_free_spiram, after_free_spiram, "SPIRAM", spiram_leak_threshold);

    internal_leak_threshold = TEST_INTERNAL_MEMORY_LEAK_THRESHOLD;
    spiram_leak_threshold = TEST_SPIRAM_MEMORY_LEAK_THRESHOLD;
}

extern "C" void app_main(void)
{
    unity_run_menu();
}
