#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "unity.h"
#include <type_traits>

static const char *TAG = "TEST DL MODEL";

// using namespace fbs;

void init_chip()
{
#if CONFIG_ESP32P4_BOOST
    //  *((uint32_t *)(0x50000008)) = 0; // turn of bypass
    asm volatile("li t0, 0x2000\n"
                 "csrrs t0, mstatus, t0\n"); /* FPU_state = 1 (initial) */
    asm volatile("li t0, 0x1\n"
                 "csrrs t0, 0x7F1, t0\n"); /* HWLP_state = 1 (initial) */
    asm volatile("li t0, 0x1\n"
                 "csrrs t0, 0x7F2, t0\n"); /* AIA_state = 1 (initial) */
#endif
}

using namespace dl;

uint8_t key[16] = {0x8a, 0x7f, 0xc9, 0x61, 0xe4, 0xe6, 0xff, 0x0a, 0xd2, 0x64, 0x36, 0x95, 0x28, 0x75, 0xae, 0x4a};

TEST_CASE("Test dl model API: load()", "[load]")
{
    ESP_LOGI(TAG, "get into app_main");
    init_chip();
    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    int internal_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, key);
    delete model;

    int internal_ram_size_second = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

    // model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, key);
    // delete model;

    fbs::FbsLoader *fbs_loader = new fbs::FbsLoader("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    fbs::FbsModel *fbs_model = fbs_loader->load(0, key);
    Model *model2 = new Model(fbs_model);
    delete model2;
    delete fbs_loader;
    delete fbs_model;

    model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    delete model;

    int internal_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);

    printf("internal ram size before: %d, second: %d, end:%d\n",
           internal_ram_size_before,
           internal_ram_size_second,
           internal_ram_size_end);
    printf("psram size before: %d, end:%d\n", psram_size_before, psram_size_end);

    TEST_ASSERT_EQUAL(true, internal_ram_size_second == internal_ram_size_end);
    TEST_ASSERT_EQUAL(true, psram_size_before == psram_size_end);
}

TEST_CASE("Test dl model API: build()", "[build]")
{
    ESP_LOGI(TAG, "get into app_main");
    init_chip();
    heap_caps_print_heap_info(MALLOC_CAP_INTERNAL);
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    delete model;
    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    for (int i = 0; i < 1; i++) {
        Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
        model->build(0);
        delete model;
    }

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    printf("ram size before: %d, ram size after:%d\n", total_ram_size_before, total_ram_size_end);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}

TEST_CASE("Test dl model API: run()", "[run]")
{
    ESP_LOGI(TAG, "get into app_main");
    init_chip();
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    delete model;

    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    model->build(128 * 1000);

    dl::tool::Latency latency;
    latency.start();
    model->run();
    latency.end();
    printf("run:%ld ms\n", latency.get_period() / 1000);

    delete model;
    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    printf("ram size before: %d, ram size after:%d\n", total_ram_size_before, total_ram_size_end);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}

// std::vector<int> shape = model->get_input_shape(0);
// int size = model->get_input_size(0);
// assign float input
// float *data_f32 = (float*)calloc(size, sizeof(float));
// int16_t *data_i16 = (int16_t*)calloc(size, sizeof(int16_t));;
// int8_t *data_i8 = (int8_t*)calloc(size, sizeof(int8_t));
// for (int i=0; i<size; i++) {
//     data_f32[i] = i*1.0;
//     if (i < DL_QUANT16_MAX)
//         data_i16[i] = i;
//     if (i< DL_QUANT8_MAX)
//         data_i8[i] = i;
// }

// Tensor<float> *input_f32 = new Tensor<float>(shape, data_f32);
// model->assign_input(0, input_f32);
// TensorBase *input = model->get_input(0);
// printf("dtype:%s\n", input->get_dtype_string());
// Tensor<int8_t> *quant_input_i8 = static_cast<Tensor<int8_t> *>(input);
// quant_input_i8->print({0,1,0,2,0,2,0,10});

// Tensor<int16_t> *input_i16 = new Tensor<int16_t>(shape, data_i16, -2);
// Tensor<int8_t> *input_i8 = new Tensor<int8_t>(shape, data_i8, -2);
// model->assign_input(0, input_i16);
// quant_input_i8 = static_cast<Tensor<int8_t> *>(model->get_input(0));
// quant_input_i8->print({0,1,0,2,0,2,0,10});

// model->assign_input(0, input_i8);
// model->assign_input(0, input_i16);
// quant_input_i8 = static_cast<Tensor<int8_t> *>(model->get_input(0));
// quant_input_i8->print({0,1,0,2,0,2,0,10});
// int8_t *data_i8 = (int8_t*)calloc(size, sizeof(int8_t));
// for (int i=0; i<size; i++) {
//         data_i8[i] = i % 128;
// }

// model->assign_input(0, );