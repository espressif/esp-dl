/*
  * ESPRESSIF MIT License
  *
  * Copyright (c) 2018 <ESPRESSIF SYSTEMS (SHANGHAI) PTE LTD>
  *
  * Permission is hereby granted for use on ESPRESSIF SYSTEMS products only, in which case,
  * it is free of charge, to any person obtaining a copy of this software and associated
  * documentation files (the "Software"), to deal in the Software without restriction, including
  * without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
  * and/or sell copies of the Software, and to permit persons to whom the Software is furnished
  * to do so, subject to the following conditions:
  *
  * The above copyright notice and this permission notice shall be included in all copies or
  * substantial portions of the Software.
  *
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
  * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
  * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
  * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  *
  */
#include <string.h>
#include <math.h>
#include "esp_system.h"
#include "lssh_forward.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

//static const char *TAG = "lssh_forward";

#if CONFIG_LSSH_SPARSE_MN_5
#define LSSH_MODULES_CONFIG sparse_mn_5_modules_config
#endif

lssh_config_t lssh_initialize_config(fptp_t min_face, fptp_t score_threshold, fptp_t nms_threshold, int image_height, int image_width)
{
    lssh_config_t config = {0};
    config.min_face = min_face;
    config.score_threshold = score_threshold;
    config.nms_threshold = nms_threshold;
    lssh_update_image_shape(&config, image_height, image_width);
    config.mode = DL_XTENSA_IMPL;
    return config;
}

void lssh_update_image_shape(lssh_config_t *config, int image_height, int image_width)
{
    if (config->min_face == LSSH_MODULES_CONFIG.module_config[0].anchor_size[0])
    {
        config->free_image = false;
        config->resized_scale = 1.0;
        config->resized_width = image_width;
        config->resized_height = image_height;
    }
    else
    {
        config->free_image = true;
        config->resized_scale = 1.0f * LSSH_MODULES_CONFIG.module_config[0].anchor_size[0] / config->min_face;
        config->resized_width = round(image_width * config->resized_scale);
        config->resized_height = round(image_height * config->resized_scale);
    }

    int short_side = min(config->resized_height, config->resized_width);
    config->enabled_top_k = 0;
    for (size_t i = 0; i < LSSH_MODULES_CONFIG.number; i++)
    {
        if (short_side >= LSSH_MODULES_CONFIG.module_config[i].boundary)
            config->enabled_top_k++;
        else
            break;
    }
    assert(config->enabled_top_k > 0);
    // ESP_LOGW(TAG, "The \'min_face\' should < %d ", min(image_height, image_width) * LSSH_MODULES_CONFIG[0].anchor_size[0] / LSSH_MODULES_CONFIG[0].boundary);
}

box_array_t *lssh_detect_object(dl_matrix3du_t *image, lssh_config_t config)
{
    /**
     * @brief resize image
     * 
     */
    dl_matrix3du_t *resized_image = image;
    if (config.free_image)
    {
        resized_image = dl_matrix3du_alloc(1, config.resized_width, config.resized_height, image->c);
        image_resize_linear(resized_image->item, image->item, resized_image->w, resized_image->h, resized_image->c, image->w, image->h);
    }

    /**
     * @brief net operation
     * 
     */
#if CONFIG_LSSH_SPARSE_MN_5
#if CONFIG_LSSH_WITH_LANDMARK
    lssh_module_result_t *module_result = sparse_mn_5_q_with_landmark(resized_image, config.free_image, config.enabled_top_k, DL_XTENSA_IMPL);
#else
    lssh_module_result_t *module_result = sparse_mn_5_q_without_landmark(resized_image, config.free_image, config.enabled_top_k, DL_XTENSA_IMPL);
#endif
#endif

    /**
     * @brief filter by score
     * 
     */
    image_list_t **origin_head = (image_list_t **)dl_lib_calloc(config.enabled_top_k, sizeof(image_list_t *), 0);
    image_list_t all_box_list = {NULL};

    for (size_t i = 0; i < config.enabled_top_k; i++)
    {
        fptp_t *category = module_result[i].category->item;
        fptp_t *box = module_result[i].box_offset->item;
#if CONFIG_LSSH_WITH_LANDMARK
        fptp_t *landmark = module_result[i].landmark_offset->item;
        origin_head[i] = image_get_valid_boxes(category,
                                               box,
                                               landmark,
                                               module_result[i].category->h,
                                               module_result[i].category->n,
                                               module_result[i].category->w,
                                               LSSH_MODULES_CONFIG.module_config[i].anchor_size,
                                               config.score_threshold,
                                               LSSH_MODULES_CONFIG.module_config[i].stride,
                                               config.resized_scale,
                                               true);
#else
        origin_head[i] = image_get_valid_boxes(category,
                                               box,
                                               NULL,
                                               module_result[i].category->h,
                                               module_result[i].category->n,
                                               module_result[i].category->w,
                                               LSSH_MODULES_CONFIG.module_config[i].anchor_size,
                                               config.score_threshold,
                                               LSSH_MODULES_CONFIG.module_config[i].stride,
                                               config.resized_scale,
                                               true);
#endif
        if (origin_head[i])
            image_sort_insert_by_score(&all_box_list, origin_head[i]);

        lssh_module_result_free(module_result[i]);
    }
    dl_lib_free(module_result);

    /**
     * @brief nms
     * 
     */
    image_nms_process(&all_box_list, config.nms_threshold, false);

    /**
     * @brief build up result
     * 
     */
    box_array_t *targets_list = NULL;
    if (all_box_list.len)
    {
        targets_list = (box_array_t *)dl_lib_calloc(1, sizeof(box_array_t), 0);
        targets_list->len = all_box_list.len;
        targets_list->score = (fptp_t *)dl_lib_calloc(targets_list->len, sizeof(fptp_t), 0);
        targets_list->box = (box_t *)dl_lib_calloc(targets_list->len, sizeof(box_t), 0);
#if CONFIG_LSSH_WITH_LANDMARK
        targets_list->landmark = (landmark_t *)dl_lib_calloc(targets_list->len, sizeof(landmark_t), 0);
#endif

        image_box_t *t = all_box_list.head;
        for (int i = 0; i < all_box_list.len; i++, t = t->next)
        {
            targets_list->box[i] = t->box;
#if CONFIG_LSSH_WITH_LANDMARK
            targets_list->landmark[i] = t->landmark;
#endif
        }
    }

    for (int i = 0; i < config.enabled_top_k; i++)
    {
        if (origin_head[i])
        {
            dl_lib_free(origin_head[i]->origin_head);
            dl_lib_free(origin_head[i]);
        }
    }
    dl_lib_free(origin_head);

    return targets_list;
}
