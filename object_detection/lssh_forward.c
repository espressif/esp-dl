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

static const char *TAG = "lssh_forward";

#if CONFIG_LSSH_SPARSE_MN_5
#define LSSH_MODULES_CONFIG sparse_mn_5_modules_config
#endif

typedef struct tag_tree_point
{
    fptp_t *score;
    fptp_t *box;
#if CONFIG_LSSH_WITH_LANDMARK
    fptp_t *landmark;
#endif
    struct tag_tree_point *left;
    struct tag_tree_point *right;
} tree_point_t;

typedef struct tag_list_pint
{
    fptp_t score;
    box_t box;
    fptp_t area;
#if CONFIG_LSSH_WITH_LANDMARK
    landmark_t landmark;
#endif
    struct tag_list_pint *next;
} list_point_t;

tree_point_t *__binary_tree_insert(tree_point_t *head, fptp_t *score, fptp_t *box
#if CONFIG_LSSH_WITH_LANDMARK
                                   ,
                                   fptp_t *landmark
#endif
)
{
    if (head == NULL)
    {
        head = (tree_point_t *)dl_lib_calloc(1, sizeof(tree_point_t), 0);
        head->score = score;
        head->box = box;
#if CONFIG_LSSH_WITH_LANDMARK
        head->landmark = landmark;
#endif
        head->left = NULL;
        head->right = NULL;

        return head;
    }

    if (score[0] > head->score[0])
        head->left = __binary_tree_insert(head->left, score, box
#if CONFIG_LSSH_WITH_LANDMARK
                                          ,
                                          landmark
#endif
        );
    else
        head->right = __binary_tree_insert(head->right, score, box
#if CONFIG_LSSH_WITH_LANDMARK
                                           ,
                                           landmark
#endif
        );

    return head;
}

list_point_t *__binary_tree_sort(tree_point_t *in_head, list_point_t *out_head)
{
    if (in_head->left)
        out_head = __binary_tree_sort(in_head->left, out_head);

    out_head->score = in_head->score[0];
    size_t i = 0;
    for (; i < 4; i++)
    {
        out_head->box.box_p[i] = in_head->box[i];
#if CONFIG_LSSH_WITH_LANDMARK
        out_head->landmark.landmark_p[i] = in_head->landmark[i];
#endif
    }

#if CONFIG_LSSH_WITH_LANDMARK
    for (; i < 10; i++)
        out_head->landmark.landmark_p[i] = in_head->landmark[i];
#endif

    out_head->area = (out_head->box.box_p[2] - out_head->box.box_p[0] + 1) * (out_head->box.box_p[3] - out_head->box.box_p[1] + 1);

    out_head->next = out_head + 1;
    out_head++;

    if (in_head->right)
        out_head = __binary_tree_sort(in_head->right, out_head);

    return out_head;
}

// void lssh_binary_tree_print(tree_point_t *in_head)
// {
//     if (in_head->left)
//         lssh_binary_tree_print(in_head->left);

//     printf("%f\n", in_head->score[0]);

//     if (in_head->right)
//         lssh_binary_tree_print(in_head->right);

//     return;
// }

void __binary_tree_free(tree_point_t *head)
{
    if (head->left)
        __binary_tree_free(head->left);

    if (head->right)
        __binary_tree_free(head->right);

    dl_lib_free(head);
}

int __nms(list_point_t *in, int length, fptp_t threshold)
{
    while (in)
    {
        list_point_t *rest_next_point = in->next;
        list_point_t *rest_prev_point = in;

        while (rest_next_point)
        {
            box_t inter_box;
            inter_box.box_p[0] = DL_IMAGE_MAX(in->box.box_p[0], rest_next_point->box.box_p[0]);
            inter_box.box_p[1] = DL_IMAGE_MAX(in->box.box_p[1], rest_next_point->box.box_p[1]);
            inter_box.box_p[2] = DL_IMAGE_MIN(in->box.box_p[2], rest_next_point->box.box_p[2]);
            inter_box.box_p[3] = DL_IMAGE_MIN(in->box.box_p[3], rest_next_point->box.box_p[3]);

            fptp_t inter_w = inter_box.box_p[2] - inter_box.box_p[0] + 1;
            fptp_t inter_h = inter_box.box_p[3] - inter_box.box_p[1] + 1;

            if (inter_w > 0 && inter_h > 0)
            {
                fptp_t inter_area = inter_w * inter_h;
                fptp_t iou = inter_area / (in->area + rest_next_point->area - inter_area + 1e-8);
                if (iou > threshold)
                {
                    length--;
                    // Delete duplicated box
                    // Here we cannot free a single box, because these boxes are allocated by calloc, we need to free all the calloced memory together.
                    rest_prev_point->next = rest_next_point->next;
                    rest_next_point = rest_next_point->next;
                    continue;
                }
            }

            rest_prev_point = rest_next_point;
            rest_next_point = rest_next_point->next;
        }

        in = in->next;
    }
    return length;
}

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

    int targets_number = 0;

    // TODO: resize image
    dl_matrix3du_t *resized_image = image;
    if (config.free_image)
    {
        resized_image = dl_matrix3du_alloc(1, config.resized_width, config.resized_height, image->c);
        image_resize_linear(resized_image->item, image->item, resized_image->w, resized_image->h, resized_image->c, image->w, image->h);
    }

    // TODO: net operation
#if CONFIG_LSSH_SPARSE_MN_5
#if CONFIG_LSSH_WITH_LANDMARK
    lssh_module_result_t *module_result = sparse_mn_5_q_with_landmark(resized_image, config.free_image, config.enabled_top_k, DL_XTENSA_IMPL);
#else
    lssh_module_result_t *module_result = sparse_mn_5_q_without_landmark(resized_image, config.free_image, config.enabled_top_k, DL_XTENSA_IMPL);
#endif
#endif

    // TODO: filter
    tree_point_t *targets_tree = NULL;
    for (size_t i = 0; i < config.enabled_top_k; i++)
    {
        fptp_t *category = module_result[i].category->item;
        fptp_t *box = module_result[i].box_offset->item;
#if CONFIG_LSSH_WITH_LANDMARK
        fptp_t *landmark = module_result[i].landmark_offset->item;
#endif

        for (size_t y = 0; y < module_result[i].category->n; y++)
        {
            for (size_t x = 0; x < module_result[i].category->h; x++)
            {
                for (size_t c = 0; c < module_result[i].category->w; c++)
                {
                    if (category[1] > config.score_threshold)
                    {
                        int anchor_size = LSSH_MODULES_CONFIG.module_config[i].anchor_size[c];

                        int anchor_left_up_x = x * LSSH_MODULES_CONFIG.module_config[i].stride;
                        int anchor_left_up_y = y * LSSH_MODULES_CONFIG.module_config[i].stride;

                        box[0] = (box[0] * anchor_size + anchor_left_up_x) / config.resized_scale;
                        box[1] = (box[1] * anchor_size + anchor_left_up_y) / config.resized_scale;
                        box[2] = (box[2] * anchor_size + anchor_left_up_x + anchor_size) / config.resized_scale;
                        box[3] = (box[3] * anchor_size + anchor_left_up_y + anchor_size) / config.resized_scale;

#if CONFIG_LSSH_WITH_LANDMARK
                        landmark[0] = (landmark[0] * anchor_size + anchor_left_up_x) / config.resized_scale;
                        landmark[1] = (landmark[1] * anchor_size + anchor_left_up_y) / config.resized_scale;
                        landmark[2] = (landmark[2] * anchor_size + anchor_left_up_x) / config.resized_scale;
                        landmark[3] = (landmark[3] * anchor_size + anchor_left_up_y) / config.resized_scale;
                        landmark[4] = (landmark[4] * anchor_size + anchor_left_up_x) / config.resized_scale;
                        landmark[5] = (landmark[5] * anchor_size + anchor_left_up_y) / config.resized_scale;
                        landmark[6] = (landmark[6] * anchor_size + anchor_left_up_x) / config.resized_scale;
                        landmark[7] = (landmark[7] * anchor_size + anchor_left_up_y) / config.resized_scale;
                        landmark[8] = (landmark[8] * anchor_size + anchor_left_up_x) / config.resized_scale;
                        landmark[9] = (landmark[9] * anchor_size + anchor_left_up_y) / config.resized_scale;
#endif

                        targets_tree = __binary_tree_insert(targets_tree, category + 1, box
#if CONFIG_LSSH_WITH_LANDMARK
                                                            ,
                                                            landmark
#endif
                        );
                        targets_number++;
                    }
                    category += 2;
                    box += 4;
#if CONFIG_LSSH_WITH_LANDMARK
                    landmark += 10;
#endif
                }
            }
        }
    }

    if (!targets_tree)
    {
        lssh_module_results_free(module_result, config.enabled_top_k);
        return NULL;
    }

    /**
     * @brief convert lssh_binary_tree to increase_list
     * 
     */
    list_point_t *targets_increase_list = (list_point_t *)dl_lib_calloc(targets_number, sizeof(list_point_t), 0);
    list_point_t *targets_increase_list_last = __binary_tree_sort(targets_tree, targets_increase_list) - 1;
    targets_increase_list_last->next = NULL;

    lssh_module_results_free(module_result, config.enabled_top_k);
    __binary_tree_free(targets_tree);

    /**
     * @brief nms
     * 
     */
    targets_number = __nms(targets_increase_list, targets_number, config.nms_threshold);

    /**
     * @brief convert targets_increase_list to targets_list
     * 
     */
    box_array_t *targets_list = NULL;
    if (targets_number)
    {
        targets_list = (box_array_t *)dl_lib_calloc(1, sizeof(box_array_t), 0);
        targets_list->score = (fptp_t *)dl_lib_calloc(targets_number, sizeof(fptp_t), 0);
        targets_list->box = (box_t *)dl_lib_calloc(targets_number, sizeof(box_t), 0);
#if CONFIG_LSSH_WITH_LANDMARK
        targets_list->landmark = (landmark_t *)dl_lib_calloc(targets_number, sizeof(landmark_t), 0);
#endif
        targets_list->len = targets_number;

        list_point_t *t = targets_increase_list;
        for (size_t i = 0; i < targets_number; i++, t = t->next)
        {
            targets_list->score[i] = t->score;
            targets_list->box[i] = t->box;
#if CONFIG_LSSH_WITH_LANDMARK
            targets_list->landmark[i] = t->landmark;
#endif
        }
    }
    dl_lib_free(targets_increase_list);

    return targets_list;
}
