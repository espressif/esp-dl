#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "esp_log.h"
#include "fr_forward.h"
#include "freertos/FreeRTOS.h"
#include "rom/ets_sys.h"
#include "esp_partition.h"

static const char *TAG = "face_recognition";

void face_id_init(face_id_list *l, uint8_t size, uint8_t confirm_times)
{
    l->head = 0;
    l->tail = 0;
    l->count = 0;
    l->size = size;
    l->confirm_times = confirm_times;
    l->id_list = (dl_matrix3d_t **)calloc(size, sizeof(dl_matrix3d_t *));
}

dl_matrix3dq_t *transform_frmn_input(dl_matrix3du_t *image)
{
    dl_matrix3d_t *image_3d = dl_matrix3d_alloc(image->n,
                                                image->w,
                                                image->h,
                                                image->c);

    fptp_t r = 0;
    fptp_t *a = image_3d->item;
    uc_t *b = image->item;
    uint32_t count = (image->n) * (image->w) * (image->h) * (image->c);
    for (uint32_t i = 0; i < count; i++)
    {
        r = *b++;
        r = (r - 127.5) * (0.0078125);
        *a++ = r;
    }
    dl_matrix3dq_t *image_3dq = dl_matrixq_from_matrix3d_qmf(image_3d, -10);
    dl_matrix3d_free(image_3d);
    return image_3dq;
}

dl_matrix3du_t *aligned_face_alloc()
{
    return dl_matrix3du_alloc(1,
                              FACE_WIDTH,
                              FACE_HEIGHT,
                              3);
}

int8_t align_face(box_array_t *onet_boxes,
                  dl_matrix3du_t *src,
                  dl_matrix3du_t *dest)
{
    fptp_t angle;
    fptp_t ratio;
    fptp_t ne_ratio;
    fptp_t eye_dist;
    fptp_t le_x;
    fptp_t le_y;
    fptp_t re_x;
    fptp_t re_y;
    fptp_t no_x;
    fptp_t no_y;
    fptp_t center[2]; // some params of face align

    le_x = onet_boxes->landmark[0].landmark_p[LEFT_EYE_X];
    le_y = onet_boxes->landmark[0].landmark_p[LEFT_EYE_Y];
    re_x = onet_boxes->landmark[0].landmark_p[RIGHT_EYE_X];
    re_y = onet_boxes->landmark[0].landmark_p[RIGHT_EYE_Y];
    no_x = onet_boxes->landmark[0].landmark_p[NOSE_X];
    no_y = onet_boxes->landmark[0].landmark_p[NOSE_Y];

    ne_ratio = (pow(no_x - le_x, 2) + pow(no_y - le_y, 2)) / (pow(no_x - re_x, 2) + pow(no_y - re_y, 2));
    if (ne_ratio <= NOSE_EYE_RATIO_THRES_MIN || ne_ratio >= NOSE_EYE_RATIO_THRES_MAX) //RATIO_THRES is to avoid small faces and NOSE_EYE_RATIO_THRES is to keep the face front
    {
        ESP_LOGI(TAG, "ne_ratio  : %f", ne_ratio);
        return ESP_FAIL;
    }

    angle = -atan((re_y - le_y) / (re_x - le_x));
    eye_dist = sqrt(pow(re_y - le_y, 2) + pow(re_x - le_x, 2));
    ratio = eye_dist / EYE_DIST_SET;

    center[0] = (re_x + le_x + no_x + no_x) * 0.25;
    center[1] = (re_y + le_y + no_y + no_y) * 0.25;

    ESP_LOGD(TAG, "Left-eye  : (%f,%f)", le_x, le_y);
    ESP_LOGD(TAG, "Right-eye : (%f,%f)", re_x, re_y);
    ESP_LOGD(TAG, "Nose      : (%f,%f)", no_x, no_y);
    ESP_LOGD(TAG, "Angle     : %f", angle);
    ESP_LOGD(TAG, "Eye_dist  : %f", eye_dist);
    ESP_LOGD(TAG, "Ratio     : %f", ratio);
    ESP_LOGD(TAG, "Center    : (%f,%f)", center[0], center[1]);
    ESP_LOGD(TAG, "ne_ratio  : %f", ne_ratio);

    image_cropper(dest,
            src,
            angle,
            ratio,
            center);
    return ESP_OK;
}

dl_matrix3d_t *get_face_id(dl_matrix3du_t *aligned_face)
{
    dl_matrix3d_t *face_id = NULL;
    dl_matrix3dq_t *mobileface_in = transform_frmn_input(aligned_face);
#if CONFIG_XTENSA_IMPL
    dl_matrix3dq_t *face_id_q = frmn_q(mobileface_in, DL_XTENSA_IMPL);
#else
    dl_matrix3dq_t *face_id_q = frmn_q(mobileface_in, DL_C_IMPL);
#endif
    face_id = dl_matrix3d_from_matrixq(face_id_q);
    dl_matrix3dq_free(face_id_q);
    return face_id;
}

fptp_t cos_distance(dl_matrix3d_t *id_1,
                    dl_matrix3d_t *id_2)
{
    assert(id_1->c == id_2->c);
    uint16_t c = id_1->c;
    fptp_t l2_norm_1 = 0;
    fptp_t l2_norm_2 = 0;
    fptp_t dist = 0;
    for (int i = 0; i < c; i++)
    {
        l2_norm_1 += ((id_1->item[i]) * (id_1->item[i]));
        l2_norm_2 += ((id_2->item[i]) * (id_2->item[i]));
    }
    l2_norm_1 = sqrt(l2_norm_1);
    l2_norm_2 = sqrt(l2_norm_2);
    for (uint16_t i = 0; i < c; i++)
    {
        dist += ((id_1->item[i]) * (id_2->item[i]) / (l2_norm_1 * l2_norm_2));
    }
    return dist;
}

fptp_t euclidean_distance(dl_matrix3d_t *id_1,
                          dl_matrix3d_t *id_2)
{
    assert(id_1->c == id_2->c);
    uint16_t c = id_1->c;
    fptp_t l2_norm_1 = 0;
    fptp_t l2_norm_2 = 0;
    fptp_t dist = 0;
    for (int i = 0; i < c; i++)
    {
        l2_norm_1 += ((id_1->item[i]) * (id_1->item[i]));
        l2_norm_2 += ((id_2->item[i]) * (id_2->item[i]));
    }
    l2_norm_1 = sqrt(l2_norm_1);
    l2_norm_2 = sqrt(l2_norm_2);

    for (int i = 0; i < c; i++)
    {
        fptp_t tmp = ((id_1->item[i]) / l2_norm_1) - ((id_2->item[i]) / l2_norm_2);
        dist += (tmp * tmp);
    }
    return dist;
}

void add_face_id(dl_matrix3d_t *dest_id,
                 dl_matrix3d_t *src_id)
{
    fptp_t *dest_item = dest_id->item;
    fptp_t *src_item = src_id->item;
    for (int i = 0; i < src_id->c; i++)
    {
        (*dest_item++) += (*src_item++);
    }
}

void devide_face_id(dl_matrix3d_t *id, uint8_t num)
{
    fptp_t *in1 = id->item;
    for (int i = 0; i < id->c; i++)
    {
        (*in1++) /= num;
    }
}

int8_t recognize_face(face_id_list *l,
                        dl_matrix3du_t *algined_face)
{
    fptp_t similarity = 0;
    fptp_t max_similarity = -1;
    int8_t matched_id = -1;
    dl_matrix3d_t *face_id = NULL;

    face_id = get_face_id(algined_face);

    for (uint16_t i = 0; i < l->count; i++)
    {
        uint8_t head = (l->head + i) % l->size;
        similarity = cos_distance(l->id_list[head], face_id);

        if (similarity > max_similarity)
        {
            max_similarity = similarity;
            matched_id = head;
        }
    }

    if (max_similarity < FACE_REC_THRESHOLD)
    {
        matched_id = -1;
    }
    dl_matrix3d_free(face_id);

    ESP_LOGI(TAG, "\nSimilarity: %.6f, id: %d", max_similarity, matched_id);

    return matched_id;
}

int8_t enroll_face(face_id_list *l, 
                dl_matrix3du_t *aligned_face)
{
    static int8_t confirm_counter = 0;

    // add new_id to dest_id
    dl_matrix3d_t *new_id = get_face_id(aligned_face);

    if (l->count < l->size)
        l->id_list[l->tail] = dl_matrix3d_alloc(1, 1, 1, FACE_ID_SIZE);

    add_face_id(l->id_list[l->tail], new_id);
    dl_matrix3d_free(new_id);

    confirm_counter++;

    if (confirm_counter == l->confirm_times)
    {
        devide_face_id(l->id_list[l->tail], l->confirm_times);
        confirm_counter = 0;

        l->tail = (l->tail + 1) % l->size;
        l->count++;
        // Overlap head
        if (l->count > l->size)
        {
            l->head = (l->head + 1) % l->size;
            l->count = l->size;
        }

        return 0;
    }

    return l->confirm_times - confirm_counter;
}

uint8_t delete_face(face_id_list *l)
{
    if (l->count == 0)
        return 0;

    if (l->id_list[l->head])
        dl_matrix3d_free(l->id_list[l->head]);

    l->head = (l->head + 1) % l->size;
    l->count--;
    return l->count;
}

