#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "esp_log.h"
#include "fr_forward.h"
#include "freertos/FreeRTOS.h"
#include "rom/ets_sys.h"
#include "esp_partition.h"

static const char *TAG = "face_recognition";

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
    fptp_t center_offset;
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

    angle = -atan((re_y - le_y) / (re_x - le_x));
    eye_dist = sqrt(pow(re_y - le_y, 2) + pow(re_x - le_x, 2));
    ratio = eye_dist / EYE_DIST_SET;
    center_offset = ratio * CENTER_OFFSET;
    center[0] = (re_x + le_x) / 2.0f + center_offset * sin(angle);
    center[1] = (re_y + le_y) / 2.0f + center_offset * cos(angle);

    ne_ratio = (pow(no_x - le_x, 2) + pow(no_y - le_y, 2)) / (pow(no_x - re_x, 2) + pow(no_y - re_y, 2));

    ESP_LOGI(TAG, "Left-eye  : (%f,%f)", le_x, le_y);
    ESP_LOGI(TAG, "Right-eye : (%f,%f)", re_x, re_y);
    ESP_LOGI(TAG, "Nose      : (%f,%f)", no_x, no_y);
    ESP_LOGI(TAG, "Angle     : %f", angle);
    ESP_LOGI(TAG, "Eye_dist  : %f", eye_dist);
    ESP_LOGI(TAG, "Ratio     : %f", ratio);
    ESP_LOGI(TAG, "Center    : (%f,%f)", center[0], center[1]);
    ESP_LOGI(TAG, "ne_ratio  : %f", ne_ratio);

    if (ratio > RATIO_THRES && NOSE_EYE_RATIO_THRES_MIN < ne_ratio && ne_ratio < NOSE_EYE_RATIO_THRES_MAX) //RATIO_THRES is to avoid small faces and NOSE_EYE_RATIO_THRES is to keep the face front
    {
        image_cropper(dest,
                      src,
                      angle,
                      ratio,
                      center);
        return ESP_OK;
    }
    else
        return ESP_FAIL;
}

dl_matrix3d_t *get_face_id(dl_matrix3du_t *aligned_face)
{
    dl_matrix3d_t *face_id = NULL;
    dl_matrix3dq_t *mobileface_in = transform_frmn_input(aligned_face);
    dl_matrix3dq_t *face_id_q = frmn_q(mobileface_in);
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

void devide_face_id(dl_matrix3d_t *id,
                    uint16_t n)
{
    fptp_t *in1 = id->item;
    for (int i = 0; i < id->c; i++)
    {
        (*in1++) /= n;
    }
}

uint16_t recognize_face(dl_matrix3du_t *algined_face,
                        dl_matrix3d_t **id_list,
                        fptp_t threshold,
                        uint16_t enrolled_id_number)
{
    fptp_t similarity = 0;
    fptp_t max_similarity = -1;
    uint16_t matched_id = 0;
    dl_matrix3d_t *face_id = NULL;

    face_id = get_face_id(algined_face);

    for (uint16_t i = 0; i < enrolled_id_number; i++)
    {
        similarity = cos_distance(id_list[i], face_id);

        ESP_LOGI(TAG, "Similarity: %.6f", similarity);

        if ((similarity > threshold) && (similarity > max_similarity))
        {
            max_similarity = similarity;
            matched_id = i + 1;
        }
    }

    dl_matrix3d_free(face_id);

    return matched_id;
}

int8_t enroll(dl_matrix3du_t *aligned_face,
              dl_matrix3d_t *dest_id,
              int8_t enroll_confirm_times)
{
    if (enroll_confirm_times < 0)
    {
        ESP_LOGE(TAG, "[enroll_confirm_times] should be set to >= 1");
        return -1;
    }

    static int8_t confirm_counter = 0;

    // add new_id to dest_id
    dl_matrix3d_t *new_id = dl_matrix3d_alloc(1, 1, 1, FACE_ID_SIZE);
    new_id = get_face_id(aligned_face);
    add_face_id(dest_id, new_id);
    dl_matrix3d_free(new_id);

    confirm_counter++;

    if (confirm_counter == enroll_confirm_times)
    {
        devide_face_id(dest_id, enroll_confirm_times);
        confirm_counter = 0;

        return 0;
    }

    return enroll_confirm_times - confirm_counter;
}

int8_t enroll_to_flash(dl_matrix3du_t *aligned_face,
              dl_matrix3d_t *dest_id,
              int8_t enroll_confirm_times,
              uint16_t enrolled_id_number)
{
    if (enroll_confirm_times < 0)
    {
        ESP_LOGE(TAG, "[enroll_confirm_times] should be set to >= 1");
        return -1;
    }

    static int8_t confirm_counter = 0;

    // add new_id to dest_id
    dl_matrix3d_t *new_id = dl_matrix3d_alloc(1, 1, 1, FACE_ID_SIZE);
    new_id = get_face_id(aligned_face);
    add_face_id(dest_id, new_id);
    dl_matrix3d_free(new_id);

    confirm_counter++;

    if (confirm_counter == enroll_confirm_times)
    {
        devide_face_id(dest_id, enroll_confirm_times);
        confirm_counter = 0;

        const esp_partition_t *pt = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_NVS, FLASH_PARTITION_NAME);
        if (pt == NULL){
            ESP_LOGE("Pt", "Not found");
            return -2;
        }
        float mat[FACE_ID_SIZE] = {0};
        int flash_info_flag = FLASH_INFO_FLAG;
        int id_number = enrolled_id_number;
        if(id_number%2 == 1){
            esp_partition_read(pt, 4096+(id_number-1)*2048, mat, 2048);
            esp_partition_erase_range(pt, 4096+(id_number-1)*2048, 4096);
            esp_partition_write(pt, 4096+(id_number-1)*2048, mat, 2048);
            esp_partition_write(pt, 4096+id_number*2048, dest_id->item, 2048); 
        }else{
            esp_partition_erase_range(pt, 4096+id_number*2048, 4096);
            esp_partition_write(pt, 4096+id_number*2048, dest_id->item, 2048);
        }
        id_number++;
        esp_partition_erase_range(pt, 0, 4096);
        esp_partition_write(pt, 0, &flash_info_flag, sizeof(int));
        esp_partition_write(pt, sizeof(int), &id_number, sizeof(int));


        return 0;
    }

    return enroll_confirm_times - confirm_counter;
}

uint16_t read_id_from_flash(dl_matrix3d_t **id_list)
{
    const esp_partition_t *pt = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_NVS, FLASH_PARTITION_NAME);
    if (pt == NULL){
        ESP_LOGE("Pt", "Not found");
        return 0;
    }

    int flash_info_flag = 0;
    int enrolled_id_number = 0;
    esp_partition_read(pt, 0, &flash_info_flag, sizeof(int));
    if((flash_info_flag != FLASH_INFO_FLAG)){
        ESP_LOGE("Read", "No ID Infomation");
        return 0;
    }else{
        esp_partition_read(pt, sizeof(int), &enrolled_id_number, sizeof(int));
        if(enrolled_id_number > 0){
            for(int i=0;i<enrolled_id_number;i++){
                id_list[i] = dl_matrix3d_alloc(1,1,1,FACE_ID_SIZE);
                esp_partition_read(pt, 4096+i*2048, id_list[i]->item, 2048);
            }
            return (uint16_t)enrolled_id_number;
        }else{
            return 0;
        }
    } 
}

uint16_t delete_id_in_flash(int delte_number)
{
    const esp_partition_t *pt = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_NVS, FLASH_PARTITION_NAME);
    if (pt == NULL){
        ESP_LOGE("Pt", "Not found");
        return -1;
    }

    int flash_info_flag = 0;
    int enrolled_id_number = 0;
    esp_partition_read(pt, 0, &flash_info_flag, sizeof(int));
    if((flash_info_flag != FLASH_INFO_FLAG)){
        ESP_LOGE("Read", "No ID Infomation");
        return 0;
    }else{
        esp_partition_read(pt, sizeof(int), &enrolled_id_number, sizeof(int));
        if(enrolled_id_number > 0){
            if((delte_number == -1)||(delte_number > enrolled_id_number)){
                enrolled_id_number = 0;
            }else{
                enrolled_id_number -= delte_number;
            }
            esp_partition_erase_range(pt, 0, 4096);
            esp_partition_write(pt, 0, &flash_info_flag, sizeof(int));
            esp_partition_write(pt, sizeof(int), &enrolled_id_number, sizeof(int));
            return (uint16_t)enrolled_id_number;
        }else{
            return 0;
        }
    } 
}


