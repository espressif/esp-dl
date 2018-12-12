#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "esp_log.h"
#include "fr_flash.h"
#include "freertos/FreeRTOS.h"
#include "rom/ets_sys.h"
#include "esp_partition.h"

static const char *TAG = "fr_flash";
int8_t enroll_face_id_to_flash(face_id_list *l,
              dl_matrix3du_t *aligned_face)
{
    int8_t left_sample = enroll_face(l, aligned_face);
    if (left_sample == 0)
    {
        const esp_partition_t *pt = esp_partition_find_first(FR_FLASH_TYPE, FR_FLASH_SUBTYPE, FR_FLASH_PARTITION_NAME);
        if (pt == NULL){
            ESP_LOGE(TAG, "Not found");
            return -2;
        }

        const int block_len = FACE_ID_SIZE * sizeof(float);
        const int block_num = (4096 + block_len - 1) / block_len;
        float *backup_buf = (float *)calloc(1, block_len);
        int flash_info_flag = FR_FLASH_INFO_FLAG;
        uint8_t enroll_id_idx = (l->tail - 1) % l->size;

        if(enroll_id_idx % block_num == 0)
        {
            // save the other block TODO: if block != 2
            esp_partition_read(pt, 4096 + (enroll_id_idx + 1) * block_len, backup_buf, block_len);

            esp_partition_erase_range(pt, 4096 + enroll_id_idx * block_len, 4096);

            esp_partition_write(pt, 4096 + enroll_id_idx * block_len, l->id_list[enroll_id_idx]->item, block_len);
            esp_partition_write(pt, 4096 + (enroll_id_idx + 1) * block_len, backup_buf, block_len); 
        }
        else
        {
            // save the other block TODO: if block != 2
            esp_partition_read(pt, 4096 + (enroll_id_idx - 1) * block_len, backup_buf, block_len);

            esp_partition_erase_range(pt, 4096 + (enroll_id_idx - 1) * block_len, 4096);

            esp_partition_write(pt, 4096 + (enroll_id_idx - 1) * block_len, backup_buf, block_len);
            esp_partition_write(pt, 4096 + enroll_id_idx * block_len, l->id_list[enroll_id_idx]->item, block_len); 
        }

        esp_partition_erase_range(pt, 0, 4096);
        esp_partition_write(pt, 0, &flash_info_flag, sizeof(int));
        esp_partition_write(pt, sizeof(int), l, sizeof(face_id_list));

        return 0;
    }

    return left_sample;
}

int8_t read_face_id_from_flash(face_id_list *l)
{
    const esp_partition_t *pt = esp_partition_find_first(FR_FLASH_TYPE, FR_FLASH_SUBTYPE, FR_FLASH_PARTITION_NAME);
    if (pt == NULL){
        ESP_LOGE(TAG, "Not found");
        return -1;
    }

    int flash_info_flag = 0;

    esp_partition_read(pt, 0, &flash_info_flag, sizeof(int));
    if(flash_info_flag != FR_FLASH_INFO_FLAG)
    {
        ESP_LOGE(TAG, "No ID Infomation");
        return -2;
    }

    uint8_t size = l->size;
    uint8_t confirm_times = l->confirm_times;
    dl_matrix3d_t **id_list = l->id_list;

    esp_partition_read(pt, sizeof(int), l, sizeof(face_id_list));
    const int block_len = FACE_ID_SIZE * sizeof(float);

    assert(l->size == size);
    assert(l->confirm_times == confirm_times);

    for(int i = 0; i < l->count; i++)
    {
        uint8_t head = (l->head + i) % size;
        id_list[head] = dl_matrix3d_alloc(1, 1, 1, FACE_ID_SIZE);
        esp_partition_read(pt, 4096 + head * block_len, id_list[head]->item, block_len);
    }

    l->id_list = id_list;

    return l->count;
}

int8_t delete_face_id_in_flash(face_id_list *l)
{
    delete_face(l);

    const esp_partition_t *pt = esp_partition_find_first(FR_FLASH_TYPE, FR_FLASH_SUBTYPE, FR_FLASH_PARTITION_NAME);
    if (pt == NULL){
        ESP_LOGE(TAG, "Not found");
        return -1;
    }

    int flash_info_flag = 0;
    esp_partition_read(pt, 0, &flash_info_flag, sizeof(int));
    if((flash_info_flag != FR_FLASH_INFO_FLAG))
    {
        ESP_LOGE(TAG, "No ID Infomation");
        return -2;
    }

    esp_partition_erase_range(pt, 0, 4096);
    esp_partition_write(pt, 0, &flash_info_flag, sizeof(int));
    esp_partition_write(pt, sizeof(int), l, sizeof(face_id_list));
    return l->count;
}
