#pragma once

#if __cplusplus
extern "C"
{
#endif

#include "image_util.h"
#include "dl_lib.h"
#include "frmn.h"

#define FACE_WIDTH 56
#define FACE_HEIGHT 56
#define FACE_ID_SIZE 512
#define FACE_REC_THRESHOLD 0.7

#define LEFT_EYE_X 0
#define LEFT_EYE_Y 1
#define RIGHT_EYE_X 6
#define RIGHT_EYE_Y 7
#define NOSE_X 4
#define NOSE_Y 5
#define EYE_DIST_SET 19.5f
#define CENTER_OFFSET 1.5f
#define RATIO_THRES 2.0f
#define NOSE_EYE_RATIO_THRES_MIN 0.8f
#define NOSE_EYE_RATIO_THRES_MAX 1.25f

#define FLASH_INFO_FLAG 12138
#define FLASH_PARTITION_NAME "fr"


    /**
     * @brief Alloc memory for aligned face.
     * 
     * @return dl_matrix3du_t*          Size: 1xFACE_WIDTHxFACE_HEIGHTx3
     */
    dl_matrix3du_t *aligned_face_alloc();

    /**
     * @brief Align detected face to average face according to landmark
     * 
     * @param onet_boxes        Output of MTMN with box and landmark
     * @param src               Image matrix, rgb888 format
     * @param dest              Output image
     * @return ESP_OK           Input face is good for recognition
     * @return ESP_FAIL         Input face is not good for recognition
     */
    int8_t align_face(box_array_t *onet_boxes,
                      dl_matrix3du_t *src,
                      dl_matrix3du_t *dest);

    /**
     * @brief Add src_id to dest_id
     * 
     * @param dest_id 
     * @param src_id 
     */
    void add_face_id(dl_matrix3d_t *dest_id,
                     dl_matrix3d_t *src_id);

    /**
     * @brief Match face with the id_list, and return matched_id.
     * 
     * @param algined_face          An aligned face
     * @param id_list               An ID list
     * @param threshold             The threshold of recognition
     * @param enrolled_id_number    The number of enrolled id
     * @return uint16_t             Matched face id
     */
    uint16_t recognize_face(dl_matrix3du_t *algined_face,
                            dl_matrix3d_t **id_list,
                            fptp_t threshold,
                            uint16_t enrolled_id_number);

    /**
     * @brief Produce face id according to the input aligned face, and save it to dest_id.
     * 
     * @param aligned_face          An aligned face
     * @param dest_id               Saves final face id or accumulation id in process
     * @param enroll_confirm_times  Confirm times for each face id enrollment
     * @return -1                   Wrong input enroll_confirm_times
     * @return 0                    Enrollment finish
     * @return >=1                  The left piece of aligned faces should be input
     */
    int8_t enroll(dl_matrix3du_t *aligned_face,
                  dl_matrix3d_t *dest_id,
                  int8_t enroll_confirm_times);

    /**
     * @brief Produce face id according to the input aligned face, and save it to dest_id and flash.
     * 
     * @param aligned_face          An aligned face
     * @param dest_id               Saves final face id or accumulation id in process
     * @param enroll_confirm_times  Confirm times for each face id enrollment
     * @enrolled_id_number          The number of enrolled id
     * @return -2                   Flash partition not found
     * @return -1                   Wrong input enroll_confirm_times
     * @return 0                    Enrollment finish
     * @return >=1                  The left piece of aligned faces should be input
     */
    int8_t enroll_to_flash(dl_matrix3du_t *aligned_face,
              dl_matrix3d_t *dest_id,
              int8_t enroll_confirm_times,
              uint16_t enrolled_id_number);

    /**
     * @brief Read the enrolled face IDs from the flash.
     * 
     * @param id_list               An ID list which stores the IDs read from the flash
     * @return uint16_t             The number of enrolled face IDs
     */
    uint16_t read_id_from_flash(dl_matrix3d_t **id_list);

    /**
     * @brief Delete the enrolled face IDs in the flash.
     * 
     * @param delte_number          The Number of IDs to be deleted in flash
     * @return uint16_t             The number of IDs remaining in flash
     */
    uint16_t delete_id_in_flash(int delte_number);

#if __cplusplus
}
#endif
