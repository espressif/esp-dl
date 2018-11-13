#pragma once

#if __cplusplus
extern "C"
{
#endif

#include <math.h>
#include <stdio.h>
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

    /**
     * @brief Transform a image to the FRMN format. First normalize, then quantize.
     * 
     * @param image                     Image matrix, rgb888 format
     * @return dl_matrix3dq_t*          A normalized and quantized image format
     */
    dl_matrix3dq_t *transform_frmn_input(dl_matrix3du_t *image);

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
     * @brief Calculate cos distance between id_1 and id_2
     * 
     * @param id_1 
     * @param id_2 
     * @return fptp_t 
     */
    fptp_t cos_distance(dl_matrix3d_t *id_1,
                        dl_matrix3d_t *id_2);

    /**
     * @brief Calculate euclidean distance between id_1 and id_2
     * 
     * @param id_1 
     * @param id_2 
     * @return fptp_t 
     */
    fptp_t euclidean_distance(dl_matrix3d_t *id_1,
                              dl_matrix3d_t *id_2);

    /**
     * @brief Add src_id to dest_id
     * 
     * @param dest_id 
     * @param src_id 
     */
    void add_face_id(dl_matrix3d_t *dest_id,
                     dl_matrix3d_t *src_id);

    /**
     * @brief Devide id by n
     * 
     * @param id 
     * @param n 
     */
    void devide_face_id(dl_matrix3d_t *id,
                        uint16_t n);

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

#if __cplusplus
}
#endif
