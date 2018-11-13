#pragma once

#if __cplusplus
extern "C"
{
#endif

#include "dl_lib.h"

#define FACE_WIDTH 56
#define FACE_HEIGHT 56
#define LOGIN_CONFIRM_TIMES 3
#define FACE_REC_THRESHOLD 0.7

    /**
     * @brief 
     * 
     * @param in 
     */
    void print_matrix(dl_matrix3d_t *in);

    /**
     * @brief 
     * 
     * @param in 
     */
    void print_matrix_q(dl_matrix3dq_t *in);

    /**
     * @brief 
     * 
     * @param in 
     * @return dl_matrix3d_t* 
     */
    dl_matrix3d_t *frmn(dl_matrix3d_t *in);

    /**
     * @brief 
     * 
     * @param in 
     * @return dl_matrix3dq_t* 
     */
    dl_matrix3dq_t *frmn_q(dl_matrix3dq_t *in);

#if __cplusplus
}
#endif
