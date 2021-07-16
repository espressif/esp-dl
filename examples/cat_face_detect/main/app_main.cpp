#include <stdio.h>
#include "image.hpp"
#include "cat_face_detect_mn03.hpp"

extern "C" void app_main(void)
{
    // detector
    CatFaceDetectMN03 cat_face_detect(0.4F, 0.3F, 10, 0.3F);

    // inference
    std::list<dl::detect::result_t> &results = cat_face_detect.infer((uint8_t *)IMAGE_ELEMENT, {IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL});

    int i = 0;
    for (std::list<dl::detect::result_t>::iterator prediction = results.begin(); prediction != results.end(); prediction++, i++)
    {
        printf("[%d] score: %f, box: [%d, %d, %d, %d]\n", i, prediction->score, prediction->box[0], prediction->box[1], prediction->box[2], prediction->box[3]);
    }
}