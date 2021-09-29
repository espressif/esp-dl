#include <stdio.h>
#include <iostream>
#include "image.hpp"
#include "color_detector.hpp"

using namespace dl;
using namespace std;

extern "C" void app_main(void)
{
    // set color detection model.
    ColorDetector detector({{}, {}, {}, {}}, {}, true);

    // set the Tensor of test image. 
    Tensor<uint8_t> rgby;
    rgby.set_element((uint8_t *)rgby_image).set_shape(rgby_shape).set_auto_free(false);

    // get color threshold of each color block in the image. 
    // In the test image, we divided the picture into four regions, corresponding to four colors: red, green, blue and yellow.
    vector<uint8_t> thresh_r = detector.cal_color_thresh(rgby, {0, 0, 128, 32});
    vector<uint8_t> thresh_g = detector.cal_color_thresh(rgby, {0, 32, 128, 64});
    vector<uint8_t> thresh_b = detector.cal_color_thresh(rgby, {0, 64, 128, 96});
    vector<uint8_t> thresh_y = detector.cal_color_thresh(rgby, {0, 96, 128, 128});

    // set color threshold and area threshold.
    detector.color_thresh = {thresh_r, thresh_g, thresh_b, thresh_y};
    detector.area_thresh = {100, 100, 100, 100};

    // detect the four color blocks in the test picture based on the threshold set above.
    std::vector<std::vector<components_stats_t>> &results = detector.detect(rgby);

    // print the detection results.
    printf("\ncolor number: %d\n\n", results.size());
    for (int i = 0; i < results.size(); ++i)
    {
        printf("color %d: detected box :%d\n", i, results[i].size());
        for (int j = 0; j < results[i].size(); ++j)
        {
            printf("center: (%d, %d)\n", results[i][j].center[0], results[i][j].center[1]);
            printf("box: (%d, %d), (%d, %d)\n", results[i][j].box[0], results[i][j].box[1], results[i][j].box[2], results[i][j].box[3]);
            printf("area: %d\n", results[i][j].area);
        }
        printf("\n");
    }
}