#include "test_fft.h"

float cal_snr_item(float x, float y)
{
    x *= x;

    y *= y;
    if (y < 1e-10) {
        y = 1e-10;
    }

    return 10 * log10f(x / y);
}

float get_snr(const float *x, const float *gt, int size)
{
    float mean_db = 0;
    int count = 0;
    for (int i = 0; i < size; i++) {
        // printf("%d %f %f %f\n", i, gt[i], x[i], gt[i] - x[i]);
        if (fabs(gt[i]) > 1e-6) {
            float snr_db = cal_snr_item(gt[i], gt[i] - x[i]);
            mean_db += snr_db;
            count++;
        }
    }
    return mean_db / count;
}

float get_rmse(const float *x, const float *gt, int size)
{
    float rmse = 0;
    for (int i = 0; i < size; i++) {
        // printf("%d %f %f %f\n", i, gt[i], x[i], gt[i] - x[i]);
        rmse += (gt[i] - x[i]) * (gt[i] - x[i]);
        // printf("%d - %f %f\n", i, gt[i] - x[i], rmse);
    }
    return sqrt(rmse / size + 1e-7);
}

bool check_fft_results(const float *x, const float *gt, int size, float snr_threshold, float rmse_threshold)
{
    float snr = get_snr(x, gt, size);
    float rmse = get_rmse(x, gt, size);
    bool pass = true;
    printf("snr: %f, rmse: %f ", snr, rmse);
    if (snr < snr_threshold) {
        pass = false;
    }
    if (rmse > rmse_threshold) {
        pass = false;
    }
    if (pass) {
        printf("pass\n");
    } else {
        printf("fail\n");
    }
    return pass;
}

bool check_is_same(const float *x, const float *gt, int size, float threshold)
{
    for (int i = 0; i < size; i++) {
        if (fabs(x[i] - gt[i]) > threshold) {
            return false;
        }
    }
    printf("check is same pass\n");
    return true;
}
