#include "unity.h"
#include <iostream>
#include <limits.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "dl_nn_pad.hpp"
#include "dl_variable.hpp"

#include "dl_layer_max2d.hpp"
#include "dl_layer_pad.hpp"

using namespace dl;
using namespace std;

TEST_CASE("PAD", "shape")
{
    printf("--------------------------------------------\n");
    Tensor<int8_t> x = Tensor<int8_t>::arange(2 * 3 * 4);
    x.reshape({2, 3, 4}).print({}, "x");
    Tensor<int8_t> out;
    out = nn::pad(x, {1, 0, 1, 2, 0, 1}, {-1, -2, -3, -4, -5, -6}, PADDING_CONSTANT);
    out.print({}, "PADDING_CONSTANT 1");

    out = nn::pad(x, {0, 1, 2, 1, 1, 0}, {}, PADDING_EDGE);
    out.print({}, "PADDING_EDGE 1");

    out = nn::pad(x, {0, 0, 2, 1, 1, 0}, {}, PADDING_EDGE);
    out.print({}, "PADDING_EDGE 2");

    out = nn::pad(x, {0, 0, 0, 1, 0, 0}, {}, PADDING_EDGE);
    out.print({}, "PADDING_EDGE 3");

    out = nn::pad(x, {1, 1, 1, 2, 1, 1}, {}, PADDING_REFLECT);
    out.print({}, "PADDING_REFLECT 1");

    out = nn::pad(x, {1, 0, 1, 2, 0, 1}, {}, PADDING_REFLECT);
    out.print({}, "PADDING_REFLECT 2");

    out = nn::pad(x, {1, 2, 1, 1, 1, 1}, {}, PADDING_SYMMETRIC);
    out.print({}, "PADDING_SYMMETRIC 1");

    out = nn::pad(x, {1, 0, 1, 2, 0, 1}, {}, PADDING_SYMMETRIC);
    out.print({}, "PADDING_SYMMETRIC 2");

    x.expand_dims({1});
    out = nn::pad(x, {1, 0, 1, 2, 0, 1, 1, 1}, {}, PADDING_REFLECT);
    out.print({}, "PADDING_REFLECT 3");

    out = nn::pad(x, {0, 1, 1, 0, 1, 2, 0, 1}, {}, PADDING_SYMMETRIC);
    out.print({}, "PADDING_SYMMETRIC 3");

    x.flatten().expand_dims({0});
    out = nn::pad(x, {1, 2, 0, 1}, {}, PADDING_REFLECT);
    out.print({}, "PADDING_REFLECT 4");

    out = nn::pad(x, {0, 1, 0, 0}, {}, PADDING_REFLECT);
    out.print({}, "PADDING_REFLECT 5");

    out = nn::pad(x, {2, 1, 1, 0}, {}, PADDING_SYMMETRIC);
    out.print({}, "PADDING_SYMMETRIC 4");

    out = nn::pad(x, {1, 0, 2, 2}, {}, PADDING_SYMMETRIC);
    out.print({}, "PADDING_SYMMETRIC 5");
    printf("\n\n++++++++++++++++++++++++++++++++++++++++++++\n\n");

    dl::layer::Pad<int8_t> padlayer({1}, {-1});
    padlayer.build(x.reshape({2, 3, -1}), true);
    Tensor<int8_t> &output = padlayer.call(x);
    output.print({}, "layer");
}
