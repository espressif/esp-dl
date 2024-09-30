#pragma once

#include <stdint.h>
#include <stdlib.h>

inline void random_value(int8_t &value, int32_t low = INT8_MIN, int32_t high = INT8_MAX)
{
    value = (rand() % (high - low + 1)) + low;
}

inline void random_array(int8_t *array, const int length)
{
    for (size_t i = 0; i < length; i++) random_value(array[i]);
}

inline void random_value(int16_t &value, int32_t low = INT16_MIN, int32_t high = INT16_MAX)
{
    value = (rand() % (high - low + 1)) + low;
}

inline void random_array(int16_t *array, const int length)
{
    for (size_t i = 0; i < length; i++) random_value(array[i]);
}
