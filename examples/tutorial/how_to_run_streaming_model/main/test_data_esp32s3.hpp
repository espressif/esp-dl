
#pragma once

#include <stdint.h>

#define TEST_INPUT_CHANNELS 64
#define STREAMING_WINDOW_SIZE 3
#define STREAMING_NUMBER 12
#define TIME_SERIES_LENGTH (STREAMING_NUMBER * STREAMING_WINDOW_SIZE)

// NWC layout
const int8_t test_inputs[TIME_SERIES_LENGTH][TEST_INPUT_CHANNELS] = {
    {32, -4, -24, -11, 0,   6,  -80, 24, -39, 49,  55, -45, 37,  7,  -51, -2, -58, -23, 10,  7,   -49, -59,
     48, 16, -13, 44,  -26, 31, 19,  -2, 2,   -93, -1, -26, -5,  33, 25,  -8, -15, 30,  -39, -32, 42,  48,
     35, -9, -30, -49, -22, 24, -33, 36, 17,  -3,  8,  2,   -48, 35, 7,   71, 43,  19,  2,   -23},
    {-8, -38, -13, 17,  36, -52, 19, 8,   31,  40, 13,  19,  -35, -37, 48, 31,  48,  7,  18, -33, -49, -4,
     62, -12, 20,  -25, 24, 69,  48, -2,  -20, -4, -40, -7,  -20, -40, 9,  55,  -47, -5, 10, -33, 31,  -13,
     39, -29, 33,  0,   55, 7,   34, -30, -3,  68, -39, -40, 89,  -2,  37, -27, 15,  -3, 2,  -38},
    {15, 20,  13,  -64, 27,  -42, 40,  6,  -38, 38, 37,  6,  0,   53,  -23, 30,  -22, 20, 0,   3,  -27, -59,
     17, 27,  -30, -9,  -29, -49, -9,  -1, -26, -4, -88, -3, -13, -28, -14, -28, 0,   35, -22, 30, -64, 21,
     -9, -81, -40, 13,  19,  -37, -15, 41, -42, 9,  28,  10, 4,   35,  -34, 16,  0,   74, -14, 4},
    {-40, 12,  32,  -17, 38,  3,   -26, -10, 17,  -9,  -42, -46, -25, -28, 2,   -29, -13, 35, 29,  2,  29, -5,
     -9,  -25, 64,  9,   -30, -23, 51,  -38, -42, -43, -22, -5,  25,  -15, -32, -27, -47, 8,  -62, 53, 27, 49,
     36,  -31, -50, 82,  15,  -47, -18, 27,  44,  34,  -17, -39, 39,  -3,  -10, -33, -21, 34, -12, 33},
    {46,  -37, 8,  4,   -13, -29, 26,  -2,  -19, -12, -19, 21,  40, 1,  -9,  49, 31,  -22, -26, -17, 54,  7,
     39,  -11, 10, -25, -8,  17,  -22, 27,  7,   -21, -18, 14,  -8, 14, -19, 51, 28,  -37, -15, -5,  -32, -18,
     100, -75, 7,  8,   5,   70,  -22, -57, 14,  37,  43,  -62, 43, 89, -50, 34, -22, 30,  10,  -30},
    {-45, 31, -20, 14, 17,  -18, 12, -1, 29,  92,  39, -27, -8,  20, -3,  -2, 9,   57,  29, -62, 8,  1,
     -18, -4, 37,  37, -35, -25, 41, 12, -26, 6,   49, 49,  0,   45, 40,  -5, -6,  -15, -2, -5,  27, -9,
     4,   35, 59,  1,  -3,  -54, 20, -9, 14,  -28, 44, 15,  -13, 16, -56, 15, -30, 14,  0,  1},
    {45, 35,  -46, 7,   -31, -18, 26, 49,  -18, 82, -9,  -13, 17,  -23, 28, -49, 15,  -42, 5,   -34, 18, -6,
     9,  -17, 88,  40,  38,  -19, 12, -58, -43, 26, 19,  6,   40,  -33, -8, -40, 64,  -6,  32,  -13, 20, 43,
     33, 23,  -10, -16, -20, -64, -1, 34,  4,   0,  -15, 2,   -26, 5,   53, -42, -30, 69,  -20, 3},
    {30, -21, -30, -16, 19, 19,  -20, 34, 22,  -20, -49, -47, 9,   12, -11, -57, 10, 39, 9,   -4, -26, 18,
     21, -41, 42,  44,  25, -13, 5,   15, 39,  2,   -7,  -33, -38, 68, 3,   -1,  6,  -2, -20, -5, 18,  -3,
     18, -22, 28,  -7,  27, -10, -26, 30, -44, 10,  -27, 2,   -28, 13, 29,  3,   -1, 12, 25,  16},
    {-15, 18,  53,  6,   20,  23, 23, 10,  -21, -39, -31, -24, 10, 31,  -12, 42,  -21, 17,  10,  14, 10, 37,
     34,  -31, -34, -19, -53, 10, 46, 27,  4,   10,  -25, -27, 6,  -10, -35, -46, -17, -32, 40,  0,  46, 4,
     -15, 3,   -19, 9,   -8,  34, 41, -18, 44,  -31, 35,  3,   41, 97,  12,  -51, 58,  -40, -37, -19},
    {7,   -36, 4,   -37, 23,  10,  7,   -17, 59,  -53, 22, 19, 17,  30,  -24, 7,  37,  -74, 3,  16, -44, -21,
     22,  33,  10,  -6,  -12, 28,  -41, -16, 50,  4,   35, 0,  -10, 0,   16,  22, -41, 10,  17, 22, 5,   -8,
     -46, 19,  -11, 13,  -32, -19, -39, 27,  -27, -1,  26, 11, 11,  -21, 30,  15, 46,  -44, 0,  27},
    {29, -45, -13, -31, -8, 14,  32,  -58, 7,   -20, -9, 20,  22,  52,  -36, -32,  31, -25, 53,  -31, -2, -10,
     49, 9,   16,  -11, 10, -17, -14, 78,  -29, -9,  -7, -24, -52, 23,  10,  -111, 38, 1,   -38, -41, 1,  -5,
     13, -7,  61,  15,  -4, -47, -62, 33,  75,  35,  29, -16, -56, -39, -2,  -112, 5,  -17, -66, -7},
    {-8, 19,  43,  21,  -52, 46,  18,  23, 76,  -37, -54, -20, -16, 36,  17, 18,  -9,  19,  35,  0,   35, 36,
     2,  30,  14,  31,  -9,  -13, 52,  46, -42, 10,  73,  -32, -6,  67,  1,  1,   -44, 40,  -22, -10, -1, 11,
     7,  -33, -28, -43, -22, 65,  -17, 17, 56,  15,  44,  -2,  17,  -36, 28, -19, -51, -32, -13, 15},
    {-2, -27, -37, 0,  21,  29, 21,  18,  20,  -38, -2, 34,  21,  -36, 22, 4,   48,  -34, 5,   35,  -7,  13,
     -2, -9,  32,  10, -2,  20, -26, -33, 15,  -77, 34, 56,  -48, 4,   10, -20, 28,  33,  -17, -15, -59, 27,
     11, 18,  -39, -2, -10, 2,  -8,  -9,  -34, 8,   3,  -48, -47, -8,  56, -35, -24, -38, 0,   -15},
    {-25, 27, -13, 17,  -42, -44, -41, -55, -36, 14,  -65, 38,  32,  52,  -49, -8, -24, -17, 74, 5,   14, -28,
     30,  0,  -14, -25, 21,  -20, 16,  13,  -7,  4,   -18, -35, -10, -3,  3,   48, -20, -30, 19, -12, 0,  48,
     -27, -4, 3,   9,   -27, 13,  28,  19,  18,  -22, -10, 15,  -15, -30, 18,  24, 29,  -63, 26, 36},
    {-32, 38,  28, -17, -2,  33,  77,  21,  23, -1,  18, 17, -13, 77,  5,   78,  -13, -5,  -3, -52, -25, 0,
     -1,  9,   7,  -4,  -18, -44, 27,  -17, -2, -6,  8,  62, 51,  -21, -13, -40, 28,  -70, 35, 18,  -54, 0,
     4,   -38, 5,  74,  17,  -22, -39, 6,   24, -28, 17, 31, 39,  12,  18,  64,  -83, -29, 79, -22},
    {18, -49, 74,  -5, -36, 12, 15, 11,  -4,  27,  -85, 52, -7,  -7,  31,  36, -29, 16,  29,  44, -26, -9,
     2,  -2,  -21, 18, 35,  73, 21, -20, -4,  -2,  -16, 20, -31, 14,  -26, 16, -7,  -30, -33, 89, -31, -11,
     28, 8,   40,  -4, 53,  19, 0,  58,  -18, -23, -29, -1, 4,   -41, -18, 77, 4,   97,  -25, -4},
    {38,  5,  24,  -50, 27, -3,  -38, -9,  7,   9,   -23, -13, -21, -27, 53,  43,  2,   -52, -13, 9,  4,   30,
     -16, 0,  25,  46,  -9, 34,  -31, 61,  84,  -11, -15, -25, 14,  -6,  70,  -26, 68,  -28, -10, 64, -66, 48,
     -21, 44, -11, 9,   34, -56, 41,  -20, -69, -2,  13,  12,  19,  9,   -36, -21, -59, -14, 12,  53},
    {47,  -33, 54, -14, 30,  -38, -59, 71,  -14, -12, 47,  -20, -47, -60, 20,  30,  -27, -29, -6,  76,  9,   -34,
     2,   -33, 7,  28,  -48, -30, 20,  -18, 51,  14,  23,  3,   -21, -24, -25, 25,  -23, -58, -64, -61, -52, 27,
     -57, 35,  62, 60,  23,  -44, 10,  -2,  -30, -10, -53, 9,   -58, 60,  -20, -34, 31,  62,  -35, -34},
    {-24, 2,   -46, 9,   23, -11, -58, 21, 43, -16, -21, -7,  -27, 17,  22,  16,  18,  21,  -64, 19,  11, -4,
     45,  -60, -22, -4,  -4, 21,  15,  23, 9,  80,  17,  -45, -3,  -10, -49, -49, -17, 14,  -5,  -13, 15, -6,
     61,  30,  26,  -23, 38, -8,  29,  51, 14, -8,  40,  -43, 14,  -53, -15, -19, -37, -55, 71,  -17},
    {-4,  -22, -12, 61, 13,  -42, -22, 80,  22,  4,  19, 28,  23,  34,  11,  17,  -2,  23,  -5, -1, -11, -24,
     -50, -36, 10,  20, 4,   81,  -15, 41,  6,   6,  -6, -47, 45,  -43, 73,  24,  -61, -64, -1, 32, 10,  33,
     24,  3,   30,  56, -43, -12, -27, -17, -41, 11, 7,  8,   -28, 2,   -59, -18, -6,  2,   13, 8},
    {-21, 32,  -13, -37, 7,  75, -13, 18, -15, 15, 14,  -45, 24,  25,  -17, -3,  60, -33, 51, 4,   -15, -29,
     -47, -52, -1,  -25, 14, 31, -2,  33, 30,  32, 27,  -5,  -43, 18,  -45, -22, -5, 24,  -7, -15, -38, 7,
     10,  -3,  3,   -38, 10, -8, -37, 14, 16,  12, -27, -24, 6,   -23, -10, 18,  -9, 10,  46, 50},
    {-42, -70, -8,  -20, -25, 51, -6,  -7, -19, -3,  -31, -10, 9,   -29, -58, -11, 11,  3,   4,  21, 40, 49,
     -44, -16, 46,  13,  32,  49, -26, 5,  5,   3,   -20, 20,  -54, 47,  -33, -53, -38, -34, 44, 58, -7, 9,
     14,  54,  -46, -27, -21, 58, 28,  30, -23, -26, -4,  5,   -21, 2,   7,   13,  45,  55,  49, 19},
    {9,  53,  13, -42, -36, 25, 28, -3,  2,  -49, -47, -22, 24,  35, -26, -25, 16,  27, 15, -10, -16, 25,
     29, 1,   -4, 12,  18,  2,  27, 17,  59, -29, -4,  5,   -36, -7, 37,  -24, -86, 80, -2, -14, 4,   36,
     -9, -34, -9, 26,  -29, 37, 13, -25, 4,  -28, -71, -53, -22, 41, 38,  -8,  5,   1,  50, -35},
    {-35, -18, -36, 24, 8,   64,  70, 26, 34, -59, -67, 12,  -3,  -12, 27,  -47, -30, -36, 94,  -71, -13, -20,
     -68, 16,  -6,  -2, 55,  -20, 59, 28, 40, -30, 33,  40,  16,  16,  4,   -41, -3,  19,  3,   11,  17,  -53,
     13,  -8,  -37, 28, -28, 33,  -4, 13, 25, -3,  -39, -79, -27, -20, -36, 60,  -19, 14,  -23, 4},
    {30,  -15, -6,  -6,  -42, -2, -11, 48, -33, 17,  -59, -30, -38, -5, -20, 15,  -21, 35, 3,  34, -36, -34,
     -29, 49,  -22, -11, -22, 49, -21, -2, 31,  33,  36,  0,   29,  42, 26,  -8,  30,  20, 22, -9, -12, 7,
     -38, 0,   -23, -1,  -69, 59, 32,  44, 101, -21, 49,  3,   3,   19, 57,  -16, -65, 13, -8, -40},
    {63, -38, 31, -28, -82, 28, -18, -32, -7, 44, -78, -34, -25, -35, -5, -9,  -29, 5,   5,   27,  60,  -25,
     55, 39,  60, -13, -4,  34, 27,  9,   -9, 50, -53, 48,  -12, 24,  -3, -9,  10,  -54, -26, -47, -69, 44,
     13, 5,   4,  -26, 48,  8,  33,  -32, 2,  10, 77,  -10, -34, -4,  35, -13, -5,  22,  30,  -17},
    {-65, 10, 17,  -18, 6,  -39, -10, -11, -37, 48, -13, -11, 56,  -52, -11, 24,  62,  34,  51,  26, -6,  28,
     -10, 15, 2,   -26, -3, 37,  -44, 12,  13,  33, 4,   -1,  -55, -26, -14, 75,  22,  -56, -14, 20, -20, 49,
     33,  26, -15, 9,   29, 33,  5,   -19, 31,  35, -18, 20,  -27, 71,  -64, -12, -21, -62, -51, 13},
    {-12, 27, -19, 8,  -15, -63, 39, -30, 6,   32, 50,  25,  13,  36, 56, -20, 11,  -37, -52, 8,  -16, 24,
     -86, 70, 3,   46, -12, -19, 7,  -3,  -9,  14, -12, -42, -31, 63, 23, -48, 3,   -6,  -25, 8,  -29, -28,
     16,  39, -60, 9,  26,  -8,  -6, 22,  -35, 52, 20,  10,  33,  24, 7,  2,   -31, -58, 8,   -41},
    {-34, 28,  -28, 9,  -57, 16, 40,  -5, 15, -10, 5,  -34, 12, 1,   0,   -11, 15,  -19, -5,  -15, 3,   -35,
     -12, 88,  -42, 2,  31,  -8, 10,  11, 14, -61, 18, 20,  -2, -12, 14,  31,  -47, 3,   -44, 79,  -11, 24,
     14,  -23, 71,  27, 6,   29, -78, 6,  22, -4,  18, 37,  -8, 1,   -40, 6,   5,   -4,  10,  -19},
    {-29, -60, -10, 8,   -1, 15,  -46, 17,  6,   -29, -17, -22, 16,  12,  -34, 23,  47,  58, 8,  -7, 24,  42,
     6,   -29, 19,  -57, -5, -56, 12,  -19, 57,  7,   7,   -46, -25, -25, -13, -17, -2,  24, 48, 15, -64, 36,
     -56, 48,  13,  15,  17, -44, -9,  15,  -25, -19, -47, 32,  -27, 20,  30,  -20, -16, 60, -5, -35},
    {2,   3,   6,   -26, 5,  40,  6,  -34, -7,  2,  -2,  -15, -48, 0,   28,  -59, 58,  -5, -52, 19,  -10, 30,
     -4,  -6,  4,   -1,  53, -49, -3, 58,  -15, 42, 61,  -30, -61, 37,  -14, -3,  -26, 87, 14,  -35, 63,  -24,
     -16, -19, -24, 9,   57, -24, 48, 13,  -12, 17, -14, -45, -32, -38, -46, -44, 32,  5,  7,   10},
    {-50, -23, 20, -2,  -50, -21, 45,  -29, 5,   -21, 61,  28,  62,  49,  -5, -31, -9,  38, 20,  15,  1,  -14,
     26,  10,  37, 3,   -51, -18, -38, -23, 1,   -29, 40,  -43, -28, 0,   40, 31,  -42, -6, -45, -45, -7, 20,
     -25, -1,  41, -27, 5,   9,   -38, 30,  -31, 59,  -42, -15, 18,  -67, 25, -34, -6,  21, -34, -30},
    {39, 10,  8,   -10, -26, -13, 36,  -14, 42, 4,  -19, 32,  -32, 39,  29, -14, -27, -21, -19, 18, 41, -53,
     -3, -16, 13,  14,  21,  -20, -15, -17, 27, -8, -10, 18,  -50, -58, 9,  14,  -16, -4,  10,  14, 88, -20,
     0,  -12, -40, 29,  7,   32,  25,  0,   11, -5, 34,  -47, 13,  -28, -4, -2,  28,  25,  -36, -11},
    {17, -48, 1,  22,  48, 3,  17, -23, -10, -58, 0,   -10, 10,  37, 37,  38,  -1,  -17, 58,  11,  34,  -4,
     0,  13,  40, -20, 8,  -5, 1,  64,  -19, 21,  43,  -25, -84, -4, 22,  -17, -75, -4,  76,  -56, -44, -2,
     -6, -44, 36, 106, 59, 23, 5,  -39, 20,  6,   -27, -21, 16,  21, -11, -26, 20,  -1,  -21, -32},
    {-24, -3, 32,  -3,  -11, -15, -18, -35, -40, -3, 28,  33, -2,  -32, 43,  -12, 9,   4,   -45, -66, -6,  -48,
     28,  37, 29,  -22, -8,  54,  35,  80,  -18, 16, -18, 22, 35,  99,  4,   63,  -47, -24, -12, -19, -19, 16,
     -35, 19, -22, 20,  -67, 43,  -2,  49,  49,  -7, 7,   -2, -45, 13,  -37, 64,  77,  33,  -11, -3},
    {12,  21,  18, 14,  53,  -29, -3,  10, -24, 20, 22,  -12, 19,  -44, -60, -6, 43,  -13, 17,  -25, 99, 12,
     -65, -36, 21, -28, 14,  -62, -25, 49, -13, -4, -31, -18, 66,  -18, 35,  9,  -34, 35,  -58, 28,  35, 24,
     21,  11,  27, -38, -43, 105, 19,  -7, -6,  23, 25,  42,  -12, -39, 47,  57, -28, -38, -6,  -24},

};

const int8_t test_outputs[] = {
    -15,  70,  9,   -10, -8,  -18, 7,   -64,  5,   12,  -50, -21, -12,  -6,  15,   -80,  -62, 37,  71,   40,  20,  14,
    -20,  12,  23,  53,  6,   -32, -4,  -31,  18,  -41, 10,  5,   -37,  37,  41,   -60,  -43, 19,  65,   83,  13,  -2,
    -1,   -64, -15, -26, -7,  11,  40,  -123, -30, -19, 28,  12,  14,   -49, -83,  18,   -60, 41,  -31,  -21, 5,   33,
    2,    -51, -26, -26, 37,  -58, 37,  -42,  -22, -12, -6,  -37, 7,    -59, -69,  40,   73,  65,  26,   3,   2,   16,
    11,   41,  -15, -29, 10,  4,   -3,  -38,  -7,  3,   -30, 24,  -7,   -45, -16,  28,   46,  28,  35,   8,   32,  -48,
    -18,  -7,  8,   -12, 43,  -14, -19, -42,  14,  34,  5,   -6,  -59,  15,  -71,  11,   -33, -27, -20,  75,  7,   -14,
    -15,  -5,  -14, -66, 3,   17,  -47, -24,  -43, 12,  35,  -78, -50,  45,  86,   63,   -17, 2,   -20,  14,  -4,  84,
    -27,  -40, -22, -21, 12,  -67, 33,  7,    -76, -5,  7,   -63, -11,  -23, 64,   56,   6,   10,  5,    -54, -45, -23,
    19,   19,  44,  -85, -23, -31, 1,   4,    -3,  -32, -36, 39,  -66,  41,  -21,  -42,  -3,  74,  13,   -12, -21, -33,
    2,    -81, 36,  -6,  -49, -2,  -1,  1,    15,  -91, -68, 36,  51,   70,  26,   15,   -20, 17,  11,   53,  -26, -64,
    -6,   -25, 12,  -49, 22,  16,  -44, 7,    23,  -43, -10, -16, 49,   72,  21,   15,   8,   -85, -18,  -24, -29, 3,
    38,   -88, 8,   -23, 22,  13,  11,  -45,  -68, 68,  -65, 12,  -31,  -30, 5,    74,   1,   -23, -13,  -2,  14,  -81,
    18,   26,  -46, -19, -28, -19, 42,  -79,  -52, 49,  100, 75,  4,    32,  -9,   10,   -3,  52,  -26,  -24, 19,  -34,
    14,   -67, 3,   8,   -33, 11,  15,  -58,  -30, 14,  53,  72,  30,   7,   12,   -60,  -34, -8,  -30,  12,  36,  -80,
    -10,  -26, 2,   18,  33,  -15, -61, 22,   -60, 42,  -43, -23, -19,  73,  -1,   10,   -10, -18, 27,   -65, 1,   6,
    -50,  -8,  -21, -9,  10,  -63, -67, 39,   62,  72,  20,  -3,  17,   21,  12,   50,   -52, -25, 8,    -16, 26,  -26,
    11,   -29, -59, 32,  9,   -50, -45, 37,   65,  44,  50,  -4,  23,   -31, -46,  -5,   2,   11,  79,   -62, -51, -38,
    11,   7,   21,  -12, -84, -19, -40, 55,   -36, -40, -13, 66,  6,    -6,  20,   -46,  34,  -56, -30,  -32, -50, 6,
    -6,   4,   8,   -77, -74, 3,   45,  26,   24,  19,  -9,  48,  12,   48,  0,    -34,  -12, 1,   29,   -19, 4,   32,
    -27,  25,  30,  -55, -22, 7,   42,  68,   36,  2,   49,  -63, -21,  -5,  -2,   -51,  68,  -41, -27,  -26, 45,  -15,
    -1,   -70, -67, 28,  -65, 31,  -19, -7,   3,   59,  0,   -13, -9,   -16, 38,   -74,  5,   4,   -48,  9,   -19, -16,
    31,   -69, -61, 39,  54,  58,  24,  25,   11,  41,  -2,  40,  -37,  -25, 5,    -14,  18,  -56, -3,   10,  -26, 13,
    20,   -39, -52, 17,  24,  70,  30,  11,   24,  -54, -41, -9,  -25,  -10, 37,   -71,  -26, -6,  18,   16,  59,  -13,
    -72,  -15, -69, 47,  -41, -22, 1,   52,   -11, -27, 0,   -18, 30,   -86, 21,   -35,  -53, 7,   -18,  -24, 40,  -84,
    -128, 25,  75,  42,  24,  -10, 31,  11,   11,  40,  -57, -30, -13,  -5,  23,   -8,   -12, -10, -43,  26,  27,  -81,
    -52,  27,  53,  54,  43,  5,   -16, -62,  -64, 4,   -17, 3,   37,   -39, -35,  -38,  -21, 28,  42,   -10, -75, -5,
    -32,  29,  -60, -25, 23,  77,  15,  -4,   -9,  -52, 39,  -81, -6,   -45, -28,  -31,  25,  -59, 14,   -88, -53, 41,
    62,   30,  30,  4,   2,   26,  7,   26,   -10, -30, 5,   -32, 14,   -26, -11,  2,    -43, 31,  -17,  -31, -55, 48,
    34,   69,  44,  11,  39,  -23, -28, -37,  -27, -21, 34,  -83, -5,   -23, 23,   13,   37,  -21, -104, 4,   -74, 47,
    -40,  -5,  -4,  86,  9,   19,  -4,  -27,  46,  -87, -21, -17, -62,  13,  -28,  -19,  53,  -43, -46,  43,  71,  54,
    42,   39,  -2,  64,  7,   77,  -59, -5,   19,  -13, 3,   -50, -2,   38,  4,    10,   -18, -57, -40,  17,  55,  58,
    23,   20,  22,  -17, -24, 5,   -17, -3,   48,  -46, -45, 9,   46,   2,   44,   -30,  -91, -29, -73,  58,  -44, -15,
    16,   81,  2,   1,   -24, -47, 50,  -80,  29,  -14, -41, -10, 21,   -36, 12,   -101, -67, 46,  54,   52,  45,  -1,
    6,    7,   27,  34,  -40, -75, 13,  3,    11,  -1,  -10, 4,   -44,  36,  17,   -87,  -73, 52,  34,   62,  46,  11,
    -1,   -30, -23, -22, -63, 13,  61,  -111, 19,  -9,  46,  37,  34,   -47, -110, 30,   -37, 52,  -61,  -16, 20,  71,
    9,    -22, -11, -34, 29,  -65, -2,  -44,  -65, -6,  20,  -42, -7,   -62, -45,  18,   65,  57,  32,   8,   -7,  40,
    11,   23,  -14, -10, 25,  -45, 33,  -21,  8,   20,  -34, 10,  -18,  -35, -20,  31,   48,  72,  33,   18,  51,  -45,
    -26,  -21, -8,  -30, 51,  -48, -11, -36,  35,  34,  24,  2,   -105, 19,  -78,  16,   -39, -4,  12,   65,  4,   2,
    26,   -20, 63,  -88, -8,  -4,  -32, 7,    1,   -31, 15,  -66, -51,  57,  45,   50,   29,  12,  -5,   28,  -16, 52,
    -49,  -27, 4,   -13, -4,  -17, 2,   -10,  -42, 18,  9,   -28, -61,  18,  34,   72,   24,  -11, 34,   -51, -35, -1,
    -17,  8,   39,  -92, -34, -21, 23,  2,    28,  -42, -51, 19,  -82,  55,  -21,  -37,  1,   60,  16,   2,   -9,  -27,
    22,   -66, 28,  -27, -60, -4,  -2,  -22,  28,  -63, -72, 40,  73,   80,  16,   -9,   21,  20,  -22,  42,  -45, -48,
    -22,  -14, 29,  -28, -32, -1,  -44, 45,   13,  -64, -42, 46,  71,   71,  86,   -7,   15,  -49, -38,  -24, -37, -28,
    42,   -53, -13, -41, 9,   36,  2,   -17,  -89, 31,  -40, 46,  -58,  -23, -7,   29,   6,   0,   -10,  -10, -9,  -81,
    42,   13,  -5,  -35, -25, -15, 23,  -98,  -57, 71,  81,  78,  -9,   9,   -4,   -15,  23,  31,  -4,   -30, -9,  -33,
    0,    -93, -9,  -9,  -38, 25,  29,  -60,  -34, 12,  37,  51,  29,   11,  -4,   -38,  -24, -17, -22,  -7,  28,  -90,
    4,    -6,  23,  10,  1,   -24, -60, 25,   -68, 67,  -38, -1,  3,    51,  -21,  -31,  -2,  -18, 22,   -74, 22,  -24,
    -62,  -7,  -22, -47, 24,  -54, -74, 48,   49,  57,  47,  35,  34,   42,  25,   47,   -13, -17, 4,    -45, 14,  -73,
    -21,  2,   -14, 16,  20,  -44, -46, 18,   50,  75,  44,  2,   30,   -48, -36,  5,    -25, -9,  20,   -59, -33, 10,
    0,    30,  82,  12,  -80, -32, -66, 49,   -63, 11,  -27, 26,  -13,  -42, -7,   -44,  1,   -65, 12,   -30, -52, 27,
    -20,  -9,  37,  -51, -87, 27,  50,  56,   59,  17,  5,   54,  -4,   55,  -29,  -26,  1,   31,  31,   -56, -6,  7,
    -8,   -6,  31,  -53, -32, 12,  36,  52,   49,  38,  7,   -67, -27,  -5,  -14,  -26,  39,  -18, -19,  -7,  11,  8,
    55,   -8,  -79, -21, -37, 28,  -57, -28,  1,   36,  -15, -14, -31,  -8,  28,   -58,  23,  -18, -18,  -28, -4,  -52,
    30,   -78, -61, 26,  55,  40,  20,  -6,   -1,  32,  12,  10,  -15,  -20, -10,  -28,  6,   -83, -9,   -19, -30, 4,
    16,   -19, -48, 31,  -9,  69,  51,  -8,   26,  -23, -43, -25, -9,   -29, 31,   -83,  -6,  -17, 8,    34,  50,  22,
    -79,  -24, -69, 58,  -40, -5,  0,   27,   8,   -3,  4,   -30, 29,   -86, 15,   28,   -57, 12,  -9,   -42, 46,  -48,
    -69,  36,  61,  64,  20,  10,  -2,  0,    -10, 45,  -59, -30, -4,   16,  21,   -51,  6,   5,   -56,  20,  22,  -44,
    -44,  11,  48,  85,  35,  -13, 13,  -55,  -14, -24, -41, -31, 80,   -90, -31,  -10,  -3,  -5,  25,   -15, -70, -9,
    -89,  37,  -14, -20, 7,   59,  -2,  -8,   -23, 1,   32,  -89, -3,   -15, -19,  -31,  -18, -21, 32,   -65, -64, 57,
    53,   60,  9,   24,  4,   21,  6,   42,   -27, -21, -1,  -30, 13,   -52, -14,  0,    -22, 12,  10,   -39, -58, 40,
    29,   57,  23,  -2,  13,  -44, -40, -5,   -25, 5,   51,  -79, -29,  -7,  12,   18,   42,  4,   -76,  -21, -58, 56,
    -39,  -22, 2,   76,  -25, -20, -3,  -6,   17,  -81, 12,  -25, -22,  -53, -12,  -56,  9,   -86, -89,  51,  58,  46,
    -1,   3,   17,  13,  22,  21,  1,   -20,  17,  -57, 19,  -37, 0,    -26, -37,  19,   10,  -45, -57,  34,  38,  57,
    44,   -3,  6,   -31, -66, -25, -8,  14,   7,   -95, -31, -15, 2,    31,  50,   -9,   -83, -6,  -53,  55,  -43, -26,
    23,   55,  -8,  -18, -10, -26, 33,  -90,  16,  -18, -35, -18, 6,    -44, 9,    -87,  -57, 56,  55,   75,  17,  -11,
    15,   9,   1,   28,  -39, -56, -3,  -19,  15,  -21, 16,  -11, -68,  4,   15,   -50,  -47, 14,  38,   50,  20,  2,
    21,   -63, -29, -38, -37, -11, 48,  -91,  0,   -16, 12,  18,  32,   -22, -60,  31,   -67, 32,  -52,  -43, 46,  57,
    -3,   -18, -36, -11, 30,  -77, 14,  9,    -12, -25, -13, -12, 8,    -48, -71,  48,   69,  87,  -6,   10,  1,   8,
    -2,   24,  -28, -34, 19,  -25, 24,  -23,  -12, 34,  -16, 53,  26,   -84, -43,  32,   51,  68,  42,   11,  30,  -58,
    -20,  -28, -62, -37, 72,  -79, -1,  11,   31,  -3,  8,   -31, -75,  9,   -66,  64,   -36, -16, 19,   74,  -9,  -43,
    1,    -29, 36,  -97, 13,  -33, -9,  -32,  -21, -34, 13,  -71, -107, 38,  44,   36,   6,   -25, 1,    9,   21,  18,
    -6,   -33, 10,  -25, 22,  -6,  3,   -4,   -45, 21,  45,  -65, -52,  10,  32,   73,   34,  -5,  2,    -24, -29, -15,
    -20,  -9,  26,  -93, -11, 16,  1,   9,    56,  -37, -61, 16,  -54,  73,  -37,  -36,  -4,  62,  2,    -27, -1,  -48,
    11,   -96, 3,   -44, -34, -28, -10, -26,  25,  -98, -95, 40,  84,   29,  32,   9,    -2,  33,  10,   44,  14,  -48,
    21,   -26, 34,  -29, -3,  -13, -22, 16,   33,  -74, -39, 20,  30,   61,  69,   18,   -2,  -36, -42,  -20, -19, -14,
    -1,   -77, -5,  -18, 7,   41,  49,  -33,  -85, 13,  -33, 48,  -47,  -40, 7,    62,   -6,  -11, -13,  6,   30,  -88,
    1,    -4,  -35, -19, -34, -7,  29,  -49,  -73, 45,  50,  58,  -5,   21,  5,    29,   -11, 46,  -45,  -8,  -6,  -24,
    2,    -41, -10, 13,  -20, 16,  11,  -51,  -54, 21,  38,  48,  11,   -6,  2,    -50,  -47, 1,   -7,   0,   46,  -69,
    -48,  -11, 3,   -13, 41,  6,   -47, -3,   -73, 52,  -16, -25, -14,  96,  18,   3,    -14, -41, 36,   -50, -21, -6,
    -87,  24,  -23, -25, 19,  -80, -57, 34,   81,  32,  29,  -1,  2,    17,  6,    59,   -68, -35, 24,   9,   33,  -13,
    16,   19,  -70, 46,  6,   -75, -34, 6,    75,  84,  40,  -7,  42,   -54, -24,  -23,  -9,  -30, 68,   -43, -57, -66,
    40,   16,  20,  -55, -89, -13, -53, 25,   -38, -32, 16,  83,  -6,   26,  -20,  -15,  55,  -62, 3,    14,  -31, -23,
    -16,  -19, 22,  -67, -62, 57,  45,  44,   20,  12,  6,   18,  2,    40,  -16,  -38,  17,  -47, 13,   -23, 1,   -11,
    -46,  48,  -1,  -35, -54, 31,  25,  76,   47,  -20, 41,  -23, -37,  -9,  -30,  5,    42,  -96, -25,  -20, 19,  -6,
    45,   -34, -82, 10,  -69, 66,  -27, -14,  8,   50,  -11, -10, -9,   3,   36,   -75,  -8,  -9,  -26,  -14, -13, -32,
    20,   -66, -49, 61,  44,  66,  15,  21,   3,   26,  2,   27,  -35,  -22, 2,    -35,  12,  -49, -2,   -17, -29, 9,
    20,   -31, -59, 39,  32,  47,  30,  3,    19,  -43, -40, -7,  -17,  -8,  33,   -68,  -27, -15, 8,    11,  45,  8,
    -74,  -29, -50, 55,  -39, -35, 20,  73,   12,  -16, -1,  -46, 62,   -79, -1,   -37,  -17, -38, -22,  -76, 14,  -68,
    -61,  55,  73,  41,  52,  16,  6,   34,   6,   50,  -26, -20, 33,   -7,  3,    -23,  -10, 7,   -31,  17,  -3,  -47,
    -32,  36,  44,  39,  41,  1,   40,  -37,  -16, -14, -9,  -22, 56,   -64, -22,  -19,  8,   20,  42,   -7,  -79, -39,
    -57,  46,  -36, -44, -16, 62,  14,  -9,   -18, -40, 53,  -72, -12,  -7,  -17,  -4,   10,  -38, 29,   -67, -40, 35,
    77,   36,  32,  -15, -30, 12,  -9,  55,   -77, -43, 21,  22,  -10,  -32, 33,   -3,   -58, 17,  -7,   -40, -33, 35,
    43,   48,  47,  -19, 30,  -29, -10, -24,  -13, -2,  90,  -67, -32,  -56, 23,   -1,   -11, -34, -66,  -12, -62, 30,
    -12,  -68, 30,  54,  29,  11,  10,  -28,  30,  -87, 1,   -2,  -25,  7,   -10,  -15,  18,  -84, -72,  49,  95,  70,
    7,    -1,  -37, 10,  1,   54,  -33, -28,  -2,  -24, 9,   -32, 19,   32,  -46,  24,   0,   -60, -20,  23,  41,  59,
    25,   2,   31,  -57, -11, -25, -40, -16,  54,  -56, -23, -33, 7,    -9,  -12,  -70,  -41, 34,  -73,  43,  -17, -37};
