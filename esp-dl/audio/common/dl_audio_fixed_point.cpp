#include "dl_audio_fixed_point.hpp"

namespace dl {
namespace audio {

// round(ln(1 + i/64) * 2^31)
static const uint32_t s_ln_tab[64] = {
    0x00000000u, 0x01fc0a8bu, 0x03f05362u, 0x05dd163eu, 0x07c28c30u, 0x09a0ebcbu, 0x0b786945u, 0x0d49369du,
    0x0f1383b7u, 0x10d77e7du, 0x129552f8u, 0x144d2b6du, 0x15ff3071u, 0x17ab8902u, 0x19525a9du, 0x1af3c94fu,
    0x1c8ff7c8u, 0x1e27076eu, 0x1fb9186du, 0x214649c5u, 0x22ceb957u, 0x245283f8u, 0x25d1c576u, 0x274c98abu,
    0x28c31784u, 0x2a355b0eu, 0x2ba37b7fu, 0x2d0d903du, 0x2e73afedu, 0x2fd5f077u, 0x3134670eu, 0x328f2838u,
    0x33e647d9u, 0x3539d935u, 0x3689eef8u, 0x37d69b3bu, 0x391fef8fu, 0x3a65fcfcu, 0x3ba8d40au, 0x3ce884c4u,
    0x3e251ebfu, 0x3f5eb11fu, 0x40954a97u, 0x41c8f970u, 0x42f9cb91u, 0x4427ce79u, 0x45530f4cu, 0x467b9ad2u,
    0x47a17d7au, 0x48c4c35fu, 0x49e5784au, 0x4b03a7b5u, 0x4c1f5ccdu, 0x4d38a276u, 0x4e4f834du, 0x4f6409aau,
    0x50763fa1u, 0x51862f08u, 0x5293e177u, 0x539f6047u, 0x54a8b499u, 0x55afe757u, 0x56b50131u, 0x57b80aa5u,
};

// round(2^31 / (1 + i/64))
static const uint32_t s_inv_tab[64] = {
    0x80000000u, 0x7e07e07eu, 0x7c1f07c2u, 0x7a44c6b0u, 0x78787878u, 0x76b981dbu, 0x75075075u, 0x73615a24u,
    0x71c71c72u, 0x70381c0eu, 0x6eb3e453u, 0x6d3a06d4u, 0x6bca1af3u, 0x6a63bd82u, 0x69069069u, 0x67b23a54u,
    0x66666666u, 0x6522c3f3u, 0x63e7063eu, 0x62b2e43eu, 0x61861862u, 0x60606060u, 0x5f417d06u, 0x5e293206u,
    0x5d1745d1u, 0x5c0b8170u, 0x5b05b05bu, 0x5a05a05au, 0x590b2164u, 0x58160581u, 0x572620aeu, 0x563b48c2u,
    0x55555555u, 0x54741facu, 0x5397829du, 0x52bf5a81u, 0x51eb851fu, 0x511be196u, 0x50505050u, 0x4f88b2f4u,
    0x4ec4ec4fu, 0x4e04e04eu, 0x4d4873edu, 0x4c8f8d29u, 0x4bda12f7u, 0x4b27ed36u, 0x4a7904a8u, 0x49cd42e2u,
    0x49249249u, 0x487ede05u, 0x47dc11f7u, 0x473c1ab7u, 0x469ee584u, 0x46046046u, 0x456c797eu, 0x44d72045u,
    0x44444444u, 0x43b3d5b0u, 0x4325c53fu, 0x429a042au, 0x42108421u, 0x4189374cu, 0x41041041u, 0x40810204u,
};

// round(2^31 / sqrt((i + 16.5) / 16)), initial 1/sqrt(m) for m in [1, 4)
static const uint32_t s_rsqrt_tab[48] = {
    0x7e0bb221u, 0x7a64336bu, 0x77099efbu, 0x73f1f68du, 0x7114f644u, 0x6e6bb6e9u, 0x6bf06762u, 0x699e16d0u,
    0x67708af9u, 0x65641faeu, 0x6375ad16u, 0x61a27320u, 0x5fe808fcu, 0x5e444fafu, 0x5cb56711u, 0x5b39a4c7u,
    0x59cf8cbcu, 0x5875cadeu, 0x572b2de0u, 0x55eea2c4u, 0x54bf311au, 0x539bf7cdu, 0x52842a5fu, 0x51770e8fu,
    0x5073fa50u, 0x4f7a5202u, 0x4e8986eau, 0x4da115dau, 0x4cc08605u, 0x4be767f5u, 0x4b1554a6u, 0x4a49ecb3u,
    0x4984d7a4u, 0x48c5c34bu, 0x480c6332u, 0x4758701cu, 0x46a9a794u, 0x45ffcb80u, 0x455aa1cbu, 0x44b9f40bu,
    0x441d8f3bu, 0x43854374u, 0x42f0e3aeu, 0x4260458eu, 0x41d3412au, 0x4149b0e5u, 0x40c3713bu, 0x404060a1u,
};

static inline uint32_t mulhu(uint32_t a, uint32_t b)
{
    return (uint32_t)(((uint64_t)a * b) >> 32);
}

int64_t ln_u64_q36(uint64_t v)
{
    int n = 63 - __builtin_clzll(v);
    uint32_t m = n >= 31 ? (uint32_t)(v >> (n - 31)) : (uint32_t)(v << (31 - n)); // [1, 2) in Q31
    uint32_t f = m & 0x7fffffffu;
    int i = f >> 25;
    uint32_t d = f & 0x01ffffffu;
    // m = (1 + i/64) * (1 + r), r < 1/64
    uint32_t r = mulhu(d << 6, s_inv_tab[i]); // Q36
    uint32_t r2 = (uint32_t)(((uint64_t)r * r) >> 36);
    uint32_t r3 = (uint32_t)(((uint64_t)r2 * r) >> 36);
    uint32_t r4 = (uint32_t)(((uint64_t)r2 * r2) >> 36);
    int64_t ln1p = (int64_t)r - (r2 >> 1) + r3 / 3 - (r4 >> 2);
    return ((int64_t)s_ln_tab[i] << 5) + ln1p + (int64_t)n * LN2_Q36;
}

uint64_t sqrt_u64_q8(uint64_t v)
{
    if (v == 0) {
        return 0;
    }
    int e = (63 - __builtin_clzll(v)) >> 1; // v = m * 4^e, m in [1, 4)
    int sh = 2 * e - 30;
    uint32_t m = sh >= 0 ? (uint32_t)(v >> sh) : (uint32_t)(v << -sh); // Q30
    uint32_t y = s_rsqrt_tab[(m >> 26) - 16];                          // 1/sqrt(m) in Q31
    for (int it = 0; it < 2; it++) {
        uint32_t my2 = mulhu(m, mulhu(y, y)); // Q28
        y = mulhu(y, (3u << 28) - my2) << 3;
    }
    uint32_t s = mulhu(m, y); // sqrt(m) in Q29
    int k = e - 21;
    return k >= 0 ? (uint64_t)s << k : (uint64_t)((s + (1u << (-k - 1))) >> -k);
}

} // namespace audio
} // namespace dl
