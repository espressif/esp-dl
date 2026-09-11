#pragma once

/**
 * @brief Private Attention helpers for dl_base_attention.cpp and white-box tests.
 *
 * Internal linkage: each includer gets its own copy of the lookup tables.
 */

#include "dl_base_attention.hpp"
#include "dl_define.hpp"
#include "dl_tensor_base.hpp"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace dl {
namespace base {
namespace {

constexpr int ATTENTION_QUERY_TILE_SIZE = 32; /*!< Q block height in tokens */
constexpr int ATTENTION_KEY_TILE_SIZE = 64;   /*!< K/V tile width in tokens */
#ifndef ESP_DL_ATTENTION_MAX_KV_TOKENS
#define ESP_DL_ATTENTION_MAX_KV_TOKENS 2048 /*!< admission bound on Lk only; no buffer is sized from it */
#endif
constexpr int ATTENTION_MAX_KV_TOKENS = ESP_DL_ATTENTION_MAX_KV_TOKENS; /*!< max admitted key length */
static_assert(ATTENTION_MAX_KV_TOKENS >= ATTENTION_KEY_TILE_SIZE,
              "ATTENTION_MAX_KV_TOKENS must be at least one tile wide");
static_assert(ATTENTION_MAX_KV_TOKENS <= 8192, "ATTENTION_MAX_KV_TOKENS is capped at 8192, as in esp-ppq");

constexpr int ATTENTION_TASK_COUNT = 2; /*!< dual-core task count */
#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
constexpr int ATTENTION_TILE_BUFFER_COUNT = ATTENTION_TASK_COUNT; /*!< one tile buffer per task */
#else
constexpr int ATTENTION_TILE_BUFFER_COUNT = 1; /*!< single-core builds only keep task 0's buffer */
#endif

/**
 * @brief Element strides of one Q/K/V/O tensor under BHND or BNHD.
 */
struct AttentionStrides {
    int batch; /*!< batch stride in elements */
    int head;  /*!< head stride in elements */
    int pos;   /*!< token stride in elements */
};

/**
 * @brief Build the (batch, head, pos) strides for one Attention tensor.
 *
 * @param layout   BHND or BNHD
 * @param H        number of heads
 * @param L        sequence length
 * @param D        head dimension
 *
 * @return strides in elements
 */
inline AttentionStrides attention_strides(AttentionLayout layout, int H, int L, int D)
{
    if (layout == AttentionLayout::BHND) {
        return {H * L * D, L * D, D};
    }
    return {L * H * D, D, H * D};
}

/**
 * @brief Logical Attention dimensions decoded from Q and K shapes.
 */
struct AttentionDims {
    int B;   /*!< batch */
    int Hq;  /*!< query heads */
    int Lq;  /*!< query length */
    int D;   /*!< head dimension */
    int Hkv; /*!< key/value heads */
    int Lk;  /*!< key length */
};

/**
 * @brief Decode B/Hq/Lq/D/Hkv/Lk from 4-D Q and K shapes.
 *
 * @param layout   BHND or BNHD
 * @param q        Q shape [B, H, L, D] or [B, L, H, D]
 * @param k        K shape in the same layout
 *
 * @return decoded dimensions
 */
inline AttentionDims attention_dims(AttentionLayout layout, const std::vector<int> &q, const std::vector<int> &k)
{
    AttentionDims dims;
    dims.B = q[0];
    dims.D = q[3];
    if (layout == AttentionLayout::BNHD) {
        dims.Lq = q[1];
        dims.Hq = q[2];
        dims.Lk = k[1];
        dims.Hkv = k[2];
    } else {
        dims.Hq = q[1];
        dims.Lq = q[2];
        dims.Hkv = k[1];
        dims.Lk = k[2];
    }
    return dims;
}

/**
 * @brief Convert a tensor exponent to its dequantization scale 2^exponent.
 *
 * @param exponent   tensor exponent
 *
 * @return 2^exponent
 */
inline float dequant_scale_from_exponent(int exponent)
{
    // ldexp is defined for the full int range; a later is_positive_normal()
    // check rejects inf / 0 / subnormals. Do not use `1u << exponent`:
    // shift counts outside [0, 31] are undefined on this target.
    return std::ldexp(1.0f, exponent);
}

/**
 * @brief Convert an output exponent to the inverse scale used when storing integers.
 *
 * @param exponent   output tensor exponent
 *
 * @return 2^(-exponent), or 0 when exponent is INT_MIN
 */
inline float rescale_to_int_from_exponent(int exponent)
{
    if (exponent == INT_MIN) {
        return 0.0f;
    }
    return std::ldexp(1.0f, -exponent);
}

/**
 * @brief exp(x) for x in [-16, 0], 1025 entries + linear interp (<= 3.1e-5 relative err).
 */
constexpr int FLOAT_EXP_LUT_ENTRIES = 1025;
constexpr float FLOAT_EXP_LUT_MAX_NEGATIVE = 16.0f;
constexpr float FLOAT_EXP_LUT_STEP = FLOAT_EXP_LUT_MAX_NEGATIVE / (FLOAT_EXP_LUT_ENTRIES - 1);
constexpr float FLOAT_EXP_LUT_INVERSE_STEP = 1.0f / FLOAT_EXP_LUT_STEP;

/**
 * @brief lookup_precise() reads table[] over r = -x in [0, FLOAT_EXP_PRECISE_MAX_R]; the table is 0
 * from r = 32 on, so callers may clamp r to FLOAT_EXP_PRECISE_MAX_R.
 */
constexpr float FLOAT_EXP_PRECISE_MAX_R = 33.0f;
constexpr int FLOAT_EXP_PRECISE_ENTRIES = static_cast<int>(FLOAT_EXP_PRECISE_MAX_R * FLOAT_EXP_LUT_INVERSE_STEP) + 1;

/**
 * @brief fp32 bits of exp(-i/64), i = 0..1024.
 *
 * Must match esp-ppq/.../espdl_attention_exp_lut.py::FLOAT_EXP_LUT_BITS.
 */
constexpr uint32_t FLOAT_EXP_LUT_BITS[1025] = {
    0x3f800000u, 0x3f7c07f5u, 0x3f781fabu, 0x3f7446e3u, 0x3f707d60u, 0x3f6cc2e4u, 0x3f691735u, 0x3f657a17u, 0x3f61eb51u,
    0x3f5e6aaau, 0x3f5af7e9u, 0x3f5792d8u, 0x3f543b41u, 0x3f50f0edu, 0x3f4db3a8u, 0x3f4a833eu, 0x3f475f7du, 0x3f444832u,
    0x3f413d2bu, 0x3f3e3e38u, 0x3f3b4b29u, 0x3f3863cfu, 0x3f3587fcu, 0x3f32b780u, 0x3f2ff231u, 0x3f2d37e0u, 0x3f2a8863u,
    0x3f27e38fu, 0x3f254939u, 0x3f22b937u, 0x3f203361u, 0x3f1db78fu, 0x3f1b4598u, 0x3f18dd55u, 0x3f167ea0u, 0x3f142954u,
    0x3f11dd4au, 0x3f0f9a5du, 0x3f0d606bu, 0x3f0b2f4eu, 0x3f0906e5u, 0x3f06e70cu, 0x3f04cfa1u, 0x3f02c083u, 0x3f00b992u,
    0x3efd7558u, 0x3ef98764u, 0x3ef5a908u, 0x3ef1da07u, 0x3eee1a24u, 0x3eea6922u, 0x3ee6c6c7u, 0x3ee332d9u, 0x3edfad1eu,
    0x3edc355du, 0x3ed8cb60u, 0x3ed56ef0u, 0x3ed21fd7u, 0x3ecedde0u, 0x3ecba8d7u, 0x3ec88088u, 0x3ec564c2u, 0x3ec25552u,
    0x3ebf5207u, 0x3ebc5ab2u, 0x3eb96f22u, 0x3eb68f29u, 0x3eb3ba99u, 0x3eb0f145u, 0x3eae3301u, 0x3eab7f9fu, 0x3ea8d6f5u,
    0x3ea638d9u, 0x3ea3a521u, 0x3ea11ba2u, 0x3e9e9c36u, 0x3e9c26b4u, 0x3e99baf3u, 0x3e9758cfu, 0x3e950020u, 0x3e92b0c2u,
    0x3e906a8eu, 0x3e8e2d61u, 0x3e8bf917u, 0x3e89cd8du, 0x3e87aa9fu, 0x3e85902du, 0x3e837e13u, 0x3e817431u, 0x3e7ee4cdu,
    0x3e7af126u, 0x3e770d2fu, 0x3e7338a8u, 0x3e6f7355u, 0x3e6bbcfau, 0x3e68155au, 0x3e647c3cu, 0x3e60f165u, 0x3e5d749eu,
    0x3e5a05aeu, 0x3e56a45eu, 0x3e535079u, 0x3e5009c9u, 0x3e4cd019u, 0x3e49a337u, 0x3e4682efu, 0x3e436f0fu, 0x3e406766u,
    0x3e3d6bc4u, 0x3e3a7bf8u, 0x3e3797d4u, 0x3e34bf2au, 0x3e31f1ccu, 0x3e2f2f8du, 0x3e2c7841u, 0x3e29cbbdu, 0x3e2729d5u,
    0x3e249260u, 0x3e220534u, 0x3e1f8229u, 0x3e1d0916u, 0x3e1a99d3u, 0x3e18343au, 0x3e15d825u, 0x3e13856du, 0x3e113bedu,
    0x3e0efb81u, 0x3e0cc405u, 0x3e0a9555u, 0x3e086f4fu, 0x3e0651cfu, 0x3e043cb5u, 0x3e022fdfu, 0x3e002b2bu, 0x3dfc5cf5u,
    0x3df8735au, 0x3df49946u, 0x3df0ce7bu, 0x3ded12beu, 0x3de965d2u, 0x3de5c77cu, 0x3de23782u, 0x3ddeb5adu, 0x3ddb41c3u,
    0x3dd7db8cu, 0x3dd482d4u, 0x3dd13764u, 0x3dcdf908u, 0x3dcac78bu, 0x3dc7a2bau, 0x3dc48a64u, 0x3dc17e57u, 0x3dbe7e61u,
    0x3dbb8a54u, 0x3db8a1ffu, 0x3db5c535u, 0x3db2f3c6u, 0x3db02d87u, 0x3dad724bu, 0x3daac1e7u, 0x3da81c2eu, 0x3da580f7u,
    0x3da2f019u, 0x3da06969u, 0x3d9decc0u, 0x3d9b79f6u, 0x3d9910e3u, 0x3d96b162u, 0x3d945b4cu, 0x3d920e7bu, 0x3d8fcaccu,
    0x3d8d9019u, 0x3d8b5e3fu, 0x3d89351bu, 0x3d87148bu, 0x3d84fc6cu, 0x3d82ec9cu, 0x3d80e4fcu, 0x3d7dcad3u, 0x3d79db8cu,
    0x3d75fbe2u, 0x3d722b98u, 0x3d6e6a71u, 0x3d6ab831u, 0x3d67149cu, 0x3d637f79u, 0x3d5ff88du, 0x3d5c7fa2u, 0x3d59147eu,
    0x3d55b6ecu, 0x3d5266b5u, 0x3d4f23a4u, 0x3d4bed86u, 0x3d48c427u, 0x3d45a754u, 0x3d4296dcu, 0x3d3f928du, 0x3d3c9a38u,
    0x3d39adacu, 0x3d36ccbbu, 0x3d33f737u, 0x3d312cf2u, 0x3d2e6dc0u, 0x3d2bb976u, 0x3d290fe6u, 0x3d2670e8u, 0x3d23dc51u,
    0x3d2151f8u, 0x3d1ed1b4u, 0x3d1c5b5du, 0x3d19eeccu, 0x3d178bdau, 0x3d153261u, 0x3d12e23bu, 0x3d109b43u, 0x3d0e5d55u,
    0x3d0c284cu, 0x3d09fc06u, 0x3d07d860u, 0x3d05bd38u, 0x3d03aa6cu, 0x3d019fdau, 0x3cff3ac4u, 0x3cfb45c8u, 0x3cf76081u,
    0x3cf38ab0u, 0x3cefc417u, 0x3cec0c7bu, 0x3ce8639fu, 0x3ce4c94au, 0x3ce13d42u, 0x3cddbf4eu, 0x3cda4f35u, 0x3cd6ecc2u,
    0x3cd397bdu, 0x3cd04ff2u, 0x3ccd152cu, 0x3cc9e738u, 0x3cc6c5e2u, 0x3cc3b0f8u, 0x3cc0a84au, 0x3cbdaba6u, 0x3cbabaddu,
    0x3cb7d5bfu, 0x3cb4fc1fu, 0x3cb22dcfu, 0x3caf6aa2u, 0x3cacb26cu, 0x3caa0500u, 0x3ca76235u, 0x3ca4c9e1u, 0x3ca23bd9u,
    0x3c9fb7f4u, 0x3c9d3e0cu, 0x3c9acdf7u, 0x3c98678fu, 0x3c960aaeu, 0x3c93b72du, 0x3c916ce8u, 0x3c8f2bbau, 0x3c8cf37eu,
    0x3c8ac412u, 0x3c889d52u, 0x3c867f1cu, 0x3c84694eu, 0x3c825bc7u, 0x3c805665u, 0x3c7cb212u, 0x3c78c724u, 0x3c74ebc4u,
    0x3c711fb2u, 0x3c6d62b2u, 0x3c69b489u, 0x3c6614fau, 0x3c6283ceu, 0x3c5f00c9u, 0x3c5b8bb5u, 0x3c582459u, 0x3c54ca80u,
    0x3c517df3u, 0x3c4e3e7fu, 0x3c4b0beeu, 0x3c47e60eu, 0x3c44ccadu, 0x3c41bf99u, 0x3c3ebea0u, 0x3c3bc994u, 0x3c38e044u,
    0x3c360282u, 0x3c333021u, 0x3c3068f2u, 0x3c2daccau, 0x3c2afb7du, 0x3c2854e0u, 0x3c25b8c8u, 0x3c23270cu, 0x3c209f82u,
    0x3c1e2203u, 0x3c1bae65u, 0x3c194483u, 0x3c16e434u, 0x3c148d54u, 0x3c123fbdu, 0x3c0ffb4au, 0x3c0dbfd7u, 0x3c0b8d40u,
    0x3c096361u, 0x3c074219u, 0x3c052945u, 0x3c0318c4u, 0x3c011074u, 0x3bfe206bu, 0x3bfa2fd0u, 0x3bf64ed8u, 0x3bf27d45u,
    0x3beebad9u, 0x3beb075au, 0x3be7628bu, 0x3be3cc32u, 0x3be04417u, 0x3bdcc9ffu, 0x3bd95db4u, 0x3bd5feffu, 0x3bd2adaau,
    0x3bcf6980u, 0x3bcc324du, 0x3bc907ddu, 0x3bc5e9fdu, 0x3bc2d87du, 0x3bbfd329u, 0x3bbcd9d3u, 0x3bb9ec4bu, 0x3bb70a61u,
    0x3bb433e9u, 0x3bb168b3u, 0x3baea894u, 0x3babf360u, 0x3ba948ebu, 0x3ba6a90bu, 0x3ba41395u, 0x3ba18860u, 0x3b9f0744u,
    0x3b9c9019u, 0x3b9a22b6u, 0x3b97bef6u, 0x3b9564b2u, 0x3b9313c4u, 0x3b90cc08u, 0x3b8e8d58u, 0x3b8c5791u, 0x3b8a2a90u,
    0x3b880631u, 0x3b85ea53u, 0x3b83d6d3u, 0x3b81cb91u, 0x3b7f90d8u, 0x3b7b9a86u, 0x3b77b3efu, 0x3b73dcd2u, 0x3b7014f4u,
    0x3b6c5c17u, 0x3b68b1ffu, 0x3b651673u, 0x3b618939u, 0x3b5e0a17u, 0x3b5a98d6u, 0x3b57353eu, 0x3b53df1au, 0x3b509633u,
    0x3b4d5a57u, 0x3b4a2b50u, 0x3b4708ebu, 0x3b43f2f8u, 0x3b40e943u, 0x3b3deb9du, 0x3b3af9d7u, 0x3b3813bfu, 0x3b353929u,
    0x3b3269e7u, 0x3b2fa5cbu, 0x3b2cecaau, 0x3b2a3e57u, 0x3b279aa9u, 0x3b250174u, 0x3b227290u, 0x3b1fedd2u, 0x3b1d7314u,
    0x3b1b022cu, 0x3b189af5u, 0x3b163d48u, 0x3b13e8ffu, 0x3b119df4u, 0x3b0f5c03u, 0x3b0d2308u, 0x3b0af2dfu, 0x3b08cb65u,
    0x3b06ac78u, 0x3b0495f6u, 0x3b0287beu, 0x3b0081adu, 0x3afd074bu, 0x3af91b0bu, 0x3af53e5eu, 0x3af17104u, 0x3aedb2c1u,
    0x3aea035au, 0x3ae66293u, 0x3ae2d032u, 0x3adf4bffu, 0x3adbd5c0u, 0x3ad86d3eu, 0x3ad51244u, 0x3ad1c49au, 0x3ace840du,
    0x3acb5069u, 0x3ac82979u, 0x3ac50f0cu, 0x3ac200f0u, 0x3abefef5u, 0x3abc08e9u, 0x3ab91e9eu, 0x3ab63fe5u, 0x3ab36c8fu,
    0x3ab0a471u, 0x3aade75du, 0x3aab3528u, 0x3aa88da6u, 0x3aa5f0acu, 0x3aa35e12u, 0x3aa0d5aeu, 0x3a9e5758u, 0x3a9be2e6u,
    0x3a997833u, 0x3a971718u, 0x3a94bf6eu, 0x3a927110u, 0x3a902bd9u, 0x3a8defa5u, 0x3a8bbc50u, 0x3a8991b7u, 0x3a876fb7u,
    0x3a85562eu, 0x3a8344fbu, 0x3a813bfbu, 0x3a7e7620u, 0x3a7a8431u, 0x3a76a1eau, 0x3a72cf0du, 0x3a6f0b5du, 0x3a6b569eu,
    0x3a67b094u, 0x3a641906u, 0x3a608fb9u, 0x3a5d1475u, 0x3a59a703u, 0x3a56472bu, 0x3a52f4b8u, 0x3a4faf74u, 0x3a4c772bu,
    0x3a494ba9u, 0x3a462cbdu, 0x3a431a33u, 0x3a4013dbu, 0x3a3d1984u, 0x3a3a2affu, 0x3a37481du, 0x3a3470afu, 0x3a31a488u,
    0x3a2ee37cu, 0x3a2c2d5eu, 0x3a298203u, 0x3a26e140u, 0x3a244aebu, 0x3a21bedbu, 0x3a1f3ce6u, 0x3a1cc4e6u, 0x3a1a56b2u,
    0x3a17f224u, 0x3a159714u, 0x3a13455fu, 0x3a10fcddu, 0x3a0ebd6cu, 0x3a0c86e6u, 0x3a0a5929u, 0x3a083411u, 0x3a06177du,
    0x3a04034au, 0x3a01f757u, 0x39ffe709u, 0x39fbef61u, 0x39f80779u, 0x39f42f11u, 0x39f065ecu, 0x39ecabcdu, 0x39e9007au,
    0x39e563b6u, 0x39e1d549u, 0x39de54f9u, 0x39dae28fu, 0x39d77dd2u, 0x39d4268eu, 0x39d0dc8cu, 0x39cd9f98u, 0x39ca6f7eu,
    0x39c74c0cu, 0x39c4350du, 0x39c12a53u, 0x39be2babu, 0x39bb38e6u, 0x39b851d4u, 0x39b57648u, 0x39b2a613u, 0x39afe108u,
    0x39ad26fcu, 0x39aa77c2u, 0x39a7d32fu, 0x39a5391au, 0x39a2a959u, 0x39a023c2u, 0x399da82du, 0x399b3673u, 0x3998ce6du,
    0x39966ff3u, 0x39941ae1u, 0x3991cf10u, 0x398f8c5cu, 0x398d52a1u, 0x398b21bbu, 0x3988f988u, 0x3986d9e4u, 0x3984c2adu,
    0x3982b3c3u, 0x3980ad04u, 0x397d5ca1u, 0x39796f0fu, 0x39759113u, 0x3971c271u, 0x396e02ecu, 0x396a5246u, 0x3966b046u,
    0x39631cb1u, 0x395f974eu, 0x395c1fe4u, 0x3958b63cu, 0x39555a20u, 0x39520b59u, 0x394ec9b4u, 0x394b94fau, 0x39486cfbu,
    0x39455182u, 0x3942425eu, 0x393f3f5fu, 0x393c4853u, 0x39395d0du, 0x39367d5cu, 0x3933a912u, 0x3930e004u, 0x392e2204u,
    0x392b6ee5u, 0x3928c67eu, 0x392628a3u, 0x3923952bu, 0x39210becu, 0x391e8cbeu, 0x391c1779u, 0x3919abf6u, 0x39174a0du,
    0x3914f199u, 0x3912a274u, 0x39105c79u, 0x390e1f84u, 0x390beb71u, 0x3909c01du, 0x39079d64u, 0x39058326u, 0x39037140u,
    0x39016791u, 0x38fecbf2u, 0x38fad8aeu, 0x38f6f517u, 0x38f320f0u, 0x38ef5bfbu, 0x38eba5fcu, 0x38e7feb8u, 0x38e465f3u,
    0x38e0db75u, 0x38dd5f05u, 0x38d9f06bu, 0x38d68f6fu, 0x38d33bddu, 0x38cff57fu, 0x38ccbc20u, 0x38c98f8du, 0x38c66f93u,
    0x38c35c00u, 0x38c054a3u, 0x38bd594bu, 0x38ba69c8u, 0x38b785edu, 0x38b4ad8au, 0x38b1e071u, 0x38af1e77u, 0x38ac676fu,
    0x38a9bb2eu, 0x38a71988u, 0x38a48253u, 0x38a1f567u, 0x389f729bu, 0x389cf9c5u, 0x389a8abfu, 0x38982562u, 0x3895c988u,
    0x3893770au, 0x38912dc3u, 0x388eed8fu, 0x388cb64bu, 0x388a87d1u, 0x38886201u, 0x388644b6u, 0x38842fd0u, 0x3882232cu,
    0x38801eabu, 0x387c4459u, 0x38785b1fu, 0x3874816bu, 0x3870b6ffu, 0x386cfb9fu, 0x38694f0fu, 0x3865b113u, 0x38622173u,
    0x385e9ff5u, 0x385b2c61u, 0x3857c67fu, 0x38546e1bu, 0x385122fdu, 0x384de4f1u, 0x384ab3c4u, 0x38478f42u, 0x3844773au,
    0x38416b78u, 0x383e6bceu, 0x383b780au, 0x38388ffeu, 0x3835b37bu, 0x3832e253u, 0x38301c59u, 0x382d6161u, 0x382ab140u,
    0x38280bc9u, 0x382570d3u, 0x3822e035u, 0x382059c4u, 0x381ddd59u, 0x381b6accu, 0x381901f6u, 0x3816a2b0u, 0x38144cd4u,
    0x3812003du, 0x380fbcc6u, 0x380d824bu, 0x380b50a8u, 0x380927bau, 0x3807075eu, 0x3804ef74u, 0x3802dfd8u, 0x3800d86au,
    0x37fdb213u, 0x37f9c32eu, 0x37f5e3e5u, 0x37f213fau, 0x37ee5331u, 0x37eaa14du, 0x37e6fe13u, 0x37e36949u, 0x37dfe2b6u,
    0x37dc6a21u, 0x37d8ff52u, 0x37d5a214u, 0x37d25230u, 0x37cf0f71u, 0x37cbd9a3u, 0x37c8b093u, 0x37c5940eu, 0x37c283e2u,
    0x37bf7fdfu, 0x37bc87d3u, 0x37b99b90u, 0x37b6bae7u, 0x37b3e5aau, 0x37b11babu, 0x37ae5cbeu, 0x37aba8b6u, 0x37a8ff6au,
    0x37a660adu, 0x37a3cc57u, 0x37a1423du, 0x379ec237u, 0x379c4c1eu, 0x3799dfc9u, 0x37977d13u, 0x379523d4u, 0x3792d3e8u,
    0x37908d29u, 0x378e4f72u, 0x378c1aa1u, 0x3789ee92u, 0x3787cb21u, 0x3785b02du, 0x37839d95u, 0x37819336u, 0x377f21e0u,
    0x377b2d47u, 0x37774861u, 0x377372efu, 0x376facb5u, 0x376bf576u, 0x37684cf6u, 0x3764b2fbu, 0x3761274bu, 0x375da9aeu,
    0x375a39ebu, 0x3756d7ccu, 0x3753831bu, 0x37503ba2u, 0x374d012cu, 0x3749d387u, 0x3746b27fu, 0x37439de3u, 0x37409580u,
    0x373d9927u, 0x373aa8a7u, 0x3737c3d2u, 0x3734ea79u, 0x37321c6fu, 0x372f5987u, 0x372ca194u, 0x3729f46cu, 0x372751e3u,
    0x3724b9cfu, 0x37222c06u, 0x371fa861u, 0x371d2eb6u, 0x371abedeu, 0x371858b2u, 0x3715fc0cu, 0x3713a8c5u, 0x37115eb9u,
    0x370f1dc3u, 0x370ce5bfu, 0x370ab68au, 0x37089000u, 0x370671feu, 0x37045c64u, 0x37024f10u, 0x370049e1u, 0x36fc996du,
    0x36f8aee2u, 0x36f4d3e1u, 0x36f1082eu, 0x36ed4b8cu, 0x36e99dbeu, 0x36e5fe8au, 0x36e26db7u, 0x36deeb0au, 0x36db764cu,
    0x36d80f45u, 0x36d4b5bfu, 0x36d16985u, 0x36ce2a62u, 0x36caf821u, 0x36c7d290u, 0x36c4b97cu, 0x36c1acb4u, 0x36beac06u,
    0x36bbb744u, 0x36b8ce3cu, 0x36b5f0c2u, 0x36b31ea7u, 0x36b057beu, 0x36ad9bdbu, 0x36aaead1u, 0x36a84476u, 0x36a5a89fu,
    0x36a31723u, 0x36a08fd8u, 0x369e1297u, 0x369b9f37u, 0x36993590u, 0x3696d57du, 0x36947ed8u, 0x3692317au, 0x368fed40u,
    0x368db204u, 0x368b7fa4u, 0x368955fbu, 0x368734e8u, 0x36851c49u, 0x36830bfbu, 0x368103deu, 0x367e07a3u, 0x367a176au,
    0x367636d3u, 0x3672659fu, 0x366ea392u, 0x366af06eu, 0x36674bfau, 0x3663b5fbu, 0x36602e38u, 0x365cb477u, 0x36594882u,
    0x3655ea21u, 0x3652991fu, 0x364f5546u, 0x364c1e63u, 0x3648f442u, 0x3645d6b0u, 0x3642c57cu, 0x363fc074u, 0x363cc769u,
    0x3639da29u, 0x3636f888u, 0x36342256u, 0x36315766u, 0x362e978cu, 0x362be29bu, 0x36293869u, 0x362698cau, 0x36240394u,
    0x3621789fu, 0x361ef7c2u, 0x361c80d4u, 0x361a13aeu, 0x3617b02au, 0x36155620u, 0x3613056cu, 0x3610bde9u, 0x360e7f71u,
    0x360c49e1u, 0x360a1d16u, 0x3607f8edu, 0x3605dd44u, 0x3603c9f8u, 0x3601bee9u, 0x35ff77ecu, 0x35fb81fdu, 0x35f79bc7u,
    0x35f3c50au, 0x35effd8au, 0x35ec450au, 0x35e89b4eu, 0x35e5001cu, 0x35e1733au, 0x35ddf46fu, 0x35da8384u, 0x35d72041u,
    0x35d3ca70u, 0x35d081dcu, 0x35cd4650u, 0x35ca1798u, 0x35c6f582u, 0x35c3dfdcu, 0x35c0d673u, 0x35bdd918u, 0x35bae79bu,
    0x35b801ccu, 0x35b5277du, 0x35b25881u, 0x35af94aau, 0x35acdbcdu, 0x35aa2dbdu, 0x35a78a51u, 0x35a4f15du, 0x35a262b8u,
    0x359fde39u, 0x359d63b9u, 0x359af30fu, 0x35988c13u, 0x35962ea1u, 0x3593da92u, 0x35918fc0u, 0x358f4e08u, 0x358d1544u,
    0x358ae552u, 0x3588be0eu, 0x35869f56u, 0x35848908u, 0x35827b03u, 0x35807525u, 0x357cee9eu, 0x357902c0u, 0x35752673u,
    0x35715978u, 0x356d9b93u, 0x3569ec88u, 0x35664c1bu, 0x3562ba14u, 0x355f3638u, 0x355bc04fu, 0x35585823u, 0x3554fd7cu,
    0x3551b025u, 0x354e6feau, 0x354b3c95u, 0x354815f4u, 0x3544fbd5u, 0x3541ee05u, 0x353eec54u, 0x353bf693u, 0x35390c90u,
    0x35362e1fu, 0x35335b10u, 0x35309337u, 0x352dd668u, 0x352b2475u, 0x35287d36u, 0x3525e07eu, 0x35234e24u, 0x3520c5ffu,
    0x351e47e7u, 0x351bd3b3u, 0x3519693cu, 0x3517085cu, 0x3514b0ecu, 0x351262c8u, 0x35101dcau, 0x350de1ceu, 0x350baeb0u,
    0x3509844du, 0x35076282u, 0x3505492du, 0x3503382du, 0x35012f61u, 0x34fe5d4fu, 0x34fa6bc2u, 0x34f689dcu, 0x34f2b75fu,
    0x34eef40du, 0x34eb3fabu, 0x34e799fcu, 0x34e402c7u, 0x34e079d3u, 0x34dcfee6u, 0x34d991c9u, 0x34d63246u, 0x34d2e025u,
    0x34cf9b33u, 0x34cc633au, 0x34c93808u, 0x34c61969u, 0x34c3072cu, 0x34c00120u, 0x34bd0713u, 0x34ba18d7u, 0x34b7363du,
    0x34b45f16u, 0x34b19335u, 0x34aed26eu, 0x34ac1c93u, 0x34a9717bu, 0x34a6d0f9u, 0x34a43ae5u, 0x34a1af15u, 0x349f2d5fu,
    0x349cb59cu, 0x349a47a5u, 0x3497e352u, 0x3495887eu, 0x34933702u, 0x3490eeb9u, 0x348eaf80u, 0x348c7932u, 0x348a4babu,
    0x348826c9u, 0x34860a69u, 0x3483f66au, 0x3481eaabu, 0x347fce14u, 0x347bd6d0u, 0x3477ef49u, 0x34741741u, 0x34704e7au,
    0x346c94b9u, 0x3468e9c1u, 0x34654d57u, 0x3461bf43u, 0x345e3f4bu, 0x345acd36u, 0x345768cfu, 0x345411deu, 0x3450c82eu,
    0x344d8b8bu, 0x344a5bc1u, 0x3447389cu, 0x344421ebu, 0x3441177cu, 0x343e191fu, 0x343b26a4u, 0x34383fdau, 0x34356495u,
    0x343294a7u, 0x342fcfe1u, 0x342d1619u, 0x342a6722u, 0x3427c2d2u, 0x342528feu, 0x3422997cu, 0x34201424u, 0x341d98ceu,
    0x341b2751u, 0x3418bf86u, 0x34166148u, 0x34140c6fu, 0x3411c0d8u, 0x340f7e5cu, 0x340d44d9u, 0x340b142au, 0x3408ec2cu,
    0x3406ccbdu, 0x3404b5bbu, 0x3402a704u, 0x3400a078u, 0x33fd43ebu, 0x33f956bbu, 0x33f57921u, 0x33f1aadeu,
};

/**
 * @brief fp32 exp LUT used by softmax. table[0, 1025) is FLOAT_EXP_LUT_BITS; [1025, 2048) is
 * table[i - 1024] * exp(-16); the rest is 0.
 */
struct AttentionFloatExpLut {
    float table[FLOAT_EXP_PRECISE_ENTRIES]; /*!< [0, FLOAT_EXP_PRECISE_ENTRIES) exp values */
    AttentionFloatExpLut()
    {
        static_assert(FLOAT_EXP_LUT_ENTRIES == 1025, "AttentionFloatExpLut length must match FLOAT_EXP_LUT_BITS");
        for (int i = 0; i < FLOAT_EXP_LUT_ENTRIES; ++i) {
            std::memcpy(&table[i], &FLOAT_EXP_LUT_BITS[i], sizeof(float));
        }
        constexpr int steps = FLOAT_EXP_LUT_ENTRIES - 1;
        for (int i = FLOAT_EXP_LUT_ENTRIES; i < 2 * steps; ++i) {
            table[i] = table[i - steps] * table[steps];
        }
        for (int i = 2 * steps; i < FLOAT_EXP_PRECISE_ENTRIES; ++i) {
            table[i] = 0.0f;
        }
    }
    /**
     * @brief Interpolated exp(x) for x in [-16, 0].
     *
     * Truncation alone is biased ~1.6% low.
     *
     * @param x   logit, typically <= 0
     *
     * @return interpolated table value
     */
    inline float lookup(float x) const
    {
        if (x >= 0.0f)
            return 1.0f;
        const float pos = -x * FLOAT_EXP_LUT_INVERSE_STEP;
        int idx = static_cast<int>(pos);
        if (idx >= FLOAT_EXP_LUT_ENTRIES - 1) {
            return table[FLOAT_EXP_LUT_ENTRIES - 1];
        }
        if (idx < 0)
            idx = 0;
        const float frac = pos - static_cast<float>(idx);
        return table[idx] + (table[idx + 1] - table[idx]) * frac;
    }

    /**
     * @brief exp(x) to a few float ulps via table[i] * cubic Taylor of exp(-d). lookup() is biased up
     * to 3e-5 high, which int16 rows over a long Lk accumulate past the +-1 step Model::test() allows.
     *
     * Below -32 the result is under 1.3e-14 and flushes to 0.
     *
     * @param x   logit, typically <= 0
     *
     * @return table[i] * exp(-d)
     */
    inline float lookup_precise(float x) const
    {
        if (x >= 0.0f)
            return 1.0f;
        const float r = -x;
        if (!(r < 2.0f * FLOAT_EXP_LUT_MAX_NEGATIVE))
            return 0.0f;
        const int idx = static_cast<int>(r * FLOAT_EXP_LUT_INVERSE_STEP);
        const float d = r - static_cast<float>(idx) * FLOAT_EXP_LUT_STEP;
        const float p = 1.0f - d * (1.0f - d * (0.5f - d * (1.0f / 6.0f)));
        return table[idx] * p;
    }
};

AttentionFloatExpLut attention_float_exp_lut_data{};

/**
 * @brief Return the process-wide fp32 exp LUT.
 *
 * @return LUT singleton
 */
inline const AttentionFloatExpLut &attention_float_exp_lut()
{
    return attention_float_exp_lut_data;
}

/**
 * @brief int32 QK dot of a slot excluded from softmax (causal frontier or -inf mask).
 */
constexpr int32_t QK_DOTS_MASKED = INT32_MIN;

/**
 * @brief Summary of one (Q block, K tile) of a constant attn_mask.
 */
enum : uint8_t {
    MASK_TILE_ZERO = 0,    /*!< all 0: no mask work */
    MASK_TILE_FINITE = 1,  /*!< finite, not all 0 */
    MASK_TILE_MIXED = 2,   /*!< holds -inf */
    MASK_TILE_DROPPED = 3, /*!< all -inf: the whole tile is a softmax no-op and is skipped */
};

/**
 * @brief The component builds with -ffast-math, which lets GCC fold float compares against
 * infinity; test the bit pattern instead.
 */
inline bool is_neg_inf(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u == 0xff800000u;
}

/**
 * @brief Finite, positive and normal (a subnormal's reciprocal overflows), by bit pattern for the
 * same reason.
 */
inline bool is_positive_normal(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u >= 0x00800000u && u < 0x7f800000u;
}

/**
 * @brief Load one attn_mask element as an additive float score.
 *
 * @param mask_args   mask payload and dtype
 * @param idx         linear element index
 *
 * @return 0 / -inf for BOOL, or the FLOAT value
 */
inline float load_mask_element(const AttentionMaskArgs &mask_args, int idx)
{
    if (mask_args.dtype == DATA_TYPE_BOOL) {
        return static_cast<const uint8_t *>(mask_args.data)[idx] ? 0.f : -std::numeric_limits<float>::infinity();
    }
    return static_cast<const float *>(mask_args.data)[idx];
}

/**
 * @brief One cached V tile: both int8 and int16 use a d-block-major layout spanning D_pad rows of
 * 64 tokens (16 int8 or 8 int16 dimensions per block).
 */
inline size_t attention_v_tile_bytes(int D_pad, size_t elem_bytes)
{
    return static_cast<size_t>(D_pad) * static_cast<size_t>(ATTENTION_KEY_TILE_SIZE) * elem_bytes;
}

/**
 * @brief Element size of one Q/K/V/O value.
 *
 * @tparam qkv_dtype   FLOAT / INT16 / INT8
 *
 * @return size in bytes
 */
template <dtype_t qkv_dtype>
constexpr int qkv_element_size()
{
    if constexpr (qkv_dtype == DATA_TYPE_FLOAT) {
        return static_cast<int>(sizeof(float));
    } else if constexpr (qkv_dtype == DATA_TYPE_INT16) {
        return static_cast<int>(sizeof(int16_t));
    } else {
        return static_cast<int>(sizeof(int8_t));
    }
}

/**
 * @brief Per-task tile workspace used by one Attention forward.
 */
struct AttentionTileBuffer {
    union {
        alignas(16) uint8_t q_tile[sizeof(float) * ATTENTION_QUERY_TILE_SIZE * ATTENTION_MAX_HEAD_DIM]; /*!< packed Q */
        alignas(16) float q_values[ATTENTION_QUERY_TILE_SIZE * ATTENTION_MAX_HEAD_DIM]; /*!< dequantized Q */
    };
    alignas(16) float v_values[ATTENTION_KEY_TILE_SIZE * ATTENTION_MAX_HEAD_DIM]; /*!< dequantized V tile */
    alignas(16) float acc[ATTENTION_QUERY_TILE_SIZE * ATTENTION_MAX_HEAD_DIM];    /*!< online P·V accumulator */
    union {
        alignas(16) float exp_scores[ATTENTION_QUERY_TILE_SIZE * ATTENTION_KEY_TILE_SIZE]; /*!< fp32 exp scores */
        alignas(16) int8_t quantized_exp_scores[ATTENTION_QUERY_TILE_SIZE * ATTENTION_KEY_TILE_SIZE]; /*!< int8 exp */
    };
    union {
        alignas(16) float scores[ATTENTION_QUERY_TILE_SIZE * ATTENTION_KEY_TILE_SIZE]; /*!< fp32 QK scores */
        alignas(16) int32_t dots[ATTENTION_QUERY_TILE_SIZE * ATTENTION_KEY_TILE_SIZE]; /*!< int32 QK dots */
    };
    float exp_scale[ATTENTION_QUERY_TILE_SIZE];     /*!< per-row dequant scale of the exp scores */
    float row_max[ATTENTION_QUERY_TILE_SIZE];       /*!< online softmax row max */
    int32_t dot_row_max[ATTENTION_QUERY_TILE_SIZE]; /*!< online softmax row max in dot space */
    float row_sum[ATTENTION_QUERY_TILE_SIZE];       /*!< online softmax row sum */
    alignas(16) int32_t qacc_buffer[32];            /*!< QACC dumps: 16 int32 per st.qacc; int16 P·V keeps two */
    alignas(16) int32_t tile_dot_max[ATTENTION_QUERY_TILE_SIZE]; /*!< current int8 QK-tile maxima */
    alignas(16) int32_t exp_scratch[5];                          /*!< four broadcast words and one saved CFG word */
};

/**
 * @brief Bytes of the per-node tile workspace.
 *
 * @return ATTENTION_TILE_BUFFER_COUNT * sizeof(AttentionTileBuffer)
 */
constexpr size_t attention_tile_buffer_bytes()
{
    return sizeof(AttentionTileBuffer) * static_cast<size_t>(ATTENTION_TILE_BUFFER_COUNT);
}

/**
 * @brief Wait until every task of this node has reached the same barrier.
 *
 * @param barrier      shared phase/count
 * @param task_count   number of tasks that must arrive
 */
inline void attention_wait_at_barrier(AttentionTaskBarrier &barrier, int task_count)
{
    if (task_count <= 1) {
        return;
    }
    const int my_phase = __atomic_load_n(&barrier.bar_phase, __ATOMIC_RELAXED);
    const int arrived = __atomic_add_fetch(&barrier.bar_count, 1, __ATOMIC_ACQ_REL);
    if (arrived == task_count) {
        __atomic_store_n(&barrier.bar_count, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&barrier.bar_phase, my_phase + 1, __ATOMIC_RELEASE);
    } else {
        int spins = 0;
        while (__atomic_load_n(&barrier.bar_phase, __ATOMIC_ACQUIRE) == my_phase) {
            if (++spins >= 64) {
                taskYIELD();
                spins = 0;
            }
        }
    }
}

} // namespace
} // namespace base
} // namespace dl
