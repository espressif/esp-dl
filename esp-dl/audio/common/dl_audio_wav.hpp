#include "dl_audio_common.hpp"

namespace dl {
namespace audio {

typedef struct {
    int16_t *data;
    uint32_t length;
    int sample_rate;
    int bits_per_sample;
    int channels;
} dl_audio_t;

dl_audio_t *decode_wav(const uint8_t *data, int data_len);

void print_audio_info(dl_audio_t *data);

} // namespace audio
} // namespace dl
