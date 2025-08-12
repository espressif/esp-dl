
#include "dl_audio_wav.hpp"
namespace dl {
namespace audio {

#define TAG(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | (d))

static uint32_t read_tag(const uint8_t **data)
{
    uint32_t tag = 0;
    tag = (tag << 8) | *(*data)++;
    tag = (tag << 8) | *(*data)++;
    tag = (tag << 8) | *(*data)++;
    tag = (tag << 8) | *(*data)++;
    return tag;
}

static uint32_t read_int32(const uint8_t **data)
{
    uint32_t value = 0;
    value |= *(*data)++ << 0;
    value |= *(*data)++ << 8;
    value |= *(*data)++ << 16;
    value |= *(*data)++ << 24;
    return value;
}

static uint16_t read_int16(const uint8_t **data)
{
    uint16_t value = 0;
    value |= *(*data)++ << 0;
    value |= *(*data)++ << 8;
    return value;
}

dl_audio_t *decode_wav(const uint8_t *data, int data_len)
{
    const uint8_t *end = data + data_len;
    dl_audio_t *audio = (dl_audio_t *)calloc(1, sizeof(dl_audio_t));
    if (!audio)
        return NULL;

    int format = 0;
    int data_found = 0;
    const uint8_t *audio_data = NULL;
    uint32_t audio_data_len = 0;

    while (data < end) {
        uint32_t tag = read_tag(&data);
        if (data + 4 > end)
            break;
        uint32_t length = read_int32(&data);
        if (data + length > end)
            break;

        if (tag == TAG('R', 'I', 'F', 'F') && length >= 4) {
            if (data + 4 > end)
                break;
            uint32_t wave_tag = read_tag(&data);
            length -= 4;
            if (wave_tag != TAG('W', 'A', 'V', 'E')) {
                data += length;
                continue;
            }

            // Process subchunks
            while (length >= 8) {
                uint32_t subtag = read_tag(&data);
                uint32_t sublength = read_int32(&data);
                length -= 8;
                if (sublength > length)
                    break;

                if (subtag == TAG('f', 'm', 't', ' ')) {
                    if (sublength < 16)
                        break;
                    format = read_int16(&data);
                    audio->channels = read_int16(&data);
                    audio->sample_rate = read_int32(&data);
                    read_int32(&data); // byte rate
                    read_int16(&data); // block align
                    audio->bits_per_sample = read_int16(&data);
                    data += sublength - 16;
                } else if (subtag == TAG('d', 'a', 't', 'a')) {
                    audio_data = data;
                    audio_data_len = sublength;
                    data += sublength;
                    data_found = 1;
                } else {
                    data += sublength;
                }
                length -= sublength;
            }
            if (length > 0)
                data += length;
        } else {
            data += length;
        }
    }

    if (!data_found || format != 1 || audio->bits_per_sample != 16) {
        free(audio);
        return NULL;
    }

    audio->length = audio_data_len / (audio->bits_per_sample / 8 * audio->channels);
    audio->data = (int16_t *)malloc(audio_data_len);
    if (!audio->data) {
        free(audio);
        return NULL;
    }
    memcpy(audio->data, audio_data, audio_data_len);

    return audio;
}

void print_audio_info(dl_audio_t *data)
{
    if (!data) {
        printf("No audio data\n");
        return;
    }
    printf("Audio Info:\n");
    printf("  Sample Rate: %d Hz\n", data->sample_rate);
    printf("  Channels: %d\n", data->channels);
    printf("  Bits per Sample: %d\n", data->bits_per_sample);
    printf("  Length: %ld samples\n", data->length);
    printf("  Data Size: %ld bytes\n", sizeof(int16_t) * data->length * data->channels);
}

} // namespace audio
} // namespace dl
