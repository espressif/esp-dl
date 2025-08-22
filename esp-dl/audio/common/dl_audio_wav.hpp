#include "dl_audio_common.hpp"

namespace dl {
namespace audio {
/**
 * @brief Structure representing audio data.
 *
 * This structure contains the audio data buffer and its associated metadata,
 * such as length, sample rate, bits per sample, and number of channels.
 */
typedef struct {
    int16_t *data;       /**< Pointer to the audio data buffer. */
    uint32_t length;     /**< Length of the audio data buffer (number of samples). */
    int sample_rate;     /**< Sample rate of the audio data (in Hz). */
    int bits_per_sample; /**< Number of bits per audio sample. */
    int channels;        /**< Number of audio channels (e.g., 1 for mono, 2 for stereo). */
} dl_audio_t;

/**
 * @brief Decodes WAV audio data from a memory buffer.
 *
 * This function takes a buffer containing WAV audio data and decodes it into
 * a `dl_audio_t` structure. The caller is responsible for freeing the memory
 * allocated for the returned `dl_audio_t` structure.
 *
 * @param data Pointer to the buffer containing the WAV audio data.
 * @param data_len Length of the WAV audio data buffer (in bytes).
 * @return Pointer to a `dl_audio_t` structure containing the decoded audio data,
 *         or `nullptr` if decoding fails.
 */
dl_audio_t *decode_wav(const uint8_t *data, int data_len);

/**
 * @brief Prints information about the audio data.
 *
 * This function outputs the metadata of the given `dl_audio_t` structure,
 * such as sample rate, bits per sample, number of channels, and data length.
 *
 * @param data Pointer to the `dl_audio_t` structure containing the audio data.
 */
void print_audio_info(dl_audio_t *data);

} // namespace audio
} // namespace dl
