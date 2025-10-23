#include "speaker_verification.hpp"
#include "esp_log.h"
#include "bsp/esp-bsp.h"

extern const uint8_t _binary_audio_a_wav_start[] asm("_binary_audio_a_wav_start");
extern const uint8_t _binary_audio_a_wav_end[]   asm("_binary_audio_a_wav_end");
extern const uint8_t _binary_audio_b_wav_start[] asm("_binary_audio_b_wav_start");
extern const uint8_t _binary_audio_b_wav_end[]   asm("_binary_audio_b_wav_end");
extern const uint8_t _binary_audio_c_wav_start[] asm("_binary_audio_c_wav_start");
extern const uint8_t _binary_audio_c_wav_end[]   asm("_binary_audio_c_wav_end");

const char *TAG = "speaker_verification";

extern "C" void app_main(void)
{
#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    SpeakerVerification* verifier = new SpeakerVerification();

    float* embed_a = verifier->run(
        _binary_audio_a_wav_start, _binary_audio_a_wav_end - _binary_audio_a_wav_start);
    float* embed_b = verifier->run(
        _binary_audio_b_wav_start, _binary_audio_b_wav_end - _binary_audio_b_wav_start);
    float* embed_c = verifier->run(
        _binary_audio_c_wav_start, _binary_audio_c_wav_end - _binary_audio_c_wav_start);
    
    float similarity1 = verifier->compute_similarity(embed_a, embed_b);
    ESP_LOGI(TAG, "Cosine similarity between audio a and b: %.4f", similarity1);
    float similarity2 = verifier->compute_similarity(embed_b, embed_c);
    ESP_LOGI(TAG, "Cosine similarity between audio b and c: %.4f", similarity2);

    free(embed_a);
    free(embed_b);
    free(embed_c);
    delete verifier;

#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
