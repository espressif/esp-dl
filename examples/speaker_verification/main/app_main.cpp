#include "dl_audio_verification_database.hpp"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "speaker_verification.hpp"
#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
#include "bsp/esp-bsp.h"
#endif

extern const uint8_t _binary_audio_a_wav_start[] asm("_binary_audio_a_wav_start");
extern const uint8_t _binary_audio_a_wav_end[] asm("_binary_audio_a_wav_end");
extern const uint8_t _binary_audio_b_wav_start[] asm("_binary_audio_b_wav_start");
extern const uint8_t _binary_audio_b_wav_end[] asm("_binary_audio_b_wav_end");
extern const uint8_t _binary_audio_c_wav_start[] asm("_binary_audio_c_wav_start");
extern const uint8_t _binary_audio_c_wav_end[] asm("_binary_audio_c_wav_end");

static const char *TAG = "speaker_verification";

extern "C" void app_main(void)
{
#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    // The verification database is persisted to a file, so mount a filesystem.
    esp_vfs_spiffs_conf_t spiffs_conf = {
        .base_path = "/spiffs",
        .partition_label = "storage",
        .max_files = 2,
        .format_if_mount_failed = true,
    };
    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&spiffs_conf));

    SpeakerVerification *verifier = new SpeakerVerification();

    dl::audio::AudioVerificationDatabase db("/spiffs/spker.db", verifier->get_embedding_dim());
    if (!db.is_valid()) {
        ESP_LOGE(TAG, "Failed to initialize verification database.");
        delete verifier;
        esp_vfs_spiffs_unregister("storage");
#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
        bsp_sdcard_unmount();
#endif
        return;
    }
    // Start from a clean database every run.
    db.clear();

    // --- Enroll: store a single reference embedding for "speaker1" -----------
    float *embed_a = verifier->run(_binary_audio_a_wav_start, _binary_audio_a_wav_end - _binary_audio_a_wav_start);
    if (!embed_a) {
        ESP_LOGE(TAG, "Failed to extract enrollment embedding for audio_a.");
        delete verifier;
        esp_vfs_spiffs_unregister("storage");
#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
        bsp_sdcard_unmount();
#endif
        return;
    }
    db.enroll("speaker1", embed_a);
    free(embed_a);

    // Persist the database. Subspace verification needs >=2 enrollments, so with
    // a single embedding build() skips the subspace and only cosine verification
    // is meaningful below.
    db.build();
    db.print();

    // --- Verify audio_b and audio_c against the enrolled reference -----------
    const struct {
        const char *name;
        const uint8_t *start;
        const uint8_t *end;
    } test_audios[] = {
        {"audio_b", _binary_audio_b_wav_start, _binary_audio_b_wav_end},
        {"audio_c", _binary_audio_c_wav_start, _binary_audio_c_wav_end},
    };

    for (const auto &t : test_audios) {
        float *target = verifier->run(t.start, (size_t)(t.end - t.start));
        if (!target) {
            ESP_LOGE(TAG, "Failed to extract test embedding for %s.", t.name);
            continue;
        }
        ESP_LOGI(TAG, "Testing %s against enrolled speaker1:", t.name);
        db.verify_max_cosine(target, 0.25f); // >0.25 => SAME speaker
        free(target);
    }

    delete verifier;
    esp_vfs_spiffs_unregister("storage");

#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
