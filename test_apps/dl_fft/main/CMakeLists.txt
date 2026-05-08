set(srcs "app_main.cpp"
         "test_fft.c"
         "test_dl_fft.cpp"
        #  "test_dsp_fft.cpp"
         "test_kiss_fft.cpp"
         "kiss_fft/kiss_fft.c"
         "kiss_fft/kiss_fftr.c"
         "kiss_fft/kiss_fftnd.c"
         "kiss_fft/kiss_fftndr.c")

set(include_dirs "test_data"
                 "kiss_fft")

# Same WAV as dl_audio tests; copied here so EMBED_FILES path is stable (_binary_wav_embed_test_wav_*).
set(DL_AUDIO_TEST_WAV "${CMAKE_CURRENT_LIST_DIR}/../../dl_audio/test_cases/test.wav")
set(WAV_EMBED_DIR "${CMAKE_CURRENT_LIST_DIR}/wav_embed")
file(MAKE_DIRECTORY "${WAV_EMBED_DIR}")
file(COPY "${DL_AUDIO_TEST_WAV}" DESTINATION "${WAV_EMBED_DIR}")

set(requires    unity
                dl_fft
                esp_timer
                esp-dl)

idf_component_register(SRCS ${srcs}
                       INCLUDE_DIRS ${include_dirs}
                       REQUIRES ${requires}
                       EMBED_FILES "wav_embed/test.wav"
                       WHOLE_ARCHIVE)


component_compile_options(-Wno-error=format=-Wno-format)
