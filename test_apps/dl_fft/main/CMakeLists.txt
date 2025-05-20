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

set(requires    unity
                dl_fft
                esp_timer)

idf_component_register(SRCS ${srcs}
                       INCLUDE_DIRS ${include_dirs}
                       REQUIRES ${requires}
                       WHOLE_ARCHIVE)


component_compile_options(-Wno-error=format=-Wno-format)
