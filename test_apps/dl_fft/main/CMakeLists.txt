set(srcs app_main.cpp
         test_fft.c
         test_dl_fft.cpp
         test_dsp_fft.cpp
         test_kiss_fft.cpp)

set(include_dirs "test_data")

set(requires    unity
                dl_fft
                esp_timer)

idf_component_register(SRCS ${srcs}
                       INCLUDE_DIRS ${include_dirs}
                       REQUIRES ${requires}
                       WHOLE_ARCHIVE)


component_compile_options(-Wno-error=format=-Wno-format)
