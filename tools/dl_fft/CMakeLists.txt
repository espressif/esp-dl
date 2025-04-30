idf_build_get_property(target IDF_TARGET)

set(src_dirs        .
                    base
                        )

set(include_dirs    .
                    base
                    )

if(CONFIG_IDF_TARGET_ESP32)
    list(APPEND src_dirs        "base/esp32" )

elseif(CONFIG_IDF_TARGET_ESP32S3)
    list(APPEND src_dirs        "base/esp32s3" )

elseif(CONFIG_IDF_TARGET_ESP32P4)
    list(APPEND src_dirs        "base/esp32p4" )

endif()

if(CONFIG_DL_FFT_WITH_KISS_FFT)
list(APPEND src_dirs            "kiss_fft" )
list(APPEND include_dirs        "kiss_fft" )
endif()

set(requires       esp-dsp)

idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

component_compile_options(-ffast-math -O2)
