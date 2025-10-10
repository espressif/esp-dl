idf_build_get_property(target IDF_TARGET)

set(src_dirs        ./dl/tool/src
                    ./dl/tensor/src
                    ./dl/base
                    ./dl/math/src
                    ./dl/model/src
                    ./dl/module/src
                    ./fbs_loader/src
                    ./vision/detect
                    ./vision/image
                    ./vision/recognition
                    ./vision/classification
                    ./audio/common
                    ./audio/speech_features
                    )

set(include_dirs    ./dl
                    ./dl/tool/include
                    ./dl/tensor/include
                    ./dl/base
                    ./dl/base/isa
                    ./dl/math/include
                    ./dl/model/include
                    ./dl/module/include
                    ./fbs_loader/include
                    ./vision/detect
                    ./vision/image
                    ./vision/image/isa
                    ./vision/recognition
                    ./vision/classification
                    ./audio/common
                    ./audio/speech_features
                    )

if(CONFIG_IDF_TARGET_ESP32)
    list(APPEND src_dirs        dl/tool/isa/xtensa)
    list(APPEND src_dirs        dl/base/isa/xtensa)
    list(APPEND include_dirs    dl/base/isa/xtensa)

elseif(CONFIG_IDF_TARGET_ESP32S2)

elseif(CONFIG_IDF_TARGET_ESP32S3)
    list(APPEND src_dirs        dl/tool/isa/xtensa)
    list(APPEND src_dirs        dl/tool/isa/tie728)
    list(APPEND src_dirs        dl/base/isa/xtensa)
    list(APPEND src_dirs        dl/base/isa/tie728)
    list(APPEND include_dirs    dl/base/isa/tie728)
    list(APPEND include_dirs    dl/base/isa/xtensa)

elseif(CONFIG_IDF_TARGET_ESP32C3)

elseif(CONFIG_IDF_TARGET_ESP32P4)
    list(APPEND src_dirs        dl/tool/isa/esp32p4)
    list(APPEND src_dirs        dl/base/isa/esp32p4)
    list(APPEND include_dirs    dl/base/isa/esp32p4)
    list(APPEND src_dirs        ./vision/image/isa/esp32p4)
endif()

set(requires        dl_fft
                    esp_mm
                    esp_new_jpeg
                    esp_driver_jpeg
                    esp_driver_ppa
                    esp_partition
                    esp_timer
                    mbedtls
                    spi_flash)

idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

component_compile_options(-ffast-math -O3 -Wno-error=format=-Wno-format)
add_prebuilt_library(fbs_model fbs_loader/lib/${IDF_TARGET}/libfbs_model.a)
target_link_libraries(${COMPONENT_LIB} PRIVATE fbs_model)

# component_compile_options(-ffast-math -frtti -O3 -Wno-error=format=-Wno-format)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-array-bounds
                                                -Wno-deprecated-copy
                                                -Wno-strict-aliasing
                                                -Wno-overloaded-virtual)
