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

file(GLOB exclude_srcs RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/vision/image/dl_image_pixel_cvt_dispatch_*.cpp)

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
    file(GLOB ext_exclude_srcs RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/vision/image/isa/esp32p4/dl_esp32p4_cvt_color_*.S ${CMAKE_CURRENT_LIST_DIR}/vision/image/isa/esp32p4/dl_esp32p4_resize_nn_*.S)
    list(APPEND exclude_srcs ${ext_exclude_srcs})
endif()

set(requires        dl_fft
                    esp_hw_support
                    esp_mm
                    esp_new_jpeg
                    esp_driver_jpeg
                    esp_driver_ppa
                    esp_partition
                    esp_timer
                    mbedtls
                    spi_flash)

if (CONFIG_PIX_CVT_GRAY_TO_GRAY_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_gray2gray.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_gray2gray.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_gray2gray.S)
endif()
if (CONFIG_PIX_CVT_HSV_TO_HSV_MASK_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_hsv2hsv_mask.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_hsv2hsv_mask.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_hsv2hsv_mask.S)
endif()
if (CONFIG_PIX_CVT_RGB565_TO_GRAY_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_rgb5652gray.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_rgb5652gray.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_rgb5652gray.S)
endif()
if (CONFIG_PIX_CVT_RGB565_TO_HSV_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_rgb5652hsv.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_rgb5652hsv.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_rgb5652hsv.S)
endif()
if (CONFIG_PIX_CVT_RGB565_TO_RGB565_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_rgb5652rgb565.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_rgb5652rgb565.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_rgb5652rgb565.S)
endif()
if (CONFIG_PIX_CVT_RGB565_TO_RGB888_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_rgb5652rgb888.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_rgb5652rgb888.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_rgb5652rgb888.S)
endif()
if (CONFIG_PIX_CVT_RGB888_TO_GRAY_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_rgb8882gray.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_rgb8882gray.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_rgb8882gray.S)
endif()
if (CONFIG_PIX_CVT_RGB888_TO_HSV_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_rgb8882hsv.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_rgb8882hsv.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_rgb8882hsv.S)
endif()
if (CONFIG_PIX_CVT_RGB888_TO_RGB565_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_rgb8882rgb565.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_rgb8882rgb565.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_rgb8882rgb565.S)
endif()
if (CONFIG_PIX_CVT_RGB888_TO_RGB888_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_rgb8882rgb888.cpp)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_cvt_color_rgb8882rgb888.S)
    list(REMOVE_ITEM exclude_srcs vision/image/isa/esp32p4/dl_esp32p4_resize_nn_rgb8882rgb888.S)
endif()
if (CONFIG_PIX_CVT_YUV_TO_RGB888_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_yuv2rgb888.cpp)
endif()
if (CONFIG_PIX_CVT_YUV_TO_RGB565_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_yuv2rgb565.cpp)
endif()
if (CONFIG_PIX_CVT_YUV_TO_GRAY_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_yuv2gray.cpp)
endif()
if (CONFIG_PIX_CVT_YUV_TO_HSV_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_yuv2hsv.cpp)
endif()
if (CONFIG_PIX_CVT_YUV_TO_YUV_SUPPORT)
    list(REMOVE_ITEM exclude_srcs vision/image/dl_image_pixel_cvt_dispatch_yuv2yuv.cpp)
endif()

idf_component_register(SRC_DIRS ${src_dirs} EXCLUDE_SRCS ${exclude_srcs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

component_compile_options(-ffast-math -O3 -Wno-error=format=-Wno-format)
add_prebuilt_library(fbs_model fbs_loader/lib/${IDF_TARGET}/libfbs_model.a)
target_link_libraries(${COMPONENT_LIB} PRIVATE fbs_model)

# component_compile_options(-ffast-math -frtti -O3 -Wno-error=format=-Wno-format)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-array-bounds
                                                -Wno-deprecated-copy
                                                -Wno-strict-aliasing
                                                -Wno-overloaded-virtual)
