set(include_dirs    include/
                    include/tool
                    include/typedef
                    include/image
                    include/math
                    include/nn
                    include/tvm
                    include/layer
                    include/detect
                    include/model_zoo)

idf_component_register(SRCS dummy.c INCLUDE_DIRS ${include_dirs})

if(IDF_TARGET STREQUAL "esp32")
    set(lib_dir ${COMPONENT_DIR}/lib/esp32)
elseif(IDF_TARGET STREQUAL "esp32s2")
    set(lib_dir ${COMPONENT_DIR}/lib/esp32s2)
elseif(IDF_TARGET STREQUAL "esp32s3")
    set(lib_dir ${COMPONENT_DIR}/lib/esp32s3)
elseif(IDF_TARGET STREQUAL "esp32c3")    
    set(lib_dir ${COMPONENT_DIR}/lib/esp32c3)
endif()

add_prebuilt_library(dl ${lib_dir}/libdl.a)
add_prebuilt_library(cat_face_detect ${lib_dir}/libcat_face_detect.a)
add_prebuilt_library(color_detect ${lib_dir}/libcolor_detect.a)
add_prebuilt_library(human_face_detect ${lib_dir}/libhuman_face_detect.a)
add_prebuilt_library(mfn ${lib_dir}/libmfn.a)

target_link_libraries(${COMPONENT_LIB} INTERFACE cat_face_detect color_detect human_face_detect mfn dl)

