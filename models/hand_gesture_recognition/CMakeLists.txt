idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl
                    hand_detect)

set(packed_model ${BUILD_DIR}/espdl_models/hand_gesture_cls.espdl)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

if (NOT CONFIG_HAND_GESTURE_CLS_MODEL_IN_SDCARD)
    if (IDF_TARGET STREQUAL "esp32s3")
        set(models_dir ${COMPONENT_DIR}/models/s3)
    elseif (IDF_TARGET STREQUAL "esp32p4")
        set(models_dir ${COMPONENT_DIR}/models/p4)
    endif()

    file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
    set(models)
    if(CONFIG_FLASH_HAND_GESTURE_CLS_MOBILENETV2_0_5_S8_V1)
        list(APPEND models ${models_dir}/mobilenetv2_0_5_128_128_gesture.espdl)
    endif()


    set(pack_model_exe ${espdl_dir}/fbs_loader/pack_espdl_models.py)
    add_custom_command(
        OUTPUT ${packed_model}
        COMMENT "Move and Pack models..."
        COMMAND python ${pack_model_exe} --model_path ${models} --out_file ${packed_model}
        DEPENDS ${models}
        VERBATIM)


    if(CONFIG_HAND_GESTURE_CLS_MODEL_IN_FLASH_RODATA)
        target_add_aligned_binary_data(${COMPONENT_LIB} ${packed_model} BINARY)
    endif()

    if(CONFIG_HAND_GESTURE_CLS_MODEL_IN_FLASH_PARTITION)
        add_custom_target(hand_gesture_cls_model ALL DEPENDS ${packed_model})
        add_dependencies(flash hand_gesture_cls_model)
        esptool_py_flash_to_partition(flash "hand_gesture_cls" ${packed_model})
    endif()
endif()
