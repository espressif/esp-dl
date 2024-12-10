set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl)

set(packed_model ${BUILD_DIR}/espdl_models/pedestrian_detect.espdl)
if(CONFIG_PEDESTRIAN_DETECT_MODEL_IN_FLASH_RODATA)
    set(embed_files ${packed_model})
else()
    set(embed_files)
endif()

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})

if (IDF_TARGET STREQUAL "esp32s3")
    set(models_dir ${COMPONENT_DIR}/models/s3)
elseif (IDF_TARGET STREQUAL "esp32p4")
    set(models_dir ${COMPONENT_DIR}/models/p4)
endif()

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
set(models)
if(CONFIG_PEDESTRIAN_DETECT_PICO_S8_V1)
    list(APPEND models ${models_dir}/pedestrian_detect_pico_s8_v1.espdl)
endif()

set(pack_model_exe ${COMPONENT_DIR}/pack_model.py)
add_custom_command(
    OUTPUT ${packed_model}
    COMMENT "Move and Pack models..."
    COMMAND python ${pack_model_exe} --model_path ${models} --out_file ${packed_model}
    DEPENDS ${models}
    VERBATIM)

add_custom_target(pedestrian_detect_model ALL DEPENDS ${packed_model})

if(CONFIG_PEDESTRIAN_DETECT_MODEL_IN_FLASH_PARTITION)
    add_dependencies(flash pedestrian_detect_model)
    esptool_py_flash_to_partition(flash "pedestrian_det" ${packed_model})
endif()
