set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl
                    human_face_detect
                    fatfs
                    vfs
                    spiffs)

if (IDF_TARGET STREQUAL "esp32s3")
    list(APPEND requires esp32_s3_eye)
    set(models_dir ${COMPONENT_DIR}/models/s3)
elseif (IDF_TARGET STREQUAL "esp32p4")
    list(APPEND requires esp32_p4_function_ev_board)
    set(models_dir ${COMPONENT_DIR}/models/p4)
endif()

set(packed_model ${BUILD_DIR}/espdl_models/human_face_feat.espdl)
if(CONFIG_HUMAN_FACE_FEAT_MODEL_IN_FLASH_RODATA)
    set(embed_files ${packed_model})
else()
    set(embed_files)
endif()

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
set(models)
if(CONFIG_HUMAN_FACE_FEAT_MFN_S8_V1)
    list(APPEND models ${models_dir}/human_face_feat_mfn_s8_v1.espdl)
endif()
if(CONFIG_HUMAN_FACE_FEAT_MBF_S8_V1)
    list(APPEND models ${models_dir}/human_face_feat_mbf_s8_v1.espdl)
endif()

set(pack_model_exe ${COMPONENT_DIR}/pack_model.py)
add_custom_command(
    OUTPUT ${packed_model}
    COMMENT "Move and Pack models..."
    COMMAND python ${pack_model_exe} --model_path ${models} --out_file ${packed_model}
    DEPENDS ${models}
    VERBATIM)

add_custom_target(human_face_feat_model ALL DEPENDS ${packed_model})

if(CONFIG_HUMAN_FACE_FEAT_MODEL_IN_FLASH_PARTITION)
    add_dependencies(flash human_face_feat_model)
    esptool_py_flash_to_partition(flash "human_face_feat" ${packed_model})
endif()
