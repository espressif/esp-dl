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
set(requires        esp-dl)

set(packed_model ${BUILD_DIR}/espdl_models/pp_ocr_v6.espdl)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

if (NOT CONFIG_PP_OCR_V6_MODEL_IN_SDCARD)
    if (IDF_TARGET STREQUAL "esp32p4" OR IDF_TARGET STREQUAL "esp32s31")
        set(models_dir ${COMPONENT_DIR}/models/p4)
    endif()

    file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
    # Detection is required by PPOCRV6 and has no runtime model choice, so it
    # must always be present in every non-SD-card package.
    set(models ${models_dir}/pp_ocr_v6_det_s8.espdl)
    if(CONFIG_FLASH_PP_OCR_V6_REC_S8)
        list(APPEND models ${models_dir}/pp_ocr_v6_rec_s8.espdl)
    endif()
    if(CONFIG_FLASH_PP_OCR_V6_REC_S16)
        list(APPEND models ${models_dir}/pp_ocr_v6_rec_s16.espdl)
    endif()
    if(CONFIG_FLASH_PP_OCR_V6_REC_S16_W640)
        list(APPEND models ${models_dir}/pp_ocr_v6_rec_s16_w640.espdl)
    endif()

    set(pack_model_exe ${espdl_dir}/fbs_loader/pack_espdl_models.py)
    add_custom_command(
        OUTPUT ${packed_model}
        COMMENT "Pack PP-OCRv6 models..."
        COMMAND python ${pack_model_exe} --model_path ${models} --out_file ${packed_model}
        DEPENDS ${models}
        VERBATIM)

    if(CONFIG_PP_OCR_V6_MODEL_IN_FLASH_RODATA)
        target_add_aligned_binary_data(${COMPONENT_LIB} ${packed_model} BINARY)
    endif()

    if(CONFIG_PP_OCR_V6_MODEL_IN_FLASH_PARTITION)
        add_custom_target(pp_ocr_v6_model ALL DEPENDS ${packed_model})
        add_dependencies(flash pp_ocr_v6_model)
        esptool_py_flash_to_partition(flash "pp_ocr_v6" ${packed_model})
    endif()
endif()
