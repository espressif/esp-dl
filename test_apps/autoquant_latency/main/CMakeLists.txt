set(srcs app_main.cpp)

set(requires esp-dl)

idf_component_register(SRCS ${srcs}
                       REQUIRES ${requires}
                       WHOLE_ARCHIVE)

if(CONFIG_MODEL_FILE_PATH)
    set(model_path ${CONFIG_MODEL_FILE_PATH})

    set(MVMODEL_EXE ${PROJECT_DIR}/../../esp-dl/fbs_loader/pack_espdl_models.py)
    idf_build_get_property(build_dir BUILD_DIR)
    set(image_file ${build_dir}/espdl_models/models.espdl)

    add_custom_command(
        OUTPUT ${image_file}
        COMMENT "Move and Pack models..."
        COMMAND python ${MVMODEL_EXE} --model_path=${model_path} --out_file=${image_file}
        DEPENDS ${CONFIG_MODEL_FILE_PATH}
        VERBATIM)

    add_custom_target(espdl_models ALL DEPENDS ${image_file})
    add_dependencies(flash espdl_models)

    partition_table_get_partition_info(size "--partition-name model" "size")
    partition_table_get_partition_info(offset "--partition-name model" "offset")

    if("${size}" AND "${offset}")
        esptool_py_flash_to_partition(flash "model" "${image_file}")
    else()
        message(WARNING "model partition not found in partition table")
    endif()
else()
    message(STATUS "CONFIG_MODEL_FILE_PATH not set yet; model packing skipped (set-target phase).")
endif()

component_compile_options(-Wno-error=format=-Wno-format)
