set(srcs app_main.cpp
         test_dl_model.cpp
         test_dl_api.cpp)

set(requires    unity
                esp-dl)

idf_component_register(SRCS ${srcs}
                       REQUIRES ${requires}
                       WHOLE_ARCHIVE)


if(CONFIG_MODEL_FILE_PATH)
    set(model_path ${CONFIG_MODEL_FILE_PATH})
else()
    set(model_path "${PROJECT_DIR}/models/conv2d")
endif()

message("PROJECT_DIR: ${PROJECT_DIR}")

set(MVMODEL_EXE ${PROJECT_DIR}/../../tools/pack_models/pack_espdl_models.py)
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
    set(message "Failed to find model in partition table file"
                "Please add a line(Name=model, Size>recommended size in log) to the partition file.")
endif()