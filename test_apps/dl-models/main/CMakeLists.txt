set(srcs app_main.cpp
         test_dl_model.cpp
         test_dl_op.cpp)

set(requires    unity
                esp-dl)

idf_component_register(SRCS ${srcs}
                       REQUIRES ${requires}
                       WHOLE_ARCHIVE)

if(CONFIG_MODEL_FILE_PATH)
    message("CONFIG_MODEL_FILE_PATH is defined")
endif()
message("CONFIG_MODEL_FILE_PATH: ${CONFIG_MODEL_FILE_PATH}")
message("PROJECT_DIR: ${PROJECT_DIR}")
# set(models_file ${PROJECT_DIR}/models/esp_model.bin)
partition_table_get_partition_info(size "--partition-name model" "size")
message("partition size: ${size}")
# esptool_py_flash_to_partition(flash "model" "${CONFIG_MODEL_FILE_PATH}")

if("${size}")
    esptool_py_flash_to_partition(flash "model" "${CONFIG_MODEL_FILE_PATH}")
endif()
