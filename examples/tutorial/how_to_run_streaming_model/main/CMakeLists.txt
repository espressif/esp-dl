set(srcs    app_main.cpp)

set(requires    esp-dl)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()

set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# if (IDF_TARGET STREQUAL "esp32s3")
#     set(embed_files models/s3/model.espdl)
# elseif (IDF_TARGET STREQUAL "esp32p4")
#     set(embed_files models/p4/model.espdl)
# endif()

set(packed_model ${BUILD_DIR}/espdl_models/streaming_models.espdl)

idf_component_register(SRCS ${srcs} REQUIRES ${requires})

if (IDF_TARGET STREQUAL "esp32s3")
    set(models_dir ${PROJECT_DIR}/main/models/s3)
elseif (IDF_TARGET STREQUAL "esp32p4")
    set(models_dir ${PROJECT_DIR}/main/models/p4)
endif()

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
file(GLOB_RECURSE models "${models_dir}/*.espdl")

set(pack_model_exe ${espdl_dir}/fbs_loader/pack_espdl_models.py)
add_custom_command(
    OUTPUT ${packed_model}
    COMMENT "Move and Pack models..."
    COMMAND python ${pack_model_exe} --model_path ${models} --out_file ${packed_model}
    DEPENDS ${models}
    VERBATIM)

target_add_aligned_binary_data(${COMPONENT_LIB} ${packed_model} BINARY)
