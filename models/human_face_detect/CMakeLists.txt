set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl)

set(HUMAN_FACE_DETECT_MODEL ${BUILD_DIR}/espdl_models/human_face_detect.espdl)

set(embed_files     ${HUMAN_FACE_DETECT_MODEL})

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})

set(MVMODEL_EXE ${COMPONENT_DIR}/pack_model.py)

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)

if(IDF_TARGET STREQUAL "esp32s3")
    file(GLOB MODEL_FILE_PATH "${COMPONENT_DIR}/models/*esp32s3.espdl")
elseif (IDF_TARGET STREQUAL "esp32p4")
    file(GLOB MODEL_FILE_PATH "${COMPONENT_DIR}/models/*esp32p4.espdl")
endif()

message(${MODEL_FILE_PATH})

add_custom_command(
    OUTPUT ${HUMAN_FACE_DETECT_MODEL}
    COMMENT "Move and Pack models..."
    COMMAND python ${MVMODEL_EXE} --model_path ${MODEL_FILE_PATH} --out_file ${HUMAN_FACE_DETECT_MODEL}
    DEPENDS ${CONFIG_MODEL_FILE_PATH}
    VERBATIM)
