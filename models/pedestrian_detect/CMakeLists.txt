set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl)

set(PEDESTRIAN_DETECT_MODEL ${BUILD_DIR}/espdl_models/pedestrian_detect.espdl)

message(${PEDESTRIAN_DETECT_MODEL})

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
if(IDF_TARGET STREQUAL "esp32s3")
    file(COPY ${COMPONENT_DIR}/models/pedestrian_s3.espdl DESTINATION ${BUILD_DIR}/espdl_models)
    file(RENAME ${BUILD_DIR}/espdl_models/pedestrian_s3.espdl ${PEDESTRIAN_DETECT_MODEL})
elseif (IDF_TARGET STREQUAL "esp32p4")
    file(COPY ${COMPONENT_DIR}/models/pedestrian_p4.espdl DESTINATION ${BUILD_DIR}/espdl_models)
    file(RENAME ${BUILD_DIR}/espdl_models/pedestrian_p4.espdl ${PEDESTRIAN_DETECT_MODEL})
endif()

set(embed_files     ${PEDESTRIAN_DETECT_MODEL})

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})
