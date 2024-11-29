set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl
                    esp_jpeg)

set(IMAGENET_CLS_MODEL ${BUILD_DIR}/espdl_models/imagenet_cls.espdl)

message(${IMAGENET_CLS_MODEL})

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
file(COPY ${COMPONENT_DIR}/models/mobilenet_v2.espdl DESTINATION ${BUILD_DIR}/espdl_models)
file(RENAME ${BUILD_DIR}/espdl_models/mobilenet_v2.espdl ${IMAGENET_CLS_MODEL})

set(embed_files     ${IMAGENET_CLS_MODEL})

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})
