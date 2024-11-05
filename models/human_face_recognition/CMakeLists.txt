set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl
                    human_face_detect
                    fatfs
                    vfs
                    spiffs)

set(FACE_RECOGNITION_MODEL ${BUILD_DIR}/espdl_models/mfn.espdl)

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
if(IDF_TARGET STREQUAL "esp32s3")
    file(COPY ${COMPONENT_DIR}/models/mfn_s3.espdl DESTINATION ${BUILD_DIR}/espdl_models)
    file(RENAME ${BUILD_DIR}/espdl_models/mfn_s3.espdl ${FACE_RECOGNITION_MODEL})
    list(APPEND requires esp32_s3_eye)
elseif (IDF_TARGET STREQUAL "esp32p4")
    file(COPY ${COMPONENT_DIR}/models/human_face_recognition_mfn_s8_p4_v1.espdl DESTINATION ${BUILD_DIR}/espdl_models)
    file(RENAME ${BUILD_DIR}/espdl_models/human_face_recognition_mfn_s8_p4_v1.espdl ${FACE_RECOGNITION_MODEL})
    if (CONFIG_HUMAN_FACE_RECOGNITION_MODEL EQUAL 0)
        file(COPY ${COMPONENT_DIR}/models/human_face_recognition_mfn_s8_p4_v1.espdl DESTINATION ${BUILD_DIR}/espdl_models)
        file(RENAME ${BUILD_DIR}/espdl_models/human_face_recognition_mfn_s8_p4_v1.espdl ${FACE_RECOGNITION_MODEL})
    elseif (CONFIG_HUMAN_FACE_RECOGNITION_MODEL EQUAL 1)
        file(COPY ${COMPONENT_DIR}/models/human_face_recognition_mbf_s8_p4_v1.espdl DESTINATION ${BUILD_DIR}/espdl_models)
        file(RENAME ${BUILD_DIR}/espdl_models/human_face_recognition_mbf_s8_p4_v1.espdl ${FACE_RECOGNITION_MODEL})
    endif()
    list(APPEND requires esp32_p4_function_ev_board)
endif()

set(embed_files     ${FACE_RECOGNITION_MODEL})

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})
