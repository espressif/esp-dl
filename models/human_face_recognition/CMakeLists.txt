set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl
                    human_face_detect
                    fatfs
                    vfs
                    spiffs)

set(HUMAN_FACE_FEAT_MODEL ${BUILD_DIR}/espdl_models/human_face_feat.espdl)
file(REMOVE ${BUILD_DIR}/human_face_feat.espdl.S)

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
if(IDF_TARGET STREQUAL "esp32s3")
    if (CONFIG_HUMAN_FACE_FEAT_MFN_S8_V1)
        file(COPY ${COMPONENT_DIR}/models/human_face_feat_mfn_s8_s3_v1.espdl DESTINATION ${BUILD_DIR}/espdl_models)
        file(RENAME ${BUILD_DIR}/espdl_models/human_face_feat_mfn_s8_s3_v1.espdl ${HUMAN_FACE_FEAT_MODEL})
    elseif (CONFIG_HUMAN_FACE_FEAT_MBF_S8_V1)
        file(COPY ${COMPONENT_DIR}/models/human_face_feat_mbf_s8_s3_v1.espdl DESTINATION ${BUILD_DIR}/espdl_models)
        file(RENAME ${BUILD_DIR}/espdl_models/human_face_feat_mbf_s8_s3_v1.espdl ${HUMAN_FACE_FEAT_MODEL})
    endif()
    list(APPEND requires esp32_s3_eye)
elseif (IDF_TARGET STREQUAL "esp32p4")
    if (CONFIG_HUMAN_FACE_FEAT_MFN_S8_V1)
        file(COPY ${COMPONENT_DIR}/models/human_face_feat_mfn_s8_p4_v1.espdl DESTINATION ${BUILD_DIR}/espdl_models)
        file(RENAME ${BUILD_DIR}/espdl_models/human_face_feat_mfn_s8_p4_v1.espdl ${HUMAN_FACE_FEAT_MODEL})
    elseif (CONFIG_HUMAN_FACE_FEAT_MBF_S8_V1)
        file(COPY ${COMPONENT_DIR}/models/human_face_feat_mbf_s8_p4_v1.espdl DESTINATION ${BUILD_DIR}/espdl_models)
        file(RENAME ${BUILD_DIR}/espdl_models/human_face_feat_mbf_s8_p4_v1.espdl ${HUMAN_FACE_FEAT_MODEL})
    endif()
    list(APPEND requires esp32_p4_function_ev_board)
endif()

set(embed_files     ${HUMAN_FACE_FEAT_MODEL})

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})
