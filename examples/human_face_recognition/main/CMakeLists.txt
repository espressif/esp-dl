set(src_dirs        ./)

set(include_dirs    ./)

set(requires        human_face_recognition
                    esp_jpeg)

file(GLOB embed_files "${PROJECT_DIR}/faces/*.jpg")

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})
