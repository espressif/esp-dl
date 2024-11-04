set(src_dirs        ./)

set(include_dirs    ./)

set(requires        human_face_detect
                    esp_jpeg)

set(embed_files     "human_face.jpg")

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})
