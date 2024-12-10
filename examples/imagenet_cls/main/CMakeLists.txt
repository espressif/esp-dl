set(src_dirs        ./)

set(include_dirs    ./)

set(requires        imagenet_cls)

set(embed_files     "cat.jpg")

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})
