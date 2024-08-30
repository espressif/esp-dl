# set(srcs

# test_xxx.cpp
# ) # for debug

set(src_dirs

.
)

set(include_dirs

.
)

set(requires

unity
dl
)

idf_component_register(
    SRCS ${srcs} 
    SRC_DIRS ${src_dirs} 
    INCLUDE_DIRS ${include_dirs} 
    REQUIRES ${requires}
)

