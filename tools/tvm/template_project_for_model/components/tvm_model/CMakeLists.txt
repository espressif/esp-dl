idf_build_get_property(target IDF_TARGET)



set(srcs_dir        
                    ./model/
                    ./model/codegen/host/src
                    )

set(include_dirs    
                    ./crt_config
                    ./model
                    ./model/codegen/host/include
                    ./model/runtime/include
                    )

set(requires esp-dl)

idf_component_register(SRC_DIRS ${srcs_dir} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

component_compile_options(-Os -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-but-set-parameter)
