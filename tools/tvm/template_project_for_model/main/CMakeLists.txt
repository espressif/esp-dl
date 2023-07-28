set(srcs            app_main.c)

set(include_dirs    .)

set(requires    esp_timer
                esp-dl
                tvm_model)

idf_component_register(SRCS ${srcs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})
