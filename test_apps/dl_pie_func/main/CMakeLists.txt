set(srcs app_main.cpp
         test_dotprod.cpp
         test_reduce_l2.cpp)

file(GLOB embed_files  ${PROJECT_DIR}/test_cases/test*)

set(requires    unity
                esp-dl
                esp-dsp
                esp_timer)


idf_component_register(SRCS ${srcs}
                       REQUIRES ${requires}
                       EMBED_FILES ${embed_files}
                       WHOLE_ARCHIVE)
