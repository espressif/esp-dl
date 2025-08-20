set(srcs app_main.cpp
        test_dl_audio.cpp)

file(GLOB embed_files  ${PROJECT_DIR}/test_cases/test*)

set(requires    unity
                esp-dl)


idf_component_register(SRCS ${srcs}
                       REQUIRES ${requires}
                       EMBED_FILES ${embed_files}
                       WHOLE_ARCHIVE)
