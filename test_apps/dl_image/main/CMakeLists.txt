set(src_dirs .)

file(GLOB embed_files  ${PROJECT_DIR}/imgs/*.jpg)

set(requires    unity
                esp-dl)

if (IDF_TARGET STREQUAL "esp32s3")
    list(APPEND requires esp32_s3_eye_noglib
                            esp_lcd)
elseif (IDF_TARGET STREQUAL "esp32p4")
    list(APPEND requires esp32_p4_function_ev_board_noglib
                            esp_lcd)
endif()

idf_component_register(SRC_DIRS ${src_dirs}
                       REQUIRES ${requires}
                       EMBED_FILES ${embed_files}
                       WHOLE_ARCHIVE)
