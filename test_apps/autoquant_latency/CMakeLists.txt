# AutoQuant latency-only firmware (no Unity / no IDF test_apps extras). See main/app_main.cpp.
cmake_minimum_required(VERSION 3.5)

add_compile_options(-fdiagnostics-color=always)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Only the esp-dl component; do not add $IDF_PATH/tools/test_apps/components (pulls test_utils, enlarges ELF).
set(EXTRA_COMPONENT_DIRS "../../esp-dl")

project(autoquant_latency)
