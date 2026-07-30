# ESP-IDF project boilerplate (order matters).
cmake_minimum_required(VERSION 3.5)

add_compile_options(-fdiagnostics-color=always)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(pp_ocr_v6)
