# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)

add_compile_options(-fdiagnostics-color=always)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

if(IDF_VERSION_MAJOR GREATER_EQUAL 6)
    set(EXTRA_COMPONENT_DIRS "$ENV{IDF_PATH}/tools/test_apps/components"
                            "../../esp-dl")
else()
    set(EXTRA_COMPONENT_DIRS "$ENV{IDF_PATH}/tools/unit-test-app/components"
                            "../../esp-dl")
endif()

project(dl_ops_test)
