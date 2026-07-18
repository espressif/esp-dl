# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)

add_compile_options(-fdiagnostics-color=always)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(speaker_verification)

# esp_video uses the legacy ISP sample-point names to remain compatible with
# ESP-IDF 5.3. They are aliases of the new names, but are deprecated in IDF 6.
# The component is optional and is not included in every target configuration.
if(IDF_VERSION_MAJOR GREATER_EQUAL 6 AND TARGET __idf_espressif__esp_video)
    target_compile_options(__idf_espressif__esp_video PRIVATE -Wno-deprecated-declarations)
endif()
