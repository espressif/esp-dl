# SPDX-License-Identifier: Apache-2.0
#
# Registration API for esp-dl compile-time op/kernel stripping.
# Call from model-component CMakeLists.txt AFTER idf_component_register().
#
#   idf_build_get_property(build_components BUILD_COMPONENTS)
#   if("espressif__esp-dl" IN_LIST build_components)
#       idf_component_get_property(_d espressif__esp-dl COMPONENT_DIR)
#   elseif("esp-dl" IN_LIST build_components)
#       idf_component_get_property(_d esp-dl COMPONENT_DIR)
#   endif()
#   include(${_d}/cmake/compile_register.cmake)
#   esp_dl_compile_add_yml(.../model.compile_req.yml)
#
# Always probe BUILD_COMPONENTS before idf_component_get_property; a missing
# component name makes get_property fail hard.
#
# Paths are stored in ESPDL_COMPILE_REQS and packed in esp_dl_compile_finalize()
# (deferred to the end of the project, after every component CMakeLists runs).

include_guard(GLOBAL)

function(esp_dl_compile_add_yml yml)
    get_filename_component(yml "${yml}" ABSOLUTE)
    if(NOT EXISTS "${yml}")
        message(FATAL_ERROR "esp_dl_compile_add_yml: file not found: ${yml}")
    endif()
    idf_build_set_property(ESPDL_COMPILE_REQS "${yml}" APPEND)
    message(STATUS "esp-dl: compile_add_yml '${yml}'")
endfunction()

function(esp_dl_compile_add_ymls)
    foreach(yml ${ARGN})
        esp_dl_compile_add_yml("${yml}")
    endforeach()
endfunction()
