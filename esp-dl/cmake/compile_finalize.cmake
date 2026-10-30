# SPDX-License-Identifier: Apache-2.0
#
# Called at end of project configure (via cmake_language DEFER), after every
# component CMakeLists has run esp_dl_compile_add_*.

# idf_component_get_property errors if the component is absent; probe BUILD_COMPONENTS first.
function(_esp_dl_compile_comp_prop out_var prop)
    idf_build_get_property(build_components BUILD_COMPONENTS)
    set(_val "")
    foreach(_name IN LISTS ARGN)
        if("${_name}" IN_LIST build_components)
            idf_component_get_property(_val ${_name} ${prop})
            break()
        endif()
    endforeach()
    set(${out_var} "${_val}" PARENT_SCOPE)
endfunction()

function(esp_dl_compile_finalize)
    idf_build_get_property(_done ESPDL_COMPILE_CONFIG_DONE)
    if(_done)
        return()
    endif()
    idf_build_set_property(ESPDL_COMPILE_CONFIG_DONE 1)

    _esp_dl_compile_comp_prop(espdl_dir COMPONENT_DIR espressif__esp-dl esp-dl)
    _esp_dl_compile_comp_prop(espdl_lib COMPONENT_LIB espressif__esp-dl esp-dl)
    if(NOT espdl_dir OR NOT espdl_lib)
        message(FATAL_ERROR "esp-dl: component dir/lib not found")
    endif()
    get_target_property(espdl_bin "${espdl_lib}" BINARY_DIR)
    if(NOT espdl_bin)
        message(FATAL_ERROR "esp-dl: no BINARY_DIR for ${espdl_lib}")
    endif()

    idf_build_get_property(python PYTHON)
    idf_build_get_property(reqs ESPDL_COMPILE_REQS)
    idf_build_get_property(target IDF_TARGET)

    set(_ops_yml "${espdl_dir}/spec/ops.yml")
    set(_kernels_yml "${espdl_dir}/spec/kernels.yml")
    set(_conv_yml "${espdl_dir}/spec/select/Conv.yml")
    set(_header "${espdl_bin}/dl_compile_config.h")
    set(_srcs_cmake "${espdl_bin}/dl_compile_srcs.cmake")
    set(_kernel_inc "${espdl_bin}/dl_kernel.inc")
    set(_conv_sel "${espdl_bin}/dl_conv_select.inc")
    set(_module_inc "${espdl_bin}/dl_module_includes.inc")
    set(_module_reg "${espdl_bin}/dl_module_register.inc")
    set(_script "${espdl_dir}/cmake/compile_finalize.py")

    set(_cmd "${python}" "${_script}"
        --ops "${_ops_yml}"
        --kernels "${_kernels_yml}"
        --conv-yml "${_conv_yml}"
        --target "${target}"
        --component-dir "${espdl_dir}"
        --header "${_header}"
        --srcs "${_srcs_cmake}"
        --kernel-inc "${_kernel_inc}"
        --conv-select "${_conv_sel}"
        --module-includes-inc "${_module_inc}"
        --module-register-inc "${_module_reg}")
    if(reqs)
        list(APPEND _cmd --req ${reqs})
    endif()

    execute_process(
        COMMAND ${_cmd}
        RESULT_VARIABLE _rv
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    if(NOT _rv EQUAL 0)
        message(FATAL_ERROR
            "compile_finalize.py failed (${_rv})\n${_out}${_err}")
    endif()

    include("${_srcs_cmake}")
    set(_abs_srcs)
    foreach(_src ${ESPDL_COMPILE_SRCS})
        list(APPEND _abs_srcs "${espdl_dir}/${_src}")
    endforeach()
    if(_abs_srcs)
        target_sources("${espdl_lib}" PRIVATE ${_abs_srcs})
    endif()
    # Lookup table for generated dl_kernel.inc. Not an op source, so it stays
    # out of ops.yml. Macro-only ISA .S files are #included by kernel .S files.
    target_sources("${espdl_lib}" PRIVATE "${espdl_dir}/dl/base/dl_kernel.cpp")
    target_include_directories("${espdl_lib}" BEFORE PRIVATE "${espdl_bin}")

    set_property(DIRECTORY "${CMAKE_SOURCE_DIR}" APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_ops_yml}" "${_kernels_yml}" "${_conv_yml}" "${_script}")
    if(reqs)
        set_property(DIRECTORY "${CMAKE_SOURCE_DIR}" APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${reqs})
    endif()
endfunction()
