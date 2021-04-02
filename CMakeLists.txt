idf_build_get_property(target IDF_TARGET)

set(lib     libcat_face_detector.a
            libhuman_face_detector.a
            libdl.a)

if(${IDF_TARGET} STREQUAL "esp32s3")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/lib")
endif()

set(COMPONENT_ADD_INCLUDEDIRS   include/dl
                                include/model)

register_component()

target_link_libraries(${COMPONENT_TARGET}   INTERFACE   ${links})
target_link_libraries(${COMPONENT_TARGET}   INTERFACE   ${lib})

