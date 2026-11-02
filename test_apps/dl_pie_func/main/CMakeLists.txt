	set(srcs app_main.cpp
	         test_dotprod.cpp
	         test_lut.cpp
	         test_reduce_l2.cpp
	         test_reduce_l1.cpp
	         test_reduce_max_min.cpp
	         test_reduce_abs_max.cpp
	         test_reduce_sum.cpp
	         test_attention.cpp
	         test_attention_kernels.cpp)

file(GLOB embed_files  ${PROJECT_DIR}/test_cases/test*)

set(requires    unity
                esp-dl
                esp-dsp
                esp_timer)


idf_component_register(SRCS ${srcs}
                       REQUIRES ${requires}
                       EMBED_FILES ${embed_files}
                       WHOLE_ARCHIVE)

# Match the production flags used by the header-only Attention softmax helpers.
set_source_files_properties(test_attention.cpp PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math")
