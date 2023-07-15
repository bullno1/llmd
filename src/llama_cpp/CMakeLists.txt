add_library(llmd_llama_cpp "llama_cpp.c")

target_include_directories(llmd_llama_cpp PUBLIC ${LLMD_INCLUDE_DIR})
target_link_libraries(llmd_llama_cpp PRIVATE llmd_utils llama)
if (NOT MSVC)
	target_compile_options(llmd_llama_cpp PRIVATE
		# llama.h contains this
		-Wno-strict-prototypes)
endif ()
