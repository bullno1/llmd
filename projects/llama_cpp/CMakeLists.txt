set(SOURCES "src/llama_cpp.c")

setup_library(llmd_llama_cpp LLMD_LLAMA_CPP_STATIC ${SOURCES})

target_include_directories(llmd_llama_cpp PUBLIC "./include")
target_link_libraries(llmd_llama_cpp PUBLIC llmd_core_interface)
target_link_libraries(llmd_llama_cpp PRIVATE llmd_utils llama)

if (NOT MSVC)
	target_compile_options(llmd_llama_cpp PRIVATE
		# llama.h contains this
		-Wno-strict-prototypes
	)
endif ()
