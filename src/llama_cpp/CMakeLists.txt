set(SOURCES "llama_cpp.c")

if (LLMD_LLAMA_CPP_STATIC)
	add_library(llmd_llama_cpp STATIC ${SOURCES})
else ()
	add_library(llmd_llama_cpp SHARED ${SOURCES})
	target_compile_definitions(llmd_llama_cpp PUBLIC LLMD_LLAMA_CPP_SHARED)
	target_compile_definitions(llmd_llama_cpp PRIVATE LLMD_LLAMA_CPP_BUILD)
	set_target_properties(llmd_llama_cpp PROPERTIES C_VISIBILITY_PRESET hidden)
endif ()

target_include_directories(llmd_llama_cpp PUBLIC ${LLMD_INCLUDE_DIR})
target_link_libraries(llmd_llama_cpp PRIVATE llmd_utils llama)

if (NOT MSVC)
	target_compile_options(llmd_llama_cpp PRIVATE
		# llama.h contains this
		-Wno-strict-prototypes
	)
	target_link_options(llmd_llama_cpp PRIVATE
		-Wl,-ffunction-sections
		-Wl,-fdata-sections
		-Wl,--gc-sections
		-Wl,--exclude-libs,ALL
		-Wl,--no-whole-archive
	)
endif ()
