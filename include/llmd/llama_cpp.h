#ifndef LLMD_LLAMA_CPP_H
#define LLMD_LLAMA_CPP_H

#include <llmd/core.h>
#include <llama.h>

#ifdef LLMD_LLAMA_CPP_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_LLAMA_CPP_BUILD
#            define LLMD_LLAMA_CPP_API __declspec(dllexport)
#        else
#            define LLMD_LLAMA_CPP_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_LLAMA_CPP_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_LLAMA_CPP_API
#endif

struct llmd_llama_cpp_driver_config {
	struct llama_context_params context_params;
	int max_contexts;
};

#ifdef __cplusplus
extern "C" {
#endif

LLMD_LLAMA_CPP_API enum llmd_error
llmd_create_llama_cpp_driver(
	struct llmd_host* host,
	struct llmd_llama_cpp_driver_config config,
	struct llmd_driver** driver_out
);

LLMD_LLAMA_CPP_API enum llmd_error
llmd_destroy_llama_cpp_driver(
	struct llmd_driver* driver
);

#ifdef LLMD_LLAMA_CPP_BUILD

LLMD_LLAMA_CPP_API enum llmd_error
llmd_begin_create_driver(
	struct llmd_host* host,
	struct llmd_llama_cpp_driver_config** config
);

LLMD_LLAMA_CPP_API enum llmd_error
llmd_set_driver_config(
	struct llmd_host* host,
	struct llmd_llama_cpp_driver_config* config,
	const char* section,
	const char* key,
	const char* value
);

LLMD_LLAMA_CPP_API enum llmd_error
llmd_end_create_driver(
	struct llmd_host* host,
	struct llmd_llama_cpp_driver_config* config,
	struct llmd_driver** driver_out
);

LLMD_LLAMA_CPP_API enum llmd_error
llmd_destroy_driver(
	struct llmd_driver* driver
);

#endif

#ifdef __cplusplus
}
#endif

#endif
