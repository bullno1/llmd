#ifndef LLMD_LOADER_H
#define LLMD_LOADER_H

#include <llmd/core.h>

#ifdef LLMD_LOADER_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_LOADER_BUILD
#            define LLMD_LOADER_API __declspec(dllexport)
#        else
#            define LLMD_LOADER_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_LOADER_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_LOADER_API
#endif

struct llmd_loader_handle {
	struct llmd_driver* driver;
	void* internal;
};

#ifdef __cplusplus
extern "C" {
#endif

LLMD_LOADER_API enum llmd_error
llmd_load_driver(
	struct llmd_host* host,
	const char* driver_path,
	const char* config_file_path,
	const char* config_file_content,
	struct llmd_loader_handle* loader_handle
);

LLMD_LOADER_API enum llmd_error
llmd_unload_driver(
	struct llmd_loader_handle* loader_handle
);

#ifdef __cplusplus
}
#endif

#ifdef LLMD_LOADER_INTERNAL

// Loader contract

typedef enum llmd_error
(*llmd_begin_create_driver_t)(struct llmd_host* host, void** config_out);

typedef enum llmd_error
(*llmd_set_driver_config_t)(
	struct llmd_host* host,
	void* config,
	const char* section,
	const char* key,
	const char* value
);

typedef enum llmd_error
(*llmd_end_create_driver_t)(
	struct llmd_host* host,
	void* config,
	struct llmd_driver** driver_out
);

typedef enum llmd_error
(*llmd_destroy_driver_t)(struct llmd_driver* driver);

#endif

#endif
