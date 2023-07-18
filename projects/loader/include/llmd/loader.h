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

struct llmd_driver_loader;

#ifdef __cplusplus
extern "C" {
#endif

LLMD_LOADER_API enum llmd_error
llmd_begin_load_driver(
	struct llmd_host* host,
	const char* driver_path,
	struct llmd_driver_loader** loader_out
);

LLMD_LOADER_API enum llmd_error
llmd_config_driver(
	struct llmd_driver_loader* loader,
	const char* section,
	const char* key,
	const char* value
);

LLMD_LOADER_API enum llmd_error
llmd_load_driver_config_from_file(
	struct llmd_driver_loader* loader,
	const char* ini_file
);

LLMD_LOADER_API enum llmd_error
llmd_load_driver_config_from_string(
	struct llmd_driver_loader* loader,
	const char* ini_string
);

LLMD_LOADER_API enum llmd_error
llmd_end_load_driver(
	struct llmd_driver_loader* loader,
	struct llmd_driver** driver_out
);

LLMD_LOADER_API enum llmd_error
llmd_unload_driver(
	struct llmd_driver_loader* loader
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
