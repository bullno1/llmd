#ifndef LLMD_DRIVER_LOADER_H
#define LLMD_DRIVER_LOADER_H

#include <llmd/common.h>

#ifdef LLMD_DRIVER_LOADER_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_DRIVER_LOADER_BUILD
#            define LLMD_DRIVER_LOADER_API __declspec(dllexport)
#        else
#            define LLMD_DRIVER_LOADER_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_DRIVER_LOADER_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_DRIVER_LOADER_API
#endif

struct llmd_driver_config_file {
	const char* file_path;
	const char* file_content;
};

#ifdef __cplusplus
extern "C" {
#endif

LLMD_DRIVER_LOADER_API enum llmd_error
llmd_load_driver(
	struct llmd_host* host,
	const char* driver_path,
	const char* config_file_path,
	const char* config_file_content,
	struct llmd_driver** driver_out
);

LLMD_DRIVER_LOADER_API enum llmd_error
llmd_unload_driver(
	struct llmd_driver* driver
);

#ifdef __cplusplus
}
#endif

#endif
