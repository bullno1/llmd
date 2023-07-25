#ifndef LLMD_IPC_CLIENT_H
#define LLMD_IPC_CLIENT_H

#include <llmd/core.h>

#ifdef LLMD_IPC_CLIENT_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_IPC_CLIENT_BUILD
#            define LLMD_IPC_CLIENT_API __declspec(dllexport)
#        else
#            define LLMD_IPC_CLIENT_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_IPC_CLIENT_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_IPC_CLIENT_API
#endif

struct llmd_ipc_driver_config {
	const char* name;
};

#ifdef __cplusplus
extern "C" {
#endif

LLMD_IPC_CLIENT_API enum llmd_error
llmd_create_ipc_driver(
	struct llmd_host* host,
	struct llmd_ipc_driver_config* config,
	struct llmd_driver** driver_out
);

LLMD_IPC_CLIENT_API enum llmd_error
llmd_destroy_ipc_driver(
	struct llmd_driver* driver
);

#ifdef LLMD_IPC_CLIENT_BUILD

LLMD_IPC_CLIENT_API enum llmd_error
llmd_begin_create_driver(
	struct llmd_host* host,
	struct llmd_ipc_driver_config** config
);

LLMD_IPC_CLIENT_API enum llmd_error
llmd_set_driver_config(
	struct llmd_host* host,
	struct llmd_ipc_driver_config* config,
	const char* section,
	const char* key,
	const char* value
);

LLMD_IPC_CLIENT_API enum llmd_error
llmd_end_create_driver(
	struct llmd_host* host,
	struct llmd_ipc_driver_config* config,
	struct llmd_driver** driver_out
);

LLMD_IPC_CLIENT_API enum llmd_error
llmd_destroy_driver(
	struct llmd_driver* driver
);

#endif

#ifdef __cplusplus
}
#endif

#endif
