#ifndef LLMD_IPC_SERVER_H
#define LLMD_IPC_SERVER_H

#include <llmd/core.h>

#ifdef LLMD_IPC_SERVER_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_IPC_SERVER_BUILD
#            define LLMD_IPC_SERVER_API __declspec(dllexport)
#        else
#            define LLMD_IPC_SERVER_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_IPC_SERVER_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_IPC_SERVER_API
#endif

struct llmd_ipc_server_config {
	const char* name;
};

struct llmd_ipc_server;

#ifdef __cplusplus
extern "C" {
#endif

LLMD_IPC_SERVER_API enum llmd_error
llmd_create_ipc_server(
	struct llmd_host* host,
	struct llmd_ipc_server_config* config,
	struct llmd_driver* driver,
	struct llmd_ipc_server** server_out
);

LLMD_IPC_SERVER_API enum llmd_error
llmd_start_ipc_server(
	struct llmd_ipc_server* server
);

LLMD_IPC_SERVER_API enum llmd_error
llmd_stop_ipc_server(
	struct llmd_ipc_server* server
);

LLMD_IPC_SERVER_API enum llmd_error
llmd_destroy_ipc_server(
	struct llmd_ipc_server* server
);

#ifdef __cplusplus
}
#endif

#endif
