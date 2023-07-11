#ifndef LLMD_SERVER_H
#define LLMD_SERVER_H

#ifdef LLMD_SERVER_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_SERVER_BUILD
#            define LLMD_SERVER_API __declspec(dllexport)
#        else
#            define LLMD_SERVER_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_SERVER_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_SERVER_API
#endif

struct llmd_host;
struct llmd_session;
struct llmd_server;

struct llmd_server_config {
	const char* endpoint_name;
	int max_num_clients;
	int required_uid;
	int required_gid;
};

#ifdef __cplusplus
extern "C" {
#endif

LLMD_SERVER_API enum llmd_error
llmd_create_server(
	struct llmd_host* host,
	struct llmd_session* session,
	struct llmd_server** server
);

LLMD_SERVER_API enum llmd_error
llmd_start_server(
	struct llmd_server* server
);

LLMD_SERVER_API enum llmd_error
llmd_stop_server(
	struct llmd_server* server
);

LLMD_SERVER_API enum llmd_error
llmd_destroy_server(
	struct llmd_server* server
);

#ifdef __cplusplus
}
#endif

#endif
