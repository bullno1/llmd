#include <stddef.h>
#include <llmd/core.h>
#include <llmd/ipc/server.h>
#include <argparse.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include "common.h"

struct llmd_ipc_server* server = NULL;

static void
stop_server(int signum) {
	(void)signum;
	fprintf(stderr, "Stopping server\n");
	if (server != NULL) {
		llmd_stop_ipc_server(server);
	}
}

int
main(int argc, const char* argv[]) {
	struct driver_config driver_config = { 0 };
	struct llmd_ipc_server_config server_config = {
		.name = "llmd-ipc",
	};

	struct argparse_option options[] = {
		COMMON_OPTIONS,
		OPT_END()
	};
	struct argparse argparse;
	argparse_init(&argparse, options, NULL, ARGPARSE_STOP_AT_NON_OPTION);
	argparse_describe(&argparse, "Run an IPC server", NULL);
	argparse_parse(&argparse, argc, argv);

	enum llmd_error status = LLMD_OK;
	struct llmd_driver_loader* loader = NULL;
	struct llmd_driver* driver = NULL;

	LLMD_CHECK(load_driver(&driver_config, &argparse, &loader, &driver));
	LLMD_CHECK(llmd_create_ipc_server(NULL, &server_config, driver, &server));

	signal(SIGINT, stop_server);
	signal(SIGTERM, stop_server);
	LLMD_CHECK(llmd_start_ipc_server(server));
end:
	if (server != NULL) {
		llmd_destroy_ipc_server(server);
	}

	if (loader != NULL) {
		llmd_unload_driver(loader);
	}

	cleanup_driver_config(&driver_config);

	return status == LLMD_OK ? 0 : 1;
}
