#include <stddef.h>
#include <llmd/loader.h>
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
    const char* driver_path = NULL;
    const char* config_path = NULL;

	struct config config = {
		.num_driver_configs = 0,
	};
	struct llmd_ipc_server_config server_config = {
		.name = "llmd-ipc",
	};

	struct argparse_option options[] = {
		OPT_HELP(),
		{
			.type = ARGPARSE_OPT_STRING,
			.short_name = 'd',
			.long_name = "driver",
			.value = &driver_path,
			.help = "Path to driver",
		},
		{
			.type = ARGPARSE_OPT_STRING,
			.short_name = 'c',
			.long_name = "config",
			.value = &config_path,
			.help = "Path to config file",
		},
		{
			.type = ARGPARSE_OPT_STRING,
			.short_name = 's',
			.long_name = "set",
			.help = "Set driver config directly. For example: --set=main.model_path=custom_path",
			.callback = parse_driver_config,
			.value = &config.tmp_string,
			.data = (intptr_t)(void*)&config,
		},
		{
			.type = ARGPARSE_OPT_STRING,
			.short_name = 'n',
			.long_name = "name",
			.value = &server_config.name,
			.help = "Name of server",
		},
		OPT_END()
	};
	struct argparse argparse;
	argparse_init(&argparse, options, NULL, ARGPARSE_STOP_AT_NON_OPTION);
	argparse_describe(&argparse, "Run an IPC server", NULL);
	argparse_parse(&argparse, argc, argv);

	if (driver_path == NULL) {
		fprintf(stderr, "Driver path is missing.\n\n");
		argparse_usage(&argparse);
		return 1;
	}

	enum llmd_error status = LLMD_OK;
	struct llmd_driver_loader* loader = NULL;
	struct llmd_driver* driver = NULL;

	LLMD_CHECK(llmd_begin_load_driver(NULL, driver_path, &loader));

	if (config_path != NULL) {
		LLMD_CHECK(llmd_load_driver_config_from_file(loader, config_path));
	}

	for (unsigned int i = 0; i < config.num_driver_configs; ++i) {
		const struct driver_config_skv* driver_config = &config.driver_configs[i];
		status = llmd_config_driver(
			loader,
			driver_config->section, driver_config->key, driver_config->value
		);

		if (status != LLMD_OK) {
			fprintf(
				stderr,
				"Driver rejects config %s.%s=%s\n",
				driver_config->section, driver_config->key, driver_config->value
			);
			goto end;
		}
	}
	LLMD_CHECK(llmd_end_load_driver(loader, &driver));

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

	for (unsigned int i = 0; i < config.num_driver_configs; ++i) {
		const struct driver_config_skv* driver_config = &config.driver_configs[i];
		free((void*)driver_config->section);
		free((void*)driver_config->key);
		free((void*)driver_config->value);
	}

	return status == LLMD_OK ? 0 : 1;
}
