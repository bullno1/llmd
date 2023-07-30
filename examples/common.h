#ifndef LLMD_EXAMPLE_COMMON_H
#define LLMD_EXAMPLE_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <argparse.h>
#include <llmd/loader.h>

#define MAX_DRIVER_CONFIG_ENTRIES 16

#define LLMD_CHECK(op) \
	if ((status = (op)) != LLMD_OK) { \
		fprintf(stderr, "%s returns %d (%s)", #op, status, llmd_error_to_str(status));\
		goto end; \
	}

struct ini_entry {
	const char* section;
	const char* key;
	const char* value;
};

struct driver_config {
	const char* driver_path;
	const char* config_file_path;

	unsigned int num_config_entries;
	struct ini_entry config_entries[MAX_DRIVER_CONFIG_ENTRIES];

	const char* tmp_string;
};

static inline int
parse_driver_config(
	struct argparse* argparse,
	const struct argparse_option* option
) {
	(void)argparse;

	struct driver_config* config = (void*)option->data;
	if (config->num_config_entries >= MAX_DRIVER_CONFIG_ENTRIES) {
		fprintf(stderr, "Too many driver config entries.\n");
		return -2;
	}

	char* dot_pos = strchr(config->tmp_string, '.');
	char* eq_pos = strchr(config->tmp_string, '=');
	if (
		dot_pos == NULL
		|| eq_pos == NULL
		|| eq_pos < dot_pos
	) {
		fprintf(stderr, "Invalid config string %s\n", config->tmp_string);
		return -1;
	}

	struct ini_entry* entry = &config->config_entries[config->num_config_entries++];
	entry->section = strndup(config->tmp_string, dot_pos - config->tmp_string);
	entry->key = strndup(dot_pos + 1, eq_pos - dot_pos - 1);
	entry->value = strndup(eq_pos + 1, strlen(config->tmp_string) - (eq_pos - config->tmp_string));

	return 0;
}

static inline enum llmd_error
load_driver(
	struct driver_config* driver_config,
	struct argparse* argparse,
	struct llmd_driver_loader** loader,
	struct llmd_driver** driver
) {
	if (driver_config->driver_path == NULL) {
		fprintf(stderr, "Driver path is missing.\n\n");
		argparse_usage(argparse);
		return LLMD_ERR_INVALID;
	}

	enum llmd_error status;

	LLMD_CHECK(llmd_begin_load_driver(NULL, driver_config->driver_path, loader));

	if (driver_config->config_file_path != NULL) {
		LLMD_CHECK(llmd_load_driver_config_from_file(*loader, driver_config->config_file_path));
	}

	for (unsigned int i = 0; i < driver_config->num_config_entries; ++i) {
		const struct ini_entry* config_entry = &driver_config->config_entries[i];
		status = llmd_config_driver(
			*loader,
			config_entry->section, config_entry->key, config_entry->value
		);

		if (status != LLMD_OK) {
			fprintf(
				stderr,
				"Driver rejects config %s.%s=%s\n",
				config_entry->section, config_entry->key, config_entry->value
			);
			goto end;
		}
	}

	LLMD_CHECK(llmd_end_load_driver(*loader, driver));
end:
	return status;
}

static inline void
cleanup_driver_config(struct driver_config* driver_config) {
	for (unsigned int i = 0; i < driver_config->num_config_entries; ++i) {
		const struct ini_entry* entry = &driver_config->config_entries[i];
		free((void*)entry->section);
		free((void*)entry->key);
		free((void*)entry->value);
	}
}

#endif
