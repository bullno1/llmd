#ifndef LLMD_EXAMPLE_COMMON_H
#define LLMD_EXAMPLE_COMMON_H

#include <stdio.h>
#include <string.h>
#include <argparse.h>

#define MAX_DRIVER_CONFIGS 16

#define LLMD_CHECK(op) \
	if ((status = (op)) != LLMD_OK) { \
		fprintf(stderr, "%s returns %d", #op, status);\
		goto end; \
	}

struct driver_config_skv {
	const char* section;
	const char* key;
	const char* value;
};

struct config {
	unsigned int num_driver_configs;
	struct driver_config_skv driver_configs[MAX_DRIVER_CONFIGS];
	const char* tmp_string;
};

static inline int
parse_driver_config(
	struct argparse* argparse,
	const struct argparse_option* option
) {
	(void)argparse;

	struct config* config = (void*)option->data;
	if (config->num_driver_configs >= MAX_DRIVER_CONFIGS) {
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

	struct driver_config_skv* cli_config = &config->driver_configs[config->num_driver_configs++];
	cli_config->section = strndup(config->tmp_string, dot_pos - config->tmp_string);
	cli_config->key = strndup(dot_pos + 1, eq_pos - dot_pos - 1);
	cli_config->value = strndup(eq_pos + 1, strlen(config->tmp_string) - (eq_pos - config->tmp_string));

	return 0;
}

#endif
