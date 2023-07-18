#include "llmd/loader.h"
#include <stddef.h>
#include <llmd/core.h>
#include <llmd/sampling.h>
#include <argparse.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_CLI_INI 16

#define LLMD_CHECK(op) \
	if ((status = (op)) != LLMD_OK) { \
		fprintf(stderr, "%s returns %d", #op, status);\
		goto end; \
	}

struct ini_config {
	const char* section;
	const char* key;
	const char* value;
};

struct cli_config {
	unsigned int num_cli_ini;
	struct ini_config ini_config[MAX_CLI_INI];
	const char* tmp_string;
};

static int
parse_cli_ini(
	struct argparse* argparse,
	const struct argparse_option* option
) {
	(void)argparse;

	struct cli_config* config = (void*)option->data;
	if (config->num_cli_ini >= MAX_CLI_INI) {
		fprintf(stderr, "Too many config from argument\n");
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

	struct ini_config* ini_config = &config->ini_config[config->num_cli_ini++];
	ini_config->section = strndup(config->tmp_string, dot_pos - config->tmp_string);
	ini_config->key = strndup(dot_pos + 1, eq_pos - dot_pos - 1);
	ini_config->value = strndup(eq_pos + 1, strlen(config->tmp_string) - (eq_pos - config->tmp_string));

	return 0;
}

int
main(int argc, const char* argv[]) {
    const char* driver_path = NULL;
    const char* config_path = NULL;
    const char* prompt = "Hello";

	struct cli_config cli_config = {
		.num_cli_ini = 0,
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
			.short_name = 'p',
			.long_name = "prompt",
			.value = &prompt,
			.help = "The prompt",
		},
		{
			.type = ARGPARSE_OPT_STRING,
			.short_name = 's',
			.long_name = "set",
			.help = "Set custom config. For example: --set=main.model_path=custom_path",
			.callback = parse_cli_ini,
			.value = &cli_config.tmp_string,
			.data = (intptr_t)(void*)&cli_config,
		},
		OPT_END()
	};
	struct argparse argparse;
	argparse_init(&argparse, options, NULL, ARGPARSE_STOP_AT_NON_OPTION);
	argparse_describe(&argparse, "Feed a prompt to a model and run until copmletion", NULL);
	argparse_parse(&argparse, argc, argv);

	if (driver_path == NULL) {
		fprintf(stderr, "Driver path is missing\n");
		return 1;
	}

	enum llmd_error status = LLMD_OK;
	struct llmd_driver_loader* loader = NULL;
	struct llmd_driver* driver = NULL;
	struct llmd_session* session = NULL;
	struct llmd_context* context = NULL;
	struct llmd_generate_handle* gen_handle = NULL;
	llmd_token_t* prompt_buf = NULL;
	struct llmd_sampling_candidates candidates = { 0 };

	LLMD_CHECK(llmd_begin_load_driver(NULL, driver_path, &loader));

	if (config_path != NULL) {
		LLMD_CHECK(llmd_load_driver_config_from_file(loader, config_path));
	}

	for (unsigned int i = 0; i < cli_config.num_cli_ini; ++i) {
		const struct ini_config* config = &cli_config.ini_config[i];
		status = llmd_config_driver(
			loader,
			config->section, config->key, config->value
		);

		if (status != LLMD_OK) {
			fprintf(
				stderr,
				"Driver rejects config %s.%s=%s\n",
				config->section, config->key, config->value
			);
			goto end;
		}
	}

	LLMD_CHECK(llmd_end_load_driver(loader, &driver));
	LLMD_CHECK(llmd_create_session(NULL, driver, &session));
	LLMD_CHECK(llmd_create_context(session, LLMD_CONTEXT_MIN_UPLOAD, &context));

	const llmd_token_t* tokens;
	unsigned int num_tokens;
	LLMD_CHECK(llmd_tokenize(context, prompt, strlen(prompt), &tokens, &num_tokens));

	fprintf(stderr, "Tokenized prompt to %d tokens\n", num_tokens);
	for (unsigned int i = 0; i < num_tokens; ++i) {
		const char* str;
		LLMD_CHECK(llmd_decode_token(context, tokens[i], &str, NULL));

		fprintf(stderr, "[%d] = %d (%s)\n", i, tokens[i], str);
	}

	prompt_buf = malloc(sizeof(llmd_token_t) * (num_tokens + 1));
	struct llmd_model_info model_info;
	LLMD_CHECK(llmd_get_model_info(session, &model_info));
	prompt_buf[0] = model_info.bos_token;
	memcpy(prompt_buf + 1, tokens, sizeof(llmd_token_t) * num_tokens);

	candidates.scores = malloc(sizeof(float) * model_info.vocab_size);
	candidates.ids = malloc(sizeof(llmd_token_t) * model_info.vocab_size);

	struct llmd_sampling_default_rng_state rng_state;
	struct llmd_sampling_mirostat_v2_state mirostat = {
		.tau = 5.f,
		.eta = 0.1f,
		.mu = 10.f,
	};

	LLMD_CHECK(llmd_begin_generate(context, &gen_handle));
	LLMD_CHECK(llmd_generate_next(
		gen_handle,
		prompt_buf, num_tokens + 1,
		0,
		candidates.scores
	));
	unsigned int offset = num_tokens + 1;

	// TODO: better seed
	struct llmd_sampling_rng rng = llmd_sampling_init_default_rng(&rng_state, 0);
	(void)rng;
	(void)mirostat;
	while (true) {
		for (unsigned int i = 0; i < model_info.vocab_size; ++i) {
			candidates.ids[i] = i;
		}
		candidates.num_candidates = model_info.vocab_size;
		candidates.sorted = false;

		llmd_sampling_apply_temperature(&candidates, 0.8f);
		llmd_sampling_apply_softmax(&candidates);
		llmd_token_t next_token = llmd_sampling_pick_weighted_random(&candidates, &rng);

		if (next_token == model_info.eos_token) {
			break;
		}

		const char* text;
		LLMD_CHECK(llmd_decode_token(context, next_token, &text, NULL));
		printf("%s", text);
		fflush(stdout);

		LLMD_CHECK(llmd_generate_next(
			gen_handle,
			&next_token, 1,
			offset++,
			candidates.scores
		));
	}

	LLMD_CHECK(llmd_end_generate(gen_handle));
	gen_handle = NULL;
end:
	if (candidates.scores) {
		free(candidates.scores);
		free(candidates.ids);
	}

	if (prompt_buf != NULL) {
		free(prompt_buf);
	}

	if (gen_handle != NULL) {
		llmd_end_generate(gen_handle);
	}

	if (context != NULL) {
		llmd_destroy_context(context);
	}

	if (session != NULL) {
		llmd_destroy_session(session);
	}

	for (unsigned int i = 0; i < cli_config.num_cli_ini; ++i) {
		const struct ini_config* config = &cli_config.ini_config[i];
		free((void*)config->section);
		free((void*)config->key);
		free((void*)config->value);
	}

	if (loader != NULL) {
		llmd_unload_driver(loader);
	}

	return status == LLMD_OK ? 0 : 1;
}
