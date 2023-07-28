#include <stddef.h>
#include <llmd/loader.h>
#include <llmd/core.h>
#include <llmd/sampling.h>
#include <stdlib.h>
#include "common.h"

#define READ_BLOCK 1024

int
main(int argc, const char* argv[]) {
    const char* driver_path = NULL;
    const char* config_path = NULL;
	char* prompt = malloc(READ_BLOCK);
	size_t prompt_len = 0;
	size_t prompt_buf_size = READ_BLOCK;

	struct config config = {
		.num_driver_configs = 0,
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
		OPT_END()
	};
	struct argparse argparse;
	argparse_init(&argparse, options, NULL, ARGPARSE_STOP_AT_NON_OPTION);
	argparse_describe(&argparse, "Feed a prompt to a model from stdin and run until completion", NULL);
	argparse_parse(&argparse, argc, argv);

	if (driver_path == NULL) {
		fprintf(stderr, "Driver path is missing.\n\n");
		argparse_usage(&argparse);
		return 1;
	}

	enum llmd_error status = LLMD_OK;
	struct llmd_driver_loader* loader = NULL;
	struct llmd_driver* driver = NULL;
	struct llmd_session* session = NULL;
	struct llmd_context* context = NULL;
	struct llmd_generate_handle* gen_handle = NULL;
	llmd_token_t* prompt_buf = NULL;
	float* scratch_buf = NULL;
	float* logits = NULL;
	struct llmd_sampling_mirostat_v2_state mirostat = {
		.tau = 5.f,
		.eta = 0.1f,
		.mu = 10.f,
	};

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
	LLMD_CHECK(llmd_create_session(NULL, driver, &session));

	struct llmd_model_info model_info;
	LLMD_CHECK(llmd_get_model_info(session, &model_info));
	logits = malloc(sizeof(float) * model_info.vocab_size);
	scratch_buf = malloc(sizeof(float) * model_info.vocab_size);
	mirostat.scratch_buf = scratch_buf;

	// Read prompt from stdin
	char buf[READ_BLOCK];
	size_t num_chars;
	while ((num_chars = fread(buf, 1, READ_BLOCK, stdin)) != 0) {
		if ((num_chars + prompt_len) > prompt_buf_size) {
			prompt_buf_size *= 2;
			prompt = realloc(prompt, prompt_buf_size);
			if (prompt == NULL) {
				fprintf(stderr, "OOM\n");
				goto end;
			}
		}

		memcpy(prompt + prompt_len, buf, num_chars);
		prompt_len += num_chars;
	}

	LLMD_CHECK(llmd_create_context(session, LLMD_CONTEXT_MIN_UPLOAD, &context));

	const llmd_token_t* tokens;
	unsigned int num_tokens;
	LLMD_CHECK(llmd_tokenize(context, prompt, prompt_len, &tokens, &num_tokens));

	fprintf(stderr, "Tokenized prompt to %d tokens\n", num_tokens);
	for (unsigned int i = 0; i < num_tokens; ++i) {
		const char* str;
		LLMD_CHECK(llmd_decode_token(context, tokens[i], &str, NULL));

		fprintf(stderr, "[%d] = %d (%s)\n", i, tokens[i], str);
	}

	prompt_buf = malloc(sizeof(llmd_token_t) * (num_tokens + 1));
	prompt_buf[0] = model_info.bos_token;
	memcpy(prompt_buf + 1, tokens, sizeof(llmd_token_t) * num_tokens);

	struct llmd_sampling_default_rng_state rng_state;

	LLMD_CHECK(llmd_begin_generate(context, &gen_handle));
	LLMD_CHECK(llmd_generate_next(
		gen_handle,
		prompt_buf, num_tokens + 1,
		0,
		logits
	));
	unsigned int offset = num_tokens + 1;

	// TODO: better seed
	struct llmd_sampling_rng rng = llmd_sampling_init_default_rng(&rng_state, 0);
	while (true) {
		llmd_sampling_apply_temperature(model_info.vocab_size, logits, 0.8f);
		llmd_token_t next_token = llmd_sampling_pick_mirostat_v2(
			model_info.vocab_size, logits, &rng, &mirostat
		);

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
			logits
		));
	}

	LLMD_CHECK(llmd_end_generate(gen_handle));
	gen_handle = NULL;
end:
	if (prompt) {
		free(prompt);
	}

	if (scratch_buf) {
		free(scratch_buf);
	}

	if (logits) {
		free(logits);
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

	for (unsigned int i = 0; i < config.num_driver_configs; ++i) {
		const struct driver_config_skv* driver_config = &config.driver_configs[i];
		free((void*)driver_config->section);
		free((void*)driver_config->key);
		free((void*)driver_config->value);
	}

	if (loader != NULL) {
		llmd_unload_driver(loader);
	}

	return status == LLMD_OK ? 0 : 1;
}
