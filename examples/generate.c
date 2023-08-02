#include <stddef.h>
#include <llmd/loader.h>
#include <llmd/core.h>
#include <llmd/sampling.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include "common.h"

#define READ_BLOCK 1024

int
main(int argc, const char* argv[]) {
	char* prompt = malloc(READ_BLOCK);
	const char* input_path = "-";
	FILE* input_file = stdin;
	size_t prompt_len = 0;
	size_t prompt_buf_size = READ_BLOCK;
	int seed = 0;
	float temperature = 0.8f;
	unsigned int n_last = -1;
	struct llmd_sampling_ring_buf* ring_buf = NULL;
	float repetition_penalty = 1.10f;
	float frequency_penalty = 0.00f;
	float presence_penalty  = 0.00f;

	struct driver_config driver_config = { 0 };

	struct argparse_option options[] = {
		COMMON_OPTIONS,
		OPT_GROUP("Generation options"),
		{
			.type = ARGPARSE_OPT_STRING,
			.long_name = "file",
			.short_name = 'f',
			.help = "The input file, use '-' for stdin",
			.value = &input_path,
		},
		{
			.type = ARGPARSE_OPT_INTEGER,
			.long_name = "seed",
			.help = "The seed for RNG (default: 0, use -1 for a random seed)",
			.value = &seed,
		},
		{
			.type = ARGPARSE_OPT_FLOAT,
			.long_name = "temperature",
			.help = "The temperature",
			.value = &temperature,
		},
		{
			.type = ARGPARSE_OPT_INTEGER,
			.long_name = "n-last",
			.help = "Number of characters to use for penalty. (default: -1 = context length)",
			.value = &n_last,
		},
		{
			.type = ARGPARSE_OPT_FLOAT,
			.long_name = "repetition-penalty",
			.help = "Repetition penalty",
			.value = &repetition_penalty,
		},
		{
			.type = ARGPARSE_OPT_FLOAT,
			.long_name = "presence-penalty",
			.help = "Presence penalty",
			.value = &presence_penalty,
		},
		{
			.type = ARGPARSE_OPT_FLOAT,
			.long_name = "frequency-penalty",
			.help = "Frequency penalty",
			.value = &frequency_penalty,
		},
		OPT_END()
	};
	struct argparse argparse;
	argparse_init(&argparse, options, NULL, ARGPARSE_STOP_AT_NON_OPTION);
	argparse_describe(&argparse, "Feed a prompt to a model and run until completion", NULL);
	argparse_parse(&argparse, argc, argv);

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

	LLMD_CHECK(load_driver(&driver_config, &argparse, &loader, &driver));
	LLMD_CHECK(llmd_create_session(NULL, driver, &session));

	struct llmd_model_info model_info;
	LLMD_CHECK(llmd_get_model_info(session, &model_info));
	logits = malloc(sizeof(float) * model_info.vocab_size);
	scratch_buf = malloc(sizeof(float) * model_info.vocab_size);
	mirostat.scratch_buf = scratch_buf;

	// Read prompt
	if (strcmp(input_path, "-") == 0) {
		input_file = stdin;
	} else {
		input_file = fopen(input_path, "r");
		if (input_file == NULL) {
			fprintf(stderr, "Could not open %s: %s\n", input_path, strerror(errno));
			goto end;
		}
	}

	char buf[READ_BLOCK];
	size_t num_chars;
	while ((num_chars = fread(buf, 1, READ_BLOCK, input_file)) != 0) {
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

	if (input_file != stdin) {
		fclose(input_file);
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

	ring_buf = llmd_sampling_create_ring_buf(NULL, (int)n_last > 0 ? n_last : model_info.max_context_length);
	for (unsigned int i = 0; i < num_tokens + 1; ++i) {
		llmd_sampling_ring_buf_add_token(ring_buf, prompt_buf[i]);
	}

	LLMD_CHECK(llmd_begin_generate(context, &gen_handle));
	LLMD_CHECK(llmd_generate_next(
		gen_handle,
		prompt_buf, num_tokens + 1,
		0,
		logits
	));
	unsigned int offset = num_tokens + 1;

	struct llmd_sampling_default_rng_state rng_state;
	struct llmd_sampling_rng rng = llmd_sampling_init_default_rng(&rng_state, seed == -1 ? time(NULL) : seed);
	while (true) {
		llmd_sampling_apply_repetition_penalties(
			model_info.vocab_size, logits, ring_buf, repetition_penalty
		);
		llmd_sampling_apply_frequency_and_presence_penalties(
			model_info.vocab_size, logits, ring_buf, frequency_penalty, presence_penalty
		);
		llmd_sampling_apply_temperature(model_info.vocab_size, logits, temperature);
		llmd_token_t next_token = llmd_sampling_pick_mirostat_v2(
			model_info.vocab_size, logits, &rng, &mirostat
		);

		if (next_token == model_info.eos_token) {
			break;
		}
		llmd_sampling_ring_buf_add_token(ring_buf, next_token);

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
	if (ring_buf != NULL) {
		llmd_sampling_destroy_ring_buf(ring_buf);
	}

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

	if (loader != NULL) {
		llmd_unload_driver(loader);
	}

	cleanup_driver_config(&driver_config);

	return status == LLMD_OK ? 0 : 1;
}
