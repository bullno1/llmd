#include <lm_pipeline.h>
#include <lm_pipeline/macro.h>
#include "common.h"

static void
stream_text(struct lm_pipeline_event event, void* userdata) {
	(void)userdata;

	if (event.type == LM_PIPELINE_NEW_TOKENS) {
		printf("%s", event.data.new_tokens.string);
		fflush(stdout);
	}
}

static void
knock_knock(struct lm_pipeline_ctx* ctx, void* userdata) {
	(void)userdata;

	lm_pipeline_bind(ctx);
	lm_pipeline_add_watcher(ctx, stream_text, NULL);

	var_(setup);
	var_(punch_line);

	tokens_(bos_);
	str_("A chat between a curious user and an artificial intelligence assistant. ");
	str_("The assistant gives helpful, detailed, and polite answers to the user's questions.\n");
	str_("USER: Tell me a knock knock joke.\n");
	str_("ASSISTANT: Knock knock.\n");
	str_("USER: Who's there?\n");
	str_("ASSISTANT: ");
	capture_(
		setup,
		ends_with_tokens_(eos_),
		ends_with_("USER:"), ends_with_("\n"), ends_with_(".")
	);

	to_str_(setup); tokens_(nl_);
	str_("USER: "); to_str_(setup); str_(" who?\n");
	str_("ASSISTANT:");
	capture_(punch_line, ends_with_tokens_(eos_), ends_with_("USER:"));

	printf("\n-----------------------\n");
	printf("Setup=|%s|\n", get_(setup));
	printf("Punch line=|%s|\n", get_(punch_line));
}

int
main(int argc, const char* argv[]) {
	struct driver_config driver_config = { 0 };

	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Driver options"),
		{
			.type = ARGPARSE_OPT_STRING,
			.short_name = 'd',
			.long_name = "driver",
			.value = &driver_config.driver_path,
			.help = "Path to driver",
		},
		{
			.type = ARGPARSE_OPT_STRING,
			.short_name = 'c',
			.long_name = "config",
			.value = &driver_config.config_file_path,
			.help = "Path to config file",
		},
		{
			.type = ARGPARSE_OPT_STRING,
			.short_name = 's',
			.long_name = "set",
			.help = "Set driver config directly. For example: --set=main.model_path=custom_path",
			.callback = parse_driver_config,
			.value = &driver_config.tmp_string,
			.data = (intptr_t)(void*)&driver_config,
		},
		OPT_END()
	};
	struct argparse argparse;
	argparse_init(&argparse, options, NULL, ARGPARSE_STOP_AT_NON_OPTION);
	argparse_describe(&argparse, "Feed a prompt to a model from stdin and run until completion", NULL);
	argparse_parse(&argparse, argc, argv);

	enum llmd_error status = LLMD_OK;
	struct llmd_driver_loader* loader = NULL;
	struct llmd_driver* driver = NULL;
	struct llmd_session* session = NULL;
	struct llmd_context* context = NULL;
	struct lm_pipeline_ctx* pipeline = NULL;

	LLMD_CHECK(load_driver(&driver_config, &argparse, &loader, &driver));
	LLMD_CHECK(llmd_create_session(NULL, driver, &session));
	LLMD_CHECK(llmd_create_context(session, LLMD_CONTEXT_MIN_UPLOAD, &context));

	pipeline = lm_pipeline_create_ctx(NULL);
	if (pipeline == NULL) {
		status = LLMD_ERR_OOM;
		goto end;
	}

	LLMD_CHECK(lm_pipeline_run(pipeline, knock_knock, NULL, context));

end:
	if (pipeline != NULL) {
		lm_pipeline_destroy_ctx(pipeline);
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
