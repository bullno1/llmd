#include <lm_pipeline.h>
#include <lm_pipeline/macro.h>
#include <lm_pipeline/sampling.h>
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

	printf("\n-----------------------\n");
	tokens_(bos_);
	str_(
		"### Instruction:\n"
		"This is a conversation between two characters A and B.\n"
		"A tells B a knock knock joke.\n"
		"\n"
		"### Response:\n"
		"A: Knock knock\n"
		"B: Who's there?\n"
	);
	str_("A:"); capture_(setup, ends_with_("\nB:")); str_("\n");
	str_("B: "); to_str_(setup); str_(" who?\n");
	str_("A:"); capture_(punch_line, ends_with_tokens_(eos_), ends_with_("\n"));

	printf("\n-----------------------\n");
	printf("%s", lm_pipeline_get_text_buf(ctx));

	printf("\n-----------------------\n");
	printf("%s", lm_pipeline_get_uppercase_text_buf(ctx));

	printf("\n-----------------------\n");
	printf("setup=|%s|\n", get_(setup));
	printf("punch_line=|%s|\n", get_(punch_line));
}

int
main(int argc, const char* argv[]) {
	struct driver_config driver_config = { 0 };

	struct argparse_option options[] = {
		COMMON_OPTIONS,
		OPT_END()
	};
	struct argparse argparse;
	argparse_init(&argparse, options, NULL, ARGPARSE_STOP_AT_NON_OPTION);
	argparse_describe(&argparse, "lm_pipeline demo", NULL);
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
