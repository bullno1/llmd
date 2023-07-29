#ifndef LM_PIPELINE_H
#define LM_PIPELINE_H

#include "lm_pipeline/def.h"
#include <llmd/core.h>
#include <stdbool.h>

struct lm_pipeline_var {
	unsigned int begin;
	unsigned int end;
};

enum lm_pipeline_event_type {
	LM_PIPELINE_NEW_TOKENS,
	LM_PIPELINE_CAPTURE_BEGIN,
	LM_PIPELINE_CAPTURE_END,
	LM_PIPELINE_REWIND,
	LM_PIPELINE_LLM_EVAL_BEGIN,
	LM_PIPELINE_LLM_EVAL_END,
};

struct lm_pipeline_event {
	enum lm_pipeline_event_type type;
	union {
		struct {
			const llmd_token_t* tokens;
			unsigned int num_tokens;
			const char* string;
			unsigned int num_chars;
		} new_tokens;

		struct {
			unsigned int new_pos;
		} rewind;

		struct {
			const struct lm_pipeline_var* var;
		} capture;
	} data;
};

typedef void (*lm_pipeline_logit_processor_t)(float* scores, void* userdata);
typedef llmd_token_t (*lm_pipeline_sampler_t)(float* scores, void* userdata);
typedef void (*lm_pipeline_watcher_t)(struct lm_pipeline_event event, void* userdata);
typedef void (*lm_pipeline_finalizer_t)(void* userdata);
typedef void (*lm_pipeline_t)(struct lm_pipeline_ctx* ctx, void* userdata);
typedef bool (*lm_pipeline_logit_filter_t)(llmd_token_t token_id, void* userdata);

#ifdef __cplusplus
extern "C" {
#endif

LM_PIPELINE_API struct lm_pipeline_ctx*
lm_pipeline_create_ctx(struct llmd_host* host);

LM_PIPELINE_API void
lm_pipeline_destroy_ctx(struct lm_pipeline_ctx* ctx);

LM_PIPELINE_API enum llmd_error
lm_pipeline_run(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_t exp, void* userdata,
	struct llmd_context* lm
);

LLMD_CORE_API void
lm_pipeline_abort(struct lm_pipeline_ctx* ctx, enum llmd_error error);

LM_PIPELINE_API struct llmd_model_info
lm_pipeline_get_model_info(struct lm_pipeline_ctx* ctx);

LM_PIPELINE_API struct llmd_context*
lm_pipeline_get_llm_ctx(struct lm_pipeline_ctx* ctx);

LM_PIPELINE_API void*
lm_pipeline_malloc(struct lm_pipeline_ctx* ctx, size_t size);

LM_PIPELINE_API void
lm_pipeline_add_logit_processor(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_logit_processor_t logit_processor, void* userdata
);

LM_PIPELINE_API void
lm_pipeline_add_watcher(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_watcher_t watcher, void* userdata
);

LM_PIPELINE_API void
lm_pipeline_add_finalizer(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_finalizer_t finalizer, void* userdata
);

LM_PIPELINE_API void
lm_pipeline_set_sampler(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_sampler_t sampler, void* userdata
);

LM_PIPELINE_API const llmd_token_t*
lm_pipeline_get_tokens(struct lm_pipeline_ctx* ctx);

LM_PIPELINE_API unsigned int
lm_pipeline_get_num_tokens(struct lm_pipeline_ctx* ctx);

LM_PIPELINE_API void
lm_pipeline_rewind(struct lm_pipeline_ctx* ctx, unsigned int pos);

LM_PIPELINE_API void
lm_pipeline_push_string(struct lm_pipeline_ctx* ctx, const char* string);

LM_PIPELINE_API void
lm_pipeline_push_tokens(struct lm_pipeline_ctx* ctx, const llmd_token_t* tokens, unsigned int num_tokens);

LM_PIPELINE_API void
lm_pipeline_begin_capture(struct lm_pipeline_ctx* ctx, struct lm_pipeline_var* var);

LM_PIPELINE_API void
lm_pipeline_end_capture(struct lm_pipeline_ctx* ctx, struct lm_pipeline_var* var);

LM_PIPELINE_API const char*
lm_pipeline_var_get(struct lm_pipeline_ctx* ctx, struct lm_pipeline_var* var);

LM_PIPELINE_API float*
lm_pipeline_get_next_logits(struct lm_pipeline_ctx* ctx);

LM_PIPELINE_API void
lm_pipeline_filter_next_tokens(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_logit_filter_t filter,
	void* userdata
);

LM_PIPELINE_API unsigned int
lm_pipeline_generate_one_of_strings(
	struct lm_pipeline_ctx* ctx,
	const char* strings[],
	unsigned int num_strings
);

LM_PIPELINE_API llmd_token_t
lm_pipeline_generate_one_of_tokens(
	struct lm_pipeline_ctx* ctx,
	const llmd_token_t tokens[],
	unsigned int num_tokens
);

LM_PIPELINE_API llmd_token_t
lm_pipeline_sample_next_token(struct lm_pipeline_ctx* ctx);

LM_PIPELINE_API unsigned int
lm_pipeline_check_suffix_str(
	struct lm_pipeline_ctx* ctx,
	const char* suffix,
	bool case_sensitive
);

LM_PIPELINE_API unsigned int
lm_pipeline_check_suffix_tokens(
	struct lm_pipeline_ctx* ctx,
	const llmd_token_t* tokens,
	unsigned int num_tokens
);

#ifdef __cplusplus
}
#endif

#endif
