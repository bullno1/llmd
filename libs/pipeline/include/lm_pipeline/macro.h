#ifndef LM_PIPELINE_MACRO_H
#define LM_PIPELINE_MACRO_H

#include "../lm_pipeline.h"

#define lm_pipeline_bind(ctx, llm) \
	struct lm_pipeline_ctx* lm_pipeline__ctx = ctx; \
	var_(lm_pipeline__suffix); \
	(void)lm_pipeline__ctx; \
	(void)lm_pipeline__suffix;

#define lm_pipeline_va_pack(type, ...) (type[]){ __VA_ARGS__ }

#define lm_pipeline_va_count(type, ...) \
	(sizeof(lm_pipeline_va_pack(type, __VA_ARGS__)) / sizeof(type))

#define str_(string) lm_pipeline_push_string(lm_pipeline__ctx, string)

#define tokens_(...) \
	lm_pipeline_push_tokens( \
		lm_pipeline__ctx, \
		lm_pipeline_va_pack(llmd_token_t, __VA_ARGS__), \
		lm_pipeline_va_count(llmd_token_t, __VA_ARGS__) \
	)

#define var_(name) struct lm_pipeline_var name = { 0 }

#define get_(name) lm_pipeline_var_get(var)

#define to_str_(name) str_(get(var))

#define pick_one_str_(...) \
	lm_pipeline_generate_one_of_strings( \
		lm_pipeline__ctx, \
		lm_pipeline_va_pack(const char*, __VA_ARGS__), \
		lm_pipeline_va_count(const char*, __VA_ARGS__) \
	)

#define pick_one_tok_(...) \
	lm_pipeline_generate_one_of_tokens( \
		lm_pipeline__ctx, \
		lm_pipeline_va_pack(llmd_token_t, __VA_ARGS__), \
		lm_pipeline_va_count(llmd_token_t, __VA_ARGS__) \
	)

#define capture_(var, ...) \
	do { \
		lm_pipeline_start_capture(lm_pipeline__ctx, var); \
		unsigned int lm_pipeline__suffix_len; \
		do { \
			llmd_token_t next_token = lm_pipeline_sample_next_token(lm_pipeline__ctx); \
			lm_pipeline_push_tokens(lm_pipeline__ctx, &next_token, 1); \
			lm_pipeline__suffix_len = lm_pipeline_first_suffix_match( \
				lm_pipeline_va_pack(unsigned int, __VA_ARGS__), \
				lm_pipeline_va_count(unsigned int, __VA_ARGS__), \
			); \
		} while (lm_pipeline__suffix_len == 0); \
		lm_pipeline__suffix.end = lm_pipeline_get_num_tokens(lm_pipeline__ctx); \
		lm_pipeline_rewind( \
			lm_pipeline__ctx, \
			lm_pipeline_get_num_tokens(lm_pipeline__ctx) - lm_pipeline__suffix_len \
		); \
		lm_pipeline__suffix.begin = lm_pipeline_get_num_tokens(lm_pipeline__ctx); \
		lm_pipeline_end_capture(lm_pipeline__ctx, var); \
	} while(0)

#define suffix_ lm_pipeline__suffix
#define ends_with_(str) lm_pipeline_check_suffix_str(lm_pipeline__ctx, str, false)
#define ends_with_exact_(str) lm_pipeline_check_suffix_str(lm_pipeline__ctx, str, true)
#define ends_with_tokens_(...) \
	lm_pipeline_check_suffix_tokens( \
		lm_pipeline__ctx, \
		lm_pipeline_va_pack(__VA_ARGS__), \
		lm_pipeline_va_count(__VA_ARGS__) \
	)
#define checkpoint_(name) unsigned int name = lm_pipeline_get_num_tokens(lm_pipeline__ctx)
#define rewind_(pos) lm_pipeline_rewind(lm_pipeline__ctx, pos)
#define model_info_ lm_pipeline_get_model_info(lm_pipeline__ctx)
#define nl_ model_info_.nl_token
#define bos_ model_info_.bos_token
#define eos_ model_info_.eos_token

static inline unsigned int
lm_pipeline_first_suffix_match(unsigned int matches[], unsigned int num_matches) {
	for (unsigned int i = 0; i < num_matches; ++i) {
		if (matches[i] != 0) {
			return matches[i];
		}
	}

	return 0;
}

#endif
