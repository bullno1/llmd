#include "llmd/core.h"
#include <lm_pipeline.h>
#include <lm_pipeline/sampling.h>
#include <llmd/utils/arena_allocator.h>
#include <llmd/utils/buffer.h>
#include <setjmp.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define lm_pipeline_new(var, type) \
	type* var = lm_pipeline_malloc(ctx, sizeof(type)); \
	*var = (type)

#define lm_pipeline_assert(cond, error) \
	do { \
		if (!(cond)) { lm_pipeline_abort(ctx, error); } \
	} while(0)

#define lm_pipeline_check(op) \
	if ((ctx->status = (op)) != LLMD_OK) { \
		lm_pipeline_abort(ctx, ctx->status); \
	}

struct lm_pipeline_watcher_entry {
	struct lm_pipeline_watcher_entry* next;

	void* userdata;
	lm_pipeline_watcher_t fn;
};

struct lm_pipeline_logit_processor_entry {
	struct lm_pipeline_logit_processor_entry* next;

	void* userdata;
	lm_pipeline_logit_processor_t fn;
};

struct lm_pipeline_finalizer_entry {
	struct lm_pipeline_finalizer_entry* next;

	void* userdata;
	lm_pipeline_finalizer_t fn;
};

struct lm_pipeline_ctx {
	struct llmd_host* host;
	struct llmd_arena_allocator allocator;
	struct llmd_model_info model_info;
	struct llmd_context* lm;
	struct llmd_generate_handle* gen_handle;

	unsigned int eval_offset;
	unsigned int token_offset;
	struct llmd_buffer* token_buf;

	unsigned int text_offset;
	struct llmd_buffer* text_buf;

	unsigned int uppercase_text_offset;
	struct llmd_buffer* uppercase_text_buf;

	struct llmd_buffer* logit_buf;

	enum llmd_error status;

	struct lm_pipeline_finalizer_entry* finalizers;
	struct lm_pipeline_watcher_entry* first_watcher;
	struct lm_pipeline_watcher_entry* last_watcher;
	struct lm_pipeline_logit_processor_entry* first_logit_processor;
	struct lm_pipeline_logit_processor_entry* last_logit_processor;
	lm_pipeline_sampler_t sampler;
	void* sampler_userdata;

	jmp_buf jmp;
};

static void
lm_pipeline_emit_event(
	struct lm_pipeline_ctx* ctx,
	struct lm_pipeline_event event
) {
	for (
		struct lm_pipeline_watcher_entry* watcher = ctx->first_watcher;
		watcher != NULL;
		watcher = watcher->next
	) {
		watcher->fn(event, watcher->userdata);
	}
}

struct lm_pipeline_ctx*
lm_pipeline_create_ctx(struct llmd_host* host) {
	struct lm_pipeline_ctx* ctx = llmd_malloc(host, sizeof(struct lm_pipeline_ctx));
	if (ctx == NULL) { return NULL; }

	*ctx = (struct lm_pipeline_ctx) {
		.host = host,
	};
	llmd_arena_allocator_init(host, &ctx->allocator, 2048);

	return ctx;
}

void
lm_pipeline_destroy_ctx(struct lm_pipeline_ctx* ctx) {
	llmd_arena_allocator_cleanup(&ctx->allocator);
	llmd_free(ctx->host, ctx->text_buf);
	llmd_free(ctx->host, ctx->uppercase_text_buf);
	llmd_free(ctx->host, ctx->token_buf);
	llmd_free(ctx->host, ctx->logit_buf);
	llmd_free(ctx->host, ctx);
}

enum llmd_error
lm_pipeline_run(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_t pipeline, void* userdata,
	struct llmd_context* lm
) {
	enum llmd_error status;

	struct llmd_session* session = llmd_get_session_of_context(lm);
	if ((status = llmd_get_model_info(session, &ctx->model_info)) != LLMD_OK) {
		return status;
	}

	// Ensure a large enough buffer for context tokens
	size_t required_token_buf_size = ctx->model_info.max_context_length * sizeof(llmd_token_t);
	if (llmd_buffer_size(ctx->token_buf) < required_token_buf_size) {
		ctx->token_buf = llmd_realloc_buffer(
			ctx->host, ctx->token_buf, required_token_buf_size
		);
		if (ctx->token_buf == NULL) { return LLMD_ERR_OOM; }
	}

	// Ensure a large enough buffer for context strings
	unsigned int max_token_length = 0;
	for (llmd_token_t i = 0; i < ctx->model_info.vocab_size; ++i) {
		unsigned int token_length = 0;
		if ((status = llmd_decode_token(lm, i, NULL, &token_length)) != LLMD_OK) {
			return status;
		}

		max_token_length = token_length > max_token_length ? token_length : max_token_length;
	}

	size_t required_text_buf_size = ctx->model_info.vocab_size * max_token_length + 1;
	if (llmd_buffer_size(ctx->text_buf) < required_text_buf_size) {
		ctx->text_buf = llmd_realloc_buffer(
			ctx->host, ctx->text_buf, required_text_buf_size
		);
		ctx->uppercase_text_buf = llmd_realloc_buffer(
			ctx->host, ctx->uppercase_text_buf, required_text_buf_size
		);
		if (ctx->text_buf == NULL || ctx->uppercase_text_buf == NULL) {
			return LLMD_ERR_OOM;
		}
	}

	// Ensure a large enough buffer for logits
	size_t required_logit_buf_size = ctx->model_info.vocab_size * sizeof(float);
	if (llmd_buffer_size(ctx->logit_buf) < required_logit_buf_size) {
		ctx->logit_buf = llmd_realloc_buffer(
			ctx->host, ctx->logit_buf, required_logit_buf_size
		);
		if (ctx->logit_buf == NULL) { return LLMD_ERR_OOM; }
	}

	ctx->lm = lm;
	ctx->status = LLMD_OK;
	ctx->first_logit_processor = ctx->last_logit_processor = NULL;
	ctx->first_watcher = ctx->last_watcher = NULL;
	ctx->finalizers = NULL;
	ctx->eval_offset = ctx->token_offset = 0;
	ctx->uppercase_text_offset = ctx->text_offset = 0;
	lm_pipeline_use_greedy_sampler(ctx);
	llmd_arena_allocator_reset(&ctx->allocator);
	if ((status = llmd_begin_generate(lm, &ctx->gen_handle)) != LLMD_OK) {
		return status;
	}

	if (setjmp(ctx->jmp) == 0) {
		pipeline(ctx, userdata);
	}

	llmd_end_generate(ctx->gen_handle);

	for (
		struct lm_pipeline_finalizer_entry* finalizer = ctx->finalizers;
		finalizer != NULL;
		finalizer = finalizer->next
	) {
		finalizer->fn(finalizer->userdata);
	}

	ctx->lm = NULL;
	return ctx->status;
}

void*
lm_pipeline_malloc(struct lm_pipeline_ctx* ctx, size_t size) {
	void* result = llmd_arena_allocator_malloc(&ctx->allocator, size);
	if (result == NULL) { lm_pipeline_abort(ctx, LLMD_ERR_OOM); }

	return result;
}

void
lm_pipeline_abort(struct lm_pipeline_ctx* ctx, enum llmd_error error) {
	assert(ctx->lm != NULL);
	ctx->status = error;
	longjmp(ctx->jmp, 1);
}

struct llmd_model_info
lm_pipeline_get_model_info(struct lm_pipeline_ctx* ctx) {
	return ctx->model_info;
}

void
lm_pipeline_add_logit_processor(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_logit_processor_t fn, void* userdata
) {
	assert(ctx->lm != NULL);

	lm_pipeline_new(entry, struct lm_pipeline_logit_processor_entry) {
		.userdata = userdata,
		.fn = fn,
	};

	if (ctx->first_logit_processor == NULL) {
		ctx->first_logit_processor = ctx->last_logit_processor = entry;
	} else {
		ctx->last_logit_processor->next = entry;
		ctx->last_logit_processor = entry;
	}
}

void
lm_pipeline_add_watcher(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_watcher_t fn, void* userdata
) {
	assert(ctx->lm != NULL);

	lm_pipeline_new(entry, struct lm_pipeline_watcher_entry) {
		.userdata = userdata,
		.fn = fn,
	};

	if (ctx->first_watcher == NULL) {
		ctx->first_watcher = ctx->last_watcher = entry;
	} else {
		ctx->last_watcher->next = entry;
		ctx->last_watcher = entry;
	}
}

void
lm_pipeline_add_finalizer(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_finalizer_t fn, void* userdata
) {
	assert(ctx->lm != NULL);

	lm_pipeline_new(entry, struct lm_pipeline_finalizer_entry) {
		.userdata = userdata,
		.fn = fn,
	};

	entry->next = ctx->finalizers;
	ctx->finalizers = entry;
}

void
lm_pipeline_set_sampler(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_sampler_t sampler, void* userdata
) {
	assert(ctx->lm != NULL);
	ctx->sampler_userdata = userdata;
	ctx->sampler = sampler;
}

const llmd_token_t*
lm_pipeline_get_tokens(struct lm_pipeline_ctx* ctx) {
	return (void*)ctx->token_buf->mem;
}

unsigned int
lm_pipeline_get_num_tokens(struct lm_pipeline_ctx* ctx) {
	return ctx->token_offset;
}

void
lm_pipeline_rewind(struct lm_pipeline_ctx* ctx, unsigned int pos) {
	assert(pos <= ctx->token_offset);
	unsigned int old_offset = ctx->token_offset;
	ctx->token_offset = pos;
	ctx->eval_offset = ctx->eval_offset < pos ? ctx->eval_offset : pos;

	// Rewind the text buffer
	for (unsigned int i = pos; i < old_offset; ++i) {
		unsigned int token_len;
		llmd_token_t token = ((llmd_token_t*)ctx->token_buf->mem)[i];
		lm_pipeline_check(
			llmd_decode_token(ctx->lm, token, NULL, &token_len)
		);
		ctx->text_offset -= token_len;
	}
	ctx->uppercase_text_offset = ctx->uppercase_text_offset < ctx->text_offset
		? ctx->uppercase_text_offset
		: ctx->text_offset;

	lm_pipeline_emit_event(ctx, (struct lm_pipeline_event) {
		.type = LM_PIPELINE_REWIND,
		.data = { .rewind = { .new_pos = pos } }
	});
}

void
lm_pipeline_push_string(struct lm_pipeline_ctx* ctx, const char* string) {
	const llmd_token_t* tokens;
	unsigned int num_tokens;
	lm_pipeline_check(
		llmd_tokenize(ctx->lm, string, strlen(string), &tokens, &num_tokens)
	);

	lm_pipeline_push_tokens(ctx, tokens, num_tokens);
}

void
lm_pipeline_push_tokens(
	struct lm_pipeline_ctx* ctx,
	const llmd_token_t* tokens,
	unsigned int num_tokens
) {
	lm_pipeline_assert(
		num_tokens + lm_pipeline_get_num_tokens(ctx) <= ctx->model_info.max_context_length,
		LLMD_ERR_BUF_SIZE
	);

	memcpy(
		(llmd_token_t*)ctx->token_buf->mem + ctx->token_offset,
		tokens, num_tokens * sizeof(llmd_token_t)
	);
	ctx->token_offset += num_tokens;

	unsigned int txt_start = ctx->text_offset;
	for (unsigned int i = 0; i < num_tokens; ++i) {
		const char* str;
		unsigned int num_chars;
		lm_pipeline_check(
			llmd_decode_token(ctx->lm, tokens[i], &str, &num_chars)
		);

		memcpy(ctx->text_buf->mem + ctx->text_offset, str, num_chars);
		ctx->text_offset += num_chars;
	}
	ctx->text_buf->mem[ctx->text_offset] = '\0';

	lm_pipeline_emit_event(ctx, (struct lm_pipeline_event) {
		.type = LM_PIPELINE_NEW_TOKENS,
		.data = {
			.new_tokens = {
				.tokens = tokens,
				.num_tokens = num_tokens,
				.string = ctx->text_buf->mem + txt_start,
				.num_chars = ctx->text_offset - txt_start,
			}
		}
	});
}

void
lm_pipeline_begin_capture(struct lm_pipeline_ctx* ctx, struct lm_pipeline_var* var) {
	var->begin = lm_pipeline_get_num_tokens(ctx);

	lm_pipeline_emit_event(ctx, (struct lm_pipeline_event) {
		.type = LM_PIPELINE_CAPTURE_BEGIN,
		.data = { .capture = { .var = var } }
	});
}

void
lm_pipeline_end_capture(struct lm_pipeline_ctx* ctx, struct lm_pipeline_var* var) {
	var->end = lm_pipeline_get_num_tokens(ctx);

	lm_pipeline_emit_event(ctx, (struct lm_pipeline_event) {
		.type = LM_PIPELINE_CAPTURE_END,
		.data = { .capture = { .var = var } }
	});
}

const char*
lm_pipeline_var_get(struct lm_pipeline_ctx* ctx, struct lm_pipeline_var* var) {
	assert(var->end >= var->begin);

	size_t len = var->end - var->begin;
	char* buf = lm_pipeline_malloc(ctx, len + 1);
	memcpy(buf, ctx->text_buf->mem + var->begin, len);
	buf[len] = '\0';

	return buf;
}

float*
lm_pipeline_get_next_logits(struct lm_pipeline_ctx* ctx) {
	float* logit_buf = (float*)&ctx->logit_buf->mem;

	if (ctx->eval_offset < ctx->token_offset) {
		lm_pipeline_emit_event(ctx, (struct lm_pipeline_event) {
			.type = LM_PIPELINE_LLM_EVAL_BEGIN,
		});

		lm_pipeline_check(
			llmd_generate_next(
				ctx->gen_handle,
				(llmd_token_t*)ctx->token_buf->mem + ctx->eval_offset,
				ctx->token_offset - ctx->eval_offset,
				ctx->eval_offset,
				logit_buf
			)
		);
		ctx->eval_offset = ctx->token_offset;

		lm_pipeline_emit_event(ctx, (struct lm_pipeline_event) {
			.type = LM_PIPELINE_LLM_EVAL_END,
		});

		for (
			struct lm_pipeline_logit_processor_entry* entry = ctx->first_logit_processor;
			entry != NULL;
			entry = entry->next
		) {
			entry->fn(logit_buf, entry->userdata);
		}
	}

	return logit_buf;
}

void
lm_pipeline_filter_next_tokens(
	struct lm_pipeline_ctx* ctx,
	lm_pipeline_logit_filter_t filter,
	void* userdata
) {
	float* logits = lm_pipeline_get_next_logits(ctx);
	for (unsigned int i = 0; i < ctx->model_info.vocab_size; ++i) {
		if (!filter(i, userdata)) {
			logits[i] = -INFINITY;
		}
	}
}

struct lm_prefix_set {
	struct lm_pipeline_ctx* ctx;
	char** prefixes;
	unsigned int num_prefixes;
};

static bool
lm_pipeline_match_prefixes(llmd_token_t id, struct lm_prefix_set* prefix_set) {
	struct lm_pipeline_ctx* ctx = prefix_set->ctx;
	const char* token_text;
	unsigned int text_len;
	lm_pipeline_check(
		llmd_decode_token(prefix_set->ctx->lm, id, &token_text, &text_len)
	);

	for (unsigned int i = 0; i < prefix_set->num_prefixes; ++i) {
		char* prefix = prefix_set->prefixes[i];
		if (prefix == NULL) { continue; }

		size_t prefix_len = strlen(prefix);  // TODO: calculate once

		if (text_len > prefix_len) { continue; }

		if (memcmp(token_text, prefix, text_len) == 0) {
			return true;
		}
	}

	return false;
}

unsigned int
lm_pipeline_generate_one_of_strings(
	struct lm_pipeline_ctx* ctx,
	const char* strings[],
	unsigned int num_strings
) {
	// Make a mutable copy of strings
	char** prefixes = lm_pipeline_malloc(ctx, num_strings * sizeof(void*));
	for (unsigned int i = 0; i < num_strings; ++i) {
		size_t len = strlen(strings[i]);
		char* copy = lm_pipeline_malloc(ctx, len + 1);
		memcpy(copy, strings[i], len);
		copy[len] = '\0';

		prefixes[i] = copy;
	}

	struct lm_prefix_set prefix_set = {
		.ctx = ctx,
		.num_prefixes = num_strings,
		.prefixes = prefixes,
	};

	const char* token_text;
	unsigned int text_len;

	while (true) {
		// Keep only tokens that can generate the prefix
		lm_pipeline_filter_next_tokens(
			ctx,
			(lm_pipeline_logit_filter_t)lm_pipeline_match_prefixes,
			&prefix_set
		);

		// Sample the next token
		llmd_token_t next_token = lm_pipeline_sample_next_token(ctx);
		lm_pipeline_push_tokens(ctx, &next_token, 1);

		// Modify the prefix list to keep only achievable ones
		lm_pipeline_check(
			llmd_decode_token(ctx->lm, next_token, &token_text, &text_len)
		);

		for (unsigned int i = 0; i < num_strings; ++i) {
			char* prefix = prefixes[i];
			if (prefix == NULL) { continue; }

			size_t prefix_len = strlen(prefix);  // TODO: calculate once

			if (text_len > prefix_len) {  // Overshot
				// Give up on this prefix
				prefixes[i] = NULL;
				continue;
			}

			if (memcmp(token_text, prefix, text_len) == 0) {  // Match
				if (text_len == prefix_len) {  // Exact match
					return i;
				} else {  // Prefix match
					// Skip matched chars
					prefixes[i] += text_len;
				}
			} else {
				// Give up
				prefixes[i] = NULL;
			}
		}
	}
}

struct lm_pipeline_token_set {
	const llmd_token_t* tokens;
	unsigned int num_tokens;
};

static bool
lm_pipeline_match_tokens(
	llmd_token_t token,
	struct lm_pipeline_token_set* token_set
) {
	for (unsigned int i = 0; i < token_set->num_tokens; ++i) {
		if (token == token_set->tokens[i]) {
			return true;
		}
	}

	return false;
}

llmd_token_t
lm_pipeline_generate_one_of_tokens(
	struct lm_pipeline_ctx* ctx,
	const llmd_token_t* tokens,
	unsigned int num_tokens
) {
	struct lm_pipeline_token_set token_set = {
		.tokens = tokens,
		.num_tokens = num_tokens,
	};

	lm_pipeline_filter_next_tokens(
		ctx,
		(lm_pipeline_logit_filter_t)lm_pipeline_match_tokens,
		&token_set
	);

	llmd_token_t next_token = lm_pipeline_sample_next_token(ctx);
	lm_pipeline_push_tokens(ctx, &next_token, 1);

	return next_token;
}

llmd_token_t
lm_pipeline_sample_next_token(struct lm_pipeline_ctx* ctx) {
	return ctx->sampler(lm_pipeline_get_next_logits(ctx), ctx->sampler_userdata);
}

LM_PIPELINE_API unsigned int
lm_pipeline_check_suffix_str(
	struct lm_pipeline_ctx* ctx,
	const char* suffix,
	bool case_sensitive
) {
	size_t suffix_len = strlen(suffix);
	if (suffix_len > ctx->text_offset) { return 0; }

	const char* text_buf;
	if (!case_sensitive) {
		if (ctx->uppercase_text_offset < ctx->text_offset) {
			memcpy(
				ctx->uppercase_text_buf->mem,
				ctx->text_buf->mem + ctx->uppercase_text_offset,
				ctx->text_offset - ctx->uppercase_text_offset
			);
			for (
				unsigned int i = ctx->uppercase_text_offset;
				i < ctx->text_offset;
				++i
			) {
				ctx->uppercase_text_buf->mem[i] = toupper(
					ctx->uppercase_text_buf->mem[i]
				);
			}
			ctx->uppercase_text_offset = ctx->text_offset;
		}

		text_buf = ctx->uppercase_text_buf->mem;
	} else {
		text_buf = ctx->text_buf->mem;
	}

	return memcmp(text_buf - suffix_len, suffix, suffix_len) == 0
		? suffix_len
		: 0;
}

unsigned int
lm_pipeline_check_suffix_tokens(
	struct lm_pipeline_ctx* ctx,
	const llmd_token_t* tokens, unsigned int num_tokens
) {
	if (num_tokens > ctx->token_offset) { return 0; }

	return memcmp(
		tokens,
		(llmd_token_t*)ctx->token_buf->mem + ctx->token_offset - num_tokens,
		num_tokens * sizeof(llmd_token_t)
	) ? num_tokens : 0;
}
