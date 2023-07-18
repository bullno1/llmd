#include <llmd/core.h>
#include <llmd/utils/buffer.h>
#include <llmd/utils/host.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define LLMD_TRY enum llmd_error llmd_status = LLMD_OK;
#define LLMD_EXCEPT_BEGIN llmd_except:
#define LLMD_EXCEPT_END return llmd_status;
#define LLMD_THROW(error) \
	do { llmd_status = (error); goto llmd_except; } while(0)

#define LLMD_CHECKED_MALLOC(out, host, size) \
	do { \
		if((out = llmd_malloc(host, size)) == NULL) { \
			LLMD_THROW(LLMD_ERR_OOM); \
		} \
	} while(0)

#define LLMD_CHECK_THROW(op) \
	do { \
		if ((llmd_status = (op)) != LLMD_OK) { LLMD_THROW(llmd_status); } \
	} while(0)

#define LLMD_CHECK_RETURN(op) \
	do { \
		enum llmd_error llmd_status; \
		if ((llmd_status = (op)) != LLMD_OK) { return llmd_status; } \
	} while(0)

struct llmd_session {
	struct llmd_host* host;
	struct llmd_driver* driver;

	struct llmd_model_info model_info;
	struct llmd_physical_context* free_contexts;
};

struct llmd_context {
	struct llmd_session* session;
	enum llmd_context_type type;

	struct llmd_physical_context* physical_ctx;
	bool generating;

	llmd_token_t* context_window;
	float* logits;
	struct llmd_buffer* token_buf;
	struct llmd_buffer* text_buf;
};

struct llmd_physical_context {
	int descriptor;

	llmd_token_t* context_window;
	unsigned int filled_size;
	struct llmd_context* virtual_ctx;
	struct llmd_physical_context* next;
};

static unsigned int
llmd_count_shared_prefix(
	const llmd_token_t* buf1,
	const llmd_token_t* buf2,
	unsigned int len
) {
	for (unsigned int i = 0; i < len; ++i) {
		if (buf1[i] != buf2[i]) {
			return i;
		}
	}

	return len;
}

static enum llmd_error
llmd_create_physical_ctx(
	struct llmd_session* session,
	struct llmd_physical_context** context_out
) {
LLMD_TRY
	struct llmd_driver* driver = session->driver;
	struct llmd_host* host = session->host;

	int descriptor = -1;
	struct llmd_physical_context* context = NULL;

	LLMD_CHECK_THROW(driver->interface->create_context(driver, &descriptor));
	LLMD_CHECKED_MALLOC(context, host, sizeof(*context));

	*context = (struct llmd_physical_context) {
		.descriptor = descriptor
	};
	*context_out = context;

	return LLMD_OK;
LLMD_EXCEPT_BEGIN
	llmd_free(host, context);

	if (descriptor >= 0) {
		enum llmd_error status = driver->interface->destroy_context(driver, descriptor);
		if (status != LLMD_OK) {
			llmd_log(
				host, LLMD_LOG_WARNING,
				"Error while physic freeing context %d: %d", descriptor, status
			);
		}
	}
LLMD_EXCEPT_END
}

static enum llmd_error
llmd_destroy_physical_ctx(
	struct llmd_session* session,
	struct llmd_physical_context* context
) {
	struct llmd_driver* driver = session->driver;
	struct llmd_host* host = session->host;

	llmd_free(host, context->context_window);
	enum llmd_error status = driver->interface->destroy_context(driver, context->descriptor);
	if (status != LLMD_OK) {
		llmd_log(
			host, LLMD_LOG_WARNING,
			"Error while physic freeing context %d: %d", context->descriptor, status
		);
	}

	llmd_free(host, context);

	return LLMD_OK;
}

static enum llmd_error
llmd_create_shared_physical_ctx(
	struct llmd_session* session,
	struct llmd_physical_context** context_out
) {
	struct llmd_host* host = session->host;
	struct llmd_physical_context* physical_ctx;

	if (llmd_create_physical_ctx(session, &physical_ctx) != LLMD_OK) {
		return LLMD_ERR_OOM;
	}

	physical_ctx->context_window = llmd_malloc(
		host, sizeof(llmd_token_t) * session->model_info.max_context_length
	);

	if (physical_ctx->context_window == NULL) {
		llmd_destroy_physical_ctx(session, physical_ctx);
		return LLMD_ERR_OOM;
	}

	// Link into list of free contexts for reuse after unbind
	// TODO: CAS?
	physical_ctx->next = session->free_contexts;
	session->free_contexts = physical_ctx;

	*context_out = physical_ctx;
	return LLMD_OK;
}

static enum llmd_error
llmd_bind_virtual_ctx(
	struct llmd_session* session,
	struct llmd_context* virtual_ctx,
	unsigned int num_tokens,
	unsigned int* eval_offset_out
) {
	// TODO: lock and wait for context to become free instead of OOM?
	switch (virtual_ctx->type) {
		case LLMD_CONTEXT_MIN_UPLOAD: {
			struct llmd_physical_context* chosen_context = NULL;
			unsigned int upload_size = session->model_info.max_context_length;

			for (
				struct llmd_physical_context* itr = session->free_contexts;
				itr != NULL;
				itr = itr->next
			) {
				// TODO: CAS
				if (itr->virtual_ctx != NULL) {
					continue;
				}

				unsigned int ctx_filled_size = itr->filled_size;
				unsigned int shared_prefix_length = llmd_count_shared_prefix(
					virtual_ctx->context_window, itr->context_window,
					ctx_filled_size < num_tokens ? ctx_filled_size : num_tokens
				);
				unsigned int ctx_upload_size = num_tokens - shared_prefix_length;

				if (ctx_upload_size < upload_size) {
					chosen_context = itr;
					upload_size = ctx_upload_size;
				}
			}

			if (chosen_context == NULL) {
				if (llmd_create_shared_physical_ctx(session, &chosen_context) != LLMD_OK) {
					return LLMD_ERR_OOM;
				} else {
					upload_size = num_tokens;
				}
			}

			chosen_context->virtual_ctx = virtual_ctx;
			virtual_ctx->physical_ctx = chosen_context;

			*eval_offset_out = num_tokens - upload_size;

			return LLMD_OK;
		}
		case LLMD_CONTEXT_MIN_DISCARD: {
			struct llmd_physical_context* chosen_context = NULL;
			unsigned int discard_size = session->model_info.max_context_length;

			for (
				struct llmd_physical_context* itr = session->free_contexts;
				itr != NULL;
				itr = itr->next
			) {
				// TODO: CAS
				if (itr->virtual_ctx != NULL) {
					continue;
				}

				unsigned int ctx_filled_size = itr->filled_size;
				unsigned int shared_prefix_length = llmd_count_shared_prefix(
					virtual_ctx->context_window, itr->context_window,
					ctx_filled_size < num_tokens ? ctx_filled_size : num_tokens
				);
				unsigned int ctx_discard_size = ctx_filled_size - shared_prefix_length;

				if (ctx_discard_size < discard_size) {
					chosen_context = itr;
					discard_size = ctx_discard_size;
				}
			}

			if (chosen_context == NULL || discard_size > 0) {
				// Try to create a new context to reduce discard_size
				if (llmd_create_shared_physical_ctx(session, &chosen_context) != LLMD_OK) {
					// If we never found a context
					if (chosen_context == NULL) {
						return LLMD_ERR_OOM;
					}
					// Continue with the chosen context
				} else {
					discard_size = 0;
				}
			}

			chosen_context->virtual_ctx = virtual_ctx;
			virtual_ctx->physical_ctx = chosen_context;

			*eval_offset_out = chosen_context->filled_size - discard_size;

			return LLMD_OK;
		}
		default:
			return LLMD_ERR_NOT_SUPPORTED;
	}
}

static enum llmd_error
llmd_unbind_virtual_ctx(
	struct llmd_session* session,
	struct llmd_context* virtual_ctx
) {
	(void)session;
	struct llmd_physical_context* physical_ctx = virtual_ctx->physical_ctx;

	// TODO: barrier?
	if (physical_ctx) {
		physical_ctx->virtual_ctx = NULL;
		virtual_ctx->physical_ctx = NULL;
	}

	return LLMD_OK;
}

enum llmd_error
llmd_create_session(
	struct llmd_host* host,
	struct llmd_driver* driver,
	struct llmd_session** session_out
) {
LLMD_TRY
	if (host == NULL) {
		host = &llmd_default_host;
	}

	struct llmd_session* session = NULL;
	LLMD_CHECKED_MALLOC(session, host, sizeof(struct llmd_session));

	*session = (struct llmd_session) {
		.host = host,
		.driver = driver
	};

	LLMD_CHECK_THROW(driver->interface->get_model_info(driver, &session->model_info));

	*session_out = session;
	return LLMD_OK;
LLMD_EXCEPT_BEGIN
	llmd_free(host, session);
LLMD_EXCEPT_END
}

enum llmd_error
llmd_destroy_session(
	struct llmd_session* session
) {
	for (
		struct llmd_physical_context* itr = session->free_contexts;
		itr != NULL;
		itr = itr->next
	) {
		llmd_destroy_physical_ctx(session, itr);
	}

	llmd_free(session->host, session);
	return LLMD_OK;
}

enum llmd_error
llmd_get_model_info(
	struct llmd_session* session,
	struct llmd_model_info* info_out
) {
	*info_out = session->model_info;
	return LLMD_OK;
}

enum llmd_error
llmd_create_context(
	struct llmd_session* session,
	enum llmd_context_type context_type,
	struct llmd_context** context_out
) {
LLMD_TRY
	struct llmd_host* host = session->host;

	struct llmd_context* context = NULL;
	float* logits = NULL;

	LLMD_CHECKED_MALLOC(context, host, sizeof(struct llmd_context));
	LLMD_CHECKED_MALLOC(logits, host, sizeof(*logits) * session->model_info.vocab_size);

	*context = (struct llmd_context) {
		.session = session,
		.type = context_type,
		.logits = logits,
	};

	if (context_type == LLMD_CONTEXT_DIRECT) {
		LLMD_CHECK_THROW(
			llmd_create_physical_ctx(session, &context->physical_ctx)
		);
	} else {
		LLMD_CHECKED_MALLOC(
			context->context_window, host,
			sizeof(llmd_token_t) * session->model_info.max_context_length
		);
	}

	*context_out = context;

	return LLMD_OK;
LLMD_EXCEPT_BEGIN
	llmd_free(host, logits);
	llmd_free(host, context);
LLMD_EXCEPT_END
}

enum llmd_error
llmd_destroy_context(
	struct llmd_context* context
) {
	struct llmd_session* session = context->session;
	struct llmd_host* host = session->host;

	if (context->generating) {
		llmd_log(host, LLMD_LOG_WARNING, "Context %p is still generating", (void*)context);
		llmd_unbind_virtual_ctx(session, context);
		context->generating = false;
	}

	if (context->type == LLMD_CONTEXT_DIRECT) {
		// Recycle the context
		// TODO: Lock
		struct llmd_session* session = context->session;
		struct llmd_physical_context* physical_context = context->physical_ctx;

		physical_context->context_window = llmd_malloc(
			host, sizeof(llmd_token_t) * session->model_info.max_context_length
		);

		if (!physical_context->context_window) {
			llmd_destroy_physical_ctx(session, physical_context);
		} else {
			physical_context->virtual_ctx = NULL;
			physical_context->next = session->free_contexts;
			session->free_contexts = physical_context;
		}
	}

	llmd_free(host, context->text_buf);
	llmd_free(host, context->token_buf);
	llmd_free(host, context->context_window);
	llmd_free(host, context->logits);
	llmd_free(host, context);

	return LLMD_OK;
}

enum llmd_error
llmd_tokenize(
	struct llmd_context* ctx,
	const char* string,
	unsigned int num_chars,
	const llmd_token_t** tokens_out,
	unsigned int* num_tokens_out
) {
	struct llmd_driver* driver = ctx->session->driver;
	struct llmd_host* host = ctx->session->host;

	unsigned int num_tokens = llmd_buffer_size(ctx->token_buf) / sizeof(llmd_token_t);
	enum llmd_error status = driver->interface->tokenize(
		driver,
		string, num_chars,
		(void*)ctx->token_buf->mem, &num_tokens
	);

	if (status == LLMD_ERR_BUF_SIZE) {
		ctx->token_buf = llmd_realloc_buffer(
			host, ctx->token_buf, num_tokens * sizeof(llmd_token_t)
		);
		if (ctx->token_buf == NULL) { return LLMD_ERR_OOM; }

		LLMD_CHECK_RETURN(
			driver->interface->tokenize(
				driver,
				string, num_chars,
				(void*)ctx->token_buf->mem, &num_tokens
			)
		);
	}

	if (tokens_out) {
		*tokens_out = (void*)ctx->token_buf->mem;
	}

	if (num_tokens_out) {
		*num_tokens_out = num_tokens;
	}

	return LLMD_OK;
}

enum llmd_error
llmd_decode_token(
	struct llmd_context* ctx,
	llmd_token_t token,
	const char** string_out,
	unsigned int* num_chars_out
) {
	struct llmd_driver* driver = ctx->session->driver;
	struct llmd_host* host = ctx->session->host;

	unsigned int existing_num_chars, num_chars;
   	existing_num_chars = num_chars = llmd_buffer_size(ctx->text_buf);
	enum llmd_error status = driver->interface->decode_token(
		driver,
		token,
		(void*)ctx->text_buf->mem, &num_chars
	);

	if (status == LLMD_ERR_BUF_SIZE || num_chars >= existing_num_chars) {
		// Need space for the null terminator
		ctx->text_buf = llmd_realloc_buffer(
			host, ctx->text_buf, (num_chars + 1) * sizeof(char)
		);
		if (ctx->text_buf == NULL) { return LLMD_ERR_OOM; }

		LLMD_CHECK_RETURN(
			driver->interface->decode_token(
				driver,
				token,
				(void*)ctx->text_buf->mem, &num_chars
			)
		);
	}
	ctx->text_buf->mem[num_chars] = 0;

	if (string_out) {
		*string_out = ctx->text_buf->mem;
	}

	if (num_chars_out) {
		*num_chars_out = num_chars;
	}

	return LLMD_OK;
}

enum llmd_error
llmd_begin_generate(
	struct llmd_context* ctx,
	struct llmd_generate_handle** generate_handle_out
) {
	struct llmd_session* session = ctx->session;
	struct llmd_host* host = session->host;

	if (ctx->generating) {
		llmd_log(host, LLMD_LOG_ERROR, "Context %p is already generating", (void*)ctx);
		return LLMD_ERR_INVALID;
	}

	ctx->generating = true;
	*generate_handle_out = (void*)ctx;

	return LLMD_OK;
}

enum llmd_error
llmd_generate_next(
	struct llmd_generate_handle* generate_handle,
	const llmd_token_t* tokens,
	unsigned int num_tokens,
	unsigned int offset,
	const float** logits_out
) {
	struct llmd_context* ctx = (struct llmd_context*)generate_handle;
	struct llmd_session* session = ctx->session;
	struct llmd_host* host = session->host;
	struct llmd_driver* driver = session->driver;

	if (!ctx->generating) {
		llmd_log(host, LLMD_LOG_ERROR, "Context %p is not generating", (void*)ctx);
		return LLMD_ERR_INVALID;
	}

	if (offset + num_tokens > session->model_info.max_context_length) {
		llmd_log(host, LLMD_LOG_ERROR, "Context %p will overflow", (void*)ctx);
		return LLMD_ERR_INVALID;
	}

	if (ctx->type == LLMD_CONTEXT_DIRECT) {
		LLMD_CHECK_RETURN(
			driver->interface->generate(
				driver,
				ctx->physical_ctx->descriptor,
				tokens, num_tokens, offset,
				ctx->logits
			)
		);
	} else {
		unsigned int eval_offset;
		unsigned int eval_len;
		if (ctx->physical_ctx == NULL) {
			memcpy(ctx->context_window + offset, tokens, num_tokens * sizeof(llmd_token_t));

			LLMD_CHECK_RETURN(
				llmd_bind_virtual_ctx(session, ctx, offset + num_tokens, &eval_offset)
			);

			eval_len = offset + num_tokens - eval_offset;

			memcpy(
				ctx->physical_ctx->context_window + eval_offset,
				ctx->context_window + eval_offset,
				eval_len * sizeof(llmd_token_t)
			);
		} else {
			// In multi-client system, setting offset at the end can be used to
			// extract existing data left by other clients.
			if (offset > ctx->physical_ctx->filled_size) {
				return LLMD_ERR_INVALID;
			}

			eval_offset = offset;
			eval_len = num_tokens;

			memcpy(
				ctx->physical_ctx->context_window + eval_offset,
				tokens,
				eval_len * sizeof(llmd_token_t)
			);
		}

		ctx->physical_ctx->filled_size = offset + num_tokens;

		LLMD_CHECK_RETURN(
			driver->interface->generate(
				driver,
				ctx->physical_ctx->descriptor,
				ctx->physical_ctx->context_window + eval_offset,
				eval_len, eval_offset,
				ctx->logits
			)
		);
	}

	if (logits_out) {
		*logits_out = ctx->logits;
	}

	return LLMD_OK;
}

enum llmd_error
llmd_end_generate(
	struct llmd_generate_handle* generate_handle
) {
	struct llmd_context* ctx = (struct llmd_context*)generate_handle;
	struct llmd_session* session = ctx->session;
	struct llmd_host* host = session->host;

	if (!ctx->generating) {
		llmd_log(host, LLMD_LOG_ERROR, "Context %p is not generating", (void*)ctx);
		return LLMD_ERR_INVALID;
	}

	if (ctx->physical_ctx->context_window) {
		// Copy back context window
		memcpy(
			ctx->context_window,
			ctx->physical_ctx->context_window,
			ctx->physical_ctx->filled_size * sizeof(llmd_token_t)
		);
	}

	ctx->generating = false;
	return llmd_unbind_virtual_ctx(session, ctx);
}
