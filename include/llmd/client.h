#ifndef LLMD_CLIENT_H
#define LLMD_CLIENT_H

#ifdef LLMD_CLIENT_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_CLIENT_BUILD
#            define LLMD_CLIENT_API __declspec(dllexport)
#        else
#            define LLMD_CLIENT_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_CLIENT_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_CLIENT_API
#endif

#include <llmd/core.h>

struct llmd_r_session;
struct llmd_r_context;
struct llmd_r_generate_handle;

#ifdef __cplusplus
extern "C" {
#endif

LLMD_CLIENT_API enum llmd_error
llmd_r_create_session(
	struct llmd_host* host,
	struct llmd_driver* driver,
	struct llmd_r_session** session_out
);

LLMD_CLIENT_API enum llmd_error
llmd_r_destroy_session(
	struct llmd_r_session* session
);

LLMD_CLIENT_API enum llmd_error
llmd_r_get_model_info(
	struct llmd_r_session* session,
	struct llmd_model_info* info_out
);

LLMD_CLIENT_API enum llmd_error
llmd_r_create_context(
	struct llmd_r_session* session,
	enum llmd_context_type context_type,
	struct llmd_r_context** context_out
);

LLMD_CLIENT_API enum llmd_error
llmd_r_destroy_context(
	struct llmd_r_context* context
);

LLMD_CLIENT_API enum llmd_error
llmd_r_tokenize(
	struct llmd_r_context* ctx,
	const char* string,
	unsigned int num_chars,
	const llmd_token_t** tokens_out,
	unsigned int* num_tokens_out
);

LLMD_CLIENT_API enum llmd_error
llmd_r_decode_token(
	struct llmd_r_context* ctx,
	llmd_token_t token,
	const char** str_out,
	unsigned int* num_chars_out
);

LLMD_CLIENT_API enum llmd_error
llmd_r_begin_generate(
	struct llmd_r_context* ctx,
	struct llmd_r_generate_handle** generate_handle_out
);

LLMD_CLIENT_API enum llmd_error
llmd_r_generate_next(
	struct llmd_r_generate_handle* generate_handle,
	const llmd_token_t* tokens,
	unsigned int num_tokens,
	unsigned int offset,
	const float** logits_out
);

LLMD_CLIENT_API enum llmd_error
llmd_r_end_generate(
	struct llmd_r_generate_handle* generate_handle
);

#ifdef __cplusplus
}
#endif

#endif
