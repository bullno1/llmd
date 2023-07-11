#ifndef LLMD_CORE_H
#define LLMD_CORE_H

#include <stddef.h>
#include <stdarg.h>

#ifdef LLMD_CORE_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_CORE_BUILD
#            define LLMD_CORE_API __declspec(dllexport)
#        else
#            define LLMD_CORE_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_CORE_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_CORE_API
#endif

enum llmd_error {
	LLMD_OK,
	LLMD_ERR_IO,
	LLMD_ERR_OOM,
	LLMD_ERR_BUF_SIZE,
	LLMD_ERR_INVALID,
	LLMD_ERR_NOT_SUPPORTED
};

enum llmd_context_type {
	LLMD_CONTEXT_DIRECT,
	LLMD_CONTEXT_MIN_UPLOAD,
	LLMD_CONTEXT_MIN_DISCARD,
};

enum llmd_log_level {
	LLMD_LOG_DEBUG,
	LLMD_LOG_INFO,
	LLMD_LOG_WARNING,
	LLMD_LOG_ERROR
};

struct llmd_host;

struct llmd_host_interface {
	void* (*realloc)(
		struct llmd_host* host,
		void* ptr,
		size_t size
	);

	void (*log)(
		struct llmd_host* host,
		enum llmd_log_level level,
		const char* format,
		va_list args
	);
};

struct llmd_host {
	struct llmd_host_interface* interface;
};

typedef unsigned int llmd_token_t;

struct llmd_model_info {
	llmd_token_t bos_token;
	llmd_token_t eos_token;
	llmd_token_t nl_token;

	unsigned int vocab_size;
	unsigned int max_context_length;
};

struct llmd_session;
struct llmd_context;
struct llmd_generate_handle;
struct llmd_driver;

struct llmd_driver_interface {
	enum llmd_error (*get_model_info)(
		struct llmd_driver* driver,
		struct llmd_model_info* info_out
	);

	enum llmd_error (*create_context)(
		struct llmd_driver* driver,
		int* context_descriptor_out
	);

	enum llmd_error (*destroy_context)(
		struct llmd_driver* driver,
		int context_descriptor
	);

	enum llmd_error (*generate)(
		struct llmd_driver* driver,
		int context_descriptor,
		const llmd_token_t* tokens,
		unsigned int num_tokens,
		unsigned int offset,
		float* logits_out
	);

	enum llmd_error (*tokenize)(
		struct llmd_driver* driver,
		const char* string,
		unsigned int num_chars,
		llmd_token_t* tokens_out,
		unsigned int* num_tokens_inout
	);

	enum llmd_error (*decode_token)(
		struct llmd_driver* driver,
		llmd_token_t token,
		char* string_out,
		unsigned int* num_chars_inout
	);
};

struct llmd_driver {
	struct llmd_driver_interface* interface;
};

#ifdef __cplusplus
extern "C" {
#endif

LLMD_CORE_API enum llmd_error
llmd_create_session(
	struct llmd_host* host,
	struct llmd_driver* driver,
	struct llmd_session** session_out
);

LLMD_CORE_API enum llmd_error
llmd_destroy_session(
	struct llmd_session* session
);

LLMD_CORE_API enum llmd_error
llmd_get_model_info(
	struct llmd_session* session,
	struct llmd_model_info* info_out
);

LLMD_CORE_API enum llmd_error
llmd_create_context(
	struct llmd_session* session,
	enum llmd_context_type context_type,
	struct llmd_context** context_out
);

LLMD_CORE_API enum llmd_error
llmd_destroy_context(
	struct llmd_context* context
);

LLMD_CORE_API enum llmd_error
llmd_tokenize(
	struct llmd_context* ctx,
	const char* string,
	unsigned int num_chars,
	const llmd_token_t** tokens_out,
	unsigned int* num_tokens_out
);

LLMD_CORE_API enum llmd_error
llmd_decode_token(
	struct llmd_context* ctx,
	llmd_token_t token,
	const char** str_out,
	unsigned int* num_chars_out
);

LLMD_CORE_API enum llmd_error
llmd_begin_generate(
	struct llmd_context* ctx,
	struct llmd_generate_handle** generate_handle_out
);

LLMD_CORE_API enum llmd_error
llmd_generate_next(
	struct llmd_generate_handle* generate_handle,
	const llmd_token_t* tokens,
	unsigned int num_tokens,
	unsigned int offset,
	const float** logits_out
);

LLMD_CORE_API enum llmd_error
llmd_end_generate(
	struct llmd_generate_handle* generate_handle
);

#ifdef __cplusplus
}
#endif

#endif
