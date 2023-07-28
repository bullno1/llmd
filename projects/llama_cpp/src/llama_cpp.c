#include <llmd/llama_cpp.h>
#include <llmd/core.h>
#include <llmd/utils/host.h>
#include <llmd/utils/buffer.h>
#include <llmd/utils/cfg.h>
#include <string.h>
#include <errno.h>
#include <llama.h>
#include <limits.h>

struct llmd_llama_cpp_driver {
	struct llmd_driver header;
	struct llmd_host* host;
	struct llmd_llama_cpp_driver_config* config;

	struct llmd_buffer* tmp_str_buf;
	struct llama_model* model;
	struct llama_context* contexts[];
};

static enum llmd_error
llmd_llama_cpp_get_model_info(
	struct llmd_driver* header,
	struct llmd_model_info* info_out
) {
	struct llmd_llama_cpp_driver* driver = (struct llmd_llama_cpp_driver*)header;

	info_out->bos_token = llama_token_bos();
	info_out->eos_token = llama_token_eos();
	info_out->nl_token = llama_token_nl();
	info_out->max_context_length = llama_n_ctx_from_model(driver->model);
	info_out->vocab_size = llama_n_vocab_from_model(driver->model);

	return LLMD_OK;
}

static enum llmd_error
llmd_llama_cpp_create_context(
	struct llmd_driver* header,
	int* descriptor_out
) {
	struct llmd_llama_cpp_driver* driver = (struct llmd_llama_cpp_driver*)header;

	for (unsigned int i = 0; i < driver->config->max_contexts; ++i) {
		if (driver->contexts[i] != NULL) {
			continue;
		}

		struct llama_context* ctx = llama_new_context_with_model(
			driver->model,
			driver->config->context_params
		);

		if (ctx == NULL) {
			return LLMD_ERR_IO;
		}

		driver->contexts[i] = ctx;
		*descriptor_out = i;
		return LLMD_OK;
	}

	return LLMD_ERR_OOM;
}

static enum llmd_error
llmd_llama_cpp_destroy_context(
	struct llmd_driver* header,
	int descriptor
) {
	struct llmd_llama_cpp_driver* driver = (struct llmd_llama_cpp_driver*)header;

	if (descriptor < 0 || descriptor > (int)driver->config->max_contexts) {
		return LLMD_ERR_INVALID;
	}

	struct llama_context* ctx = driver->contexts[descriptor];
	if (ctx == NULL) {
		return LLMD_ERR_INVALID;
	}

	llama_free(ctx);
	driver->contexts[descriptor] = NULL;

	return LLMD_OK;
}

static enum llmd_error
llmd_llama_cpp_tokenize(
	struct llmd_driver* header,
	const char* string,
	unsigned int num_chars,
	llmd_token_t* tokens_out,
	unsigned int* num_tokens_inout
) {
	struct llmd_llama_cpp_driver* driver = (struct llmd_llama_cpp_driver*)header;

	// Copy because we support non-null terminated string while llama.cpp
	// requires it.
	if (num_chars >= llmd_buffer_size(driver->tmp_str_buf)) {
		driver->tmp_str_buf = llmd_realloc_buffer(driver->host, driver->tmp_str_buf, num_chars + 1);
	}
	memcpy(driver->tmp_str_buf->mem, string, num_chars);
	driver->tmp_str_buf->mem[num_chars] = '\0';

	int num_tokens = llama_tokenize_with_model(
		driver->model, driver->tmp_str_buf->mem, (int*)tokens_out, *num_tokens_inout, false
	);

	if (num_tokens < 0) {
		*num_tokens_inout = -num_tokens;
		return LLMD_ERR_BUF_SIZE;
	} else {
		*num_tokens_inout = num_tokens;
		return LLMD_OK;
	}
}

static enum llmd_error
llmd_llama_cpp_decode_token(
	struct llmd_driver* header,
	llmd_token_t token,
	char* string_out,
	unsigned int* num_chars_inout
) {
	struct llmd_llama_cpp_driver* driver = (struct llmd_llama_cpp_driver*)header;

	const char* str = llama_token_to_str_with_model(driver->model, token);

	size_t len = strlen(str);
	if (len > *num_chars_inout) {
		*num_chars_inout = len;
		return LLMD_ERR_BUF_SIZE;
	} else {
		memcpy(string_out, str, len);
		*num_chars_inout = len;
		return LLMD_OK;
	}
}

static enum llmd_error
llmd_llama_cpp_generate(
	struct llmd_driver* header,
	int context_descriptor,
	const llmd_token_t* tokens,
	unsigned int num_tokens,
	unsigned int offset,
	float* logits_out
) {
	struct llmd_llama_cpp_driver* driver = (struct llmd_llama_cpp_driver*)header;

	if (context_descriptor < 0 || context_descriptor > (int)driver->config->max_contexts) {
		return LLMD_ERR_INVALID;
	}

	struct llama_context* ctx = driver->contexts[context_descriptor];
	if (ctx == NULL) {
		return LLMD_ERR_INVALID;
	}

	if (llama_eval(ctx, (const llama_token*)tokens, num_tokens, offset, 1)) {
		return LLMD_ERR_IO;
	}

	if (logits_out) {
		memcpy(
			logits_out,
			llama_get_logits(ctx),
			llama_n_vocab_from_model(driver->model) * sizeof(float)
		);
	}

	return LLMD_OK;
}

static struct llmd_driver_interface llmd_llama_cpp_driver_interface = {
	.create_context = llmd_llama_cpp_create_context,
	.destroy_context = llmd_llama_cpp_destroy_context,
	.get_model_info = llmd_llama_cpp_get_model_info,
	.tokenize = llmd_llama_cpp_tokenize,
	.decode_token = llmd_llama_cpp_decode_token,
	.generate = llmd_llama_cpp_generate,
};

enum llmd_error
llmd_create_llama_cpp_driver(
	struct llmd_host* host,
	struct llmd_llama_cpp_driver_config* config,
	struct llmd_driver** driver_out
) {
	if (host == NULL) {
		host = &llmd_default_host;
	}

	struct llmd_llama_cpp_driver* driver = llmd_malloc(
		host,
		sizeof(struct llmd_llama_cpp_driver) +
		sizeof(struct llmd_context*) * config->max_contexts
	);
	if (driver == NULL) {
		return LLMD_ERR_OOM;
	}

	struct llama_model* model = llama_load_model_from_file(
		config->model_path,
		config->context_params
	);

	if (model == NULL) {
		llmd_log(host, LLMD_LOG_ERROR, "Could not load model");
		llmd_free(host, driver);
		return LLMD_ERR_IO;
	}

	*driver = (struct llmd_llama_cpp_driver) {
		.header = {
			.interface = &llmd_llama_cpp_driver_interface,
		},
		.host = host,
		.config = config,
		.model = model,
	};

	for (unsigned int i = 0; i < config->max_contexts; ++i) {
		driver->contexts[i] = NULL;
	}

	*driver_out = &driver->header;

	return LLMD_OK;
}

enum llmd_error
llmd_destroy_llama_cpp_driver(
	struct llmd_driver* header
) {
	struct llmd_llama_cpp_driver* driver = (struct llmd_llama_cpp_driver*)header;

	for (unsigned int i = 0; i < driver->config->max_contexts; ++i) {
		struct llama_context* ctx = driver->contexts[i];
		if (ctx != NULL) {
			llama_free(ctx);
		}
	}

	llmd_free(driver->host, driver->tmp_str_buf);
	llama_free_model(driver->model);
	llmd_free(driver->host, driver);

	return LLMD_OK;
}

enum llmd_error
llmd_begin_create_driver(
	struct llmd_host* host,
	struct llmd_llama_cpp_driver_config** config_out
) {
	struct llmd_llama_cpp_driver_config* config = llmd_malloc(
		host,
		sizeof(struct llmd_llama_cpp_driver_config)
	);
	memset(config, 0, sizeof(*config));
	config->context_params = llama_context_default_params();

	*config_out = config;
	return LLMD_OK;
}

enum llmd_error
llmd_set_driver_config(
	struct llmd_host* host,
	struct llmd_llama_cpp_driver_config* config,
	const char* section,
	const char* key,
	const char* value
) {
	if (strcmp(section, "main") == 0) {
		if (strcmp(key, "model_path") == 0) {
			size_t len = strlen(value);
			config->model_path = llmd_malloc(host, len + 1);
			if (!config->model_path) {
				return LLMD_ERR_OOM;
			}
			memcpy((void*)config->model_path, value, len);
			*(char*)(&config->model_path[len]) = '\0';
			return LLMD_OK;
		} else if (strcmp(key, "max_contexts") == 0){
			return llmd_cfg_parse_uint(value, 0, INT_MAX, &config->max_contexts);
		} else {
			return LLMD_ERR_INVALID;
		}
	} else if (strcmp(section, "llama_cpp") == 0) {
		if (strcmp(key, "n_ctx") == 0) {
			return llmd_cfg_parse_int(value, 0, INT_MAX, &config->context_params.n_ctx);
		} else if (strcmp(key, "n_threads") == 0) {
			return llmd_cfg_parse_uint(value, 0, INT_MAX, &config->num_threads);
		} else if (strcmp(key, "n_gpu_layers") == 0) {
			return llmd_cfg_parse_int(value, 0, INT_MAX, &config->context_params.n_gpu_layers);
		} else if (strcmp(key, "main_gpu") == 0) {
			return llmd_cfg_parse_int(value, 0, INT_MAX, &config->context_params.main_gpu);
		} else if (strcmp(key, "low_vram") == 0) {
			return llmd_cfg_parse_bool(value, &config->context_params.low_vram);
		} else if (strcmp(key, "f16_kv") == 0) {
			return llmd_cfg_parse_bool(value, &config->context_params.f16_kv);
		} else if (strcmp(key, "use_mmap") == 0) {
			return llmd_cfg_parse_bool(value, &config->context_params.use_mmap);
		} else if (strcmp(key, "use_mlock") == 0) {
			return llmd_cfg_parse_bool(value, &config->context_params.use_mlock);
		} else {
			return LLMD_ERR_INVALID;
		}
	} else if (strcmp(section, "llmd") == 0) {
		return LLMD_OK;
	} else {
		return LLMD_ERR_INVALID;
	}
}

enum llmd_error
llmd_end_create_driver(
	struct llmd_host* host,
	struct llmd_llama_cpp_driver_config* config,
	struct llmd_driver** driver_out
) {
	if (driver_out == NULL) {
		llmd_free(host, (void*)config->model_path);
		llmd_free(host, config);
		return LLMD_OK;
	} else {
		return llmd_create_llama_cpp_driver(
			host,
			config,
			driver_out
		);
	}
}


enum llmd_error
llmd_destroy_driver(
	struct llmd_driver* header
) {
	struct llmd_llama_cpp_driver* driver = (struct llmd_llama_cpp_driver*)header;
	struct llmd_host* host = driver->host;
	struct llmd_llama_cpp_driver_config* config = driver->config;

	llmd_destroy_llama_cpp_driver(header);
	llmd_free(host, (void*)config->model_path);
	llmd_free(host, config);

	return LLMD_OK;
}
