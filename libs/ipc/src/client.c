#include <llmd/ipc/client.h>
#include "common.h"
#include <llmd/core.h>
#include <llmd/utils/host.h>
#include <stdint.h>

struct llmd_ipc_context {
	int descriptor;

	struct llmd_span context_window;
	struct llmd_span logits;
};

struct llmd_ipc_client {
	struct llmd_driver header;
	struct llmd_host* host;
	struct llmd_ipc_client_config* config;

	int ipc_sock;

	struct llmd_span session_mem;
	unsigned int num_contexts;
	struct llmd_ipc_context* contexts;
};

static enum llmd_error
llmd_ipc_begin_call(
	struct llmd_ipc_client* client,
	enum llmd_rpc_method method,
	struct llmd_rpc_buf* call_out
) {
	(void)client;
	call_out->buf_size = sizeof(call_out->tmp_buf);
	call_out->buf_cursor = 0;

	struct llmd_rpc_request_header header = {
		.version = LLMD_RPC_VERSION,
		.method = method,
	};

	return llmd_rpc_write(call_out, &header, sizeof(header));
}

static enum llmd_error
llmd_ipc_end_call(
	struct llmd_ipc_client* client,
	struct llmd_rpc_buf* call,
	unsigned int num_fds,
	int* fds_out
) {
	enum llmd_error status;
	if (num_fds > LLMD_MAX_NUM_FDS) {
		return LLMD_ERR_INVALID;
	}

	struct iovec iov = {
		.iov_base = call->tmp_buf,
		.iov_len = call->buf_cursor
	};
	struct msghdr msg = {
		.msg_iov = &iov,
		.msg_iovlen = 1,
	};
	LLMD_SYSCALL_CHECK(client->host, sendmsg(client->ipc_sock, &msg, 0));

	union {
		char buf[CMSG_SPACE(sizeof(int) * LLMD_MAX_NUM_FDS)];
		struct cmsghdr align;
	} cmsg_u;

	if (num_fds > 0) {
		msg.msg_control = cmsg_u.buf;
		msg.msg_controllen = CMSG_SPACE(sizeof(int) * num_fds);;
	}

	iov.iov_len = sizeof(call->tmp_buf);
	LLMD_SYSCALL_CHECK(client->host, recvmsg(client->ipc_sock, &msg, 0));

	if (num_fds > 0) {
		struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
		if (cmsg == NULL
			|| cmsg->cmsg_level != SOL_SOCKET
			|| cmsg->cmsg_type != SCM_RIGHTS
			|| cmsg->cmsg_len != CMSG_LEN(sizeof(int) * num_fds)
		) {
			return LLMD_ERR_IO;
		}

		memcpy(fds_out, CMSG_DATA(cmsg), num_fds * sizeof(int));
	}

	call->buf_cursor = 0;
	call->buf_size = iov.iov_len;

	struct llmd_rpc_response_header header;
	LLMD_CHECK(llmd_rpc_read(call, &header, sizeof(header)));

	return header.error;
}

static enum llmd_error
llmd_init_ipc_client(
	struct llmd_ipc_client* client
) {
	enum llmd_error status;

	LLMD_SYSCALL_CHECK(
		client->host,
		client->ipc_sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0)
	);

	struct sockaddr_un address = {
		.sun_family = AF_UNIX,
	};
	size_t name_len = strlen(client->config->name);
	if (name_len > sizeof(address.sun_path) - 2) {
		return LLMD_ERR_INVALID;
	}
	address.sun_path[0] = '\0';
	memcpy(address.sun_path + 1, client->config->name, name_len);
	address.sun_path[name_len + 1] = '\0';
	LLMD_SYSCALL_CHECK(
		client->host,
		connect(client->ipc_sock, (struct sockaddr *)&address, sizeof(address))
	);

	struct llmd_rpc_buf call;
	LLMD_CHECK(llmd_ipc_begin_call(client, LLMD_RPC_BEGIN_SESSION, &call));
	int fd = -1;
	LLMD_CHECK(llmd_ipc_end_call(client, &call, 1, &fd));

	status = llmd_ipc_setup_shared_mem(client->host, &client->session_mem, fd, PROT_READ | PROT_WRITE);
	close(fd);

	return status;
}

static enum llmd_error
llmd_ipc_client_tokenize(
	struct llmd_driver* header,
	const char* string,
	unsigned int num_chars,
	llmd_token_t* tokens_out,
	unsigned int* num_tokens_inout
) {
	enum llmd_error status;
	struct llmd_ipc_client* client = (struct llmd_ipc_client*)header;

	if (num_chars > client->session_mem.size) {
		return LLMD_ERR_INVALID;
	}

	memcpy(client->session_mem.ptr, string, num_chars);

	struct llmd_rpc_buf call;
	LLMD_CHECK(llmd_ipc_begin_call(client, LLMD_RPC_TOKENIZE, &call));
	LLMD_CHECK(llmd_rpc_write(&call, &num_chars, sizeof(num_chars)));
	LLMD_CHECK(llmd_ipc_end_call(client, &call, 0, NULL));

	unsigned int num_tokens;
	LLMD_CHECK(llmd_rpc_read(&call, &num_tokens, sizeof(num_tokens)));
	if (num_tokens > *num_tokens_inout) {
		*num_tokens_inout = num_tokens;
		return LLMD_ERR_BUF_SIZE;
	} else {
		memcpy(tokens_out, client->session_mem.ptr, sizeof(llmd_token_t) * num_tokens);
		*num_tokens_inout = num_tokens;
		return LLMD_OK;
	}
}

static enum llmd_error
llmd_ipc_client_get_model_info(
	struct llmd_driver* header,
	struct llmd_model_info* info_out
) {
	enum llmd_error status;
	struct llmd_ipc_client* client = (struct llmd_ipc_client*)header;

	struct llmd_rpc_buf call;
	LLMD_CHECK(llmd_ipc_begin_call(client, LLMD_RPC_GET_MODEL_INFO, &call));
	LLMD_CHECK(llmd_ipc_end_call(client, &call, 0, NULL));
	LLMD_CHECK(llmd_rpc_read(&call, info_out, sizeof(*info_out)));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_client_decode_token(
	struct llmd_driver* header,
	llmd_token_t token,
	char* string_out,
	unsigned int* num_chars_inout
) {
	enum llmd_error status;
	struct llmd_ipc_client* client = (struct llmd_ipc_client*)header;

	struct llmd_rpc_buf call;
	LLMD_CHECK(llmd_ipc_begin_call(client, LLMD_RPC_DECODE_TOKEN, &call));
	LLMD_CHECK(llmd_rpc_write(&call, &token, sizeof(token)));
	LLMD_CHECK(llmd_ipc_end_call(client, &call, 0, NULL));

	unsigned int num_chars;
	LLMD_CHECK(llmd_rpc_read(&call, &num_chars, sizeof(num_chars)));

	if (num_chars > *num_chars_inout) {
		*num_chars_inout = num_chars;
		return LLMD_ERR_BUF_SIZE;
	} else {
		memcpy(string_out, client->session_mem.ptr, num_chars);
		*num_chars_inout = num_chars;
		return LLMD_OK;
	}

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_client_alloc_context(
	struct llmd_ipc_client* client,
	struct llmd_ipc_context** ctx_out
) {
	for (unsigned int i = 0; i < client->num_contexts; ++i) {
		if (client->contexts[i].descriptor == -1) {
			*ctx_out = &client->contexts[i];
			return LLMD_OK;
		}
	}

	unsigned int num_contexts = client->num_contexts > 1 ? client->num_contexts : 1;
	struct llmd_ipc_context* new_contexts = llmd_realloc(
		client->host,
		client->contexts,
		num_contexts * 2 * sizeof(struct llmd_ipc_context)
	);
	if (new_contexts == NULL) {
		return LLMD_ERR_OOM;
	}

	for (unsigned int i = client->num_contexts; i < num_contexts * 2; ++i) {
		new_contexts[i] = (struct llmd_ipc_context) {
			.descriptor = -1,
		};
	}

	*ctx_out = &new_contexts[client->num_contexts];
	client->contexts = new_contexts;
	client->num_contexts = num_contexts * 2;

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_client_create_context(
	struct llmd_driver* header,
	int* descriptor_out
) {
	enum llmd_error status;
	struct llmd_ipc_client* client = (struct llmd_ipc_client*)header;

	struct llmd_ipc_context* context;
	LLMD_CHECK(llmd_ipc_client_alloc_context(client, &context));

	int context_fds[2];
	struct llmd_rpc_buf call;
	LLMD_CHECK(llmd_ipc_begin_call(client, LLMD_RPC_CREATE_CONTEXT, &call));
	LLMD_CHECK(llmd_ipc_end_call(client, &call, 2, context_fds));
	LLMD_CHECK(llmd_rpc_read(&call, &context->descriptor, sizeof(context->descriptor)));

	status = llmd_ipc_setup_shared_mem(client->host, &context->context_window, context_fds[0], PROT_READ | PROT_WRITE);
	enum llmd_error status2 = llmd_ipc_setup_shared_mem(client->host, &context->logits, context_fds[1], PROT_READ);
	close(context_fds[0]);
	close(context_fds[1]);

	if (status != LLMD_OK || status2 != LLMD_OK) {
		llmd_ipc_cleanup_shared_mem(&context->context_window);
		llmd_ipc_cleanup_shared_mem(&context->logits);
		context->descriptor = -1;

		return LLMD_ERR_IO;
	}

	*descriptor_out = context - client->contexts;
	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_client_destroy_context(
	struct llmd_driver* header,
	int descriptor
) {
	enum llmd_error status;
	struct llmd_ipc_client* client = (struct llmd_ipc_client*)header;

	if ((unsigned int)descriptor >= client->num_contexts) {
		return LLMD_ERR_INVALID;
	}

	struct llmd_ipc_context* context = &client->contexts[descriptor];
	if (context->descriptor == -1) {
		return LLMD_ERR_INVALID;
	}

	llmd_ipc_cleanup_shared_mem(&context->context_window);
	llmd_ipc_cleanup_shared_mem(&context->logits);

	struct llmd_rpc_buf call;
	LLMD_CHECK(llmd_ipc_begin_call(client, LLMD_RPC_DESTROY_CONTEXT, &call));
	LLMD_CHECK(llmd_rpc_write(&call, &context->descriptor, sizeof(context->descriptor)));
	LLMD_CHECK(llmd_ipc_end_call(client, &call, 0, NULL));

	context->descriptor = -1;

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_client_generate(
	struct llmd_driver* header,
	int context_descriptor,
	const llmd_token_t* tokens,
	unsigned int num_tokens,
	unsigned int offset,
	float* logits_out
) {
	enum llmd_error status;
	struct llmd_ipc_client* client = (struct llmd_ipc_client*)header;

	if ((unsigned int)context_descriptor >= client->num_contexts) {
		return LLMD_ERR_INVALID;
	}

	struct llmd_ipc_context* context = &client->contexts[context_descriptor];
	if (context->descriptor == -1) {
		return LLMD_ERR_INVALID;
	}

	if ((offset + num_tokens) > (context->context_window.size / sizeof(llmd_token_t))) {
		return LLMD_ERR_INVALID;
	}

	memcpy((llmd_token_t*)context->context_window.ptr + offset, tokens, num_tokens * sizeof(llmd_token_t));

	struct llmd_rpc_buf call;
	LLMD_CHECK(llmd_ipc_begin_call(client, LLMD_RPC_GENERATE, &call));
	LLMD_CHECK(llmd_rpc_write(&call, &context->descriptor, sizeof(context->descriptor)));
	LLMD_CHECK(llmd_rpc_write(&call, &num_tokens, sizeof(num_tokens)));
	LLMD_CHECK(llmd_rpc_write(&call, &offset, sizeof(offset)));
	LLMD_CHECK(llmd_ipc_end_call(client, &call, 0, NULL));

	memcpy(logits_out, context->logits.ptr, context->logits.size);

	return LLMD_OK;
}

static struct llmd_driver_interface llmd_ipc_client_interface = {
	.get_model_info = llmd_ipc_client_get_model_info,
	.tokenize = llmd_ipc_client_tokenize,
	.decode_token = llmd_ipc_client_decode_token,
	.create_context = llmd_ipc_client_create_context,
	.destroy_context = llmd_ipc_client_destroy_context,
	.generate = llmd_ipc_client_generate,
};

enum llmd_error
llmd_create_ipc_client(
	struct llmd_host* host,
	struct llmd_ipc_client_config* config,
	struct llmd_driver** driver_out
) {
	if (host == NULL) {
		host = &llmd_default_host;
	}

	struct llmd_ipc_client* client = llmd_malloc(
		host,
		sizeof(struct llmd_ipc_client)
	);
	if (client == NULL) {
		return LLMD_ERR_OOM;
	}

	*client = (struct llmd_ipc_client) {
		.header = {
			.interface = &llmd_ipc_client_interface,
		},
		.host = host,
		.config = config,
		.ipc_sock = -1,
	};

	enum llmd_error status = llmd_init_ipc_client(client);
	if (status != LLMD_OK) {
		llmd_destroy_ipc_client(&client->header);
		return status;
	}

	*driver_out = &client->header;
	return LLMD_OK;
}

enum llmd_error
llmd_destroy_ipc_client(
	struct llmd_driver* header
) {
	struct llmd_ipc_client* client = (struct llmd_ipc_client*)header;

	if (client->ipc_sock != -1) {
		close(client->ipc_sock);
	}

	if (client->session_mem.ptr != NULL) {
		llmd_ipc_cleanup_shared_mem(&client->session_mem);
	}

	llmd_free(client->host, client);
	return LLMD_OK;
}

enum llmd_error
llmd_begin_create_driver(
	struct llmd_host* host,
	struct llmd_ipc_client_config** config_out
) {
	struct llmd_ipc_client_config* config = llmd_malloc(
		host,
		sizeof(struct llmd_ipc_client_config)
	);
	memset(config, 0, sizeof(*config));

	*config_out = config;
	return LLMD_OK;
}

enum llmd_error
llmd_set_driver_config(
	struct llmd_host* host,
	struct llmd_ipc_client_config* config,
	const char* section,
	const char* key,
	const char* value
) {
	if (strcmp(section, "main") == 0) {
		if (strcmp(key, "name") == 0) {
			size_t len = strlen(value);
			config->name = llmd_malloc(host, len + 1);
			if (!config->name) {
				return LLMD_ERR_OOM;
			}
			memcpy((void*)config->name, value, len);
			*(char*)(&config->name[len]) = '\0';
			return LLMD_OK;
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
	struct llmd_ipc_client_config* config,
	struct llmd_driver** driver_out
) {
	if (driver_out == NULL) {
		llmd_free(host, (void*)config->name);
		llmd_free(host, config);
		return LLMD_OK;
	} else {
		return llmd_create_ipc_client(host, config, driver_out);
	}
}

enum llmd_error
llmd_destroy_driver(
	struct llmd_driver* header
) {
	struct llmd_ipc_client* client = (struct llmd_ipc_client*)header;
	struct llmd_host* host = client->host;
	struct llmd_ipc_client_config* config = client->config;

	llmd_destroy_ipc_client(header);
	llmd_free(host, (void*)config->name);
	llmd_free(host, config);

	return LLMD_OK;
}
