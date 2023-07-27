#define _GNU_SOURCE
#include <llmd/ipc/server.h>
#include "common.h"
#include <llmd/utils/host.h>
#include <stdbool.h>
#ifdef __linux__
#include <poll.h>
#include <fcntl.h>
#endif

struct llmd_ipc_session {
	int ipc_sock;
	struct pollfd* pollfd;

	struct llmd_span shared_mem;
	struct llmd_rpc_buf buf;

	// for next message
	unsigned int num_fds;
	int fds[LLMD_MAX_NUM_FDS];

	char output_buf[];
};

struct llmd_ipc_client_context {
	int descriptor;

	struct llmd_span context_window;
	struct llmd_span logits;

	struct llmd_ipc_session* owner;
};

struct llmd_ipc_server {
	struct llmd_host* host;
	struct llmd_ipc_server_config* config;
	struct llmd_driver* driver;

	int ipc_sock;
	bool running;

	unsigned int num_sessions;
	struct pollfd* poll_set;
	struct llmd_ipc_session** sessions;

	unsigned int num_contexts;
	struct llmd_ipc_client_context* contexts;

	size_t input_mem_size;
	size_t output_mem_size;
	struct llmd_model_info model_info;
};

static enum llmd_error
llmd_init_ipc_server(
	struct llmd_ipc_server* server
) {
	enum llmd_error status;
	LLMD_SYSCALL_CHECK(
		server->host,
		server->ipc_sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0)
	);
	struct sockaddr_un address = {
		.sun_family = AF_UNIX,
	};
	size_t name_len = strlen(server->config->name);
	if (name_len > sizeof(address.sun_path) - 2) {
		return LLMD_ERR_INVALID;
	}
	address.sun_path[0] = '\0';
	memcpy(address.sun_path + 1, server->config->name, name_len);
	address.sun_path[name_len + 1] = '\0';
	LLMD_SYSCALL_CHECK(
		server->host,
		bind(server->ipc_sock, (struct sockaddr *)&address, sizeof(address))
	);
	LLMD_SYSCALL_CHECK(server->host, listen(server->ipc_sock, 16));

	server->poll_set = llmd_malloc(server->host, sizeof(struct pollfd));
	if (server->poll_set == NULL) {
		return LLMD_ERR_OOM;
	}
	server->poll_set[0].fd = server->ipc_sock;
	server->poll_set[0].events = POLLIN;

	LLMD_CHECK(server->driver->interface->get_model_info(
		server->driver,
		&server->model_info
	));

	// Find the longest decoded token
	unsigned int max_token_len = 0;
	for (llmd_token_t token = 0; token < server->model_info.vocab_size; ++token) {
		unsigned int num_chars = 0;
		status = server->driver->interface->decode_token(
			server->driver,
			token,
			NULL,
			&num_chars
		);

		if (status != LLMD_OK && status != LLMD_ERR_BUF_SIZE) {
			return status;
		}

		if (num_chars > max_token_len) {
			max_token_len = num_chars;
		}
	}

	llmd_log(server->host, LLMD_LOG_INFO, "Longest token: %u", max_token_len);
	server->input_mem_size = max_token_len * server->model_info.max_context_length;
	server->output_mem_size =  sizeof(llmd_token_t) * server->model_info.max_context_length;
	llmd_log(server->host, LLMD_LOG_INFO, "Input memory size: %zu", server->input_mem_size);
	llmd_log(server->host, LLMD_LOG_INFO, "Output memory size: %zu", server->output_mem_size);

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_session_close(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	llmd_log(server->host, LLMD_LOG_INFO, "Session %p closed", (void*)session);
	for (unsigned int i = 0; i < server->num_contexts; ++i) {
		struct llmd_ipc_client_context* context = &server->contexts[i];
		if (context->owner != session) { continue; }

		server->driver->interface->destroy_context(server->driver, context->descriptor);

		llmd_ipc_cleanup_shared_mem(&context->context_window);
		llmd_ipc_cleanup_shared_mem(&context->logits);
		context->owner = NULL;
		context->descriptor = -1;
	}

	llmd_ipc_cleanup_shared_mem(&session->shared_mem);
	close(session->ipc_sock);

	for (unsigned int i = 0; i < session->num_fds; ++i) {
		close(session->fds[i]);
	}
	session->num_fds = 0;

	session->ipc_sock = -1;
	session->pollfd->fd = -1;
	session->pollfd->events = 0;

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_begin_response(
	struct llmd_ipc_session* session,
	enum llmd_error status
) {
	session->buf.buf_size = sizeof(session->buf.tmp_buf);
	session->buf.buf_cursor = 0;

	struct llmd_rpc_response_header header = {
		.error = status,
	};

	return llmd_rpc_write(&session->buf, &header, sizeof(header));
}

static enum llmd_error
llmd_ipc_end_response(
	struct llmd_ipc_session* session
) {
	(void)session;
	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_invalid_rpc(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	(void)server;
	enum llmd_error status;

	LLMD_CHECK(llmd_ipc_begin_response(session, LLMD_ERR_INVALID));
	LLMD_CHECK(llmd_ipc_end_response(session));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_begin_session(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	enum llmd_error status;
	if (session->shared_mem.ptr != NULL) {
		return llmd_ipc_handle_invalid_rpc(server, session);
	}

	LLMD_CHECK(llmd_ipc_begin_response(session, LLMD_OK));
	// Store into session so it will be closed on error
	session->num_fds = 1;
	int memfd;
	LLMD_SYSCALL_CHECK(
		server->host,
		session->fds[0] = memfd = memfd_create("session", MFD_CLOEXEC | MFD_ALLOW_SEALING)
	);
	LLMD_SYSCALL_CHECK(server->host, ftruncate(memfd, server->input_mem_size));
	LLMD_CHECK(llmd_ipc_setup_shared_mem(server->host, &session->shared_mem, memfd, PROT_READ | PROT_WRITE));
	LLMD_SYSCALL_CHECK(server->host, fcntl(memfd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_GROW | F_SEAL_SHRINK));

	LLMD_CHECK(llmd_ipc_end_response(session));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_get_model_info(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	enum llmd_error status;

	struct llmd_driver* driver = server->driver;
	struct llmd_model_info model_info = { 0 };
	status = driver->interface->get_model_info(driver, &model_info);

	LLMD_CHECK(llmd_ipc_begin_response(session, status));
	LLMD_CHECK(llmd_rpc_write(&session->buf, &model_info, sizeof(model_info)));
	LLMD_CHECK(llmd_ipc_end_response(session));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_tokenize(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	enum llmd_error status;

	struct llmd_driver* driver = server->driver;
	unsigned int num_chars = 0;
	LLMD_CHECK(llmd_rpc_read(&session->buf, &num_chars, sizeof(num_chars)));
	if (num_chars > server->input_mem_size) {
		return llmd_ipc_handle_invalid_rpc(server, session);
	}

	unsigned int num_tokens = server->output_mem_size;
	status = driver->interface->tokenize(
		driver,
		session->shared_mem.ptr,
		num_chars,
		(void*)session->output_buf,
		&num_tokens
	);
	memcpy(session->shared_mem.ptr, session->output_buf, num_tokens * sizeof(llmd_token_t));

	LLMD_CHECK(llmd_ipc_begin_response(session, status));
	LLMD_CHECK(llmd_rpc_write(&session->buf, &num_tokens, sizeof(num_tokens)));
	LLMD_CHECK(llmd_ipc_end_response(session));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_decode_token(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	enum llmd_error status;

	if (session->shared_mem.ptr == NULL) {
		return llmd_ipc_handle_invalid_rpc(server, session);
	}

	struct llmd_driver* driver = server->driver;
	llmd_token_t token = 0;
	LLMD_CHECK(llmd_rpc_read(&session->buf, &token, sizeof(token)));
	if (token >= server->model_info.vocab_size) {
		return llmd_ipc_handle_invalid_rpc(server, session);
	}

	unsigned int num_chars = server->input_mem_size;
	status = driver->interface->decode_token(
		driver, token, session->shared_mem.ptr, &num_chars
	);

	LLMD_CHECK(llmd_ipc_begin_response(session, status));
	LLMD_CHECK(llmd_rpc_write(&session->buf, &num_chars, sizeof(num_chars)));
	LLMD_CHECK(llmd_ipc_end_response(session));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_alloc_context(
	struct llmd_ipc_server* server,
	struct llmd_ipc_client_context** ctx_out
) {
	for (unsigned int i = 0; i < server->num_contexts; ++i) {
		if (server->contexts[i].descriptor == -1) {
			*ctx_out = &server->contexts[i];
			return LLMD_OK;
		}
	}

	unsigned int num_contexts = server->num_contexts > 1 ? server->num_contexts : 1;
	struct llmd_ipc_client_context* new_contexts = llmd_realloc(
		server->host,
		server->contexts,
		num_contexts * 2 * sizeof(struct llmd_ipc_client_context)
	);
	if (new_contexts == NULL) {
		return LLMD_ERR_OOM;
	}

	for (unsigned int i = server->num_contexts; i < num_contexts * 2; ++i) {
		new_contexts[i] = (struct llmd_ipc_client_context) {
			.descriptor = -1,
		};
	}

	*ctx_out = &new_contexts[server->num_contexts];
	server->contexts = new_contexts;
	server->num_contexts = num_contexts * 2;

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_create_context(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	enum llmd_error status;

	struct llmd_ipc_client_context* context;
	LLMD_CHECK(llmd_ipc_alloc_context(server, &context));
	int client_ctx_descriptor = context - server->contexts;

	struct llmd_driver* driver = server->driver;
	status = driver->interface->create_context(
		driver,
		&context->descriptor
	);

	if (status != LLMD_OK) {
		LLMD_CHECK(llmd_ipc_begin_response(session, status));
		LLMD_CHECK(llmd_ipc_end_response(session));
		return LLMD_OK;
	}

	context->owner = session;

	session->num_fds = 2;
	LLMD_SYSCALL_CHECK(
		server->host,
		session->fds[0] = memfd_create("context_window", MFD_CLOEXEC | MFD_ALLOW_SEALING)
	);
	LLMD_SYSCALL_CHECK(
		server->host,
		session->fds[1] = memfd_create("logits", MFD_CLOEXEC | MFD_ALLOW_SEALING)
	);

	LLMD_SYSCALL_CHECK(
		server->host,
		ftruncate(session->fds[0], sizeof(llmd_token_t) * server->model_info.max_context_length)
	);
	LLMD_CHECK(llmd_ipc_setup_shared_mem(server->host, &context->context_window, session->fds[0], PROT_READ));
	LLMD_SYSCALL_CHECK(server->host, fcntl(session->fds[0], F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_GROW | F_SEAL_SHRINK));

	LLMD_SYSCALL_CHECK(
		server->host,
		ftruncate(session->fds[1], sizeof(float) * server->model_info.vocab_size)
	);
	LLMD_CHECK(llmd_ipc_setup_shared_mem(server->host, &context->logits, session->fds[1], PROT_READ | PROT_WRITE));
	LLMD_SYSCALL_CHECK(server->host, fcntl(session->fds[1], F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_FUTURE_WRITE));

	LLMD_CHECK(llmd_ipc_begin_response(session, LLMD_OK));
	LLMD_CHECK(llmd_rpc_write(&session->buf, &client_ctx_descriptor, sizeof(client_ctx_descriptor)));
	LLMD_CHECK(llmd_ipc_end_response(session));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_destroy_context(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	enum llmd_error status;

	int descriptor;
	LLMD_CHECK(llmd_rpc_read(&session->buf, &descriptor, sizeof(descriptor)));

	if ((unsigned int)descriptor >= server->num_contexts) {
		return llmd_ipc_handle_invalid_rpc(server, session);
	}

	struct llmd_ipc_client_context* context = &server->contexts[descriptor];
	if (context->owner != session) {
		return llmd_ipc_handle_invalid_rpc(server, session);
	}

	struct llmd_driver* driver = server->driver;
	status = driver->interface->destroy_context(
		driver,
		context->descriptor
	);
	context->descriptor = -1;
	llmd_ipc_cleanup_shared_mem(&context->context_window);
	llmd_ipc_cleanup_shared_mem(&context->logits);
	context->owner = NULL;

	LLMD_CHECK(llmd_ipc_begin_response(session, status));
	LLMD_CHECK(llmd_ipc_end_response(session));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_generate(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	enum llmd_error status;

	int descriptor;
	unsigned int num_tokens, offset;
	LLMD_CHECK(llmd_rpc_read(&session->buf, &descriptor, sizeof(descriptor)));
	LLMD_CHECK(llmd_rpc_read(&session->buf, &num_tokens, sizeof(num_tokens)));
	LLMD_CHECK(llmd_rpc_read(&session->buf, &offset, sizeof(offset)));

	if ((unsigned int)descriptor >= server->num_contexts) {
		return llmd_ipc_handle_invalid_rpc(server, session);
	}

	struct llmd_ipc_client_context* context = &server->contexts[descriptor];
	if (context->owner != session) {
		return llmd_ipc_handle_invalid_rpc(server, session);
	}

	struct llmd_driver* driver = server->driver;
	status = driver->interface->generate(
		driver,
		context->descriptor,
		(llmd_token_t*)context->context_window.ptr + offset,
		num_tokens,
		offset,
		context->logits.ptr
	);

	LLMD_CHECK(llmd_ipc_begin_response(session, status));
	LLMD_CHECK(llmd_ipc_end_response(session));

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_session_readable(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	enum llmd_error status;
	struct iovec iov = {
		.iov_base = session->buf.tmp_buf,
		.iov_len = sizeof(session->buf.tmp_buf),
	};
	struct msghdr msg = {
		.msg_iov = &iov,
		.msg_iovlen = 1,
	};
	LLMD_SYSCALL_CHECK(server->host, recvmsg(session->ipc_sock, &msg, 0));

	session->buf.buf_cursor = 0;
	session->buf.buf_size = iov.iov_len;

	struct llmd_rpc_request_header request_header;
	LLMD_CHECK(llmd_rpc_read(&session->buf, &request_header, sizeof(request_header)));
	if (request_header.version != LLMD_RPC_VERSION) {
		llmd_log(server->host, LLMD_LOG_ERROR, "Session %p uses an unsupported protocol version", (void*)session);
		return llmd_ipc_handle_session_close(server, session);
	}

	switch ((enum llmd_rpc_method)request_header.method) {
		case LLMD_RPC_BEGIN_SESSION:
			status = llmd_ipc_handle_begin_session(server, session);
			break;
		case LLMD_RPC_CREATE_CONTEXT:
			status = llmd_ipc_handle_create_context(server, session);
			break;
		case LLMD_RPC_DESTROY_CONTEXT:
			status = llmd_ipc_handle_destroy_context(server, session);
			break;
		case LLMD_RPC_GENERATE:
			status = llmd_ipc_handle_generate(server, session);
			break;
		case LLMD_RPC_TOKENIZE:
			status = llmd_ipc_handle_tokenize(server, session);
			break;
		case LLMD_RPC_DECODE_TOKEN:
			status = llmd_ipc_handle_decode_token(server, session);
			break;
		case LLMD_RPC_GET_MODEL_INFO:
			status = llmd_ipc_handle_get_model_info(server, session);
			break;
		default:
			status = llmd_ipc_handle_invalid_rpc(server, session);
			break;
	}

	if (status != LLMD_OK) {
		return llmd_ipc_handle_session_close(server, session);
	}

	// Start polling for sending
	session->pollfd->events = POLLOUT | POLLERR | POLLHUP;

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_session_writeable(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session
) {
	struct iovec iov = {
		.iov_base = session->buf.tmp_buf,
		.iov_len = session->buf.buf_cursor,
	};
	struct msghdr msg = {
		.msg_iov = &iov,
		.msg_iovlen = 1,
	};
	union {
		char buf[CMSG_SPACE(sizeof(int) * LLMD_MAX_NUM_FDS)];
		struct cmsghdr align;
	} cmsg_u;

	if (session->num_fds > 0) {
		msg.msg_control = cmsg_u.buf;
		msg.msg_controllen = CMSG_SPACE(sizeof(int) * session->num_fds);

		struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
		cmsg->cmsg_level = SOL_SOCKET;
		cmsg->cmsg_type = SCM_RIGHTS;
		cmsg->cmsg_len = CMSG_LEN(sizeof(int) * session->num_fds);
		memcpy(CMSG_DATA(cmsg), session->fds, session->num_fds * sizeof(int));
	}

	if (sendmsg(session->ipc_sock, &msg, 0) < 0) {
		if (errno != EINTR) {
			llmd_log(server->host, LLMD_LOG_ERROR, "Error while seding message to %p: %s", (void*)session, strerror(errno));
			return llmd_ipc_handle_session_close(server, session);
		} else {
			return LLMD_OK;
		}
	}

	// Sent fds are no longer needed
	for (unsigned int i = 0; i < session->num_fds; ++i) {
		close(session->fds[i]);
	}
	session->num_fds = 0;

	// Go back to poll for input
	session->pollfd->events = POLLIN | POLLERR | POLLHUP;

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_handle_session(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session* session,
	short events
) {
	if ((events & (POLLERR | POLLHUP)) > 0) {
		return llmd_ipc_handle_session_close(server, session);
	} else if ((events & POLLIN) > 0) {
		return llmd_ipc_handle_session_readable(server, session);
	} else if ((events & POLLOUT) > 0) {
		return llmd_ipc_handle_session_writeable(server, session);
	}

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_alloc_session(
	struct llmd_ipc_server* server,
	struct llmd_ipc_session** session_out
) {
	for (unsigned int i = 0; i < server->num_sessions; ++i) {
		if (server->sessions[i]->ipc_sock == -1) {
			*session_out = server->sessions[i];
			return LLMD_OK;
		}
	}

	unsigned int num_sessions = server->num_sessions > 1 ? server->num_sessions : 1;

	struct pollfd* new_poll_set = llmd_realloc(
		server->host,
		server->poll_set,
		(num_sessions * 2 + 1) * sizeof(struct pollfd)
	);
	if (new_poll_set == NULL) {
		return LLMD_ERR_OOM;
	}

	for (unsigned int i = server->num_sessions + 1; i < (num_sessions * 2 + 1); ++i) {
		new_poll_set[i] = (struct pollfd) {
			.fd = -1,
		};
	}
	server->poll_set = new_poll_set;

	struct llmd_ipc_session** new_sessions = llmd_realloc(
		server->host,
		server->sessions,
		num_sessions * 2 * sizeof(void*)
	);
	if (new_sessions == NULL) {
		return LLMD_ERR_OOM;
	}

	memset(
		new_sessions + server->num_sessions, 0,
		sizeof(void*) * (num_sessions * 2 - server->num_sessions)
	);
	for (unsigned int i = server->num_sessions; i < num_sessions * 2; ++i) {
		struct llmd_ipc_session* new_session = llmd_malloc(
			server->host,
			sizeof(struct llmd_ipc_session) + server->output_mem_size
		);
		if (new_session == NULL) {
			return LLMD_ERR_OOM;
		}

		*new_session = (struct llmd_ipc_session) {
			.ipc_sock = -1,
			.pollfd = &server->poll_set[i + 1],
		};
	}

	*session_out = new_sessions[server->num_sessions];
	server->sessions = new_sessions;
	server->num_sessions = num_sessions * 2;

	return LLMD_OK;
}

static enum llmd_error
llmd_ipc_accept_client(
	struct llmd_ipc_server* server
) {
	while (true) {
		int client_sock = accept4(server->ipc_sock, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
		if (client_sock < 0) {
			if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
				llmd_log(server->host, LLMD_LOG_WARNING, "accept() returns %s", strerror(errno));
				return LLMD_ERR_IO;
			} else {
				return LLMD_OK;
			}
		}

		struct llmd_ipc_session* session;
		enum llmd_error status = llmd_ipc_alloc_session(server, &session);
		if (status != LLMD_OK) {
			close(client_sock);
			return status;
		}

		session->ipc_sock = client_sock;
		session->pollfd->fd = client_sock;
		session->pollfd->events = POLLIN | POLLERR | POLLHUP;

		llmd_log(server->host, LLMD_LOG_INFO, "Accepted session %p", (void*)session);
	}
}

static enum llmd_error
llmd_step_ipc_server(
	struct llmd_ipc_server* server
) {
	int result = poll(server->poll_set, server->num_sessions + 1, -1);
	if (result < 0) {
		if (errno != EINTR) {
			llmd_log(server->host, LLMD_LOG_WARNING, "poll() returns %s", strerror(errno));
			return LLMD_ERR_IO;
		} else {
			return LLMD_OK;
		}
	}

	enum llmd_error status;
	for (unsigned int i = 0; i < server->num_sessions + 1; ++i) {
		struct pollfd* entry = &server->poll_set[i];
		if (entry->revents == 0) { continue; }

		if (entry->fd == server->ipc_sock) {
			if ((status = llmd_ipc_accept_client(server)) != LLMD_OK) {
				llmd_log(server->host, LLMD_LOG_WARNING, "Error while accepting client: %d", status);
			}
		} else {
			struct llmd_ipc_session* session = server->sessions[i - 1];

			if ((status = llmd_ipc_handle_session(server, session, entry->revents))) {
				llmd_log(server->host, LLMD_LOG_WARNING, "Error while handling session %p: %d", (void*)session, status);
			}
		}
	}

	return LLMD_OK;
}

enum llmd_error
llmd_create_ipc_server(
	struct llmd_host* host,
	struct llmd_ipc_server_config* config,
	struct llmd_driver* driver,
	struct llmd_ipc_server** server_out
) {
	if (host == NULL) {
		host = &llmd_default_host;
	}

	struct llmd_ipc_server* server = llmd_malloc(
		host,
		sizeof(struct llmd_ipc_server)
	);
	if (server == NULL) {
		return LLMD_ERR_OOM;
	}

	*server = (struct llmd_ipc_server) {
		.host = host,
		.config = config,
		.driver = driver,
		.ipc_sock = -1,
	};

	enum llmd_error status = llmd_init_ipc_server(server);
	if (status != LLMD_OK) {
		llmd_destroy_ipc_server(server);
		return status;
	}

	*server_out = server;
	return LLMD_OK;
}

enum llmd_error
llmd_start_ipc_server(
	struct llmd_ipc_server* server
) {
	server->running = true;

	while (server->running) {
		enum llmd_error status;
		if ((status = llmd_step_ipc_server(server)) != LLMD_OK) {
			return status;
		}
	}

	return LLMD_OK;
}

enum llmd_error
llmd_stop_ipc_server(
	struct llmd_ipc_server* server
) {
	server->running = false;
	return LLMD_OK;
}

enum llmd_error
llmd_destroy_ipc_server(
	struct llmd_ipc_server* server
) {
	for (unsigned int i = 0; i < server->num_sessions; ++i) {
		struct llmd_ipc_session* session = server->sessions[i];
		if (session->ipc_sock == -1) { continue; }
		close(session->ipc_sock);

		llmd_ipc_cleanup_shared_mem(&session->shared_mem);

		for (unsigned int j = 0; j < session->num_fds; ++j) {
			close(session->fds[j]);
		}

		llmd_free(server->host, session);
	}
	llmd_free(server->host, server->sessions);
	llmd_free(server->host, server->poll_set);

	for (unsigned int i = 0; i < server->num_contexts; ++i) {
		struct llmd_ipc_client_context* context = &server->contexts[i];
		if (context->descriptor == -1) { continue; }

		server->driver->interface->destroy_context(
			server->driver,
			context->descriptor
		);

		llmd_ipc_cleanup_shared_mem(&context->context_window);
		llmd_ipc_cleanup_shared_mem(&context->logits);
	}
	llmd_free(server->host, server->contexts);

	if (server->ipc_sock != -1) {
		close(server->ipc_sock);
	}

	llmd_free(server->host, server);

	return LLMD_OK;
}
