#ifndef LLMD_IPC_COMMON_H
#define LLMD_IPC_COMMON_H

#include <stdint.h>
#include <llmd/core.h>
#include <llmd/utils/host.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <string.h>
#include <errno.h>

#define LLMD_SYSCALL_CHECK(host, op) \
	if ((op) < 0) { \
		llmd_log(host, LLMD_LOG_ERROR, #op " returns %s", strerror(errno)); \
		return LLMD_ERR_IO; \
	}
#endif

#define LLMD_MAX_NUM_FDS 2
#define LLMD_RPC_VERSION 0
#define LLMD_TMP_BUF_SIZE 1024
#define LLMD_CHECK(op) if ((status = (op)) != LLMD_OK) { return status; }

enum llmd_rpc_method {
	LLMD_RPC_BEGIN_SESSION,
	LLMD_RPC_GET_MODEL_INFO,
	LLMD_RPC_CREATE_CONTEXT,
	LLMD_RPC_DESTROY_CONTEXT,
	LLMD_RPC_GENERATE,
	LLMD_RPC_TOKENIZE,
	LLMD_RPC_DECODE_TOKEN,
};

struct llmd_rpc_request_header {
	uint8_t version;
	uint8_t method;
};

struct llmd_rpc_response_header {
	uint8_t error;
};

struct llmd_span {
	void* ptr;
	size_t size;
};

struct llmd_rpc_buf {
	char tmp_buf[LLMD_TMP_BUF_SIZE];
	size_t buf_size;
	size_t buf_cursor;
};

static inline enum llmd_error
llmd_rpc_read(struct llmd_rpc_buf* call, void* data, size_t size) {
	if (call->buf_cursor + size > call->buf_size) {
		return LLMD_ERR_IO;
	}

	memcpy(data, call->tmp_buf + call->buf_cursor, size);

	call->buf_cursor += size;
	return LLMD_OK;
}

static inline enum llmd_error
llmd_rpc_write(struct llmd_rpc_buf* call, const void* data, size_t size) {
	if (call->buf_cursor + size > call->buf_size) {
		return LLMD_ERR_IO;
	}

	memcpy(call->tmp_buf + call->buf_cursor, data, size);

	call->buf_cursor += size;
	return LLMD_OK;
}

static inline enum llmd_error
llmd_ipc_setup_shared_mem(
	struct llmd_host* host,
	struct llmd_span* shared_mem,
	int fd,
	int prot
) {
	struct stat stat;
	LLMD_SYSCALL_CHECK(host, fstat(fd, &stat));
	shared_mem->ptr = mmap(
		NULL,
		stat.st_size,
		prot,
		MAP_SHARED,
		fd,
		0
	);
	shared_mem->size = stat.st_size;
	if (shared_mem->ptr == MAP_FAILED) {
		llmd_log(host, LLMD_LOG_ERROR, "mmap() returns %s", strerror(errno));
		shared_mem->ptr = NULL;
		shared_mem->size = 0;
		return LLMD_ERR_IO;
	}

	return LLMD_OK;
}

static inline enum llmd_error
llmd_ipc_cleanup_shared_mem(
	struct llmd_span* shared_mem
) {
	if (shared_mem->ptr != NULL) {
		munmap(shared_mem->ptr, shared_mem->size);
		shared_mem->ptr = NULL;
		shared_mem->size = 0;
	}

	return LLMD_OK;
}

#endif
