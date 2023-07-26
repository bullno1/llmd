#ifndef LLMD_IPC_COMMON_H
#define LLMD_IPC_COMMON_H

#include <stdint.h>
#include <llmd/core.h>

#define LLMD_MAX_NUM_FDS 2
#define LLMD_RPC_VERSION 0

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

#endif
