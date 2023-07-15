#ifndef LLMD_COMMON_BUFFER_H
#define LLMD_COMMON_BUFFER_H

#include <stddef.h>
#include <llmd/core.h>
#include "host.h"

struct llmd_buffer {
	size_t size;
	char mem[];
};

static inline struct llmd_buffer*
llmd_realloc_buffer(
	struct llmd_host* host,
	struct llmd_buffer* buffer,
	size_t new_size
) {
	struct llmd_buffer* new_buffer = llmd_realloc(
		host, buffer, new_size + sizeof(struct llmd_buffer)
	);

	if (new_buffer) { buffer->size = new_size; }

	return new_buffer;
}

static inline size_t
llmd_buffer_size(const struct llmd_buffer* buffer) {
	return buffer != NULL ? buffer->size : 0;
}

#endif
