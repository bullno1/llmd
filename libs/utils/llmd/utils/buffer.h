#ifndef LLMD_UTILS_BUFFER_H
#define LLMD_UTILS_BUFFER_H

#include <stddef.h>
#include <llmd/core.h>
#include <stdalign.h>
#include "host.h"

#define llmd_buffer(type) type*

#define llmd_buffer_size(buffer) \
	llmd_typed_buffer_size(buffer, sizeof(*buffer))

#define llmd_resize_buffer(host, buffer, new_size) \
	llmd_resize_typed_buffer(host, buffer, new_size, sizeof(*buffer))

#define llmd_free_buffer(host, buffer) \
	llmd_free_typed_buffer(host, buffer)

struct llmd_generic_buffer {
	size_t size;
	_Alignas(max_align_t) char mem[];
};

static inline struct llmd_generic_buffer*
llmd_realloc_generic_buffer(
	struct llmd_host* host,
	struct llmd_generic_buffer* buffer,
	size_t new_size
) {
	struct llmd_generic_buffer* new_buffer = llmd_realloc(
		host, buffer, new_size + sizeof(struct llmd_generic_buffer)
	);

	if (new_buffer) { new_buffer->size = new_size; }

	return new_buffer;
}

static inline size_t
llmd_generic_buffer_size(const struct llmd_generic_buffer* buffer) {
	return buffer != NULL ? buffer->size : 0;
}

static inline struct llmd_generic_buffer*
llmd_generic_buffer_of(void* typed_buffer) {
	if (typed_buffer == NULL) {
		return NULL;
	} else {
		return (void*)((char*)typed_buffer - offsetof(struct llmd_generic_buffer, mem));
	}
}

static inline size_t
llmd_typed_buffer_size(const void* typed_buffer, size_t element_size) {
	return llmd_generic_buffer_size(llmd_generic_buffer_of((void*)typed_buffer)) / element_size;
}

static inline void*
llmd_resize_typed_buffer(
	struct llmd_host* host,
	void* typed_buffer,
	size_t new_size,
	size_t element_size
) {
	struct llmd_generic_buffer* new_generic_buffer = llmd_realloc_generic_buffer(
		host,
		llmd_generic_buffer_of(typed_buffer),
		new_size * element_size
	);

	if (new_generic_buffer == NULL) { return NULL; }

	return (char*)new_generic_buffer + offsetof(struct llmd_generic_buffer, mem);
}

static inline void
llmd_free_typed_buffer(struct llmd_host* host, void* typed_buffer) {
	llmd_free(host, llmd_generic_buffer_of(typed_buffer));
}

#endif
