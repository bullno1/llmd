#ifndef LLMD_ARENA_ALLOCATOR_H
#define LLMD_ARENA_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>
#include "host.h"

struct llmd_arena_chunk {
	struct llmd_arena_chunk* next;
	char* end;
	char start[];
};

struct llmd_arena_allocator {
	struct llmd_host* host;
	size_t chunk_size;
	char* bump_ptr;

	struct llmd_arena_chunk* free_chunks;
	struct llmd_arena_chunk* current_chunk;
};

static inline void
llmd_arena_allocator_init(
	struct llmd_host* host,
	struct llmd_arena_allocator* allocator,
	size_t chunk_size
) {
	*allocator = (struct llmd_arena_allocator) {
		.host = host,
		.chunk_size = chunk_size,
	};
}

static inline void
llmd_arena_allocator_cleanup(struct llmd_arena_allocator* allocator) {
	struct llmd_arena_chunk* chunk = allocator->current_chunk;
	while (chunk != NULL) {
		struct llmd_arena_chunk* next = chunk->next;
		llmd_free(allocator->host, chunk);
		chunk = next;
	}

	chunk = allocator->free_chunks;
	while (chunk != NULL) {
		struct llmd_arena_chunk* next = chunk->next;
		llmd_free(allocator->host, chunk);
		chunk = next;
	}
}

static inline void*
llmd_arena_allocator_align_ptr(void* ptr, size_t alignment) {
	return (void*)(((intptr_t)ptr + (intptr_t)alignment) & -(intptr_t)alignment);
}

static inline void*
llmd_arena_allocator_malloc(
	struct llmd_arena_allocator* allocator,
	size_t size
) {
	// Try current chunk
	struct llmd_arena_chunk* current_chunk = allocator->current_chunk;

	// Try to reuse existing chunks
	if (current_chunk == NULL) {
		current_chunk = allocator->free_chunks;
		if (current_chunk != NULL) {
			allocator->free_chunks = current_chunk->next;
		}
	}

	// Allocate a new chunk
	if (
		current_chunk == NULL
		|| allocator->bump_ptr >= current_chunk->end
		|| (size_t)(current_chunk->end - allocator->bump_ptr) < size
	) {
		size_t chunk_size = size > allocator->chunk_size ? size : allocator->chunk_size;
		struct llmd_arena_chunk* new_chunk = llmd_malloc(
			allocator->host,
			sizeof(struct llmd_arena_chunk) + chunk_size
		);
		if (new_chunk == NULL) {
			return NULL;
		}

		new_chunk->end = new_chunk->start + chunk_size;
		new_chunk->next = current_chunk;
		current_chunk = new_chunk;
		allocator->bump_ptr = new_chunk->start;
	}

	char* result = llmd_arena_allocator_align_ptr(allocator->bump_ptr, sizeof(void*));
	allocator->bump_ptr = llmd_arena_allocator_align_ptr(result + size, sizeof(void*));

	return result;
}

static inline void
llmd_arena_allocator_reset(
	struct llmd_arena_allocator* allocator
) {
	struct llmd_arena_chunk* chunk = allocator->current_chunk;
	while (chunk != NULL) {
		struct llmd_arena_chunk* next = chunk->next;
		chunk->next = allocator->free_chunks;
		allocator->free_chunks = chunk;
		chunk = next;
	}
}

#endif
