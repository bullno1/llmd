#include <llmd/sampling.h>
#include <llmd/utils/host.h>
#include <string.h>
#include <stdint.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-compare"
#define HASHTABLE_IMPLEMENTATION
#define HASHTABLE_MALLOC(ctx, size) llmd_malloc(ctx, size)
#define HASHTABLE_FREE(ctx, size) llmd_free(ctx, size)
#define HASHTABLE_KEYCOPY(dst, src, size) *(llmd_token_t*)(dst) = *(llmd_token_t*)(src)
#define HASHTABLE_ITEMCOPY(dst, src, size) *(unsigned int*)(dst) = *(unsigned int*)(src)
#define HASHTABLE_KEYCMP(lhs, rhs, size) *(unsigned int*)(lhs) == *(unsigned int*)(rhs)
#include <hashtable.h>
#pragma clang diagnostic pop

// From hashtable.h
static uint32_t
llmd_hash_u32( uint32_t key ) {
    key = ~key + ( key << 15 );
    key = key ^ ( key >> 12 );
    key = key + ( key << 2 );
    key = key ^ ( key >> 4 );
    key = (key + ( key << 3 ) ) + ( key << 11 );
    key = key ^ ( key >> 16);
    return key;
}

struct llmd_sampling_ring_buf {
	struct llmd_host* host;
	unsigned int size;
	unsigned int insert_index;
	llmd_token_t* tokens;
	hashtable_t frequencies;
};

struct llmd_sampling_ring_buf*
llmd_sampling_create_ring_buf(
	struct llmd_host* host,
	unsigned int size
) {
	host = host != NULL ? host : &llmd_default_host;
	struct llmd_sampling_ring_buf* ring_buf = llmd_malloc(
		host, sizeof(struct llmd_sampling_ring_buf)
	);
	if (ring_buf == NULL) {
		return NULL;
	}

	*ring_buf = (struct llmd_sampling_ring_buf) {
		.host = host,
		.size = size,
		.insert_index = 0,
		.tokens = llmd_malloc(host, sizeof(llmd_token_t) * size),
	};

	hashtable_init(
		&ring_buf->frequencies,
		sizeof(llmd_token_t),
		sizeof(unsigned int),
		size + 1, // Extra space to hold the new token when full
		host
	);

	if (ring_buf->tokens == NULL) {
		llmd_sampling_destroy_ring_buf(ring_buf);
		return NULL;
	}

	for (unsigned int i = 0; i < size; ++i) {
		ring_buf->tokens[i] = LLMD_INVALID_TOKEN;
	}

	return ring_buf;
}

void
llmd_sampling_destroy_ring_buf(
	struct llmd_sampling_ring_buf* ring_buf
) {
	struct llmd_host* host = ring_buf->host;
	llmd_free(host, ring_buf->tokens);
	hashtable_term(&ring_buf->frequencies);
	llmd_free(host, ring_buf);
}

unsigned int
llmd_sampling_ring_buf_num_unique_tokens(
	struct llmd_sampling_ring_buf* ring_buf
) {
	return (unsigned int)hashtable_count(&ring_buf->frequencies);
}

void
llmd_sampling_ring_buf_get_unique_token(
	struct llmd_sampling_ring_buf* ring_buf,
	unsigned int index,
	llmd_token_t* token_out,
	unsigned int* frequency_out
) {
	hashtable_t* frequencies = &ring_buf->frequencies;
	*token_out = ((llmd_token_t*)hashtable_keys(frequencies))[index];
	*frequency_out = ((unsigned int*)hashtable_items(frequencies))[index];
}

void
llmd_sampling_ring_buf_add_token(
	struct llmd_sampling_ring_buf* ring_buf,
	llmd_token_t token
) {
	// Write to the ring buffer
	unsigned int insert_index = ring_buf->insert_index % ring_buf->size;
	llmd_token_t old_token = ring_buf->tokens[insert_index];
	ring_buf->tokens[insert_index] = token;
	ring_buf->insert_index = insert_index + 1;

	// Update frequencies
	hashtable_t* frequencies = &ring_buf->frequencies;
	uint32_t new_token_hash = llmd_hash_u32(token);
	unsigned int* new_freq = hashtable_find(frequencies, new_token_hash, &token);
	if (new_freq != NULL) {
		*new_freq += 1;
	} else {
		unsigned int frequency = 1;
		hashtable_insert(frequencies, new_token_hash, &token, &frequency);
	}

	uint32_t old_token_hash = llmd_hash_u32(old_token);
	unsigned int* old_freq = hashtable_find(frequencies, old_token_hash, &old_token);
	if (old_freq != NULL) {
		*old_freq -= 1;

		if (*old_freq == 0) {
			hashtable_remove(frequencies, old_token_hash, &old_token);
		}
	}
}
