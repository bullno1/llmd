#include <llmd/sampling.h>
#include <llmd/utils/host.h>
#include <llmd/utils/khash.h>
#include <string.h>
#include <stdint.h>

KHASH_MAP_INIT_INT(llmd_frequency, unsigned int)

struct llmd_sampling_ring_buf {
	struct llmd_host* host;
	unsigned int size;
	unsigned int insert_index;
	llmd_token_t* tokens;
	khash_t(llmd_frequency)* frequencies;
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
		.frequencies = kh_init_llmd_frequency(host),
	};

	if (ring_buf->tokens == NULL || ring_buf->frequencies == NULL) {
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
	if (ring_buf->frequencies != NULL) {
		kh_destroy(llmd_frequency, ring_buf->frequencies);
	}
	llmd_free(host, ring_buf);
}

void
llmd_sampling_ring_buf_get_token(
	struct llmd_sampling_ring_buf* ring_buf,
	unsigned int index,
	llmd_token_t* token_out,
	unsigned int* frequency_out
) {
	if (index > ring_buf->size) {
		*token_out = LLMD_INVALID_TOKEN;
		return;
	}

	llmd_token_t token = ring_buf->tokens[index];
	*token_out = token;

	if (frequency_out != NULL) {
		khint_t itr = kh_get(llmd_frequency, ring_buf->frequencies, (int)token);

		*frequency_out = itr == kh_end(ring_buf->frequencies)
			? 0
			: kh_value(ring_buf->frequencies, itr);
	}
}

void
llmd_sampling_ring_buf_add_token(
	struct llmd_sampling_ring_buf* ring_buf,
	llmd_token_t token
) {
	unsigned int insert_index = ring_buf->insert_index % ring_buf->size;

	llmd_token_t old_token = ring_buf->tokens[insert_index];
	khint_t itr = kh_get(llmd_frequency, ring_buf->frequencies, (int)old_token);
	if (itr != kh_end(ring_buf->frequencies)) {
		unsigned int count = kh_value(ring_buf->frequencies, itr) - 1;
		kh_value(ring_buf->frequencies, itr) = count;

		if (count == 0) {
			kh_del(llmd_frequency, ring_buf->frequencies, itr);
		}
	}

	ring_buf->tokens[insert_index] = token;
	int absent;
	khint_t new_entry = kh_put(llmd_frequency, ring_buf->frequencies, (int)token, &absent);
	if (absent) {
		kh_value(ring_buf->frequencies, new_entry) = 1;
	} else {
		kh_value(ring_buf->frequencies, new_entry) += 1;
	}

	ring_buf->insert_index = insert_index + 1;
}
