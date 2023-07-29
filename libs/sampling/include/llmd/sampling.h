#ifndef LLMD_SAMPLING_H
#define LLMD_SAMPLING_H

#include <llmd/core.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef LLMD_SAMPLING_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMD_SAMPLING_BUILD
#            define LLMD_SAMPLING_API __declspec(dllexport)
#        else
#            define LLMD_SAMPLING_API __declspec(dllimport)
#        endif
#    else
#        define LLMD_SAMPLING_API __attribute__((visibility ("default")))
#    endif
#else
#    define LLMD_SAMPLING_API
#endif

// PCG from https://github.com/mattiasgustavsson/libs
struct llmd_sampling_default_rng_state {
	uint64_t state[2];
};

struct llmd_sampling_rng {
	float (*next)(void* state);
	void* state;
};

struct llmd_sampling_ring_buf;

struct llmd_sampling_mirostat_v2_state {
	float tau;
	float eta;
	float mu;
	float* scratch_buf;
};

#ifdef __cplusplus
extern "C" {
#endif

// RNG

LLMD_SAMPLING_API struct llmd_sampling_rng
llmd_sampling_init_default_rng(
	struct llmd_sampling_default_rng_state* state,
	uint32_t seed
);

// Ring buffer

LLMD_SAMPLING_API struct llmd_sampling_ring_buf*
llmd_sampling_create_ring_buf(
	struct llmd_host* host,
	unsigned int size
);


LLMD_SAMPLING_API void
llmd_sampling_destroy_ring_buf(
	struct llmd_sampling_ring_buf* ring_buf
);

LLMD_SAMPLING_API void
llmd_sampling_ring_buf_add_token(
	struct llmd_sampling_ring_buf* ring_buf,
	llmd_token_t token
);

LLMD_SAMPLING_API unsigned int
llmd_sampling_ring_buf_num_unique_tokens(
	struct llmd_sampling_ring_buf* ring_buf
);

LLMD_SAMPLING_API void
llmd_sampling_ring_buf_get_unique_token(
	struct llmd_sampling_ring_buf* ring_buf,
	unsigned int index,
	llmd_token_t* token_out,
	unsigned int* frequency_out
);

// Transform

LLMD_SAMPLING_API void
llmd_sampling_softmax(unsigned int num_items, const float* input, float* output);

LLMD_SAMPLING_API void
llmd_sampling_apply_repetition_penalties(
	unsigned int num_items, float* logits,
	struct llmd_sampling_ring_buf* recent_tokens,
	float penalty
);

LLMD_SAMPLING_API void
llmd_sampling_apply_frequency_and_presence_penalties(
	unsigned int num_items, float* logits,
	struct llmd_sampling_ring_buf* recent_tokens,
	float alpha_frequency,
	float alpha_presence
);

LLMD_SAMPLING_API void
llmd_sampling_apply_temperature(
	unsigned int num_items, float* logits,
	float temperature
);

// Pick

LLMD_SAMPLING_API llmd_token_t
llmd_sampling_pick_weighted_random(
	unsigned int num_items, const float* scores,
	struct llmd_sampling_rng* rng
);

LLMD_SAMPLING_API llmd_token_t
llmd_sampling_pick_argmax(
	unsigned int num_items, const float* logits
);

LLMD_SAMPLING_API llmd_token_t
llmd_sampling_pick_mirostat_v2(
	unsigned int num_items, const float* logits,
	struct llmd_sampling_rng* rng,
	struct llmd_sampling_mirostat_v2_state* mirostat_v2
);

#ifdef __cplusplus
}
#endif

#endif
