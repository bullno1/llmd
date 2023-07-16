#ifndef LLMD_SAMPLING_H
#define LLMD_SAMPLING_H

#include <llmd/core.h>
#include <stddef.h>
#include <stdbool.h>

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

struct llmd_sampling_candidates {
	unsigned int num_candidates;
	float* scores;
	llmd_token_t* ids;
	bool sorted;
};

struct llmd_sampling_rng {
	float (*next)(void* state);
	void* state;
};

struct llmd_sampling_ring_buf;

#ifdef __cplusplus
extern "C" {
#endif

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

LLMD_SAMPLING_API void
llmd_sampling_ring_buf_get_token(
	struct llmd_sampling_ring_buf* ring_buf,
	unsigned int index,
	llmd_token_t* token_out,
	unsigned int* frequency_out
);

// Transform

LLMD_SAMPLING_API void
llmd_sampling_apply_softmax(struct llmd_sampling_candidates* candidates);

LLMD_SAMPLING_API void
llmd_sampling_apply_repetition_penalties(
	struct llmd_sampling_candidates* candidates,
	struct llmd_sampling_ring_buf* recent_tokens,
	float penalty
);

LLMD_SAMPLING_API void
llmd_sampling_apply_frequency_and_presence_penalties(
	struct llmd_sampling_candidates* candidates,
	struct llmd_sampling_ring_buf* recent_tokens,
	float alpha_frequency,
	float alpha_presence
);

LLMD_SAMPLING_API void
llmd_sampling_apply_temperature(
	struct llmd_sampling_candidates* candidates,
	float temperature
);

// Filter

LLMD_SAMPLING_API void
llmd_sampling_filter_top_k(
	struct llmd_sampling_candidates* candidates,
	unsigned int k
);

LLMD_SAMPLING_API void
llmd_sampling_filter_top_p(
	struct llmd_sampling_candidates* candidates,
	float p,
	unsigned int min_keep
);

// Pick

LLMD_SAMPLING_API llmd_token_t
llmd_sampling_pick_weighted_random(
	struct llmd_sampling_candidates* candidates,
	struct llmd_sampling_rng* rng
);

LLMD_SAMPLING_API llmd_token_t
llmd_sampling_pick_max(
	struct llmd_sampling_candidates* candidates
);

LLMD_SAMPLING_API llmd_token_t
llmd_sampling_pick_mirostat_v2(
	struct llmd_sampling_candidates* candidates,
	float tau, float eta, float * mu
);

#ifdef __cplusplus
}
#endif

#endif
