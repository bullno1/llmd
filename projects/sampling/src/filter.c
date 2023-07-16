#include <llmd/sampling.h>

#define llmd_implement_indirect_qsort(CTX, START, END, SWAP, CMP, RECURSE) \
	if (START >= END) { return; } \
	size_t ptr = START; \
	for (size_t i = START; i < END; ++i) { \
		if (CMP(CTX, i, END) < 0) { \
			SWAP(CTX, i, ptr); \
			++ptr; \
		} \
	} \
	SWAP(CTX, END, ptr); \
	RECURSE(CTX, START, ptr - 1); \
	RECURSE(CTX, ptr + 1, END);

static inline void
llmd_swap_candidates(
	struct llmd_sampling_candidates* candidates,
	size_t idx1,
	size_t idx2
) {
	float tmp_score = candidates->scores[idx1];
	candidates->scores[idx1] = candidates->scores[idx2];
	candidates->scores[idx2] = tmp_score;

	llmd_token_t tmp_token = candidates->ids[idx1];
	candidates->ids[idx1] = candidates->ids[idx2];
	candidates->ids[idx2] = tmp_token;
}

static inline int
llmd_cmp_candidates(
	struct llmd_sampling_candidates* candidates,
	size_t idx1,
	size_t idx2
) {
	return (int)(candidates->scores[idx2] - candidates->scores[idx1]);
}

static void
llmd_sampling_qsort_candidates_desc(
	struct llmd_sampling_candidates* candidates,
	size_t start,
	size_t end
) {
	llmd_implement_indirect_qsort(
		candidates,
		start,
		end,
		llmd_swap_candidates,
		llmd_cmp_candidates,
		llmd_sampling_qsort_candidates_desc
	);
}

static void
llmd_sampling_sort_candidates_desc(
	struct llmd_sampling_candidates* candidates
) {
	if (candidates->num_candidates > 1) {
		llmd_sampling_qsort_candidates_desc(
			candidates, 0, candidates->num_candidates - 1
		);
	}
}

LLMD_SAMPLING_API void
llmd_sampling_filter_top_k(
	struct llmd_sampling_candidates* candidates,
	unsigned int k
) {
	k = k < candidates->num_candidates ? k : candidates->num_candidates;

	llmd_sampling_sort_candidates_desc(candidates);

	candidates->num_candidates = k;
}
