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
llmd_sampling_qsort_candidates(
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
		llmd_sampling_qsort_candidates
	);
}

static void
llmd_sampling_sort_candidates(
	struct llmd_sampling_candidates* candidates
) {
	if (candidates->sorted) {
		return;
	}

	if (candidates->num_candidates > 1) {
		llmd_sampling_qsort_candidates(
			candidates, 0, candidates->num_candidates - 1
		);
	}

	candidates->sorted = true;
}

void
llmd_sampling_filter_top_k(
	struct llmd_sampling_candidates* candidates,
	unsigned int k
) {
	k = k < candidates->num_candidates ? k : candidates->num_candidates;

	llmd_sampling_sort_candidates(candidates);

	candidates->num_candidates = k;
}

void
llmd_sampling_filter_top_p(
	struct llmd_sampling_candidates* candidates,
	float p,
	unsigned int min_keep
) {
	llmd_sampling_sort_candidates(candidates);

	float sum = 0.f;
	unsigned int i;
	for (i = 0; i < candidates->num_candidates; ++i) {
		sum += candidates->scores[i];

		if (sum >= p && i + 1 >= min_keep) {
			break;
		}
	}

	candidates->num_candidates = i + 1;
}
