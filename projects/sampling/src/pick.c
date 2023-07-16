#include <llmd/sampling.h>

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
