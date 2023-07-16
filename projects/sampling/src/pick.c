#include <llmd/sampling.h>
#include <math.h>

llmd_token_t
llmd_sampling_pick_weighted_random(
	struct llmd_sampling_candidates* candidates,
	struct llmd_sampling_rng* rng
) {
	float sum = 0.f;
	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		sum += candidates->scores[i];
	}

	float threshold = rng->next(rng->state) * sum;

	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		sum -= threshold;

		if (sum <= 0) {
			return candidates->ids[i];
		}
	}

	return candidates->ids[candidates->num_candidates - 1];
}

llmd_token_t
llmd_sampling_pick_max(
	struct llmd_sampling_candidates* candidates
) {
	float max_score = -INFINITY;
	llmd_token_t token = LLMD_INVALID_TOKEN;
	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		if (candidates->scores[i] > max_score) {
			max_score = candidates->scores[i];
			token = candidates->ids[i];
		}
	}

	return token;
}

LLMD_SAMPLING_API llmd_token_t
llmd_sampling_pick_mirostat_v2(
	struct llmd_sampling_candidates* candidates,
	float tau, float eta, float * mu
);
