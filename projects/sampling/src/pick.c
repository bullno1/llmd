#include <llmd/sampling.h>
#include <math.h>

static inline unsigned int
llmd_sampling_pick_weighted_random_idx(
	struct llmd_sampling_candidates* candidates,
	struct llmd_sampling_rng* rng
) {
	float sum = 0.f;
	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		sum += candidates->scores[i];
	}

	float threshold = rng->next(rng->state) * sum;

	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		threshold -= candidates->scores[i];

		if (threshold <= 0) {
			return i;
		}
	}

	return candidates->num_candidates > 0
		? candidates->num_candidates - 1
		: 0;
}

llmd_token_t
llmd_sampling_pick_weighted_random(
	struct llmd_sampling_candidates* candidates,
	struct llmd_sampling_rng* rng
) {
	return candidates->ids[llmd_sampling_pick_weighted_random_idx(candidates, rng)];
}

llmd_token_t
llmd_sampling_pick_max_score(
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
	struct llmd_sampling_rng* rng,
	struct llmd_sampling_mirostat_v2_state* mirostat_v2
) {
	float mu = mirostat_v2->mu;
	unsigned int size = candidates->num_candidates;
	float excluded_scores = 0.f;
	float sum = 0.f;
	for (unsigned int i = 0; i < size; ++i) {
		float score = candidates->scores[i];
		float surprise = -log2f(score);
		sum += score;
		if (surprise > mu && size > 1) {
			excluded_scores += score;
			// Overwrite with the last element
			size -= 1;
			candidates->scores[i] = candidates->scores[size];
			candidates->ids[i] = candidates->ids[size];
		}

		if (size == 1) {
			break;
		}
	}
	candidates->sorted = false;
	candidates->num_candidates = size;

	float new_sum = sum - excluded_scores;
	for (unsigned int i = 0; i < size; ++i) {
		candidates->scores[i] /= new_sum;
	}

	unsigned int chosen_idx = llmd_sampling_pick_weighted_random_idx(candidates, rng);

	float observed_surprise = -log2f(candidates->scores[chosen_idx]);
	float error = observed_surprise - mirostat_v2->tau;
	mirostat_v2->mu = mu - mirostat_v2->eta * error;

	return candidates->ids[chosen_idx];
}
