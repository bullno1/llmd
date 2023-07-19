#include <llmd/sampling.h>
#include <math.h>

llmd_token_t
llmd_sampling_pick_weighted_random(
	unsigned int num_items, const float* scores,
	struct llmd_sampling_rng* rng
) {
	float sum = 0.f;
	for (unsigned int i = 0; i < num_items; ++i) {
		sum += scores[i];
	}

	float threshold = rng->next(rng->state) * sum;

	for (unsigned int i = 0; i < num_items; ++i) {
		threshold -= scores[i];

		if (threshold <= 0) {
			return i;
		}
	}

	return 0;
}

llmd_token_t
llmd_sampling_pick_argmax(
	unsigned int num_items, const float* scores
) {
	float max_score = -INFINITY;
	llmd_token_t token = LLMD_INVALID_TOKEN;
	for (unsigned int i = 0; i < num_items; ++i) {
		if (scores[i] > max_score) {
			max_score = scores[i];
			token = i;
		}
	}

	return token;
}

LLMD_SAMPLING_API llmd_token_t
llmd_sampling_pick_mirostat_v2(
	unsigned int num_items, const float* logits,
	struct llmd_sampling_rng* rng,
	struct llmd_sampling_mirostat_v2_state* mirostat_v2
) {
	float* scratch_buf = mirostat_v2->scratch_buf;
	llmd_sampling_softmax(num_items, logits, scratch_buf);

	float mu = mirostat_v2->mu;
	unsigned int num_items_left = num_items;
	for (unsigned int i = 0; i < num_items; ++i) {
		float score = scratch_buf[i];
		float surprise = -log2f(score);

		if (surprise > mu && num_items_left > 1) {
			--num_items_left;
			scratch_buf[i] = -INFINITY;
		} else {
			scratch_buf[i] = logits[i];
		}
	}

	llmd_sampling_softmax(num_items, scratch_buf, scratch_buf);

	llmd_token_t token = llmd_sampling_pick_weighted_random(num_items, scratch_buf, rng);

	float observed_surprise = -log2f(scratch_buf[token]);
	float error = observed_surprise - mirostat_v2->tau;
	mirostat_v2->mu = mu - mirostat_v2->eta * error;

	return token;
}
