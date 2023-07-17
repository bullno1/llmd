#include <llmd/sampling.h>
#include <assert.h>
#include <math.h>

void
llmd_sampling_apply_softmax(struct llmd_sampling_candidates* candidates) {
	float max_score = -INFINITY;
	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		if (candidates->scores[i] > max_score) {
			max_score = candidates->scores[i];
		}
	}

	float sum = 0.f;
	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		float p = expf(candidates->scores[i] - max_score);
		sum += p;
		candidates->scores[i] = p;
	}

	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		candidates->scores[i] /= sum;
	}
}

void
llmd_sampling_apply_temperature(
	struct llmd_sampling_candidates* candidates,
	float temperature
) {
	for (unsigned int i = 0; i < candidates->num_candidates; ++i) {
		candidates->scores[i] /= temperature;
	}
}

void
llmd_sampling_apply_repetition_penalties(
	struct llmd_sampling_candidates* candidates,
	struct llmd_sampling_ring_buf* recent_tokens,
	float penalty
) {
	for (
		unsigned int i = 0;
		i < llmd_sampling_ring_buf_num_unique_tokens(recent_tokens);
		++i
	) {
		llmd_token_t token;
		unsigned int frequency;
		llmd_sampling_ring_buf_get_unique_token(recent_tokens, i, &token, &frequency);

		float score = candidates->scores[token];
		if (score > 0) {
			candidates->scores[token] = score / penalty;
		} else {
			candidates->scores[token] = score * penalty;
		}
	}
}

void
llmd_sampling_apply_frequency_and_presence_penalties(
	struct llmd_sampling_candidates* candidates,
	struct llmd_sampling_ring_buf* recent_tokens,
	float alpha_frequency,
	float alpha_presence
) {
	for (
		unsigned int i = 0;
		i < llmd_sampling_ring_buf_num_unique_tokens(recent_tokens);
		++i
	) {
		llmd_token_t token;
		unsigned int frequency;
		llmd_sampling_ring_buf_get_unique_token(recent_tokens, i, &token, &frequency);

		candidates->scores[token] -= (float)frequency * alpha_frequency + (float)(frequency > 0) * alpha_presence;
	}
}
