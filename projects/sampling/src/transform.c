#include <llmd/sampling.h>
#include <assert.h>
#include <math.h>

void
llmd_sampling_softmax(unsigned int num_items, const float* input, float* output) {
	float max_score = -INFINITY;
	for (unsigned int i = 0; i < num_items; ++i) {
		if (input[i] > max_score) {
			max_score = input[i];
		}
	}

	float sum = 0.f;
	for (unsigned int i = 0; i < num_items; ++i) {
		float p = expf(input[i] - max_score);
		sum += p;
		output[i] = p;
	}

	for (unsigned int i = 0; i < num_items; ++i) {
		output[i] /= sum;
	}
}

void
llmd_sampling_apply_temperature(
	unsigned int num_items, float* logits,
	float temperature
) {
	for (unsigned int i = 0; i < num_items; ++i) {
		logits[i] /= temperature;
	}
}

void
llmd_sampling_apply_repetition_penalties(
	unsigned int num_items, float* logits,
	struct llmd_sampling_ring_buf* recent_tokens,
	float penalty
) {
	(void)num_items;
	for (
		unsigned int i = 0;
		i < llmd_sampling_ring_buf_num_unique_tokens(recent_tokens);
		++i
	) {
		llmd_token_t token;
		unsigned int frequency;
		llmd_sampling_ring_buf_get_unique_token(recent_tokens, i, &token, &frequency);

		float logit = logits[token];
		logits[token] = logit > 0 ? logit / penalty : logit * penalty;
	}
}

void
llmd_sampling_apply_frequency_and_presence_penalties(
	unsigned int num_items, float* logits,
	struct llmd_sampling_ring_buf* recent_tokens,
	float alpha_frequency,
	float alpha_presence
) {
	(void)num_items;
	for (
		unsigned int i = 0;
		i < llmd_sampling_ring_buf_num_unique_tokens(recent_tokens);
		++i
	) {
		llmd_token_t token;
		unsigned int frequency;
		llmd_sampling_ring_buf_get_unique_token(recent_tokens, i, &token, &frequency);

		logits[token] -= (float)frequency * alpha_frequency + (float)(frequency > 0) * alpha_presence;
	}
}
