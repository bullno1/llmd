#include <lm_pipeline/sampling.h>
#include <lm_pipeline.h>
#include <math.h>

static inline llmd_token_t
lm_pipeline_argmax_sampler(float* scores, unsigned int num_entries, void* userdata) {
	(void)userdata;

	if (num_entries == 0) { return LLMD_INVALID_TOKEN; }

	llmd_token_t best_token = LLMD_INVALID_TOKEN;
	float best_score = -INFINITY;
	for (unsigned int i = 0; i < num_entries; ++i) {
		if (scores[i] > best_score) {
			best_score = scores[i];
			best_token = i;
		}
	}

	return best_token;
}

void
lm_pipeline_use_argmax_sampler(struct lm_pipeline_ctx* ctx) {
	lm_pipeline_set_sampler(ctx, lm_pipeline_argmax_sampler, NULL);
}
