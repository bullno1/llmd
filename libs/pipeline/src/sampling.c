#include <lm_pipeline/sampling.h>
#include <lm_pipeline.h>
#include <llmd/sampling.h>
#include <math.h>

static llmd_token_t
lm_pipeline_argmax_sampler(float* scores, unsigned int num_entries, void* userdata) {
	(void)userdata;

	return llmd_sampling_pick_argmax(num_entries, scores);
}

struct mirostat_v2 {
	struct llmd_sampling_rng rng;
	struct llmd_sampling_mirostat_v2_state state;
};

static llmd_token_t
lm_pipeline_mirostat_sampler(float* scores, unsigned int num_entries, void* userdata) {
	struct mirostat_v2* mirostat = userdata;

	return llmd_sampling_pick_mirostat_v2(num_entries, scores, &mirostat->rng, &mirostat->state);
}

void
lm_pipeline_use_argmax_sampler(struct lm_pipeline_ctx* ctx) {
	lm_pipeline_set_sampler(ctx, lm_pipeline_argmax_sampler, NULL);
}

void
lm_pipeline_use_mirostat_sampler(struct lm_pipeline_ctx* ctx, float tau, float eta) {
	struct mirostat_v2* mirostat = lm_pipeline_malloc(ctx, sizeof(struct mirostat_v2));
	struct llmd_sampling_default_rng_state* rng_state = lm_pipeline_malloc(ctx, sizeof(struct llmd_sampling_default_rng_state));
	*mirostat = (struct mirostat_v2) {
		.state = {
			.tau = tau,
			.eta = eta,
			.mu = tau * 2,
			.scratch_buf = lm_pipeline_malloc(ctx, lm_pipeline_get_model_info(ctx).vocab_size * sizeof(float)),
		},
		.rng = llmd_sampling_init_default_rng(rng_state, 0)
	};
	lm_pipeline_set_sampler(ctx, lm_pipeline_mirostat_sampler, mirostat);
}
