#ifndef LM_PIPELINE_SAMPLING_H
#define LM_PIPELINE_SAMPLING_H

#include "lm_pipeline/def.h"

LM_PIPELINE_API void
lm_pipeline_use_argmax_sampler(struct lm_pipeline_ctx* ctx);

LM_PIPELINE_API void
lm_pipeline_use_mirostat_sampler(struct lm_pipeline_ctx* ctx, float tau, float eta);

#endif
