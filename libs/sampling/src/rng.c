#include <llmd/sampling.h>
#define RND_IMPLEMENTATION
#include <rnd.h>

struct llmd_sampling_rng
llmd_sampling_init_default_rng(
	struct llmd_sampling_default_rng_state* state,
	uint32_t seed
) {
	rnd_pcg_seed((rnd_pcg_t*)state, seed);
	return (struct llmd_sampling_rng){
		.next = (float(*)(void*))&rnd_pcg_nextf,
		.state = state
	};
}
