add_library(lm_pipeline
	"src/pipeline.c"
	"src/sampling.c"
)

target_include_directories(lm_pipeline PUBLIC "./include")
target_link_libraries(lm_pipeline PUBLIC llmd_core)
target_link_libraries(lm_pipeline PRIVATE llmd_utils llmd_sampling)
