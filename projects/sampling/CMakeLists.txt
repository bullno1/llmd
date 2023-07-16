add_library(llmd_sampling
	"src/transform.c"
	"src/ring_buf.c"
	"src/filter.c"
)
target_include_directories(llmd_sampling PUBLIC "./include")
target_link_libraries(llmd_sampling PUBLIC llmd_core_interface)
target_link_libraries(llmd_sampling PRIVATE llmd_utils)
