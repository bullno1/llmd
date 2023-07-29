add_library(llmd_sampling
	"src/rng.c"
	"src/ring_buf.c"
	"src/transform.c"
	"src/pick.c"
)

set(MATH_LIB "")
include(CheckLibraryExists)
check_library_exists(m expf "" LIBM)
if(LIBM)
	list(APPEND MATH_LIB "m")
endif()

target_include_directories(llmd_sampling PUBLIC "./include")
target_link_libraries(llmd_sampling PUBLIC llmd_core_interface)
target_link_libraries(llmd_sampling PRIVATE llmd_utils llmd_vendor ${MATH_LIB})
