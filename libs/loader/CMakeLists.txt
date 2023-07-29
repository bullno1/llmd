add_library(llmd_loader
	"src/loader.c"
	"src/dynlib_unix.c"
)
target_include_directories(llmd_loader PUBLIC "./include")
target_link_libraries(llmd_loader PUBLIC llmd_core_interface)
target_link_libraries(llmd_loader PRIVATE llmd_utils llmd_vendor inih)
