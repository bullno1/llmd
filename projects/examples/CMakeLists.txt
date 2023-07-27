SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}")

add_executable(llmd_generate "generate.c")
target_link_libraries(
	llmd_generate PRIVATE
	argparse
	llmd_vendor llmd_core llmd_loader llmd_sampling
)
set_target_properties(llmd_generate PROPERTIES OUTPUT_NAME "generate")
