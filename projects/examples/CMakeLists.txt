SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}")

add_executable(local "local.c")
target_link_libraries(
	local PRIVATE
	argparse
	llmd_vendor llmd_core llmd_loader llmd_sampling
)
