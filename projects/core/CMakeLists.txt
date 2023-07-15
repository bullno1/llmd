add_library(llmd_core "src/core.c")
target_include_directories(llmd_core PUBLIC "./include")
target_link_libraries(llmd_core PRIVATE llmd_utils)
