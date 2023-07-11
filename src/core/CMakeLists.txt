add_library(llmd_core "core.c")
target_include_directories(llmd_core PUBLIC ${LLMD_INCLUDE_DIR})
target_link_libraries(llmd_core PRIVATE llmd_common)
