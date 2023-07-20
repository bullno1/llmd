set(SOURCES "src/ipc.c")

setup_library(llmd_ipc LLMD_IPC_STATIC ${SOURCES})

target_include_directories(llmd_ipc PUBLIC "./include")
target_link_libraries(llmd_ipc PUBLIC llmd_core_interface)
target_link_libraries(llmd_ipc PRIVATE llmd_utils)
