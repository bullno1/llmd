set(SOURCES "src/client.c")

setup_library(llmd_ipc_client ${LLMD_IPC_CLIENT_STATIC} ${SOURCES})

target_include_directories(llmd_ipc_client PUBLIC "./include")
target_link_libraries(llmd_ipc_client PUBLIC llmd_core_interface)
target_link_libraries(llmd_ipc_client PRIVATE llmd_utils)
