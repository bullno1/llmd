set(CLIENT_SOURCES "src/client.c")

setup_library(llmd_ipc_client ${LLMD_IPC_CLIENT_STATIC} ${CLIENT_SOURCES})

target_include_directories(llmd_ipc_client PUBLIC "./include")
target_link_libraries(llmd_ipc_client PUBLIC llmd_core_interface)
target_link_libraries(llmd_ipc_client PRIVATE llmd_utils)

set(SERVER_SOURCES "src/server.c")

setup_library(llmd_ipc_server ${LLMD_IPC_SERVER_STATIC} ${SERVER_SOURCES})

target_include_directories(llmd_ipc_server PUBLIC "./include")
target_link_libraries(llmd_ipc_server PUBLIC llmd_core_interface)
target_link_libraries(llmd_ipc_server PRIVATE llmd_utils)
