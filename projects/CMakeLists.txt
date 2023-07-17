# Add compile option here to not leak them to dependencies
if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_subdirectory(utils)
add_subdirectory(vendor)
add_subdirectory(core)
add_subdirectory(llama_cpp)
add_subdirectory(sampling)
add_subdirectory(loader)
