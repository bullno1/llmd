if (NOT MSVC)
	add_compile_options(-fPIC)
endif ()

include(llama_cpp.cmake)
include(inih.cmake)
include(vendor.cmake)
include(argparse.cmake)
