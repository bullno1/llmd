add_library(inih "inih/ini.c")
target_compile_definitions(inih PUBLIC
	INI_ALLOW_NO_VALUE=1
	INI_STOP_ON_FIRST_ERROR=1
)
target_include_directories(inih PUBLIC "inih")
