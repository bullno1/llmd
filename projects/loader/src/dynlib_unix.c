#if defined(__unix__) || defined(__APPLE__)

#include "dynlib.h"
#include <dlfcn.h>

llmd_dynlib_t
llmd_load_dynlib(const char* path) {
	return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

void
llmd_unload_dynlib(llmd_dynlib_t lib) {
	dlclose(lib);
}

void*
llmd_get_proc_address(llmd_dynlib_t lib, const char* name) {
	return dlsym(lib, name);
}

#endif
