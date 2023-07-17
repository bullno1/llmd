#ifndef LLMD_DYN_LIB_H
#define LLMD_DYN_LIB_H

typedef void* llmd_dynlib_t;

#define LLMD_INVALID_DYNLIB NULL

llmd_dynlib_t
llmd_load_dynlib(const char* path);

void
llmd_unload_dynlib(llmd_dynlib_t lib);

void*
llmd_get_proc_address(llmd_dynlib_t lib, const char* name);

#endif
