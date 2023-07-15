#ifndef LLMD_DRIVER_UTILS_H
#define LLMD_DRIVER_UTILS_H

#include <llmd/core.h>
#include <stdbool.h>
#include <errno.h>
#include <stdlib.h>

static inline enum llmd_error
llmd_cfg_parse_long(const char* str, long min_val, long max_val, long* out) {
	errno = 0;
	long lvalue = strtol(str, NULL, 10);
	if (errno || lvalue < min_val || lvalue > max_val) {
		return LLMD_ERR_INVALID;
	}

	*out = lvalue;
	return LLMD_OK;
}

static inline enum llmd_error
llmd_cfg_parse_int(const char* str, int min_val, int max_val, int* out) {
	long value;
	enum llmd_error status;
	if ((status = llmd_cfg_parse_long(str, min_val, max_val, &value)) != LLMD_OK) {
		return status;
	}

	*out = value;
	return LLMD_OK;
}

static inline enum llmd_error
llmd_cfg_parse_uint(const char* str, unsigned int min_val, unsigned int max_val, unsigned int* out) {
	long value;
	enum llmd_error status;
	if ((status = llmd_cfg_parse_long(str, min_val, max_val, &value)) != LLMD_OK) {
		return status;
	}

	*out = value;
	return LLMD_OK;
}

static inline enum llmd_error
llmd_cfg_parse_bool(const char* str, bool* out) {
	long value;
	enum llmd_error status;
	if ((status = llmd_cfg_parse_long(str, 0, 1, &value)) != LLMD_OK) {
		return status;
	}

	*out = value;
	return LLMD_OK;
}

#endif
