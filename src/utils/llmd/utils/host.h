#ifndef LLMD_COMMON_HOST_H
#define LLMD_COMMON_HOST_H

#include <llmd/core.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__GNUC__) || defined(__clang__)
static void
llmd_log(
	struct llmd_host* host,
	enum llmd_log_level level,
	const char* fmt,
	...
) __attribute__((format (printf, 3, 4)));
#endif

static inline void
llmd_log(
	struct llmd_host* host,
	enum llmd_log_level level,
	const char* fmt,
	...
) {
	va_list args;
	va_start(args, fmt);
	host->interface->log(host, level, fmt, args);
	va_end(args);
}

static inline void*
llmd_realloc(
	struct llmd_host* host,
	void* ptr,
	size_t size
) {
	return host->interface->realloc(host, ptr, size);
}

static inline void*
llmd_malloc(
	struct llmd_host* host,
	size_t size
) {
	return llmd_realloc(host, NULL, size);
}

static inline void*
llmd_free(
	struct llmd_host* host,
	void* ptr
) {
	return llmd_realloc(host, ptr, 0);
}

static void*
llmd_default_realloc(struct llmd_host* host, void* ptr, size_t size) {
	(void)host;

	if (size > 0) {
		return realloc(ptr, size);
	} else {
		free(ptr);
		return NULL;
	}
}

static const char*
llmd_log_level_to_str(enum llmd_log_level level) {
	switch (level) {
		case LLMD_LOG_DEBUG: return "DEBUG";
		case LLMD_LOG_INFO: return "INFO";
		case LLMD_LOG_WARNING: return "WARNING";
		case LLMD_LOG_ERROR: return "ERROR";
		default: return "";
	}
}

static void
llmd_default_log(
	struct llmd_host* host,
	enum llmd_log_level level,
	const char* format,
	va_list args
) {
	(void)host;

	fprintf(stderr, "[%s] ", llmd_log_level_to_str(level));
	vfprintf(stderr, format, args);
	fprintf(stderr, "\n");
}

static struct llmd_host_interface llmd_default_host_interface = {
	.realloc = &llmd_default_realloc,
	.log = &llmd_default_log,
};

static struct llmd_host llmd_default_host = {
	.interface = &llmd_default_host_interface,
};

#endif
