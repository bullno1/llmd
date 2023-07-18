#define LLMD_LOADER_INTERNAL
#include "dynlib.h"
#include <llmd/loader.h>
#include <llmd/utils/host.h>
#include <ini.h>
#include <stdbool.h>
#include <string.h>

#define LLMD_LOAD_ERROR(code) do { status = code; goto error; } while (0)
#define LLMD_LOAD_CHECK(op) do { if ((status = (op)) != LLMD_OK) { goto error; } } while (0)
#define llmd_get_proc_address(type, lib, name) (type)llmd_get_proc_address(lib, name)

struct llmd_driver_loader {
	struct llmd_host* host;
	llmd_dynlib_t driver_lib;
	struct llmd_driver* driver;
	void* config;

	llmd_begin_create_driver_t begin_create_driver;
	llmd_set_driver_config_t set_driver_config;
	llmd_end_create_driver_t end_create_driver;
	llmd_destroy_driver_t destroy_driver;
};

static int
llmd_parse_driver_config(
	void* ctx,
	const char* section, const char* name, const char* value
) {
	return llmd_config_driver(ctx, section, name, value) == LLMD_OK;
}

static enum llmd_error
llmd_handle_parse_result(
	struct llmd_driver_loader* loader,
	int parse_result
) {
	if (parse_result < 0) {
		return LLMD_ERR_IO;
	} else if (parse_result > 0) {
		llmd_log(loader->host, LLMD_LOG_ERROR, "Error in config file at line %d", parse_result);
		return LLMD_ERR_INVALID;
	} else {
		return LLMD_OK;
	}
}

enum llmd_error
llmd_begin_load_driver(
	struct llmd_host* host,
	const char* driver_path,
	struct llmd_driver_loader** loader_out
) {
	enum llmd_error status = LLMD_OK;
	host = host != NULL ? host : &llmd_default_host;
	struct llmd_driver_loader* loader = llmd_malloc(host, sizeof(struct llmd_driver_loader));
	if (loader == NULL) {
		return LLMD_ERR_OOM;
	}
	*loader = (struct llmd_driver_loader) {
		.host = host,
	};

	loader->driver_lib = llmd_load_dynlib(driver_path);
	if (loader->driver_lib == LLMD_INVALID_DYNLIB) {
		llmd_log(host, LLMD_LOG_ERROR, "Could not load driver library");
		LLMD_LOAD_ERROR(LLMD_ERR_IO);
	}

	loader->begin_create_driver = llmd_get_proc_address(
		llmd_begin_create_driver_t,
		loader->driver_lib,
		"llmd_begin_create_driver"
	);
	loader->set_driver_config = llmd_get_proc_address(
		llmd_set_driver_config_t,
		loader->driver_lib,
		"llmd_set_driver_config"
	);
	loader->end_create_driver = llmd_get_proc_address(
		llmd_end_create_driver_t,
		loader->driver_lib,
		"llmd_end_create_driver"
	);
	loader->destroy_driver = llmd_get_proc_address(
		llmd_destroy_driver_t,
		loader->driver_lib,
		"llmd_destroy_driver"
	);

	if (
		loader->begin_create_driver == NULL
		|| loader->set_driver_config == NULL
		|| loader->end_create_driver == NULL
		|| loader->destroy_driver == NULL
	) {
		llmd_log(host, LLMD_LOG_ERROR, "Driver does not export the required functions");
		LLMD_LOAD_ERROR(LLMD_ERR_INVALID);
	}

	LLMD_LOAD_CHECK(loader->begin_create_driver(host, &loader->config));

	// built-in config
	LLMD_LOAD_CHECK(llmd_config_driver(loader, "llmd", "driver_path", driver_path));

	*loader_out = loader;
	return LLMD_OK;
error:
	llmd_unload_driver(loader);
	return status;
}

enum llmd_error
llmd_config_driver(
	struct llmd_driver_loader* loader,
	const char* section,
	const char* key,
	const char* value
) {
	return loader->set_driver_config(loader->host, loader->config, section, key, value);
}

enum llmd_error
llmd_load_driver_config_from_file(
	struct llmd_driver_loader* loader,
	const char* ini_file
) {
	int result = ini_parse(ini_file, llmd_parse_driver_config, loader);
	return llmd_handle_parse_result(loader, result);
}

enum llmd_error
llmd_load_driver_config_from_string(
	struct llmd_driver_loader* loader,
	const char* ini_string
) {
	int result = ini_parse_string(ini_string, llmd_parse_driver_config, loader);
	return llmd_handle_parse_result(loader, result);
}

enum llmd_error
llmd_end_load_driver(
	struct llmd_driver_loader* loader,
	struct llmd_driver** driver_out
) {
	enum llmd_error status = loader->end_create_driver(loader->host, loader->config, &loader->driver);
	loader->config = NULL;
	*driver_out = loader->driver;
	return status;
}

enum llmd_error
llmd_unload_driver(
	struct llmd_driver_loader* loader
) {
	if (loader->config != NULL) {
		loader->end_create_driver(
			loader->host,
			loader->config,
			NULL
		);
	}

	if (loader->driver != NULL) {
		loader->destroy_driver(loader->driver);
	}

	if (loader->driver_lib != LLMD_INVALID_DYNLIB) {
		llmd_unload_dynlib(loader->driver_lib);
	}

	llmd_free(loader->host, loader);

	return LLMD_OK;
}
