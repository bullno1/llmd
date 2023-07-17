#define LLMD_LOADER_INTERNAL
#include "dynlib.h"
#include <llmd/loader.h>
#include <llmd/utils/host.h>
#include <ini.h>
#include <stdbool.h>

struct llmd_loader {
	struct llmd_host* host;
	void* config;
	llmd_set_driver_config_t set_driver_config;
};

static int
llmd_parse_config(
	void* ctx,
	const char* section, const char* name, const char* value
) {
	struct llmd_loader* loader = ctx;
	return loader->set_driver_config(loader->host, loader->config, section, name, value) == LLMD_OK;
}

enum llmd_error
llmd_load_driver(
	struct llmd_host* host,
	const char* driver_path,
	const char* config_file_path,
	const char* config_file_content,
	struct llmd_loader_handle* handle_out
) {
#define LLMD_LOAD_ERROR(code) do { status = code; goto error; } while (0)
#define LLMD_LOAD_CHECK(op) do { if ((status = (op)) != LLMD_OK) { goto error; } } while (0)
#define llmd_get_proc_address(type, lib, name) (type)llmd_get_proc_address(lib, name)

	struct llmd_driver* driver = NULL;
	llmd_dynlib_t driver_lib = LLMD_INVALID_DYNLIB;
	void* driver_config = NULL;
	enum llmd_error status;

	driver_lib = llmd_load_dynlib(driver_path);
	if (driver_lib == LLMD_INVALID_DYNLIB) {
		llmd_log(host, LLMD_LOG_ERROR, "Could not load driver library");
		LLMD_LOAD_ERROR(LLMD_ERR_IO);
	}

	llmd_begin_create_driver_t begin_create_driver = llmd_get_proc_address(
		llmd_begin_create_driver_t,
		handle_out->internal, "llmd_begin_create_driver"
	);
	llmd_set_driver_config_t set_driver_config = llmd_get_proc_address(
		llmd_set_driver_config_t,
		handle_out->internal, "llmd_set_driver_config"
	);
	llmd_end_create_driver_t end_create_driver = llmd_get_proc_address(
		llmd_end_create_driver_t,
		handle_out->internal, "llmd_end_create_driver"
	);

	if (begin_create_driver == NULL
		|| set_driver_config == NULL
		|| end_create_driver == NULL
	) {
		llmd_log(host, LLMD_LOG_ERROR, "Driver does not export required symbols");

		LLMD_LOAD_ERROR(LLMD_ERR_INVALID);
	}

	LLMD_LOAD_CHECK(begin_create_driver(host, &driver_config));

	struct llmd_loader loader = {
		.host = host,
		.config = driver_config,
		.set_driver_config = set_driver_config,
	};

	LLMD_LOAD_CHECK(llmd_parse_config(&loader, "llmd", "config_file_path", config_file_path));

	int parse_config_result;
	if (config_file_content != NULL) {
		parse_config_result = ini_parse_string(config_file_content, llmd_parse_config, &loader);
	} else {
		parse_config_result = ini_parse(config_file_path, llmd_parse_config, &loader);
	}

	if (parse_config_result > 0) {
		llmd_log(host, LLMD_LOG_ERROR, "Error parsing config on line: %d", parse_config_result);
		LLMD_LOAD_ERROR(LLMD_ERR_INVALID);
	}

	if (parse_config_result < 0) {
		LLMD_LOAD_ERROR(LLMD_ERR_IO);
	}

	LLMD_LOAD_CHECK(end_create_driver(host, driver_config, &driver));
	handle_out->driver = driver;
	handle_out->internal = driver_lib;

	return LLMD_OK;
error:
	if (driver_config != NULL) {
		end_create_driver(host, driver_config, NULL);
	}

	if (driver_lib != LLMD_INVALID_DYNLIB) {
		llmd_unload_dynlib(driver_lib);
	}

	return status;
}

enum llmd_error
llmd_unload_driver(
	struct llmd_loader_handle* handle
) {
	if (handle->driver != NULL) {
		llmd_destroy_driver_t destroy_driver = llmd_get_proc_address(
			llmd_destroy_driver_t,
			handle->internal,
			"llmd_destroy_driver"
		);

		if (destroy_driver) {
			destroy_driver(handle->driver);
		}
	}

	if (handle->internal != NULL) {
		llmd_unload_dynlib(handle->internal);
	}

	return LLMD_OK;
}
