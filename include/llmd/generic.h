#ifndef LLMD_GENERIC_H
#define LLMD_GENERIC_H

#include <llmd/core.h>
#include <llmd/client.h>

#define llmd_get_model_info(session, info_out) \
	_Generic(session, \
		struct llmd_session*: llmd_get_model_info, \
		struct llmd_r_session*: llmd_r_get_model_info \
	)(session, info_out)

#endif
