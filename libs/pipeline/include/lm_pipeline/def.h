#ifndef LM_PIPELINE_DEF_H
#define LM_PIPELINE_DEF_H

struct lm_pipeline_ctx;

#ifdef LM_PIPELINE_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LM_PIPELINE_BUILD
#            define LM_PIPELINE_API __declspec(dllexport)
#        else
#            define LM_PIPELINE_API __declspec(dllimport)
#        endif
#    else
#        define LM_PIPELINE_API __attribute__((visibility ("default")))
#    endif
#else
#    define LM_PIPELINE_API
#endif

#endif
