#include <stddef.h>
#include <llmd/core.h>
#include <llmd/llama_cpp.h>
#include <llmd/sampling.h>
#define KGFLAGS_IMPLEMENTATION
#include <kgflags.h>

int
main(int argc, char* argv[]) {
    const char* model_path = NULL;
    const char* prompt = NULL;

    kgflags_string("model", NULL, "Path to model.", true, &model_path);
    kgflags_string("prompt", NULL, "Prompt.", true, &model_path);

    if (!kgflags_parse(argc, argv)) {
        kgflags_print_errors();
        kgflags_print_usage();
        return 1;
    }

	return 0;
}
