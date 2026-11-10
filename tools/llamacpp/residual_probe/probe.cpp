#include "llama.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 3) {
        return 2;
    }
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        return 3;
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 4;
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    auto * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        return 4;
    }
    llama_token tokens[] = {1, 2, 3, 4};
    auto batch = llama_batch_get_one(tokens, 4);
    if (llama_decode(context, batch) != 0) {
        return 5;
    }
    const int count = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(context, -1);
    FILE * output = std::fopen(argv[2], "wb");
    if (!output || !logits) {
        return 6;
    }
    const auto written = std::fwrite(logits, sizeof(float), count, output);
    std::fclose(output);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return written == static_cast<size_t>(count) ? 0 : 7;
}
