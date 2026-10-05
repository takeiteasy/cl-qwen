#include "llama.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

struct trace_state { const char *directory; int step = 0; };
static bool trace_node(ggml_tensor *tensor, bool ask, void *user) {
    auto *state = static_cast<trace_state *>(user);
    bool wanted = state->directory && state->step == 1 && tensor->type == GGML_TYPE_F32 && tensor->name[0];
    if (ask) return wanted;
    if (wanted) {
        std::string name(tensor->name);
        std::replace(name.begin(), name.end(), '/', '_');
        std::ofstream out(std::string(state->directory) + "/" + name + ".f32", std::ios::binary);
        std::vector<char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        out.write(bytes.data(), bytes.size());
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    std::ifstream prompt_file(argv[2], std::ios::binary);
    if (!prompt_file) return 3;
    std::string text((std::istreambuf_iterator<char>(prompt_file)), {});
    bool special = std::stoi(argv[3]) != 0;
    int limit = std::stoi(argv[4]);
    if (limit < 0) return 4;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 5;
    auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, special);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, special);
    }
    if (n <= 0) return 6;
    tokens.resize(n);
    std::printf("INPUT");
    for (auto token : tokens) std::printf(" %d", token);
    std::printf("\n");
    trace_state trace{std::getenv("CL_QWEN_TRACE")};
    auto cp = llama_context_default_params();
    if (trace.directory) { cp.cb_eval = trace_node; cp.cb_eval_user_data = &trace; }
    cp.n_ctx = std::max(256, n + limit);
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 7;
    std::ofstream logits(argv[5], std::ios::binary);
    if (!logits) return 8;
    int vocabulary = llama_vocab_n_tokens(vocab);
    auto step = [&](llama_token token) {
        auto batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch)) return false;
        ++trace.step;
        logits.write(reinterpret_cast<char *>(llama_get_logits_ith(ctx, -1)), vocabulary * sizeof(float));
        return bool(logits);
    };
    for (auto token : tokens) if (!step(token)) return 9;
    std::printf("OUTPUT");
    for (int i = 0; i < limit; ++i) {
        float *scores = llama_get_logits_ith(ctx, -1);
        auto token = static_cast<llama_token>(std::max_element(scores, scores + vocabulary) - scores);
        if (llama_vocab_is_eog(vocab, token)) break;
        std::printf(" %d", token);
        if (i + 1 < limit && !step(token)) return 10;
    }
    std::printf("\n");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
