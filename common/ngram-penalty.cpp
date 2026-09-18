#include "ngram-penalty.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace {

struct ngram_penalty_ctx {
    int32_t start_n;
    float   scale;
    float   max_penalty;
    int32_t max_m;   // longest context length looked up (a match of max_m + 1 tokens saturates the cap)
    // key: hash of (m, m-gram) -> tokens that followed that m-gram in the references
    std::unordered_map<uint64_t, std::vector<llama_token>> next;
    std::vector<llama_token> tail;
};

uint64_t hash_gram(const llama_token * toks, int32_t m) {
    uint64_t h = 1469598103934665603ULL ^ (uint64_t) m;
    for (int32_t i = 0; i < m; ++i) {
        h ^= (uint64_t) (uint32_t) toks[i];
        h *= 1099511628211ULL;
        h ^= h >> 29;
    }
    return h;
}

const char * ngram_penalty_name(const struct llama_sampler * /*smpl*/) {
    return "ngram-penalty";
}

void ngram_penalty_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (ngram_penalty_ctx *) smpl->ctx;
    ctx->tail.push_back(token);
    if ((int32_t) ctx->tail.size() > ctx->max_m) {
        ctx->tail.erase(ctx->tail.begin(), ctx->tail.begin() + (ctx->tail.size() - ctx->max_m));
    }
}

void ngram_penalty_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (ngram_penalty_ctx *) smpl->ctx;
    if (ctx->next.empty() || ctx->scale <= 0.0f) {
        return;
    }
    const int32_t tail_n = (int32_t) ctx->tail.size();
    const int32_t m_hi   = std::min(tail_n, ctx->max_m);
    const int32_t m_lo   = std::max(0, ctx->start_n - 1); // m == 0: context-free, every span token
    if (m_hi < m_lo) {
        return;
    }
    // penalty per token: the longest match wins
    std::unordered_map<llama_token, float> penalty;
    for (int32_t m = m_hi; m >= m_lo; --m) {
        const uint64_t key = hash_gram(ctx->tail.data() + (tail_n - m), m);
        auto it = ctx->next.find(key);
        if (it == ctx->next.end()) {
            continue;
        }
        const int32_t n   = m + 1;
        const float   pen = std::min(ctx->max_penalty, (float) (n - ctx->start_n + 1) * ctx->scale);
        for (llama_token t : it->second) {
            auto & slot = penalty[t];
            slot = std::max(slot, pen);
        }
    }
    if (penalty.empty()) {
        return;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        auto it = penalty.find(cur_p->data[i].id);
        if (it != penalty.end()) {
            cur_p->data[i].logit -= it->second;
        }
    }
    cur_p->sorted = false;
}

void ngram_penalty_reset(struct llama_sampler * smpl) {
    auto * ctx = (ngram_penalty_ctx *) smpl->ctx;
    ctx->tail.clear();
}

struct llama_sampler * ngram_penalty_clone(const struct llama_sampler * smpl);

void ngram_penalty_free(struct llama_sampler * smpl) {
    delete (ngram_penalty_ctx *) smpl->ctx;
}

struct llama_sampler_i ngram_penalty_iface = {
    /* .name   = */ ngram_penalty_name,
    /* .accept = */ ngram_penalty_accept,
    /* .apply  = */ ngram_penalty_apply,
    /* .reset  = */ ngram_penalty_reset,
    /* .clone  = */ ngram_penalty_clone,
    /* .free   = */ ngram_penalty_free,
    /* .backend_init  = */ nullptr,
    /* .backend_accept = */ nullptr,
    /* .backend_apply  = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct llama_sampler * ngram_penalty_clone(const struct llama_sampler * smpl) {
    const auto * src = (const ngram_penalty_ctx *) smpl->ctx;
    return llama_sampler_init(&ngram_penalty_iface, new ngram_penalty_ctx(*src));
}

} // namespace

struct llama_sampler * common_ngram_penalty_init(
        const std::vector<std::vector<llama_token>> & refs,
        int32_t start_n,
        float   scale,
        float   max_penalty) {
    auto * ctx = new ngram_penalty_ctx();
    ctx->start_n     = std::max(1, start_n);
    ctx->scale       = scale;
    ctx->max_penalty = max_penalty;
    // beyond this context length the penalty is at the cap anyway
    const int32_t steps_to_cap = scale > 0.0f ? (int32_t) std::ceil(max_penalty / scale) : 1;
    ctx->max_m = ctx->start_n - 1 + std::max(1, steps_to_cap);
    for (const auto & seq : refs) {
        const int32_t n = (int32_t) seq.size();
        for (int32_t m = std::max(0, ctx->start_n - 1); m <= ctx->max_m; ++m) {
            // m == 0 (start_n == 1): a 1-gram match is any token of the span,
            // penalized without context
            for (int32_t i = 0; i + m < n; ++i) {
                auto & nexts = ctx->next[hash_gram(seq.data() + i, m)];
                const llama_token t = seq[i + m];
                if (std::find(nexts.begin(), nexts.end(), t) == nexts.end()) {
                    nexts.push_back(t);
                }
            }
        }
    }
    return llama_sampler_init(&ngram_penalty_iface, ctx);
}
