// Tests for the repetition detector (tools/server/server-repetition.h) and
// the n-gram penalty sampler (common/ngram-penalty.h).
#include "../tools/server/server-repetition.h"
#include "ngram-penalty.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static std::vector<int32_t> repeat_block(const std::vector<int32_t> & block, int times) {
    std::vector<int32_t> out;
    for (int i = 0; i < times; ++i) {
        out.insert(out.end(), block.begin(), block.end());
    }
    return out;
}

static std::vector<int32_t> seq(int32_t from, int32_t n) {
    std::vector<int32_t> v;
    for (int32_t i = 0; i < n; ++i) v.push_back(from + i);
    return v;
}

int main() {
    repeat_stop_params p;
    p.enabled = true;

    // 1. verbatim loop: a 40-token block repeated 10 times -> flagged
    {
        auto toks = repeat_block(seq(1000, 40), 10);
        auto st = detect_repetition(toks, p);
        CHECK(st.flagged);
        CHECK(st.count >= 3);
        CHECK(st.coverage >= 0.8f);
    }
    // 2. two long names alternating in a numbered list -> NOT flagged
    //    "N. <nameA 8 tokens>\n" / "N. <nameB 8 tokens>\n", 100 lines, numbers as single digit tokens
    {
        std::vector<int32_t> toks;
        auto nameA = seq(5000, 8), nameB = seq(6000, 8);
        for (int i = 1; i <= 100; ++i) {
            for (char c : std::to_string(i)) toks.push_back(100 + (c - '0')); // digit tokens
            toks.push_back(7); // "."
            const auto & name = (i % 2) ? nameA : nameB;
            toks.insert(toks.end(), name.begin(), name.end());
            toks.push_back(8); // "\n"
        }
        auto st = detect_repetition(toks, p);
        CHECK(!st.flagged);
        CHECK(st.coverage < 0.8f);
    }
    // 3. many different names, each repeated a few times in no particular
    //    order (an LCG picks the name, so there is no long repeating cycle) -> NOT flagged
    {
        std::vector<int32_t> toks;
        uint32_t rng = 12345;
        for (int i = 0; i < 120; ++i) {
            rng = rng * 1664525u + 1013904223u;
            auto name = seq(9000 + (int32_t) ((rng >> 16) % 20) * 10, 5);
            toks.insert(toks.end(), name.begin(), name.end());
            toks.push_back(8);
        }
        auto st = detect_repetition(toks, p);
        CHECK(!st.flagged);
    }
    // 4. below min_tokens nothing is judged
    {
        auto toks = repeat_block(seq(1, 20), 10); // 200 tokens
        auto st = detect_repetition(toks, p);
        CHECK(!st.flagged);
    }
    // 5. long cycle (250 tokens x 3 = 750) caught by the 900 window, not the 300 one
    {
        auto toks = repeat_block(seq(1, 250), 4); // 1000 tokens
        auto st = detect_repetition(toks, p);
        CHECK(st.flagged);
        CHECK(st.window >= 600);
    }

    // n-gram penalty: reference "a b c d e", start_n 3, scale 1, max 10
    {
        std::vector<std::vector<llama_token>> refs = { {1, 2, 3, 4, 5} };
        llama_sampler * s = common_ngram_penalty_init(refs, /*start_n*/ 3, /*scale*/ 1.0f, /*max*/ 10.0f);
        std::vector<llama_token_data> data;
        for (llama_token t = 0; t < 8; ++t) data.push_back({t, 0.0f, 0.0f});
        llama_token_data_array cur = { data.data(), data.size(), -1, false };
        // tail "1 2": candidate 3 would make a 3-gram -> penalty 1
        llama_sampler_accept(s, 1);
        llama_sampler_accept(s, 2);
        llama_sampler_apply(s, &cur);
        CHECK(data[3].logit == -1.0f);
        CHECK(data[4].logit == 0.0f);
        CHECK(data[1].logit == 0.0f);
        // tail "1 2 3": candidate 4 would make a 4-gram -> penalty 2 (the 3-gram "2 3 4" alone would be 1)
        for (auto & d : data) d.logit = 0.0f;
        llama_sampler_accept(s, 3);
        llama_sampler_apply(s, &cur);
        CHECK(data[4].logit == -2.0f);
        // unrelated tail: nothing penalized
        for (auto & d : data) d.logit = 0.0f;
        llama_sampler_reset(s);
        llama_sampler_accept(s, 7);
        llama_sampler_accept(s, 7);
        llama_sampler_apply(s, &cur);
        for (auto & d : data) CHECK(d.logit == 0.0f);
        llama_sampler_free(s);
    }
    // cap: with scale 4 a 5-gram would be 12 -> capped at 10
    {
        std::vector<std::vector<llama_token>> refs = { {1, 2, 3, 4, 5, 6} };
        llama_sampler * s = common_ngram_penalty_init(refs, 3, 4.0f, 10.0f);
        std::vector<llama_token_data> data;
        for (llama_token t = 0; t < 8; ++t) data.push_back({t, 0.0f, 0.0f});
        llama_token_data_array cur = { data.data(), data.size(), -1, false };
        for (llama_token t : {1, 2, 3, 4}) llama_sampler_accept(s, t);
        llama_sampler_apply(s, &cur);
        CHECK(data[5].logit == -10.0f);
        llama_sampler_free(s);
    }

    // start_n 1: every span token costs scale with no context; context still escalates
    {
        std::vector<std::vector<llama_token>> refs = { {1, 2, 3} };
        llama_sampler * s = common_ngram_penalty_init(refs, /*start_n*/ 1, /*scale*/ 2.0f, /*max*/ 10.0f);
        std::vector<llama_token_data> data;
        for (llama_token t = 0; t < 8; ++t) data.push_back({t, 0.0f, 0.0f});
        llama_token_data_array cur = { data.data(), data.size(), -1, false };
        llama_sampler_apply(s, &cur); // empty tail
        CHECK(data[1].logit == -2.0f);
        CHECK(data[2].logit == -2.0f);
        CHECK(data[3].logit == -2.0f);
        CHECK(data[0].logit == 0.0f);
        CHECK(data[4].logit == 0.0f);
        for (auto & d : data) d.logit = 0.0f;
        llama_sampler_accept(s, 1);
        llama_sampler_apply(s, &cur); // tail "1": 2 extends a 2-gram -> 4; 3 is a bare 1-gram -> 2
        CHECK(data[2].logit == -4.0f);
        CHECK(data[3].logit == -2.0f);
        CHECK(data[1].logit == -2.0f);
        llama_sampler_free(s);
    }

    if (failures == 0) {
        printf("test-repetition: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test-repetition: %d failures\n", failures);
    return 1;
}
