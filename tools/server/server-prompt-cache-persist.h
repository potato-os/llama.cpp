#pragma once

// Millie fork addition: persist the in-memory server_prompt_cache to disk so
// prefilled prompt state (system prompt, conversation prefixes) survives a
// server restart. This is additive -- it serializes the cache's existing
// full-state blobs (llama_state_seq_get/set_data_ext, flags=0, which already
// includes recurrent state) and reloads them into `states`, where the stock
// reuse path (server_prompt_cache::load) picks them up unchanged.
//
// A caller-supplied validation key (model id + effective launch settings)
// is stored in the header and must match on load; a mismatch means the blobs
// were produced by a different model/kv-type/context and would decode to
// garbage, so the file is rejected rather than loaded.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "server-task.h"

namespace millie_prompt_cache {

static constexpr char     MAGIC[4] = {'M', 'L', 'P', 'C'};
static constexpr uint32_t VERSION  = 1;

template <typename T>
static bool write_pod(FILE * f, const T & v) {
    return std::fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
static bool read_pod(FILE * f, T & v) {
    return std::fread(&v, sizeof(T), 1, f) == 1;
}

// Serialize every cached prompt state (tokens + full-state blob) to `path`.
// Returns true on success. Existing file is overwritten atomically via a
// temporary + rename.
inline bool save_file(const server_prompt_cache & cache, const std::string & path, const std::string & key) {
    const std::string tmp = path + ".tmp";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    bool ok = true;
    ok = ok && std::fwrite(MAGIC, 1, 4, f) == 4;
    ok = ok && write_pod(f, VERSION);
    const uint32_t key_len = (uint32_t) key.size();
    ok = ok && write_pod(f, key_len);
    ok = ok && (key_len == 0 || std::fwrite(key.data(), 1, key_len, f) == key_len);

    const uint32_t n_states = (uint32_t) cache.states.size();
    ok = ok && write_pod(f, n_states);
    for (const auto & st : cache.states) {
        if (!ok) break;
        // get_text_tokens() is safe when a vision tower is loaded
        // (get_tokens() asserts !has_mtmd); text prompts round-trip identically.
        const llama_tokens toks = st.tokens.get_text_tokens();
        const uint32_t n_tokens = (uint32_t) toks.size();
        const uint64_t data_size = (uint64_t) st.data.size();
        ok = ok && write_pod(f, n_tokens);
        ok = ok && (n_tokens == 0 || std::fwrite(toks.data(), sizeof(llama_token), n_tokens, f) == n_tokens);
        ok = ok && write_pod(f, data_size);
        ok = ok && (data_size == 0 || std::fwrite(st.data.data(), 1, data_size, f) == data_size);
    }
    std::fclose(f);
    if (!ok) {
        std::remove(tmp.c_str());
        return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

// Load prompt states from `path` into `cache.states`. Rejects the file (returns
// false, cache untouched) if the magic/version mismatch or the stored key does
// not equal `key`. Loaded states carry no checkpoints -- the full-state blob is
// what the reuse path restores.
//
// `has_mtmd` must reflect whether the server runs with a multimodal context
// (mctx != nullptr): a slot that adopts a restored state inherits its
// server_tokens, and pushing the first media chunk into one constructed
// without mtmd support aborts on GGML_ASSERT(has_mtmd).
inline bool load_file(server_prompt_cache & cache, const std::string & path, const std::string & key, bool has_mtmd) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    char magic[4] = {};
    uint32_t version = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, MAGIC, 4) == 0;
    ok = ok && read_pod(f, version) && version == VERSION;
    uint32_t key_len = 0;
    ok = ok && read_pod(f, key_len);
    std::string file_key;
    if (ok && key_len > 0) {
        file_key.resize(key_len);
        ok = std::fread(file_key.data(), 1, key_len, f) == key_len;
    }
    if (!ok || file_key != key) {
        std::fclose(f);
        return false;
    }

    uint32_t n_states = 0;
    ok = read_pod(f, n_states);
    std::vector<server_prompt> loaded;
    for (uint32_t i = 0; ok && i < n_states; ++i) {
        uint32_t n_tokens = 0;
        ok = read_pod(f, n_tokens);
        llama_tokens toks(n_tokens);
        ok = ok && (n_tokens == 0 || std::fread(toks.data(), sizeof(llama_token), n_tokens, f) == n_tokens);
        uint64_t data_size = 0;
        ok = ok && read_pod(f, data_size);
        std::vector<uint8_t> data(data_size);
        ok = ok && (data_size == 0 || std::fread(data.data(), 1, data_size, f) == data_size);
        if (!ok) break;
        server_prompt p;
        p.tokens = server_tokens(toks, has_mtmd);
        p.data = std::move(data);
        loaded.push_back(std::move(p));
    }
    std::fclose(f);
    if (!ok) {
        return false;
    }
    for (auto & p : loaded) {
        cache.states.push_back(std::move(p));
    }
    return true;
}

} // namespace millie_prompt_cache
