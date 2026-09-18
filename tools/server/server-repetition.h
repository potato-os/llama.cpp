#pragma once

// Repetition stop: detect a generation that is looping and stop it early.
//
// Rule (from the user's spec, deliberately narrow so legitimate structured
// output is not flagged): over a trailing window of W generated tokens, if a
// single gram of some length L occurs at least `min_repeats` times
// (non-overlapping) and those occurrences cover at least `coverage` of the
// window, the generation is repeating. Several window sizes are evaluated so
// short loops are caught early and long ones are still caught.
//
// This header is dependency-free so it can be unit-tested directly.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <vector>

struct repeat_stop_params {
    bool enabled = false;
    std::vector<int32_t> windows = {300, 600, 900};
    int32_t min_tokens  = 300;  // never judge fewer generated tokens than this
    int32_t min_repeats = 3;    // the gram must occur at least this often
    float   coverage    = 0.8f; // ... covering at least this fraction of the window
    int32_t stride      = 4;    // evaluate every N generated tokens
};

struct repetition_stats {
    bool    flagged   = false;
    int32_t window    = 0;      // window size that produced the stats
    int32_t gram_len  = 0;      // L of the dominant gram
    int32_t count     = 0;      // non-overlapping occurrences
    float   coverage  = 0.0f;   // count * L / window
    size_t  gram_pos  = 0;      // index (into the token vector) of one occurrence
};

namespace repetition_detail {

inline uint64_t gram_hash(const int32_t * toks, int32_t len) {
    uint64_t h = 1469598103934665603ULL ^ (uint64_t) len;
    for (int32_t i = 0; i < len; ++i) {
        h ^= (uint64_t) (uint32_t) toks[i];
        h *= 1099511628211ULL;
        h ^= h >> 29;
    }
    return h;
}

// Gram lengths to test: every length from 3 up to 16, then geometric
// (ratio ~1.12) up to max_len. A repeating gram of length L is at least as
// well covered by the nearest tested L' <= L, so nothing is missed, only
// slightly under-measured.
inline std::vector<int32_t> gram_lengths(int32_t max_len) {
    std::vector<int32_t> out;
    for (int32_t l = 3; l <= std::min(16, max_len); ++l) {
        out.push_back(l);
    }
    double l = 16.0;
    while (true) {
        l *= 1.12;
        const int32_t li = (int32_t) l;
        if (li > max_len) {
            break;
        }
        if (out.empty() || li > out.back()) {
            out.push_back(li);
        }
    }
    return out;
}

} // namespace repetition_detail

// Evaluate one window (the last `w` tokens of `toks`).
inline repetition_stats detect_repetition_window(const std::vector<int32_t> & toks, int32_t w, const repeat_stop_params & p) {
    repetition_stats best;
    if ((int32_t) toks.size() < w || w <= 0) {
        return best;
    }
    const int32_t * tail = toks.data() + (toks.size() - w);
    best.window = w;
    const int32_t max_len = w / std::max(1, p.min_repeats);
    for (int32_t L : repetition_detail::gram_lengths(max_len)) {
        // positions of every L-gram, by hash
        std::unordered_map<uint64_t, std::vector<int32_t>> pos;
        pos.reserve(w);
        for (int32_t i = 0; i + L <= w; ++i) {
            pos[repetition_detail::gram_hash(tail + i, L)].push_back(i);
        }
        // dominant gram
        const std::vector<int32_t> * dom = nullptr;
        for (const auto & kv : pos) {
            if (dom == nullptr || kv.second.size() > dom->size()) {
                dom = &kv.second;
            }
        }
        if (dom == nullptr || (int32_t) dom->size() < p.min_repeats) {
            continue;
        }
        // non-overlapping occurrences, verified token-by-token against the
        // first one (hash collisions are excluded rather than trusted)
        const int32_t first = (*dom)[0];
        int32_t count = 0;
        int32_t last_end = -1;
        for (int32_t i : *dom) {
            if (i < last_end) {
                continue;
            }
            if (std::memcmp(tail + i, tail + first, L * sizeof(int32_t)) != 0) {
                continue;
            }
            ++count;
            last_end = i + L;
        }
        const float cov = (float) count * (float) L / (float) w;
        if (count >= p.min_repeats && cov > best.coverage) {
            best.gram_len = L;
            best.count    = count;
            best.coverage = cov;
            best.gram_pos = (toks.size() - w) + first;
        }
    }
    best.flagged = best.count >= p.min_repeats && best.coverage >= p.coverage;
    return best;
}

// Evaluate all configured windows; returns the flagged one if any, else the
// window with the highest coverage (for reporting on cap hits).
inline repetition_stats detect_repetition(const std::vector<int32_t> & toks, const repeat_stop_params & p) {
    repetition_stats best;
    if ((int32_t) toks.size() < p.min_tokens) {
        return best;
    }
    for (int32_t w : p.windows) {
        const int32_t w_eff = std::min<int32_t>(w, (int32_t) toks.size());
        if (w_eff < p.min_tokens) {
            continue;
        }
        repetition_stats st = detect_repetition_window(toks, w_eff, p);
        if (st.flagged) {
            return st;
        }
        if (st.coverage > best.coverage) {
            best = st;
        }
    }
    return best;
}
