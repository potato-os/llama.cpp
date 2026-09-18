// Copyright (c) 2026 LLMs For All, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "gguf.h"

#include <array>
#include <stdexcept>
#include <vector>

inline std::vector<std::vector<int>> llama_hot_profile(const gguf_context * ctx,
                                                       int                  layers,
                                                       int                  experts,
                                                       int                  capacity) {
    const int64_t key = gguf_find_key(ctx, "millie.expert_initial_ids");
    if (key < 0) {
        return {};
    }
    const auto invalid = [] {
        throw std::runtime_error("invalid millie.expert_initial_ids metadata");
    };
    if (layers <= 0 || experts != 256 || capacity < 8 || capacity > experts) {
        invalid();
    }
    if (gguf_get_kv_type(ctx, key) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, key) != GGUF_TYPE_UINT32) {
        invalid();
    }
    const size_t count = gguf_get_arr_n(ctx, key);
    if (count % layers != 0 || count / layers < static_cast<size_t>(capacity) ||
        count / layers > static_cast<size_t>(experts)) {
        invalid();
    }
    const size_t                  width = count / layers;
    const auto *                  ids   = static_cast<const uint32_t *>(gguf_get_arr_data(ctx, key));
    std::vector<std::vector<int>> result(layers);
    for (int layer = 0; layer < layers; ++layer) {
        std::array<bool, 256> seen{};
        for (size_t i = 0; i < width; ++i) {
            const uint32_t id = ids[layer * width + i];
            if (id >= static_cast<uint32_t>(experts) || seen[id]) {
                invalid();
            }
            seen[id] = true;
            if (i < static_cast<size_t>(capacity)) {
                result[layer].push_back(id);
            }
        }
    }
    return result;
}
