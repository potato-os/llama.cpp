// Copyright (c) 2026 LLMs For All, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "ggml.h"

#include <cstddef>
#include <vector>
struct gguf_context;
struct llama_hot_cache;
llama_hot_cache * llama_hot_create(const gguf_context * ctx, int layers, int experts, int capacity, float bias);
void              llama_hot_free(llama_hot_cache * cache);
int               llama_hot_capacity(const llama_hot_cache * cache);
void              llama_hot_attach_context(llama_hot_cache * cache);
void              llama_hot_detach_context(llama_hot_cache * cache);
void              llama_hot_bind(llama_hot_cache * cache, ggml_tensor * tensor, int fd, size_t offset, size_t stride);
ggml_tensor *     llama_hot_route(llama_hot_cache * cache, ggml_context * ctx, ggml_tensor * logits, int layer);
ggml_tensor *     llama_hot_union_weight(llama_hot_cache * cache, ggml_tensor * original, int layer);
ggml_tensor *     llama_hot_union_finish(llama_hot_cache * cache, ggml_context * ctx, ggml_tensor * cur, int layer);
void              llama_hot_abort_pending(llama_hot_cache * cache);
void              llama_hot_release_workspace(llama_hot_cache * cache);
std::vector<uint8_t> llama_hot_save_state(llama_hot_cache * cache);
void                 llama_hot_load_state(llama_hot_cache * cache, const std::vector<uint8_t> & data);
