// Copyright (c) 2026 LLMs For All, Inc.
// SPDX-License-Identifier: MIT

#pragma once

struct ggml_hot_router_state {
    int   capacity;
    float bias;
    int   tokens;
    int   miss_count;
    int   queue[256];
    int   slots[256];
    int   request_expert[8];
    int   request_slot[8];
};

struct ggml_hot_router_bridge {
    struct ggml_hot_router_state * state;
    struct ggml_hot_router_state * host_backup;
};
