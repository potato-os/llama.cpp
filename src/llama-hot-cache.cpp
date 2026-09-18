// Copyright (c) 2026 LLMs For All, Inc.
// SPDX-License-Identifier: MIT

#include "llama-hot-cache.h"

#include <stdexcept>

#ifdef LLAMA_HOT_METAL
#    include "ggml-alloc.h"
#    include "ggml-backend.h"
#    include "ggml-hot-router.h"
#    include "llama-hot-policy.h"
#    include "llama-hot-profile.h"

#    include <dispatch/dispatch.h>
#    include <fcntl.h>
#    include <TargetConditionals.h>
#    include <unistd.h>

#    include <algorithm>
#    include <atomic>
#    include <cerrno>
#    include <cmath>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <limits>
#    include <map>
#    include <memory>

static constexpr bool gpu_routing = TARGET_OS_IPHONE;

static bool metal_buffer_is_shared(ggml_backend_buffer_t buffer) {
    const auto type   = ggml_backend_buffer_get_type(buffer);
    const auto device = ggml_backend_buft_get_device(type);
    const auto reg    = ggml_backend_dev_backend_reg(device);
    using predicate   = bool (*)(ggml_backend_buffer_t);
    const auto is_shared =
        reinterpret_cast<predicate>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_buffer_is_shared"));
    return is_shared && is_shared(buffer);
}

struct llama_hot_cache {
    int                           capacity;
    float                         bias;
    std::atomic<bool>             context_attached{ false };
    std::vector<std::vector<int>> initial_profiles;

    llama_hot_cache(const gguf_context * ctx, int n_layer, int n_expert, int count, float routing_bias) :
        capacity(count),
        bias(routing_bias),
        initial_profiles(llama_hot_profile(ctx, n_layer, n_expert, count)) {}

    int llama_hot_capacity() const { return capacity; }

    struct weight {
        ggml_tensor * tensor;
        int           fd;
        size_t        offset, stride;

        ~weight() { close(fd); }
    };

    struct layer_state {
        llama_hot_cache *                    owner;
        hot_policy                           policy;
        int                                  layer_index;
        std::vector<std::unique_ptr<weight>> weights;
        std::vector<unsigned char>           staging;
        ggml_hot_router_state                gpu_initial{};
        ggml_hot_router_bridge               gpu_bridge{ &gpu_initial, &gpu_initial };
        uint64_t                             tokens = 0;
        std::array<int, 256>                 union_old_slots{}, union_indices{};
        std::vector<int>                     union_experts, union_old_queue;
        uint64_t                             union_old_tokens = 0;
        bool                                 union_pending    = false;

        explicit layer_state(llama_hot_cache * cache, int layer) :
            owner(cache),
            policy(cache->capacity,
                   1234u + layer,
                   cache->initial_profiles.empty() ? std::vector<int>{} : cache->initial_profiles.at(layer)),
            layer_index(layer) {
            gpu_initial.capacity = cache->capacity;
            gpu_initial.bias     = cache->bias;
            std::copy(policy.queue.begin(), policy.queue.end(), gpu_initial.queue);
            std::copy(policy.slots.begin(), policy.slots.end(), gpu_initial.slots);
        }

        void load(weight & w, int expert, int slot) {
            staging.resize(w.stride);
            size_t done = 0;
            while (done < w.stride) {
                ssize_t n = pread(w.fd, staging.data() + done, w.stride - done, w.offset + expert * w.stride + done);
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                if (n <= 0) {
                    GGML_ABORT("hot-cache expert read failed: %s", strerror(errno));
                }
                done += n;
            }
            ggml_backend_tensor_set(w.tensor, staging.data(), slot * w.stride, w.stride);
        }
    };

    // Reused for at most eight admitted experts per decode step.
    std::array<std::vector<unsigned char>, 3> parallel_staging;

    struct read_job {
        weight *        w;
        unsigned char * data;
        size_t          offset;
        size_t          bytes = 0;
    };

    static void read_job_one(void * ctx) {
        auto &       job   = *static_cast<read_job *>(ctx);
        size_t       done  = 0;
        const size_t bytes = job.bytes ? job.bytes : job.w->stride;
        while (done < bytes) {
            ssize_t n = pread(job.w->fd, job.data + done, bytes - done, job.offset + done);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                GGML_ABORT("parallel hot-cache read failed: %s", strerror(errno));
            }
            done += n;
        }
    }

    static void read_job_run(void * ctx, size_t index) {
        read_job_one(&(*static_cast<std::vector<read_job> *>(ctx))[index]);
    }

    void load_admissions(layer_state & s, const std::vector<std::pair<int, int>> & loads) {
        if (loads.empty() || s.weights.empty()) {
            return;
        }

        GGML_ASSERT(s.weights.size() == 3 && loads.size() <= 8);
        std::vector<read_job> jobs;
        for (size_t w = 0; w < 3; w++) {
            auto & weight = *s.weights[w];
            parallel_staging[w].resize(weight.stride * loads.size());
            for (size_t i = 0; i < loads.size(); i++) {
                // The CPU routing split waits for prior GPU work; selected slots
                // are not consumed until this callback and every read complete.
                unsigned char * dest = parallel_staging[w].data() + i * weight.stride;
                jobs.push_back({ &weight, dest, weight.offset + loads[i].first * weight.stride });
            }
        }
        dispatch_apply_f(jobs.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &jobs, read_job_run);
        for (size_t w = 0; w < 3; w++) {
            for (size_t i = 0; i < loads.size(); i++) {
                auto & weight = *s.weights[w];
                ggml_backend_tensor_set(weight.tensor, parallel_staging[w].data() + i * weight.stride,
                                        loads[i].second * weight.stride, weight.stride);
            }
        }
    }

    std::map<int, std::unique_ptr<layer_state>> layers;

    layer_state & state(int il) {
        auto & p = layers[il];
        if (!p) {
            p.reset(new layer_state(this, il));
        }
        return *p;
    }

    // Match the Metal bitonic argsort, including its ordering of exact ties.
    struct union_workspace {
        ggml_context *        ctx    = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor *         matrices[3]{};
        int                   owner = -1;
    };

    union_workspace prefill_workspace;

    static int matrix_index(const ggml_tensor * t) {
        if (strstr(t->name, "ffn_down_exps")) {
            return 0;
        }
        if (strstr(t->name, "ffn_gate_exps")) {
            return 1;
        }
        if (strstr(t->name, "ffn_up_exps")) {
            return 2;
        }
        GGML_ABORT("unknown union matrix %s", t->name);
    }

    void prepare_union(layer_state & s) {
        auto & u = prefill_workspace;
        GGML_ASSERT(u.buffer && u.owner == -1 && !s.union_pending);
        u.owner         = s.layer_index;
        s.union_pending = true;
        std::vector<read_job> jobs;
        for (auto & wp : s.weights) {
            auto & w       = *wp;
            auto * scratch = u.matrices[matrix_index(w.tensor)];
            for (size_t i = 0; i < s.union_experts.size(); ++i) {
                int    e    = s.union_experts[i];
                auto * dest = static_cast<unsigned char *>(scratch->data) + i * w.stride;
                if (s.union_old_slots[e] >= 0) {
                    std::memcpy(dest, static_cast<unsigned char *>(w.tensor->data) + s.union_old_slots[e] * w.stride,
                                w.stride);
                } else {
                    const size_t offset = w.offset + e * w.stride;
                    jobs.push_back({ &w, dest, offset, w.stride });
                }
            }
        }
        if (!jobs.empty()) {
            dispatch_apply_f(jobs.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &jobs, read_job_run);
        }
    }

    static void finish_union(ggml_tensor * dst, const ggml_tensor * src, int ith, int, void * ud) {
        if (ith) {
            return;
        }
        auto & s = *static_cast<layer_state *>(ud);
        auto & u = s.owner->prefill_workspace;
        GGML_ASSERT(s.union_pending && u.owner == s.layer_index && ggml_is_contiguous(src));
        for (int e : s.policy.queue) {
            if (s.union_old_slots[e] != s.policy.slots[e]) {
                GGML_ASSERT(s.union_indices[e] >= 0);
                for (auto & wp : s.weights) {
                    auto & w       = *wp;
                    auto * scratch = u.matrices[matrix_index(w.tensor)];
                    std::memcpy(static_cast<unsigned char *>(w.tensor->data) + s.policy.slots[e] * w.stride,
                                static_cast<unsigned char *>(scratch->data) + s.union_indices[e] * w.stride, w.stride);
                }
            }
        }
        {
            auto & gpu = *s.gpu_bridge.state;
            std::copy(s.policy.queue.begin(), s.policy.queue.end(), gpu.queue);
            std::copy(s.policy.slots.begin(), s.policy.slots.end(), gpu.slots);
            gpu.tokens     = s.tokens;
            gpu.miss_count = 0;
        }
        u.owner         = -1;
        s.union_pending = false;
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }

    static void route_op(ggml_tensor * dst, int ith, int, void * ud) {
        if (ith) {
            return;
        }
        const auto * src = dst->src[0];
        GGML_ASSERT(src->type == GGML_TYPE_F32 && src->ne[0] == 256 && ggml_nrows(src) == src->ne[1]);
        auto & s = *static_cast<layer_state *>(ud);
        if (gpu_routing && src->ne[1] == 1) {
            GGML_ASSERT(src->ne[1] == 1 && s.weights.size() == 3);
            const auto & gpu = *s.gpu_bridge.state;
            GGML_ASSERT(gpu.tokens > int(s.tokens) && gpu.miss_count >= 0 && gpu.miss_count <= 8);

            std::vector<std::pair<int, int>> loads;
            for (int i = 0; i < gpu.miss_count; i++) {
                loads.emplace_back(gpu.request_expert[i], gpu.request_slot[i]);
            }

            s.owner->load_admissions(s, loads);
            s.tokens = gpu.tokens;
            return;
        }
        GGML_ASSERT(s.weights.size() == 3);
        const bool batch_union = src->ne[1] > 1;
        if (batch_union) {
            if (gpu_routing) {
                const auto & gpu = *s.gpu_bridge.state;
                s.policy.queue.assign(gpu.queue, gpu.queue + gpu.capacity);
                std::copy(gpu.slots, gpu.slots + 256, s.policy.slots.begin());
                s.tokens = gpu.tokens;
            }
            s.union_old_slots  = s.policy.slots;
            s.union_old_queue  = s.policy.queue;
            s.union_old_tokens = s.tokens;
            s.union_indices.fill(-1);
            s.union_experts.clear();
        }
        const float bias = s.owner->bias;
        for (int token = 0; token < src->ne[1]; token++) {
            std::array<float, 256> scores;
            std::array<int, 256>   indices;
            const auto *           in =
                reinterpret_cast<const float *>(static_cast<const char *>(src->data) + token * src->nb[1]);
            for (int e = 0; e < 256; ++e) {
                scores[e]  = in[e] + (s.policy.slots[e] >= 0 ? bias : 0);
                indices[e] = e;
            }
            for (int k = 2; k <= 256; k *= 2) {
                for (int j = k / 2; j > 0; j /= 2) {
                    for (int col = 0; col < 256; col++) {
                        const int other = col ^ j;
                        if (other <= col) {
                            continue;
                        }
                        bool swap = (col & k) == 0 ? scores[indices[col]] < scores[indices[other]] :
                                                     scores[indices[col]] > scores[indices[other]];
                        if (swap) {
                            std::swap(indices[col], indices[other]);
                        }
                    }
                }
            }
            std::vector<int> selected(indices.begin(), indices.begin() + 8);

            auto loads = s.policy.update(selected);
            if (!batch_union) {
                s.owner->load_admissions(s, loads);
            }

            auto * out = reinterpret_cast<int32_t *>(static_cast<char *>(dst->data) + token * dst->nb[1]);
            for (int i = 0; i < 8; i++) {
                int e  = selected[i];
                out[i] = e;
                if (batch_union && s.union_indices[e] < 0) {
                    s.union_indices[e] = s.union_experts.size();
                    s.union_experts.push_back(e);
                }
                out[8 + i] = batch_union ? s.union_indices[e] : s.policy.slots[e];
            }
            ++s.tokens;
        }
        if (batch_union) {
            s.owner->prepare_union(s);
        } else {
            auto & gpu = *s.gpu_bridge.state;
            std::copy(s.policy.queue.begin(), s.policy.queue.end(), gpu.queue);
            std::copy(s.policy.slots.begin(), s.policy.slots.end(), gpu.slots);
            gpu.tokens     = s.tokens;
            gpu.miss_count = 0;
        }
    }

    void llama_hot_bind(ggml_tensor * tensor, int fd, size_t offset, size_t stride) {
        int il = -1;
        GGML_ASSERT(sscanf(tensor->name, "blk.%d.", &il) == 1);
        auto & s  = state(il);
        auto   w  = std::make_unique<weight>();
        w->tensor = tensor;
        w->fd     = dup(fd);
        w->offset = offset;
        w->stride = stride;
        if (w->fd < 0) {
            throw std::runtime_error("dup failed");
        }
        if (fcntl(w->fd, F_NOCACHE, 1) < 0) {
            throw std::runtime_error("could not configure expert reads");
        }
        GGML_ASSERT(tensor->ne[2] == llama_hot_capacity() && tensor->nb[2] == stride);
        for (int e : s.policy.queue) {
            s.load(*w, e, s.policy.slots[e]);
        }
        s.weights.push_back(std::move(w));
    }

    ggml_tensor * llama_hot_route(ggml_context * ctx, ggml_tensor * logits, int il) {
        ggml_tensor * args[] = { logits };
        auto * out = ggml_custom_4d(ctx, GGML_TYPE_I32, 16, logits->ne[1], 1, 1, args, 1, route_op, 1, &state(il));
        // Marker stored after the custom operation parameters.
        if (gpu_routing) {
            out->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0x484f5431;
        }
        auto * bridge = &state(il).gpu_bridge;
        std::memcpy(out->op_params + 8, &bridge, sizeof(bridge));
        return out;
    }

    ggml_tensor * llama_hot_union_weight(ggml_tensor * original, int il) {
        auto & u = prefill_workspace;
        auto & s = state(il);
        GGML_ASSERT(s.weights.size() == 3 && metal_buffer_is_shared(original->buffer));
        if (!u.ctx) {
            ggml_init_params params{ ggml_tensor_overhead() * 4, nullptr, true };
            u.ctx = ggml_init(params);
            GGML_ASSERT(u.ctx);
            for (auto & wp : s.weights) {
                auto * t      = wp->tensor;
                int    i      = matrix_index(t);
                u.matrices[i] = ggml_new_tensor_3d(u.ctx, t->type, t->ne[0], t->ne[1], 256);
                ggml_format_name(u.matrices[i], "hot_prefill_union_%d", i);
            }
            u.buffer = ggml_backend_alloc_ctx_tensors_from_buft(u.ctx, ggml_backend_buffer_get_type(original->buffer));
            GGML_ASSERT(u.buffer);
            ggml_backend_buffer_set_usage(u.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        auto * t = u.matrices[matrix_index(original)];
        GGML_ASSERT(t->type == original->type && t->ne[0] == original->ne[0] && t->ne[1] == original->ne[1]);
        return t;
    }

    ggml_tensor * llama_hot_union_finish(ggml_context * ctx, ggml_tensor * cur, int il) {
        return ggml_map_custom1(ctx, cur, finish_union, 1, &state(il));
    }

    void llama_hot_abort_pending() {
        for (auto & entry : layers) {
            auto & s = *entry.second;
            if (!s.union_pending) {
                continue;
            }
            s.policy.queue  = s.union_old_queue;
            s.policy.slots  = s.union_old_slots;
            s.tokens        = s.union_old_tokens;
            s.union_pending = false;
            s.union_experts.clear();
        }
        prefill_workspace.owner = -1;
    }

    void llama_hot_release_workspace() {
        GGML_ASSERT(prefill_workspace.owner == -1);
        if (prefill_workspace.buffer) {
            ggml_backend_buffer_free(prefill_workspace.buffer);
        }
        if (prefill_workspace.ctx) {
            ggml_free(prefill_workspace.ctx);
        }
        prefill_workspace = union_workspace{};
    }

    struct saved_layer {
        int32_t               layer;
        ggml_hot_router_state router;
    };

    std::vector<uint8_t> llama_hot_save_state() {
        std::vector<uint8_t> result;
        const uint32_t       header[3] = { 0x48535431, 1, (uint32_t) layers.size() };
        result.insert(result.end(), (const uint8_t *) header, (const uint8_t *) (header + 3));
        for (auto & entry : layers) {
            auto & s = *entry.second;
            if (s.union_pending) {
                throw std::runtime_error("cannot save unfinished layer state");
            }
            saved_layer item{};
            item.layer  = entry.first;
            item.router = *s.gpu_bridge.state;

            item.router.miss_count = 0;
            std::fill(std::begin(item.router.request_expert), std::end(item.router.request_expert), 0);
            std::fill(std::begin(item.router.request_slot), std::end(item.router.request_slot), 0);
            result.insert(result.end(), (const uint8_t *) &item, (const uint8_t *) (&item + 1));
        }
        return result;
    }

    void llama_hot_load_state(const std::vector<uint8_t> & data) {
        uint32_t header[3];
        if (data.size() < sizeof(header)) {
            throw std::runtime_error("truncated layer state");
        }
        memcpy(header, data.data(), sizeof(header));
        if (header[0] != 0x48535431 || header[1] != 1 || header[2] != layers.size() ||
            data.size() != sizeof(header) + header[2] * sizeof(saved_layer)) {
            throw std::runtime_error("incompatible layer state");
        }
        std::vector<saved_layer> saved(header[2]);
        memcpy(saved.data(), data.data() + sizeof(header), saved.size() * sizeof(saved_layer));
        size_t i = 0;
        for (auto & entry : layers) {
            const auto & x = saved[i++];
            auto &       s = *entry.second;
            if (s.union_pending || x.layer != entry.first || x.router.capacity != s.gpu_initial.capacity ||
                x.router.bias != s.gpu_initial.bias) {
                throw std::runtime_error("incompatible layer policy");
            }
            std::array<bool, 256> used{};
            std::array<bool, 256> slots{};
            for (int j = 0; j < x.router.capacity; j++) {
                const int e = x.router.queue[j];
                if (e < 0 || e >= 256 || used[e]) {
                    throw std::runtime_error("invalid layer queue");
                }
                used[e]  = true;
                int slot = x.router.slots[e];
                if (slot < 0 || slot >= x.router.capacity || slots[slot]) {
                    throw std::runtime_error("invalid layer slot");
                }
                slots[slot] = true;
            }
            for (int e = 0; e < 256; e++) {
                if (!used[e] && x.router.slots[e] != -1) {
                    throw std::runtime_error("invalid cold slot");
                }
            }
        }
        i = 0;
        for (auto & entry : layers) {
            auto &               s = *entry.second;
            auto                 x = saved[i++].router;
            std::array<int, 256> current;
            std::copy(std::begin(s.gpu_bridge.state->slots), std::end(s.gpu_bridge.state->slots), current.begin());

            for (int j = 0; j < x.capacity; j++) {
                const int e = x.queue[j], slot = x.slots[e];
                if (current[e] != slot) {
                    for (auto & w : s.weights) {
                        s.load(*w, e, slot);
                    }
                }
            }
            s.policy.queue.assign(x.queue, x.queue + x.capacity);
            std::copy(std::begin(x.slots), std::end(x.slots), s.policy.slots.begin());
            s.tokens            = x.tokens;
            s.gpu_initial       = x;
            *s.gpu_bridge.state = x;
        }
    }

    void llama_hot_clear() {
        llama_hot_release_workspace();
        layers.clear();
        initial_profiles.clear();
        for (auto & buffer : parallel_staging) {
            std::vector<unsigned char>().swap(buffer);
        }
    }
};

llama_hot_cache * llama_hot_create(const gguf_context * ctx, int layers, int experts, int capacity, float bias) {
    if (experts != 256 || capacity < 8 || capacity > 256 || !std::isfinite(bias) || bias < 0) {
        throw std::runtime_error("invalid hot expert configuration");
    }
    return new llama_hot_cache(ctx, layers, experts, capacity, bias);
}

void llama_hot_free(llama_hot_cache * cache) {
    if (cache) {
        cache->llama_hot_clear();
        delete cache;
    }
}

int llama_hot_capacity(const llama_hot_cache * cache) {
    return cache->capacity;
}

void llama_hot_attach_context(llama_hot_cache * cache) {
    if (cache->context_attached.exchange(true)) {
        throw std::runtime_error("hot experts support one context per model");
    }
}

void llama_hot_detach_context(llama_hot_cache * cache) {
    cache->context_attached = false;
}

void llama_hot_bind(llama_hot_cache * cache, ggml_tensor * tensor, int fd, size_t offset, size_t stride) {
    cache->llama_hot_bind(tensor, fd, offset, stride);
}

ggml_tensor * llama_hot_route(llama_hot_cache * cache, ggml_context * ctx, ggml_tensor * logits, int layer) {
    return cache->llama_hot_route(ctx, logits, layer);
}

ggml_tensor * llama_hot_union_weight(llama_hot_cache * cache, ggml_tensor * original, int layer) {
    return cache->llama_hot_union_weight(original, layer);
}

ggml_tensor * llama_hot_union_finish(llama_hot_cache * cache, ggml_context * ctx, ggml_tensor * cur, int layer) {
    return cache->llama_hot_union_finish(ctx, cur, layer);
}

void llama_hot_abort_pending(llama_hot_cache * cache) {
    cache->llama_hot_abort_pending();
}

void llama_hot_release_workspace(llama_hot_cache * cache) {
    cache->llama_hot_release_workspace();
}

std::vector<uint8_t> llama_hot_save_state(llama_hot_cache * cache) {
    return cache->llama_hot_save_state();
}

void llama_hot_load_state(llama_hot_cache * cache, const std::vector<uint8_t> & data) {
    cache->llama_hot_load_state(data);
}

#else
llama_hot_cache * llama_hot_create(const gguf_context *, int, int, int, float) {
    throw std::runtime_error("hot experts require the Metal backend");
}

void llama_hot_free(llama_hot_cache *) {}

int llama_hot_capacity(const llama_hot_cache *) {
    throw std::runtime_error("hot experts require the Metal backend");
}

void llama_hot_attach_context(llama_hot_cache *) {
    throw std::runtime_error("hot experts require the Metal backend");
}

void llama_hot_detach_context(llama_hot_cache *) {
    throw std::runtime_error("hot experts require the Metal backend");
}

void llama_hot_bind(llama_hot_cache *, ggml_tensor *, int, size_t, size_t) {
    throw std::runtime_error("hot experts require the Metal backend");
}

ggml_tensor * llama_hot_route(llama_hot_cache *, ggml_context *, ggml_tensor *, int) {
    throw std::runtime_error("hot experts require the Metal backend");
}

ggml_tensor * llama_hot_union_weight(llama_hot_cache *, ggml_tensor *, int) {
    throw std::runtime_error("hot experts require the Metal backend");
}

ggml_tensor * llama_hot_union_finish(llama_hot_cache *, ggml_context *, ggml_tensor *, int) {
    throw std::runtime_error("hot experts require the Metal backend");
}

void llama_hot_abort_pending(llama_hot_cache *) {
    throw std::runtime_error("hot experts require the Metal backend");
}

void llama_hot_release_workspace(llama_hot_cache *) {
    throw std::runtime_error("hot experts require the Metal backend");
}

std::vector<uint8_t> llama_hot_save_state(llama_hot_cache *) {
    throw std::runtime_error("hot experts require the Metal backend");
}

void llama_hot_load_state(llama_hot_cache *, const std::vector<uint8_t> &) {
    throw std::runtime_error("hot experts require the Metal backend");
}

#endif
