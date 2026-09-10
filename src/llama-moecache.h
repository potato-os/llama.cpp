#pragma once

// GPU-resident LRU cache for MoE expert weights that -ot pinned to host memory.
//
// Motivation (measured on Qwen3.8-Flash-Next, 512 experts / 10 routed): expert
// routing has strong temporal locality (LRU-64 hit rate ~67% over a mixed
// workload) even though the long-run distribution is near-uniform. Decode on a
// host-offloaded MoE layer is bound by host RAM bandwidth, so serving the hot
// experts from VRAM removes most of the per-token DIMM traffic.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, n_slots+1] live in the device buffer of that layer's router;
//    slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table[512] maps expert id -> slot, or n_slots when uncached.
//    One copy on device (read by get_rows to remap ids for the cache-side
//    mul_mat_id chain) and one on host (read by the CPU mul_mat_id via
//    src[3] to SKIP cached ids, zeroing their dst rows).
//  - the two down-projection outputs are summed; uncached ids contribute 0
//    through the cache chain (zero slot) and cached ids contribute 0 through
//    the CPU chain (skip), so the result is exact.
//  - llama_moe_cache_step(), called at the end of llama_context::decode(),
//    performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert
//    uploads per layer per step via ggml_backend_tensor_set.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include <cstdint>

struct llama_model;
struct ggml_tensor;

// The cache maps every uncached expert of a token to one shared dummy slot, so a token's slot-id
// row can contain the same id several times. Of the CUDA mul_mat_id paths only mul_mat_vec_q
// handles duplicate ids, and it is selected only up to MMVQ_MAX_BATCH_SIZE tokens, so this limit
// must not exceed that value (8).
#define LLAMA_MOE_CACHE_MAX_BATCH 8

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

// apply throttled LRU updates; call between graph executions only
void llama_moe_cache_step();

// Process-global phase transaction. The caller must exclude new graph/context
// execution AND synchronize all target/draft backends before quiesce() settles
// remap tables, then destroy every graph reference to cache tensors before
// release(). These calls
// do not synchronize schedulers or invalidate CUDA graphs on the caller's behalf.
//
// quiesce drains/joins the upload worker and settles all completed admissions.
// release retains authoritative host weights, capacity, slot/LRU metadata and
// host tables. restore reloads populated slots from those host weights and only
// publishes READY after every device group and the worker are ready. Failure
// leaves the cache disabled; there is no automatic retry or empty-cache fallback.
enum class llama_moe_cache_phase {
    unavailable, ready, quiescing, quiescent, released, restoring, failed,
};

bool llama_moe_cache_quiesce();
bool llama_moe_cache_release();
bool llama_moe_cache_restore();
llama_moe_cache_phase llama_moe_cache_get_phase();
uint64_t llama_moe_cache_get_generation(); // increments only on successful release/restore
int32_t llama_moe_cache_get_capacity();
bool llama_moe_cache_is_active();
