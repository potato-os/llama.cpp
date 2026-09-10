#include "llama-moecache.h"

#ifdef LLAMA_MOE_CACHE_TEST_BACKEND
#include "fake-backend.hpp"
#else
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#endif

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending
    std::vector<bool>     expert_in_flight; // don't admit the same expert twice before completion

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct device_group {
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<size_t> layers;
};

struct moe_cache {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;
    std::vector<device_group> device_groups;
    std::atomic<llama_moe_cache_phase> phase { llama_moe_cache_phase::unavailable };
    std::atomic<uint64_t> generation { 0 };

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
    bool                     worker_failed = false;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    if (mc->phase.load() != llama_moe_cache_phase::ready) {
        return; // includes short (<=8 token) tails during the entire suspended phase
    }

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > LLAMA_MOE_CACHE_MAX_BATCH) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    if (mc->phase.load() != llama_moe_cache_phase::ready) {
        return;
    }
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

bool upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return false;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
    return true;
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

bool cache_fail(moe_cache * mc, const char * message) {
    if (mc) {
        mc->phase.store(llama_moe_cache_phase::failed);
    }
    LLAMA_LOG_ERROR("moe-cache: phase transaction failed: %s; cache remains disabled\n", message);
    return false;
}

void create_device_tensors(ggml_context * ctx, llama_moe_cache_layer & pub) {
    const ggml_tensor * u = pub.up_src;
    const ggml_tensor * g = pub.gate_src;
    const ggml_tensor * d = pub.down_src;
    pub.up_c      = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], pub.n_slots + 1);
    pub.gate_c    = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], pub.n_slots + 1);
    pub.down_c    = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], pub.n_slots + 1);
    pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
    ggml_format_name(pub.up_c,      "moe_cache_up.%d",   pub.il);
    ggml_format_name(pub.gate_c,    "moe_cache_gate.%d", pub.il);
    ggml_format_name(pub.down_c,    "moe_cache_down.%d", pub.il);
    ggml_format_name(pub.dev_table, "moe_cache_tbl.%d",  pub.il);
}

bool start_worker(moe_cache * mc) {
    if (mc->worker.joinable()) {
        return false;
    }
    mc->stop = false;
    try {
        mc->worker = std::thread([mc]() {
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->todo.empty() && mc->stop) {
                        return; // stop drains every queued job, not just the current upload
                    }
                    j = mc->todo.front();
                    mc->todo.pop_front();
                }
                bool ok = false;
                try {
                    auto & ls = mc->layers[j.layer_idx];
                    ok = upload_slice(ls.pub.up_c, ls.pub.up_src, j.expert, j.slot) &&
                         upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot) &&
                         upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                } catch (const std::exception & e) {
                    LLAMA_LOG_ERROR("moe-cache: upload worker failed: %s\n", e.what());
                }
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    if (!ok) {
                        mc->worker_failed = true;
                        return;
                    }
                    j.done = true;
                    mc->done.push_back(j);
                }
            }
        });
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("moe-cache: cannot start upload worker: %s\n", e.what());
        return false;
    }
}

void stop_worker(moe_cache * mc) {
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->stop = true;
    }
    mc->wcv.notify_one();
    if (mc->worker.joinable()) {
        mc->worker.join();
    }
}

bool settle_completions(moe_cache * mc) {
    std::lock_guard<std::mutex> wlk(mc->wmtx);
    std::lock_guard<std::mutex> lk(mc->mtx);
    if (mc->worker_failed) {
        return false;
    }
    for (const auto & j : mc->done) {
        auto & ls = mc->layers[j.layer_idx];
        if (!j.done || !ls.slot_in_flight[j.slot] || !ls.expert_in_flight[j.expert] ||
                ls.slot_expert[j.slot] != -1 || ls.expert_slot[j.expert] != -1) {
            return false;
        }
        ls.slot_expert[j.slot]     = j.expert;
        ls.expert_slot[j.expert]   = j.slot;
        ls.slot_last_use[j.slot]   = ++mc->clock;
        ls.slot_in_flight[j.slot]  = false;
        ls.expert_in_flight[j.expert] = false;
        set_table_entry(ls.pub, j.expert, j.slot);
    }
    mc->done.clear();
    return true;
}

bool metadata_valid(const moe_cache * mc) {
    for (const auto & ls : mc->layers) {
        if (ls.pub.n_slots != mc->n_slots || ls.slot_expert.size() != (size_t) mc->n_slots ||
                ls.slot_last_use.size() != ls.slot_expert.size() || ls.slot_in_flight.size() != ls.slot_expert.size() ||
                ls.expert_slot.size() != (size_t) ls.pub.up_src->ne[2] || ls.expert_in_flight.size() != ls.expert_slot.size()) {
            return false;
        }
        for (size_t slot = 0; slot < ls.slot_expert.size(); ++slot) {
            const int32_t expert = ls.slot_expert[slot];
            if (ls.slot_in_flight[slot] || expert < -1 || expert >= (int32_t) ls.expert_slot.size() ||
                    (expert >= 0 && ls.expert_slot[expert] != (int32_t) slot)) {
                return false;
            }
        }
        for (size_t expert = 0; expert < ls.expert_slot.size(); ++expert) {
            const int32_t slot = ls.expert_slot[expert];
            if (ls.expert_in_flight[expert] || slot < -1 || slot >= mc->n_slots ||
                    (slot >= 0 && ls.slot_expert[slot] != (int32_t) expert)) {
                return false;
            }
        }
    }
    return true;
}

void free_device_group(moe_cache * mc, device_group & group) {
    if (group.buf) {
        if (mc) {
            mc->bufs.erase(std::remove(mc->bufs.begin(), mc->bufs.end(), group.buf), mc->bufs.end());
        }
        ggml_backend_buffer_free(group.buf);
        group.buf = nullptr;
    }
    if (group.ctx) {
        if (mc) {
            mc->ctxs.erase(std::remove(mc->ctxs.begin(), mc->ctxs.end(), group.ctx), mc->ctxs.end());
        }
        ggml_free(group.ctx);
        group.ctx = nullptr;
    }
}

void clear_device_pointers(moe_cache * mc) {
    for (auto & ls : mc->layers) {
        ls.pub.up_c = ls.pub.gate_c = ls.pub.down_c = ls.pub.dev_table = nullptr;
    }
}

std::vector<int32_t> layer_table(const layer_state & ls) {
    std::vector<int32_t> table(ls.expert_slot.size(), ls.pub.n_slots);
    for (size_t expert = 0; expert < table.size(); ++expert) {
        if (ls.expert_slot[expert] >= 0) {
            table[expert] = ls.expert_slot[expert];
        }
    }
    return table;
}

bool validate_restored_bytes(const layer_state & ls, const llama_moe_cache_layer & pub) {
    const ggml_tensor * sources[] = {pub.up_src, pub.gate_src, pub.down_src};
    const ggml_tensor * targets[] = {pub.up_c, pub.gate_c, pub.down_c};
    for (int projection = 0; projection < 3; ++projection) {
        const size_t bytes = sources[projection]->nb[2];
        std::vector<unsigned char> scratch(bytes);
        for (int32_t slot = 0; slot <= pub.n_slots; ++slot) {
            if (slot != pub.n_slots && ls.slot_expert[slot] < 0) {
                continue;
            }
            ggml_backend_tensor_get(targets[projection], scratch.data(), (size_t) slot*targets[projection]->nb[2], bytes);
            if (slot == pub.n_slots) {
                if (std::any_of(scratch.begin(), scratch.end(), [](unsigned char v) { return v != 0; })) { return false; }
            } else if (memcmp(scratch.data(), (const char *) sources[projection]->data + (size_t) ls.slot_expert[slot]*bytes, bytes)) {
                return false;
            }
        }
    }
    const auto expected = layer_table(ls);
    std::vector<int32_t> observed(expected.size());
    ggml_backend_tensor_get(pub.dev_table, observed.data(), 0, observed.size()*sizeof(int32_t));
    return observed == expected;
}

void log_transition(const moe_cache * mc, const char * event, const char * phase,
        uint64_t uploaded_weight_bytes = 0, bool validation_enabled = false, bool validation_passed = false) {
    size_t populated = 0;
    for (const auto & ls : mc->layers) {
        populated += std::count_if(ls.slot_expert.begin(), ls.slot_expert.end(), [](int32_t expert) { return expert >= 0; });
    }
    LLAMA_LOG_INFO("moe-cache-phase event=%s phase=%s generation=%" PRIu64 " capacity=%d groups=%zu populated_slots=%zu uploaded_weight_bytes=%" PRIu64 " validation_enabled=%d validation_passed=%d\n",
            event, phase, mc->generation.load(), mc->n_slots, mc->device_groups.size(), populated,
            uploaded_weight_bytes, validation_enabled ? 1 : 0, validation_passed ? 1 : 0);
}

} // namespace

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        if (n_slots <= 0) {
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // experts already on a device: nothing to cache
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = n_slots;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    create_device_tensors(ctx, ls->pub);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            if (!tables_only) {
                device_group group;
                group.buft = buft;
                group.ctx = ctx;
                group.buf = buf;
                for (size_t i = 0; i < mc->layers.size(); ++i) {
                    if (std::any_of(cands.begin(), cands.end(), [&](const cand & c) { return c.il == mc->layers[i].pub.il; })) {
                        group.layers.push_back(i);
                    }
                }
                mc->device_groups.push_back(std::move(group));
            }
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(n_slots, 0);
            ls.slot_in_flight.assign(n_slots, false);
            ls.expert_in_flight.assign(n_expert, false);

            std::vector<int32_t> dummy(n_expert, n_slots);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        if (!start_worker(mc)) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true;
            return;
        }

        mc->phase.store(llama_moe_cache_phase::ready);
        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(), n_slots, mc->max_inserts, vram/1024.0/1024.0);
    }();
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache || g_cache->phase.load() != llama_moe_cache_phase::ready) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step() {
    std::lock_guard<std::mutex> lifecycle_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc || mc->phase.load() != llama_moe_cache_phase::ready) {
        return;
    }

    // 1) publish completed uploads (sync point: no graph is executing)
    if (!settle_completions(mc)) {
        cache_fail(mc, "upload completion/metadata invariant");
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it, --budget) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || ls.expert_in_flight[id]) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < mc->n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, mc->n_slots);
            }
            ls.slot_in_flight[slot] = true;
            ls.expert_in_flight[id] = true;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
        }
        ls.pending.clear();
    }
    mc->wcv.notify_one();

    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_INFO("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}

llama_moe_cache_phase llama_moe_cache_get_phase() {
    return g_cache ? g_cache->phase.load() : llama_moe_cache_phase::unavailable;
}

uint64_t llama_moe_cache_get_generation() {
    return g_cache ? g_cache->generation.load() : 0;
}

int32_t llama_moe_cache_get_capacity() {
    return g_cache ? g_cache->n_slots : 0;
}

bool llama_moe_cache_is_active() {
    return llama_moe_cache_get_phase() == llama_moe_cache_phase::ready;
}

bool llama_moe_cache_quiesce() {
    std::lock_guard<std::mutex> lifecycle_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc || mc->phase.load() != llama_moe_cache_phase::ready) {
        return cache_fail(mc, "quiesce requires READY");
    }
    mc->phase.store(llama_moe_cache_phase::quiescing);
    try {
        stop_worker(mc); // joins the active copy and drains the finite queued jobs
        if (!settle_completions(mc) || !mc->todo.empty() || !mc->done.empty() || !metadata_valid(mc)) {
            return cache_fail(mc, "quiesce did not settle every admission");
        }
        mc->phase.store(llama_moe_cache_phase::quiescent);
        log_transition(mc, "quiesce", "quiescent");
        return true;
    } catch (const std::exception & e) {
        return cache_fail(mc, e.what());
    }
}

bool llama_moe_cache_release() {
    std::lock_guard<std::mutex> lifecycle_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc || mc->phase.load() != llama_moe_cache_phase::quiescent || mc->worker.joinable()) {
        return cache_fail(mc, "release requires joined/quiescent cache and caller graph barrier");
    }
    try {
        if (!metadata_valid(mc)) {
            return cache_fail(mc, "release metadata invariant");
        }
        for (auto & group : mc->device_groups) {
            free_device_group(mc, group);
        }
        clear_device_pointers(mc);
        mc->generation.fetch_add(1);
        mc->phase.store(llama_moe_cache_phase::released);
        log_transition(mc, "release", "released");
        return true;
    } catch (const std::exception & e) {
        return cache_fail(mc, e.what());
    }
}

bool llama_moe_cache_restore() {
    std::lock_guard<std::mutex> lifecycle_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc || mc->phase.load() != llama_moe_cache_phase::released) {
        return cache_fail(mc, "restore requires RELEASED; failed transitions cannot be retried");
    }
    mc->phase.store(llama_moe_cache_phase::restoring);
    uint64_t uploaded_weight_bytes = 0;
    std::vector<device_group> staged;
    auto discard = [&]() {
        stop_worker(mc);
        for (auto & group : staged) { free_device_group(nullptr, group); }
        for (auto & group : mc->device_groups) { free_device_group(mc, group); }
        clear_device_pointers(mc);
    };
    try {
        if (!metadata_valid(mc) || !mc->todo.empty() || !mc->done.empty()) {
            return cache_fail(mc, "restore metadata/worker invariant");
        }
        // Host metadata and authoritative expert tensors are retained, not copied
        // back from GPU. Only new device ownership is staged here.
        staged = mc->device_groups;
        mc->ctxs.reserve(mc->ctxs.size() + staged.size());
        mc->bufs.reserve(mc->bufs.size() + staged.size());
        std::vector<llama_moe_cache_layer> restored;
        restored.reserve(mc->layers.size());
        for (const auto & ls : mc->layers) { restored.push_back(ls.pub); }

        for (auto & group : staged) {
            const ggml_init_params ip = {ggml_tensor_overhead()*(group.layers.size()*4 + 8), nullptr, true};
            group.ctx = ggml_init(ip);
            if (!group.ctx) {
                discard();
                return cache_fail(mc, "restore tensor-context allocation failed");
            }
            for (size_t li : group.layers) { create_device_tensors(group.ctx, restored[li]); }
            group.buf = ggml_backend_alloc_ctx_tensors_from_buft(group.ctx, group.buft);
            if (!group.buf) {
                discard();
                return cache_fail(mc, "restore device allocation failed; no group published");
            }
            ggml_backend_buffer_clear(group.buf, 0); // includes every dummy and empty slot
            for (size_t li : group.layers) {
                const auto & ls = mc->layers[li];
                auto & pub = restored[li];
                for (int32_t slot = 0; slot < mc->n_slots; ++slot) {
                    const int32_t expert = ls.slot_expert[slot];
                    if (expert >= 0 &&
                            !(upload_slice(pub.up_c, pub.up_src, expert, slot) &&
                              upload_slice(pub.gate_c, pub.gate_src, expert, slot) &&
                              upload_slice(pub.down_c, pub.down_src, expert, slot))) {
                        discard();
                        return cache_fail(mc, "restore projection bounds failed");
                    }
                    if (expert >= 0) {
                        uploaded_weight_bytes += pub.up_src->nb[2] + pub.gate_src->nb[2] + pub.down_src->nb[2];
                    }
                }
                const auto table = layer_table(ls);
                ggml_backend_tensor_set(pub.dev_table, table.data(), 0, table.size()*sizeof(int32_t));
            }
        }

        // CUDA's synchronous tensor_set/clear complete each upload before this
        // point. The optional correctness mode reads one expert slice at a time;
        // it is off by default and never creates a full D2H weight backup.
        const char * debug = std::getenv("LLAMA_MOE_CACHE_VALIDATE_RESTORE");
        const bool validate = debug && strcmp(debug, "1") == 0;
        if (validate) {
            for (size_t li = 0; li < restored.size(); ++li) {
                if (!validate_restored_bytes(mc->layers[li], restored[li])) {
                    discard();
                    return cache_fail(mc, "restore byte/dummy/map validation failed");
                }
            }
        }
        for (size_t li = 0; li < restored.size(); ++li) {
            const auto table = layer_table(mc->layers[li]);
            ggml_backend_tensor_set(restored[li].host_table, table.data(), 0, table.size()*sizeof(int32_t));
            if (validate) {
                std::vector<int32_t> observed(table.size());
                ggml_backend_tensor_get(restored[li].host_table, observed.data(), 0, observed.size()*sizeof(int32_t));
                if (observed != table) {
                    discard();
                    return cache_fail(mc, "restore host-map validation failed");
                }
            }
        }
        if (!start_worker(mc)) {
            discard();
            return cache_fail(mc, "restore worker restart failed");
        }

        // The worker has no jobs while lookup/observation/step remain disabled.
        // Reserved ownership vectors and pointer assignments cannot allocate.
        for (auto & group : staged) {
            mc->ctxs.push_back(group.ctx);
            mc->bufs.push_back(group.buf);
        }
        for (size_t li = 0; li < restored.size(); ++li) { mc->layers[li].pub = restored[li]; }
        mc->device_groups = std::move(staged);
        mc->generation.fetch_add(1);
        mc->phase.store(llama_moe_cache_phase::ready); // single global publication for all devices
        log_transition(mc, "restore", "ready", uploaded_weight_bytes, validate, validate);
        return true;
    } catch (const std::exception & e) {
        discard();
        return cache_fail(mc, e.what());
    }
}
