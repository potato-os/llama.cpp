#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

using dot_fn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static double reference(const block_qt_ms32k4 * x, const block_q8_0 * y, int nb) {
    double sum = 0;
    for (int b = 0; b < nb; ++b) {
        int rank = 0;
        for (int g = 0; g < 8; ++g) {
            int dot = 0;
            for (int j = 0; j < 32; ++j) {
                const int k = 32*g + j;
                if ((x[b].mask[k/8] >> (k%8)) & 1) {
                    if (rank >= 128) { std::abort(); }
                    dot += ((x[b].signs[rank/8] >> (rank%8)) & 1) ? -int(y[8*b+g].qs[j]) : int(y[8*b+g].qs[j]);
                    ++rank;
                }
            }
            if (rank > 128) { std::abort(); }
            const int sc = (x[b].scales[g/2] >> (4*(g%2))) & 15;
            sum += double(ggml_fp16_to_fp32(x[b].d)) * sc * ggml_fp16_to_fp32(y[8*b+g].d) * dot;
        }
    }
    return sum;
}

static double timed(dot_fn fn, const std::vector<block_qt_ms32k4> & x,
                    const std::vector<block_q8_0> & y, int nb) {
    const int rows = int(x.size())/nb;
    float out = 0;
    volatile float sink = 0;
    const auto begin = std::chrono::steady_clock::now();
    int reps = 0;
    double seconds;
    do {
        for (int r = 0; r < rows; ++r) {
            fn(nb*256, &out, 0, x.data()+r*nb, 0, y.data(), 0, 1);
            sink = out;
        }
        ++reps;
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    } while (seconds < 0.15);
    (void) sink;
    return seconds*1e9/(reps*double(rows)*nb);
}

static bool check(const char * name, dot_fn fn, dot_fn baseline,
                  const std::vector<block_qt_ms32k4> & x, const std::vector<block_q8_0> & y, int nb) {
    double err = 0, norm = 0, worst = 0;
    size_t changed = 0;
    for (size_t r = 0; r < x.size()/nb; ++r) {
        float out;
        fn(nb*256, &out, 0, x.data()+r*nb, 0, y.data(), 0, 1);
        if (baseline) {
            float old;
            baseline(nb*256, &old, 0, x.data()+r*nb, 0, y.data(), 0, 1);
            changed += std::memcmp(&old, &out, sizeof(out)) != 0;
        }
        const double ref = reference(x.data()+r*nb, y.data(), nb);
        if (!std::isfinite(out)) { return false; }
        err += (out-ref)*(out-ref);
        norm += ref*ref;
        worst = std::max(worst, std::abs(out-ref)/(1+std::abs(ref)));
    }
    const double nmse = err/std::max(norm, 1e-30);
    std::printf("%s: rows=%zu nmse=%.3g max_scaled_error=%.3g", name, x.size()/nb, nmse, worst);
    if (baseline) {
        std::array<double, 5> a, b;
        for (size_t i = 0; i < a.size(); ++i) {
            if (i%2) { b[i] = timed(fn, x, y, nb); a[i] = timed(baseline, x, y, nb); }
            else     { a[i] = timed(baseline, x, y, nb); b[i] = timed(fn, x, y, nb); }
        }
        std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
        std::printf(" baseline=%.2f candidate=%.2f ns/block speedup=%.3fx changed=%zu", a[2], b[2], a[2]/b[2], changed);
    }
    std::puts("");
    return nmse < 1e-11 && worst < 0.002 && changed == 0;
}

int main(int argc, char ** argv) {
    // Direct dot calls bypass backend initialization. x86 conversions can
    // use CPU FP16 tables even though ARM conversions use instructions.
    ggml_cpu_init();
    ggml_init_params ip = { 1024*1024, nullptr, true };
    ggml_context * init = ggml_init(ip);
    const auto fn = ggml_get_type_traits_cpu(GGML_TYPE_QT_MS32K4)->vec_dot;
    dot_fn baseline = nullptr;
#if defined(__unix__) || defined(__APPLE__)
    if (argc > 1 && argv[1][0] != '-') {
        void * lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!lib) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
        baseline = reinterpret_cast<dot_fn>(dlsym(lib, "ggml_vec_dot_qt_ms32k4_q8_0"));
        if (!baseline) { return 1; }
    }
#endif
    std::mt19937 rng(12345);
    constexpr int nb = 32;
    std::vector<block_q8_0> y(nb*8);
    for (auto & b : y) {
        b.d = ggml_fp32_to_fp16(float(1+rng()%100)/1000);
        for (auto & q : b.qs) { q = int(rng()%255)-127; }
    }
    bool ok = true;
    std::vector<block_qt_ms32k4> exhaustive;
    for (int prefix : { 0, 61, 64, 120 }) {
        for (int pattern = 0; pattern < 6561; ++pattern) {
            block_qt_ms32k4 b = {};
            b.d = ggml_fp32_to_fp16(1);
            std::fill(std::begin(b.scales), std::end(b.scales), 0x11);
            std::fill(std::begin(b.signs), std::end(b.signs), 0xa5);
            for (int k = 0; k < prefix; ++k) { b.mask[k/8] |= 1 << (k%8); }
            int digits = pattern, rank = prefix;
            for (int j = 0; j < 8; ++j, digits /= 3) {
                const int value = digits%3;
                if (value == 0) { continue; }
                b.mask[16] |= 1 << j;
                b.signs[rank/8] &= ~(1 << (rank%8));
                if (value == 2) { b.signs[rank/8] |= 1 << (rank%8); }
                ++rank;
            }
            exhaustive.push_back(b);
        }
    }
    ok &= check("all-byte-patterns/sign-boundaries", fn, nullptr, exhaustive, y, 1);
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<block_qt_ms32k4> x(nb*128);
        for (auto & b : x) {
            b.d = ggml_fp32_to_fp16(float(1+rng()%100)/100);
            for (auto & sc : b.scales) { sc = rng(); }
            for (auto & s : b.signs) { s = rng(); }
            std::array<int, 256> positions;
            std::iota(positions.begin(), positions.end(), 0);
            if (mode != 2) { std::shuffle(positions.begin(), positions.end(), rng); }
            const int nz = mode == 0 ? 128 : mode == 3 ? 0 : rng()%129;
            for (int k = 0; k < nz; ++k) { b.mask[positions[k]/8] |= 1 << (positions[k]%8); }
        }
        const char * names[] = { "full", "under-budget", "clustered", "zero" };
        ok &= check(names[mode], fn, baseline, x, y, nb);
    }
    if (argc > 2) {
        ggml_context * meta = nullptr;
        gguf_init_params gp = { true, &meta };
        gguf_context * uf = gguf_init_from_file(argv[2], gp);
        FILE * f = std::fopen(argv[2], "rb");
        if (!uf || !f) { return 1; }
        // Sample the start, middle, and end of every ternary tensor.
        std::vector<block_qt_ms32k4> real;
        for (int64_t i = 0; i < gguf_get_n_tensors(uf); ++i) {
            if (gguf_get_tensor_type(uf, i) != GGML_TYPE_QT_MS32K4) { continue; }
            const auto * t = ggml_get_tensor(meta, gguf_get_tensor_name(uf, i));
            const size_t blocks = ggml_nelements(t)/256;
            if (blocks < nb*4) { continue; }
            for (size_t pos : { size_t(0), (blocks/2/nb)*nb, blocks-nb*4 }) {
                const size_t old = real.size();
                real.resize(old+nb*4);
                const size_t off = gguf_get_data_offset(uf)+gguf_get_tensor_offset(uf, i)+pos*sizeof(block_qt_ms32k4);
                if (fseeko(f, off, SEEK_SET) || fread(real.data()+old, sizeof(block_qt_ms32k4), nb*4, f) != nb*4) { return 1; }
            }
        }
        if (real.empty()) { return 1; }
        ok &= check("real-file", fn, baseline, real, y, nb);
        std::fclose(f); gguf_free(uf); ggml_free(meta);
    }
    ggml_free(init);
    return ok ? 0 : 1;
}
