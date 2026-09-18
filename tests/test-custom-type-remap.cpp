#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Tiny handcrafted GGUFs exercise on-disk IDs independently of the enum
// used to build the writer. No model weights or GPU initialization needed.
static void integer(std::vector<uint8_t> & out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        out.push_back(uint8_t(value >> (8*i)));
    }
}

static void string(std::vector<uint8_t> & out, const std::string & value) {
    integer(out, value.size(), 8);
    out.insert(out.end(), value.begin(), value.end());
}

static bool check(uint32_t disk_id, const std::string & declaration, ggml_type expected) {
    std::vector<uint8_t> bytes = { 'G', 'G', 'U', 'F' };
    integer(bytes, 3, 4);
    integer(bytes, 1, 8);
    integer(bytes, declaration.empty() ? 0 : 1, 8);
    if (!declaration.empty()) {
        string(bytes, "general.tensor_types");
        integer(bytes, GGUF_TYPE_ARRAY, 4);
        integer(bytes, GGUF_TYPE_STRING, 4);
        integer(bytes, 1, 8);
        string(bytes, declaration);
    }
    string(bytes, "test.weight");
    integer(bytes, 1, 4);
    integer(bytes, 256, 8);
    integer(bytes, disk_id, 4);
    integer(bytes, 0, 8);
    bytes.resize((bytes.size() + 31) / 32 * 32 + 1024, 0);
    FILE * file = std::tmpfile();
    if (!file) {
        throw std::runtime_error("tmpfile failed");
    }
    if (std::fwrite(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
        std::fclose(file);
        throw std::runtime_error("fixture write failed");
    }
    std::rewind(file);
    gguf_context * ctx = gguf_init_from_file_ptr(file, { true, nullptr });
    std::fclose(file);
    const bool ok = expected == GGML_TYPE_COUNT
        ? ctx == nullptr
        : ctx && gguf_get_tensor_type(ctx, 0) == expected;
    if (ctx) {
        gguf_free(ctx);
    }
    if (!ok) {
        std::fprintf(stderr, "FAIL: disk ID %u, declaration '%s'\n", disk_id, declaration.c_str());
    }
    return ok;
}

int main() {
    const struct { uint32_t legacy; const char * name; ggml_type type; } formats[] = {
        { 54, "Q1_sym_g32_sq4_sg8", GGML_TYPE_Q1_SYM32K },
        { 55, "Q4_sym_g16_sq8_sg8", GGML_TYPE_Q4_SYM16K },
        { 56, "Q2_sym_g32_sq4_sg8", GGML_TYPE_Q2_SYM32K4 },
        { 57, "Qt_sym_ms_g32_sq4_sg8", GGML_TYPE_QT_MS32K4 },
    };
    bool ok = true;
    for (const auto & format : formats) {
        ok &= check(format.legacy, "", format.type);
        ok &= check(401, std::string("401=") + format.name, format.type);
        // A declared custom layout must override even a valid built-in ID.
        ok &= check(0, std::string("0=") + format.name, format.type);
    }
    ok &= check(0, "0=unknown_future_layout", GGML_TYPE_COUNT);
    std::printf("%s: 13 custom type compatibility checks\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
