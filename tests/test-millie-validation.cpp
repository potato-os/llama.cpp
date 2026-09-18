#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#include <cstdio>
#include <cstring>
#include <initializer_list>

static int failures = 0;
static void check(bool expected, ggml_type type, const void * data, size_t size) {
    if (ggml_validate_row_data(type, data, size) != expected) {
        std::fprintf(stderr, "unexpected validation result for %s\n", ggml_type_name(type));
        ++failures;
    }
}
int main() {
    block_q2_sym32k4 q2[2]{};
    block_qt_ms32k4 qt[2]{};
    for (auto & b : qt) {
        b.d = ggml_fp32_to_fp16(1.0f);
        std::memset(b.mask, 0x55, sizeof(b.mask)); // 128 set bits
    }
    check(true, GGML_TYPE_Q2_SYM32K4, q2, sizeof(q2));
    check(true, GGML_TYPE_QT_MS32K4, qt, sizeof(qt));
    check(false, GGML_TYPE_QT_MS32K4, qt, sizeof(qt)-1);
    qt[1].mask[0] = 0x54;
    check(false, GGML_TYPE_QT_MS32K4, qt, sizeof(qt));
    qt[1].mask[0] = 0x57;
    check(false, GGML_TYPE_QT_MS32K4, qt, sizeof(qt));
    qt[1].mask[0] = 0x55;
    for (unsigned bad : {0x7c00u, 0x7e00u}) {
        q2[1].d = (ggml_half) bad;
        qt[1].d = (ggml_half) bad;
        check(false, GGML_TYPE_Q2_SYM32K4, q2, sizeof(q2));
        check(false, GGML_TYPE_QT_MS32K4, qt, sizeof(qt));
    }
    return failures ? 1 : 0;
}
