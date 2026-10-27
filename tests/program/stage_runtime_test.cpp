// The stage-owner rule for explicit layer boundaries. No GPU, no model.
#include <cstdio>
#include <vector>

#include "strata/program/stage_runtime.hpp"

using strata::program::stage_for_layer;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const std::vector<int64_t> one{24};
    CHECK(stage_for_layer(0, one) == 0);
    CHECK(stage_for_layer(23, one) == 0);
    CHECK(stage_for_layer(24, one) == 1);
    CHECK(stage_for_layer(47, one) == 1);
    CHECK(stage_for_layer(0, {}) == 0);
    CHECK(stage_for_layer(47, {}) == 0);

    const std::vector<int64_t> three{8, 16};
    CHECK(stage_for_layer(7, three) == 0);
    CHECK(stage_for_layer(8, three) == 1);
    CHECK(stage_for_layer(15, three) == 1);
    CHECK(stage_for_layer(16, three) == 2);
    CHECK(stage_for_layer(47, three) == 2);

    if (failed == 0) std::printf("stage_runtime_test: ok\n");
    return failed == 0 ? 0 : 1;
}
