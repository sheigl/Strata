// The prompt-stage boundary (include/strata/prefill/prefill_stage.hpp): one chunk, its hand-off and pipeline controls.
// Header-only, CPU.
#include "strata/prefill/prefill_stage.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

class FakeStage final : public strata::prefill::PrefillStage {
public:
    int calls = 0;
    int64_t n = 0, pos0 = 0;
    std::vector<int64_t> tokens;
    const void* rows = nullptr;
    const void* event = nullptr;
    int pieces = 0;
    bool single = false, piped = false;
    int drains = 0;
    int64_t reserved = 0;
    bool run_result = true;

    bool run_chunk(const int64_t* t, int64_t count, int64_t pos, std::string&) override {
        ++calls;
        n = count;
        pos0 = pos;
        tokens.assign(t, t + count);
        return run_result;
    }
    void set_handoff(const void* r, const void* e, int p, bool s, bool piped_chunk) override {
        rows = r;
        event = e;
        pieces = p;
        single = s;
        piped = piped_chunk;
    }
    bool drain_pipeline(std::string&) override {
        ++drains;
        return true;
    }
    bool reserve_uploads(int64_t T) override {
        reserved = T;
        return true;
    }
    int stage_count() const override { return 1; }
    int64_t chain_experts_streamed() const override { return 7; }
};
}  // namespace

int main() {
    using strata::prefill::PrefillStage;
    FakeStage stage;
    const int64_t tokens[] = {11, 13, 17};
    std::string err;

    check(stage.run_chunk(tokens, 3, 42, err), "run_chunk result");
    check(stage.calls == 1 && stage.n == 3 && stage.pos0 == 42, "run_chunk args");
    check(stage.tokens == std::vector<int64_t>({11, 13, 17}), "run_chunk tokens");
    check(stage.reserve_uploads(512) && stage.reserved == 512, "reserve_uploads");
    check(stage.drain_pipeline(err) && stage.drains == 1, "drain_pipeline");
    check(stage.stage_count() == 1, "stage_count");
    check(stage.chain_experts_streamed() == 7, "chain_experts_streamed");

    int row = 0, event = 0;
    stage.set_handoff(&row, &event, 2, true, false);
    check(stage.rows == &row && stage.event == &event && stage.pieces == 2 && stage.single && !stage.piped,
          "set_handoff preserves rows, event and chunk shape");

    stage.run_result = false;
    check(!stage.run_chunk(tokens, 3, 42, err), "run_chunk failure is returned");
    stage.log_mmq_direct_totals();   // the default diagnostic hook is a no-op

    if (failures != 0) return EXIT_FAILURE;
    std::printf("prefill_stage_test: ok\n");
    return EXIT_SUCCESS;
}
