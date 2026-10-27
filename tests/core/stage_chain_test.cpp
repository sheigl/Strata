// The narrow run/commit boundary of a decode stage (include/strata/core/decode_stage.hpp).  Header-only, CPU.
#include "strata/core/decode_stage.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void pool(void* user, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, float* out, int64_t layer) {
    (void) user;
    (void) x;
    (void) ids;
    (void) n_tok;
    (void) k;
    (void) out;
    (void) layer;
}

class LocalStage : public strata::core::DecodeStage {
public:
    int windows = 0;
    int commits = 0;
    int batch_windows = 0;
    int batch_commits = 0;
    int partial_kept = 0;

    bool run_decode_window(int T, const int32_t* tokens, int64_t pos0, strata::core::PoolMultiFn fn, void* user,
                           int32_t* out, std::string&) override {
        ++windows;
        for (int t = 0; t < T; ++t) out[t] = tokens[t] + (int32_t) pos0;
        (void) fn;
        (void) user;
        return true;
    }

    bool commit_decode_prefix(int n_keep, std::string&) override {
        ++commits;
        partial_kept = n_keep;
        return true;
    }

    bool supports_batch_decode() const override { return true; }

    bool run_batch_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos,
                             strata::core::PoolMultiFn fn, void* user, int32_t* out, std::string&) override {
        ++batch_windows;
        for (int t = 0; t < S; ++t) out[t] = tokens[t] + (int32_t) pos[rows[t]];
        (void) fn;
        (void) user;
        return true;
    }

    bool commit_batch_slots(std::string&) override {
        ++batch_commits;
        return true;
    }
};

class RemoteStage : public strata::core::DecodeStage {
public:
    bool run_decode_window(int T, const int32_t* tokens, int64_t pos0, strata::core::PoolMultiFn fn, void* user,
                           int32_t* out, std::string&) override {
        for (int t = 0; t < T; ++t) out[t] = tokens[t] + (int32_t) pos0;
        (void) fn;
        (void) user;
        return true;
    }

    bool commit_decode_prefix(int, std::string&) override { return true; }
};
}  // namespace

int main() {
    std::string err;
    LocalStage local;
    const int32_t tokens[] = {7, 11, 13};
    const int64_t positions[] = {100, 101};
    const int rows[] = {1, 0};
    int32_t out[4] = {};

    check(local.run_decode_window(3, tokens, 100, pool, nullptr, out, err), "local decode window");
    check(out[0] == 107 && out[1] == 111 && out[2] == 113, "local decode picks");
    check(local.commit_decode_prefix(2, err), "local partial commit");
    check(local.windows == 1 && local.commits == 1 && local.partial_kept == 2, "local run and commit order");

    check(local.supports_batch_decode(), "local stage supports batch decode");
    check(local.run_batch_slot_rows(rows, 2, tokens, positions, pool, nullptr, out, err), "local batch rows");
    check(out[0] == 108 && out[1] == 111, "local batch rows use the caller's row layout");
    check(local.commit_batch_slots(err), "local batch commit");
    check(local.batch_windows == 1 && local.batch_commits == 1, "local batch calls reached the stage");

    RemoteStage remote;
    check(remote.run_decode_window(3, tokens, 100, pool, nullptr, out, err), "remote decode window");
    check(remote.commit_decode_prefix(1, err), "remote commit");
    check(!remote.supports_batch_decode(), "remote stage does not support batch decode");

    err.clear();
    check(!remote.run_batch_slots(2, tokens, positions, pool, nullptr, out, err), "unsupported batch slots");
    check(err.find("batch windows") != std::string::npos, "unsupported batch slots names the capability");
    check(!remote.run_batch_slot_rows(rows, 2, tokens, positions, pool, nullptr, out, err), "unsupported batch rows");
    check(!remote.commit_batch_slots(err), "unsupported batch commit");
    check(!remote.commit_batch_slot_prefixes(rows, err), "unsupported batch prefix commit");

    if (failures != 0) return EXIT_FAILURE;
    std::printf("stage_chain_test: ok\n");
    return EXIT_SUCCESS;
}
