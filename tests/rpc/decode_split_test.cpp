// The worker loop and controller proxy wire, with a fake decode stage: readiness, a window hand-off, a commit, an
// invalid partial commit that must fail closed, and a clean shutdown.  No GPU or model.
#include "strata/program/rpc_worker.hpp"
#include "strata/rpc/control.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

class FakeStage final : public strata::program::RpcWorkerStage {
public:
    int windows = 0;
    int commits = 0;
    float* staging = nullptr;
    std::uint64_t floats = 0;

    bool run_window(int T, const int32_t* tokens, int64_t pos0, int32_t* out, std::string&) override {
        ++windows;
        for (int t = 0; t < T; ++t) out[t] = tokens[t];
        for (std::uint64_t i = 0; i < (std::uint64_t) T * floats; ++i) staging[i] = (float) (i + 1);
        (void) pos0;
        return true;
    }

    bool commit_window(uint32_t keep, std::string&) override {
        ++commits;
        if (keep == 0 || keep > 8) return false;
        return true;
    }
};
}  // namespace

int main() {
    using namespace strata::rpc;
    const std::string path = "/tmp/opencode/strata-rpc-decode-test-" + std::to_string((long) ::getpid()) + ".sock";
    const std::string token = "decode-test-token";
    std::remove(path.c_str());
    constexpr uint32_t floats_per_row = 3;
    std::atomic<bool> worker_ok{false};
    std::string worker_err;

    std::thread worker([&] {
        auto control = ControlChannel::worker_listen(path, token, kDefaultBulkBytes, worker_err, 10000);
        if (!control || !control->receive_bulk(worker_err)) return;
        float staging[8 * floats_per_row] = {};
        FakeStage stage;
        stage.staging = staging;
        stage.floats = floats_per_row;
        WorkerInit expected;
        expected.layer_begin = 0;
        expected.layer_end = 4;
        expected.handoff_floats = floats_per_row;
        expected.max_window = 8;
        expected.max_context = 4096;
        const int rc = strata::program::run_rpc_worker(*control, expected, staging, sizeof staging, stage, worker_err);
        if (rc == 0 || worker_err.find("COMMIT prefix") == std::string::npos) {
            worker_err = worker_err.empty() ? "worker loop did not reject the invalid commit" : worker_err;
            return;
        }
        if (stage.windows != 1 || stage.commits != 0) {
            worker_err = "worker stage counters";
            return;
        }
        worker_ok = true;
    });

    for (int i = 0; i < 200; ++i) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f != nullptr) {
            std::fclose(f);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto control = ControlChannel::controller_connect(path, token, kDefaultBulkBytes, worker_err, 10000);
    check(control != nullptr, "controller connects");

    if (control) {
        WorkerInit init;
        init.layer_begin = 0;
        init.layer_end = 4;
        init.handoff_floats = floats_per_row;
        init.max_window = 8;
        init.max_context = 4096;
        check(control->send_request(Opcode::InitStage, encode_worker_init(init), worker_err), "INIT_STAGE");
        Frame ready;
        check(control->recv(ready, worker_err) && ready.header.opcode == Opcode::Ready, "READY");

        WindowRequest window;
        window.pos0 = 5;
        window.tokens = {2, 3, 5};
        check(control->send_request(Opcode::Window, encode_window_request(window), worker_err), "WINDOW");
        Frame done;
        check(control->recv(done, worker_err) && done.header.opcode == Opcode::WindowDone, "WINDOW_DONE");
        WindowResult result;
        check(decode_window_result(done.payload, result, worker_err), "WINDOW_DONE payload");
        check(result.rows == 3 && result.float_count == 9, "WINDOW_DONE shape");
        const float* got = reinterpret_cast<const float*>(control->bulk().data());
        check(got[0] == 1.0f && got[8] == 9.0f, "bulk bytes after the worker window");

        // The fake stage committed 0 windows: an out-of-window COMMIT must be rejected by the worker, and the
        // controller must see ERROR rather than a successful partial response.
        check(control->send_request(Opcode::Commit, encode_commit_request(4), worker_err), "invalid COMMIT sent");
        Frame failed;
        check(control->recv(failed, worker_err) && failed.header.opcode == Opcode::Error,
              "invalid COMMIT fails closed");
    }
    worker.join();
    if (!worker_ok.load())
        std::fprintf(stderr, "worker diagnostic: '%s'\n", worker_err.c_str());
    check(worker_ok.load(), worker_err.empty() ? "worker completed" : worker_err.c_str());
    std::remove(path.c_str());
    if (failures != 0) return EXIT_FAILURE;
    std::printf("rpc_decode_split_test: ok\n");
    return EXIT_SUCCESS;
}
