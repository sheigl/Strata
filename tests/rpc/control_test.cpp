// The control channel and its payloads over a real local socket: worker listen/Hello/HelloAck, controller connect,
// INIT_STAGE/READY, WINDOW/WINDOW_DONE, COMMIT/COMMIT_DONE and SHUTDOWN.  No GPU or model.
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
}  // namespace

int main() {
    using namespace strata::rpc;
    const std::string path = "/tmp/opencode/strata-rpc-control-test-" + std::to_string((long) ::getpid()) + ".sock";
    const std::string token = "control-test-token";
    std::remove(path.c_str());
    std::atomic<bool> worker_ok{false};
    std::string worker_err;

    std::thread worker([&] {
        auto channel = ControlChannel::worker_listen(path, token, kDefaultBulkBytes, worker_err, 10000);
        if (!channel) return;
        if (!channel->receive_bulk(worker_err)) return;
        Frame init;
        if (!channel->recv(init, worker_err) || init.header.opcode != Opcode::InitStage) return;
        WorkerInit info;
        if (!decode_worker_init(init.payload, info, worker_err)) return;
        if (!channel->send_response(Opcode::Ready, encode_worker_init(info), worker_err)) return;
        Frame window;
        if (!channel->recv(window, worker_err) || window.header.opcode != Opcode::Window) return;
        WindowRequest request;
        if (!decode_window_request(window.payload, request, worker_err)) return;
        const float values[] = {request.tokens.empty() ? 0.0f : (float) request.tokens[0], request.pos0 == 7 ? 1.0f : 0.0f};
        std::memcpy(channel->bulk().data(), values, sizeof values);
        WindowResult result;
        result.rows = (uint32_t) request.tokens.size();
        result.float_count = 2;
        if (!channel->send_response(Opcode::WindowDone, encode_window_result(result), worker_err)) return;
        Frame commit;
        if (!channel->recv(commit, worker_err) || commit.header.opcode != Opcode::Commit) return;
        uint32_t keep = 0;
        if (!decode_commit_request(commit.payload, keep, worker_err)) return;
        if (!channel->send_response(Opcode::CommitDone, encode_commit_result(keep), worker_err)) return;
        Frame shutdown;
        if (!channel->recv(shutdown, worker_err) || shutdown.header.opcode != Opcode::Shutdown) return;
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
    auto controller = ControlChannel::controller_connect(path, token, kDefaultBulkBytes, worker_err, 10000);
    check(controller != nullptr, "controller connects and hands over the bulk region");

    if (controller) {
        WorkerInit init;
        init.layer_begin = 0;
        init.layer_end = 24;
        init.handoff_floats = 2;
        init.max_window = 8;
        init.max_context = 131072;
        check(controller->send_request(Opcode::InitStage, encode_worker_init(init), worker_err), "controller INIT_STAGE");
        Frame ready;
        check(controller->recv(ready, worker_err) && ready.header.opcode == Opcode::Ready, "worker READY");

        WindowRequest window;
        window.pos0 = 7;
        window.tokens = {11, 13, 17};
        check(controller->send_request(Opcode::Window, encode_window_request(window), worker_err), "controller WINDOW");
        Frame done;
        check(controller->recv(done, worker_err) && done.header.opcode == Opcode::WindowDone, "worker WINDOW_DONE");
        WindowResult result;
        check(decode_window_result(done.payload, result, worker_err), "WINDOW_DONE payload");
        check(result.rows == 3 && result.float_count == 2, "WINDOW_DONE shape");
        const float* got = reinterpret_cast<const float*>(controller->bulk().data());
        check(got[0] == 11.0f && got[1] == 1.0f, "bulk hand-off bytes");

        check(controller->send_request(Opcode::Commit, encode_commit_request(2), worker_err), "controller COMMIT");
        Frame committed;
        check(controller->recv(committed, worker_err) && committed.header.opcode == Opcode::CommitDone,
              "worker COMMIT_DONE");
        uint32_t keep = 0;
        check(decode_commit_result(committed.payload, keep, worker_err) && keep == 2, "COMMIT_DONE prefix");

        check(controller->send_request(Opcode::Shutdown, {}, worker_err), "controller SHUTDOWN");
    }
    worker.join();
    check(worker_ok.load(), worker_err.empty() ? "worker completed the sequence" : worker_err.c_str());
    std::remove(path.c_str());
    if (failures != 0) return EXIT_FAILURE;
    std::printf("rpc_control_test: ok\n");
    return EXIT_SUCCESS;
}
