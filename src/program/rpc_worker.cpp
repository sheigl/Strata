// src/program/rpc_worker.cpp - the worker half of the local decode vertical slice.
#include "strata/program/rpc_worker.hpp"

#include <cstring>
#include <vector>

namespace strata::program {

int run_rpc_worker(strata::rpc::ControlChannel& control, const strata::rpc::WorkerInit& expected,
                   const float* staging, std::uint64_t staging_bytes, RpcWorkerStage& stage, std::string& err) {
    std::string send_err;
    auto fail = [&](const std::string& message) -> int {
        send_err.clear();
        (void) control.send_error(message, send_err);
        err = message;
        return 1;
    };

    strata::rpc::Frame init_frame;
    if (!control.recv(init_frame, err)) {
        err = "RPC worker waiting for INIT_STAGE: " + err;
        return 1;
    }
    if (init_frame.header.opcode != strata::rpc::Opcode::InitStage) {
        err = "RPC worker received " + std::string(strata::rpc::opcode_name(init_frame.header.opcode)) +
              " before INIT_STAGE";
        return fail(err);
    }
    strata::rpc::WorkerInit init;
    if (!strata::rpc::decode_worker_init(init_frame.payload, init, err)) {
        return fail(err);
    }
    if (init.layer_begin != expected.layer_begin || init.layer_end != expected.layer_end ||
        init.handoff_floats != expected.handoff_floats || init.max_window != expected.max_window ||
        init.max_context != expected.max_context) {
        err = "RPC INIT_STAGE does not match this worker's launched geometry or hand-off";
        return fail(err);
    }
    const std::uint64_t per_row = (std::uint64_t) expected.handoff_floats * sizeof(float);
    if (per_row == 0 || (std::uint64_t) expected.max_window * per_row > staging_bytes ||
        (std::uint64_t) expected.max_window * per_row > control.bulk().size()) {
        err = "RPC hand-off staging buffer is smaller than the negotiated window";
        return fail(err);
    }
    if (!control.send_response(strata::rpc::Opcode::Ready, strata::rpc::encode_worker_init(expected), err)) return 1;

    std::vector<int32_t> out((size_t) expected.max_window);
    uint32_t last_rows = 0;
    for (;;) {
        strata::rpc::Frame frame;
        if (!control.recv(frame, err)) {
            // The controller closed the channel without a SHUTDOWN frame: the worker has no state to preserve.
            if (err.find("closed") != std::string::npos) return 0;
            return 1;
        }
        switch (frame.header.opcode) {
        case strata::rpc::Opcode::Shutdown:
            return 0;
        case strata::rpc::Opcode::Window: {
            strata::rpc::WindowRequest window;
            if (!strata::rpc::decode_window_request(frame.payload, window, err)) {
                return fail(err);
            }
            if (window.tokens.empty() || (uint32_t) window.tokens.size() > expected.max_window) {
                err = "RPC WINDOW row count is outside this worker's negotiated window";
                return fail(err);
            }
            if (!stage.run_window((int) window.tokens.size(), window.tokens.data(), (int64_t) window.pos0,
                                  out.data(), err)) {
                return fail(err);
            }
            const uint64_t floats = (uint64_t) window.tokens.size() * (uint64_t) expected.handoff_floats;
            std::memcpy(control.bulk().data(), staging, (size_t) (floats * sizeof(float)));
            strata::rpc::WindowResult result;
            result.rows = (uint32_t) window.tokens.size();
            result.float_count = floats;
            if (!control.send_response(strata::rpc::Opcode::WindowDone, strata::rpc::encode_window_result(result), err))
                return 1;
            last_rows = result.rows;
            break;
        }
        case strata::rpc::Opcode::Commit: {
            uint32_t keep = 0;
            if (!strata::rpc::decode_commit_request(frame.payload, keep, err)) {
                return fail(err);
            }
            if (keep == 0 || keep > last_rows) {
                err = "RPC COMMIT prefix is outside the last WINDOW";
                return fail(err);
            }
            if (!stage.commit_window(keep, err)) {
                return fail(err);
            }
            if (!control.send_response(strata::rpc::Opcode::CommitDone, strata::rpc::encode_commit_result(keep), err))
                return 1;
            last_rows = 0;
            break;
        }
        default:
            err = "RPC worker received unsupported opcode " + std::string(strata::rpc::opcode_name(frame.header.opcode));
            return fail(err);
        }
    }
}

}  // namespace strata::program
