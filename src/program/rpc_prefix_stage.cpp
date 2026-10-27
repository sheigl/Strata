// src/program/rpc_prefix_stage.cpp - controller half of the local decode slice: prefix over RPC, suffix locally.
#include "strata/program/rpc_prefix_stage.hpp"

#include <cstring>

namespace strata::program {
namespace {

std::string payload_error(const strata::rpc::Frame& frame) {
    return frame.payload.empty() ? std::string("the RPC worker returned ERROR")
                                 : std::string(frame.payload.begin(), frame.payload.end());
}

}  // namespace

RpcPrefixStage::RpcPrefixStage(std::unique_ptr<strata::rpc::ControlChannel> control, strata::core::Verifier& suffix,
                               float* handoff, std::uint64_t handoff_bytes, std::uint64_t handoff_floats,
                               int max_window)
    : control_(std::move(control)), suffix_(suffix), handoff_(handoff), handoff_bytes_(handoff_bytes),
      handoff_floats_(handoff_floats), max_window_(max_window) {}

RpcPrefixStage::~RpcPrefixStage() {
    std::string ignored;
    (void) shutdown(ignored);
}

bool RpcPrefixStage::wait_ready(std::string& err) {
    strata::rpc::Frame frame;
    if (!receive_expected(strata::rpc::Opcode::Ready, frame, err)) return false;
    ready_ = true;
    return true;
}

bool RpcPrefixStage::run_decode_window(int T, const int32_t* tokens, int64_t pos0, strata::core::PoolMultiFn pool,
                                       void* user, int32_t* out, std::string& err) {
    if (!ready_) {
        err = "the RPC prefix is not READY";
        return false;
    }
    if (T <= 0 || T > max_window_ || tokens == nullptr) {
        err = "the RPC prefix window is outside the negotiated size";
        return false;
    }
    strata::rpc::WindowRequest window;
    window.pos0 = (uint64_t) pos0;
    window.tokens.assign(tokens, tokens + T);
    if (!control_->send_request(strata::rpc::Opcode::Window, strata::rpc::encode_window_request(window), err))
        return false;
    strata::rpc::Frame frame;
    if (!receive_expected(strata::rpc::Opcode::WindowDone, frame, err)) return false;
    strata::rpc::WindowResult result;
    if (!strata::rpc::decode_window_result(frame.payload, result, err)) return false;
    if (result.rows != (uint32_t) T) {
        err = "RPC WINDOW_DONE returned a different row count";
        return false;
    }
    const std::uint64_t bytes = result.float_count * sizeof(float);
    if (handoff_floats_ == 0 || result.float_count != (uint64_t) T * handoff_floats_ || bytes > handoff_bytes_ ||
        bytes > control_->bulk().size()) {
        err = "RPC WINDOW_DONE hand-off size does not match this geometry";
        return false;
    }
    std::memcpy(handoff_, control_->bulk().data(), (size_t) bytes);
    last_rows_ = T;
    return suffix_.run_decode_window(T, tokens, pos0, pool, user, out, err);
}

bool RpcPrefixStage::commit_decode_prefix(int n_keep, std::string& err) {
    if (!ready_ || n_keep <= 0 || n_keep > last_rows_) {
        err = "the RPC prefix commit is outside the last window";
        return false;
    }
    if (!control_->send_request(strata::rpc::Opcode::Commit, strata::rpc::encode_commit_request((uint32_t) n_keep),
                                err))
        return false;
    strata::rpc::Frame frame;
    if (!receive_expected(strata::rpc::Opcode::CommitDone, frame, err)) return false;
    uint32_t keep = 0;
    if (!strata::rpc::decode_commit_result(frame.payload, keep, err)) return false;
    if (keep != (uint32_t) n_keep) {
        err = "RPC COMMIT_DONE returned a different prefix";
        return false;
    }
    if (!suffix_.commit_decode_prefix(n_keep, err)) return false;
    last_rows_ = 0;
    return true;
}

bool RpcPrefixStage::shutdown(std::string& err) {
    if (!ready_ || control_->state().state() == strata::rpc::ConnectionDirection::Closed) return true;
    ready_ = false;
    return control_->send_request(strata::rpc::Opcode::Shutdown, {}, err);
}

bool RpcPrefixStage::receive_expected(strata::rpc::Opcode expected, strata::rpc::Frame& frame, std::string& err) {
    if (!control_->recv(frame, err)) return false;
    if (frame.header.opcode == strata::rpc::Opcode::Error) {
        err = payload_error(frame);
        return false;
    }
    if (frame.header.opcode != expected) {
        err = "RPC controller expected " + std::string(strata::rpc::opcode_name(expected)) + ", received " +
              std::string(strata::rpc::opcode_name(frame.header.opcode));
        return false;
    }
    return true;
}

}  // namespace strata::program
