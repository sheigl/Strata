// src/rpc/control.cpp - payload codecs and the stateful control channel for the local RPC worker.
#include "strata/rpc/control.hpp"

#include <cstring>
#include <limits>

namespace strata::rpc {
namespace {

bool checked_window_size(uint64_t count, std::string& err) {
    if (count == 0 || count > kMaxControlPayloadBytes / 4) {
        err = "RPC window row count is outside the control payload limit";
        return false;
    }
    return true;
}

}  // namespace

std::vector<uint8_t> encode_worker_init(const WorkerInit& init) {
    std::vector<uint8_t> out;
    out.reserve(24);
    append_u32(out, init.layer_begin);
    append_u32(out, init.layer_end);
    append_u32(out, init.handoff_floats);
    append_u32(out, init.max_window);
    append_u64(out, init.max_context);
    return out;
}

bool decode_worker_init(const std::vector<uint8_t>& payload, WorkerInit& init, std::string& err) {
    std::size_t at = 0;
    if (!read_u32(payload.data(), payload.size(), at, init.layer_begin) ||
        !read_u32(payload.data(), payload.size(), at, init.layer_end) ||
        !read_u32(payload.data(), payload.size(), at, init.handoff_floats) ||
        !read_u32(payload.data(), payload.size(), at, init.max_window) ||
        !read_u64(payload.data(), payload.size(), at, init.max_context) || at != payload.size()) {
        err = "RPC INIT_STAGE payload is malformed";
        return false;
    }
    return true;
}

std::vector<uint8_t> encode_window_request(const WindowRequest& window) {
    std::vector<uint8_t> out;
    out.reserve(16 + window.tokens.size() * 4);
    append_u64(out, window.pos0);
    append_u32(out, static_cast<uint32_t>(window.tokens.size()));
    append_u32(out, 0);
    for (const int32_t token : window.tokens) append_u32(out, static_cast<uint32_t>(token));
    return out;
}

bool decode_window_request(const std::vector<uint8_t>& payload, WindowRequest& window, std::string& err) {
    std::size_t at = 0;
    uint32_t rows = 0, reserved = 0;
    if (!read_u64(payload.data(), payload.size(), at, window.pos0) ||
        !read_u32(payload.data(), payload.size(), at, rows) ||
        !read_u32(payload.data(), payload.size(), at, reserved) || !checked_window_size(rows, err)) {
        if (err.empty()) err = "RPC WINDOW payload is malformed";
        return false;
    }
    if (payload.size() != 16ull + static_cast<uint64_t>(rows) * 4ull) {
        err = "RPC WINDOW token count does not match its payload";
        return false;
    }
    window.tokens.resize(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        uint32_t value = 0;
        if (!read_u32(payload.data(), payload.size(), at, value)) {
            err = "RPC WINDOW token payload is truncated";
            return false;
        }
        window.tokens[i] = static_cast<int32_t>(value);
    }
    return true;
}

std::vector<uint8_t> encode_window_result(const WindowResult& result) {
    std::vector<uint8_t> out;
    out.reserve(16);
    append_u32(out, result.rows);
    append_u32(out, 0);
    append_u64(out, result.float_count);
    return out;
}

bool decode_window_result(const std::vector<uint8_t>& payload, WindowResult& result, std::string& err) {
    std::size_t at = 0;
    uint32_t reserved = 0;
    if (!read_u32(payload.data(), payload.size(), at, result.rows) ||
        !read_u32(payload.data(), payload.size(), at, reserved) ||
        !read_u64(payload.data(), payload.size(), at, result.float_count) || at != payload.size()) {
        err = "RPC WINDOW_DONE payload is malformed";
        return false;
    }
    return true;
}

std::vector<uint8_t> encode_commit_request(uint32_t keep) {
    std::vector<uint8_t> out;
    out.reserve(8);
    append_u32(out, keep);
    append_u32(out, 0);
    return out;
}

bool decode_commit_request(const std::vector<uint8_t>& payload, uint32_t& keep, std::string& err) {
    std::size_t at = 0;
    uint32_t reserved = 0;
    if (!read_u32(payload.data(), payload.size(), at, keep) ||
        !read_u32(payload.data(), payload.size(), at, reserved) || at != payload.size()) {
        err = "RPC COMMIT payload is malformed";
        return false;
    }
    return true;
}

std::vector<uint8_t> encode_commit_result(uint32_t keep) {
    std::vector<uint8_t> out;
    out.reserve(4);
    append_u32(out, keep);
    return out;
}

bool decode_commit_result(const std::vector<uint8_t>& payload, uint32_t& keep, std::string& err) {
    std::size_t at = 0;
    if (!read_u32(payload.data(), payload.size(), at, keep) || at != payload.size()) {
        err = "RPC COMMIT_DONE payload is malformed";
        return false;
    }
    return true;
}

ControlChannel::ControlChannel(std::unique_ptr<LocalTransport> transport, ConnectionDirection state,
                               uint64_t next_request_id)
    : transport_(std::move(transport)), state_() {
    state_.resume(state, next_request_id);
}

std::unique_ptr<ControlChannel> ControlChannel::worker_listen(const std::string& socket_path, const std::string& token,
                                                              std::uint64_t bulk_bytes, std::string& err,
                                                              int timeout_ms) {
    auto transport = LocalTransport::listen(socket_path, token, bulk_bytes, err, timeout_ms, true);
    if (!transport) return nullptr;
    // LocalTransport::listen's authenticated HELLO/HELLO_ACK is the supervisor handshake.  The controller continues
    // the request sequence at 2 in the loading state; there is no second HELLO at this layer.
    return std::unique_ptr<ControlChannel>(new ControlChannel(std::move(transport), ConnectionDirection::Loading, 2));
}

std::unique_ptr<ControlChannel> ControlChannel::controller_connect(const std::string& socket_path,
                                                                   const std::string& token,
                                                                   std::uint64_t bulk_bytes, std::string& err,
                                                                   int timeout_ms) {
    auto transport = LocalTransport::connect(socket_path, token, bulk_bytes, err, timeout_ms);
    if (!transport) return nullptr;
    return std::unique_ptr<ControlChannel>(new ControlChannel(std::move(transport), ConnectionDirection::Loading, 2));
}

std::unique_ptr<ControlChannel> ControlChannel::controller_adopt(int fd, std::uint64_t bulk_bytes, std::string& err,
                                                                 int timeout_ms) {
    auto transport = LocalTransport::adopt(fd, bulk_bytes, err, timeout_ms);
    if (!transport) return nullptr;
    // The supervisor consumed request 1 with HELLO; the controller continues from 2 in the loading state.
    return std::unique_ptr<ControlChannel>(new ControlChannel(std::move(transport), ConnectionDirection::Loading, 2));
}

bool ControlChannel::receive_bulk(std::string& err) { return transport_->receive_bulk(err); }

bool ControlChannel::send_request(Opcode opcode, std::vector<uint8_t> payload, std::string& err) {
    Frame frame;
    frame.payload = std::move(payload);
    frame.header.opcode = opcode;
    frame.header.request_id = state_.next_request_id();
    frame.header.payload_bytes = frame.payload.size();
    if (!state_.accepts(frame.header, err)) return false;
    return transport_->send(frame, err);
}

bool ControlChannel::recv(Frame& frame, std::string& err) {
    if (!transport_->recv(frame, err)) return false;
    return state_.accepts(frame.header, err);
}

bool ControlChannel::send_response(Opcode opcode, std::vector<uint8_t> payload, std::string& err) {
    Frame frame;
    frame.payload = std::move(payload);
    frame.header.opcode = opcode;
    frame.header.request_id = state_.active_request_id();
    frame.header.payload_bytes = frame.payload.size();
    if (!state_.accepts(frame.header, err)) return false;
    return transport_->send(frame, err);
}

bool ControlChannel::send_error(const std::string& message, std::string& err) {
    std::vector<uint8_t> payload(message.begin(), message.end());
    Frame frame;
    frame.payload = std::move(payload);
    frame.header.opcode = Opcode::Error;
    frame.header.request_id = state_.active_request_id();
    frame.header.payload_bytes = frame.payload.size();
    if (!state_.accepts(frame.header, err)) return false;
    return transport_->send(frame, err);
}

}  // namespace strata::rpc
