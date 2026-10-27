// src/rpc/protocol.cpp - validation for the Strata worker wire contract.
//
// The decoder performs bounds and state checks before any backend receives a message. This file must stay portable:
// no CUDA/HIP/SYCL/ggml types, no sockets, and no engine-side state other than the protocol's own sequence/state.
#include "strata/rpc/protocol.hpp"

#include <cstring>

namespace strata::rpc {

namespace {

void fail(std::string& err, const std::string& message) { err = message; }

bool valid_state_transition(ConnectionDirection state, Opcode opcode) {
    using C = ConnectionDirection;
    switch (opcode) {
    case Opcode::Hello:
        return state == C::Starting || state == C::Loading;
    case Opcode::HelloAck:
        return state == C::Starting || state == C::Loading;
    case Opcode::InitStage:
        return state == C::Loading;
    case Opcode::InitProgress:
        return state == C::Loading;
    case Opcode::Ready:
        return state == C::Loading;
    case Opcode::Window:
    case Opcode::PrefillChunk:
        return state == C::Ready;
    case Opcode::WindowDone:
    case Opcode::PrefillChunkDone:
        return state == C::Running;
    case Opcode::Commit:
        return state == C::Ready;
    case Opcode::CommitDone:
        return state == C::Committing;
    case Opcode::Cancel:
        return state == C::Running || state == C::Committing;
    case Opcode::CancelDone:
        return state == C::CancelRequested;
    case Opcode::Stats:
    case Opcode::Ping:
        return state == C::Loading || state == C::Ready || state == C::Running || state == C::Committing ||
               state == C::CancelRequested;
    case Opcode::StatsResponse:
    case Opcode::Pong:
        return state != C::Failed && state != C::Closed;
    case Opcode::Shutdown:
        return state != C::Closed;
    case Opcode::Error:
        return state != C::Closed;
    default:
        return false;
    }
}

}  // namespace

const char* opcode_name(Opcode opcode) {
    switch (opcode) {
    case Opcode::Invalid: return "Invalid";
    case Opcode::Hello: return "Hello";
    case Opcode::HelloAck: return "HelloAck";
    case Opcode::InitStage: return "InitStage";
    case Opcode::InitProgress: return "InitProgress";
    case Opcode::Ready: return "Ready";
    case Opcode::Window: return "Window";
    case Opcode::WindowDone: return "WindowDone";
    case Opcode::Commit: return "Commit";
    case Opcode::CommitDone: return "CommitDone";
    case Opcode::PrefillChunk: return "PrefillChunk";
    case Opcode::PrefillChunkDone: return "PrefillChunkDone";
    case Opcode::Cancel: return "Cancel";
    case Opcode::CancelDone: return "CancelDone";
    case Opcode::Stats: return "Stats";
    case Opcode::Ping: return "Ping";
    case Opcode::Shutdown: return "Shutdown";
    case Opcode::Error: return "Error";
    case Opcode::StatsResponse: return "StatsResponse";
    case Opcode::Pong: return "Pong";
    default: return "Unknown";
    }
}

const char* connection_state_name(ConnectionDirection state) {
    switch (state) {
    case ConnectionDirection::Starting: return "Starting";
    case ConnectionDirection::Loading: return "Loading";
    case ConnectionDirection::Ready: return "Ready";
    case ConnectionDirection::Running: return "Running";
    case ConnectionDirection::Committing: return "Committing";
    case ConnectionDirection::CancelRequested: return "CancelRequested";
    case ConnectionDirection::Failed: return "Failed";
    case ConnectionDirection::Closed: return "Closed";
    }
    return "Unknown";
}

bool ConnectionState::accepts(const FrameHeader& header, std::string& err) {
    err.clear();
    if (!is_valid_opcode(header.opcode)) {
        fail(err, "unknown RPC opcode");
        return false;
    }
    if (header.payload_bytes > kMaxControlPayloadBytes) {
        fail(err, "RPC control payload exceeds the protocol limit");
        return false;
    }
    if (header.bulk_bytes > kMaxBulkBytes) {
        fail(err, "RPC bulk payload exceeds the protocol limit");
        return false;
    }
    if (!valid_state_transition(state_, header.opcode)) {
        fail(err, std::string("opcode ") + opcode_name(header.opcode) + " is not valid in state " + connection_state_name(state_));
        return false;
    }

    const bool request = is_request(header.opcode);
    if (request) {
        if (header.request_id != next_request_id_) {
            fail(err, "RPC request id does not continue the connection sequence");
            return false;
        }
        ++next_request_id_;

        switch (header.opcode) {
        case Opcode::Hello:
        case Opcode::InitStage:
        case Opcode::Window:
        case Opcode::Commit:
        case Opcode::PrefillChunk:
            if (active_request_id_ != 0) {
                fail(err, "an RPC operation is already in flight");
                return false;
            }
            active_request_id_ = header.request_id;
            state_ = header.opcode == Opcode::Window || header.opcode == Opcode::PrefillChunk ? ConnectionDirection::Running
                   : header.opcode == Opcode::Commit ? ConnectionDirection::Committing
                   : state_;
            break;
        case Opcode::Cancel:
            if (active_request_id_ == 0) {
                fail(err, "cancel arrived without an operation to cancel");
                return false;
            }
            active_request_id_ = header.request_id;
            state_ = ConnectionDirection::CancelRequested;
            break;
        case Opcode::Ping:
            state_ = state_;
            break;
        case Opcode::Stats:
            break;
        case Opcode::Shutdown:
            active_request_id_ = 0;
            state_ = ConnectionDirection::Closed;
            break;
        default:
            break;
        }
        return true;
    }

    if (header.opcode == Opcode::Error) {
        active_request_id_ = 0;
        state_ = ConnectionDirection::Failed;
        return true;
    }
    if (header.request_id == active_request_id_) {
        switch (header.opcode) {
        case Opcode::InitProgress:
            state_ = ConnectionDirection::Loading;
            return true;
        case Opcode::HelloAck:
            active_request_id_ = 0;
            state_ = ConnectionDirection::Loading;
            return true;
        case Opcode::Ready:
            active_request_id_ = 0;
            state_ = ConnectionDirection::Ready;
            return true;
        case Opcode::WindowDone:
        case Opcode::PrefillChunkDone:
            active_request_id_ = 0;
            state_ = ConnectionDirection::Ready;
            return true;
        case Opcode::CommitDone:
            active_request_id_ = 0;
            state_ = ConnectionDirection::Ready;
            return true;
        case Opcode::CancelDone:
            active_request_id_ = 0;
            state_ = ConnectionDirection::Ready;
            return true;
        default:
            fail(err, std::string(opcode_name(header.opcode)) + " is not an operation response");
            return false;
        }
    }
    fail(err, "RPC response does not match the active request");
    return false;
}

void append_u8(std::vector<uint8_t>& out, uint8_t value) { out.push_back(value); }

void append_u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xffu));
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xffu));
}

void append_u64(std::vector<uint8_t>& out, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<uint8_t>((value >> shift) & 0xffull));
}

bool read_u8(const uint8_t* data, std::size_t available, std::size_t& at, uint8_t& value) {
    if (at >= available) return false;
    value = data[at++];
    return true;
}

bool read_u16(const uint8_t* data, std::size_t available, std::size_t& at, uint16_t& value) {
    if (at + 2 > available) return false;
    value = static_cast<uint16_t>(data[at]) | static_cast<uint16_t>(static_cast<unsigned>(data[at + 1]) << 8);
    at += 2;
    return true;
}

bool read_u32(const uint8_t* data, std::size_t available, std::size_t& at, uint32_t& value) {
    if (at + 4 > available) return false;
    value = static_cast<uint32_t>(data[at]) | (static_cast<uint32_t>(data[at + 1]) << 8) |
            (static_cast<uint32_t>(data[at + 2]) << 16) | (static_cast<uint32_t>(data[at + 3]) << 24);
    at += 4;
    return true;
}

bool read_u64(const uint8_t* data, std::size_t available, std::size_t& at, uint64_t& value) {
    if (at + 8 > available) return false;
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<uint64_t>(data[at + shift / 8]) << shift;
    at += 8;
    return true;
}

bool encode_frame(const Frame& frame, std::vector<uint8_t>& out, std::string& err) {
    err.clear();
    if (!is_valid_opcode(frame.header.opcode)) {
        fail(err, "cannot encode an invalid RPC opcode");
        return false;
    }
    if (frame.payload.size() > kMaxControlPayloadBytes) {
        fail(err, "cannot encode an oversized RPC control payload");
        return false;
    }
    if (frame.header.bulk_bytes > kMaxBulkBytes) {
        fail(err, "cannot encode an oversized RPC bulk size");
        return false;
    }

    out.clear();
    out.reserve(kFrameHeaderBytes + frame.payload.size());
    append_u32(out, kFrameMagic);
    append_u16(out, kProtocolVersion);
    append_u16(out, static_cast<uint16_t>(frame.header.opcode));
    append_u64(out, frame.header.request_id);
    append_u32(out, frame.header.stage_id);
    append_u32(out, frame.header.flags);
    append_u64(out, frame.payload.size());
    append_u64(out, frame.header.bulk_bytes);
    out.insert(out.end(), frame.payload.begin(), frame.payload.end());
    return true;
}

bool decode_frame(const uint8_t* data, std::size_t bytes, Frame& frame, std::string& err) {
    std::size_t consumed = 0;
    if (!decode_next_frame(data, bytes, frame, consumed, err)) return false;
    if (consumed != bytes) {
        fail(err, "RPC frame buffer contains trailing bytes");
        return false;
    }
    return true;
}

bool decode_header(const uint8_t* data, std::size_t bytes, FrameHeader& header, std::string& err) {
    err.clear();
    header = FrameHeader();
    if (data == nullptr) {
        fail(err, "RPC decoder received a null buffer");
        return false;
    }
    if (bytes < kFrameHeaderBytes) {
        fail(err, "RPC frame is shorter than its header");
        return false;
    }

    std::size_t at = 0;
    uint32_t magic = 0;
    uint16_t wire_version = 0, wire_opcode = 0;
    uint64_t request_id = 0, payload_bytes = 0, bulk_bytes = 0;
    uint32_t stage_id = 0, flags = 0;
    if (!read_u32(data, bytes, at, magic) || !read_u16(data, bytes, at, wire_version) ||
        !read_u16(data, bytes, at, wire_opcode) || !read_u64(data, bytes, at, request_id) ||
        !read_u32(data, bytes, at, stage_id) || !read_u32(data, bytes, at, flags) ||
        !read_u64(data, bytes, at, payload_bytes) || !read_u64(data, bytes, at, bulk_bytes)) {
        fail(err, "truncated RPC frame header");
        return false;
    }
    if (magic != kFrameMagic) {
        fail(err, "RPC frame has a bad magic value");
        return false;
    }
    if (wire_version < kMinProtocolVersion || wire_version > kProtocolVersion) {
        fail(err, "RPC protocol version is not supported");
        return false;
    }
    const Opcode opcode = static_cast<Opcode>(wire_opcode);
    if (!is_valid_opcode(opcode)) {
        fail(err, "RPC frame has an unknown opcode");
        return false;
    }
    if (payload_bytes > kMaxControlPayloadBytes) {
        fail(err, "RPC frame announces an oversized control payload");
        return false;
    }
    if (bulk_bytes > kMaxBulkBytes) {
        fail(err, "RPC frame announces an oversized bulk payload");
        return false;
    }

    header.opcode = opcode;
    header.request_id = request_id;
    header.stage_id = stage_id;
    header.flags = flags;
    header.payload_bytes = payload_bytes;
    header.bulk_bytes = bulk_bytes;
    return true;
}

bool assemble_frame(const FrameHeader& header, const uint8_t* payload, std::size_t payload_bytes, Frame& frame,
                    std::string& err) {
    err.clear();
    frame = Frame();
    if (payload_bytes != header.payload_bytes || (header.payload_bytes != 0 && payload == nullptr)) {
        fail(err, "RPC frame payload does not match its header");
        return false;
    }
    frame.header = header;
    if (payload_bytes != 0) frame.payload.assign(payload, payload + payload_bytes);
    return true;
}

bool decode_next_frame(const uint8_t* data, std::size_t available, Frame& frame, std::size_t& consumed, std::string& err) {
    err.clear();
    consumed = 0;
    frame = Frame();
    FrameHeader header;
    if (!decode_header(data, available, header, err)) return false;
    if (available - kFrameHeaderBytes < header.payload_bytes) {
        fail(err, "RPC frame payload is truncated");
        return false;
    }
    const std::size_t payload_size = static_cast<std::size_t>(header.payload_bytes);
    if (!assemble_frame(header, data + kFrameHeaderBytes, payload_size, frame, err)) return false;
    consumed = kFrameHeaderBytes + payload_size;
    return true;
}

}  // namespace strata::rpc
