// include/strata/rpc/protocol.hpp - backend-neutral wire framing for Strata RPC workers.
//
// This header deliberately carries no CUDA, HIP, SYCL, or ggml dependency: the same definitions are used by the
// CUDA tree, the SYCL tree, and CPU-only protocol tests. The wire layout is fixed-width little-endian. Callers must
// treat every decoded field as untrusted and pass through the validators below before acting on it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::rpc {

inline constexpr uint32_t kFrameMagic = 0x43505253u;  // 'S', 'R', 'P', 'C', written as little-endian bytes.
inline constexpr uint16_t kProtocolVersion = 1u;
inline constexpr uint16_t kMinProtocolVersion = 1u;

/// 4 + 2 + 2 + 8 + 4 + 4 + 8 + 8: magic, version, opcode, request id, stage id, flags, payload size, bulk size.
inline constexpr std::size_t kFrameHeaderBytes = 40u;
inline constexpr uint64_t kMaxControlPayloadBytes = 1024ull * 1024ull;

/// Bulk payloads live in an fd-passed region. The ceiling only prevents an untrusted header from asking for an
/// absurd allocation; the negotiated transport must be no larger than this either.
inline constexpr uint64_t kMaxBulkBytes = 16ull * 1024ull * 1024ull * 1024ull;

enum class Opcode : uint16_t {
    Invalid = 0,
    Hello = 1,
    HelloAck = 2,
    InitStage = 3,
    InitProgress = 4,
    Ready = 5,
    Window = 6,
    WindowDone = 7,
    Commit = 8,
    CommitDone = 9,
    PrefillChunk = 10,
    PrefillChunkDone = 11,
    Cancel = 12,
    CancelDone = 13,
    Stats = 14,
    Ping = 15,
    Shutdown = 16,
    Error = 17,
    StatsResponse = 18,
    Pong = 20,
};

constexpr bool is_response(Opcode opcode) {
    switch (opcode) {
    case Opcode::HelloAck:
    case Opcode::InitProgress:
    case Opcode::Ready:
    case Opcode::WindowDone:
    case Opcode::CommitDone:
    case Opcode::PrefillChunkDone:
    case Opcode::CancelDone:
    case Opcode::StatsResponse:
    case Opcode::Pong:
    case Opcode::Error:
        return true;
    default:
        return false;
    }
}

constexpr bool is_request(Opcode opcode) {
    switch (opcode) {
    case Opcode::Hello:
    case Opcode::InitStage:
    case Opcode::Window:
    case Opcode::Commit:
    case Opcode::PrefillChunk:
    case Opcode::Cancel:
    case Opcode::Stats:
    case Opcode::Ping:
    case Opcode::Shutdown:
        return true;
    default:
        return false;
    }
}

constexpr bool is_valid_opcode(Opcode opcode) { return opcode != Opcode::Invalid && (is_request(opcode) || is_response(opcode)); }

const char* opcode_name(Opcode opcode);

struct FrameHeader {
    Opcode opcode = Opcode::Invalid;
    uint64_t request_id = 0;
    uint32_t stage_id = 0;
    uint32_t flags = 0;
    uint64_t payload_bytes = 0;
    uint64_t bulk_bytes = 0;
};

struct Frame {
    FrameHeader header;
    std::vector<uint8_t> payload;

    Frame() = default;
    explicit Frame(Opcode opcode) {
        header.opcode = opcode;
    }
    Frame(Opcode opcode, uint64_t request_id, std::vector<uint8_t> payload = {}) {
        header.opcode = opcode;
        header.request_id = request_id;
        this->payload = std::move(payload);
    }
};

enum class ConnectionDirection { Starting, Loading, Ready, Running, Committing, CancelRequested, Failed, Closed };

const char* connection_state_name(ConnectionDirection state);

/// Validates both request/response sequence numbers and the one-in-flight operation rule. A response may be repeated
/// only for INIT_PROGRESS; a terminal response clears the active request. Callers should create one validator per
/// peer channel.
class ConnectionState {
public:
    bool accepts(const FrameHeader& header, std::string& err);
    ConnectionDirection state() const { return state_; }
    uint64_t active_request_id() const { return active_request_id_; }
    uint64_t next_request_id() const { return next_request_id_; }

    /// Continue a connection whose token handshake ran outside this C++ process (the supervisor's HELLO/HELLO_ACK).
    void resume(ConnectionDirection state, uint64_t next_request_id) {
        state_ = state;
        active_request_id_ = 0;
        next_request_id_ = next_request_id == 0 ? 1 : next_request_id;
    }

private:
    ConnectionDirection state_ = ConnectionDirection::Starting;
    uint64_t next_request_id_ = 1;
    uint64_t active_request_id_ = 0;
};

/// Encode one complete frame. The version is taken from the header supplied by the caller.
bool encode_frame(const Frame& frame, std::vector<uint8_t>& out, std::string& err);

/// Decode and validate only a frame header. The caller is responsible for reading the announced payload separately.
bool decode_header(const uint8_t* data, std::size_t bytes, FrameHeader& header, std::string& err);

/// Assemble a frame from a validated header and its already-read payload.
bool assemble_frame(const FrameHeader& header, const uint8_t* payload, std::size_t payload_bytes, Frame& frame, std::string& err);

/// Decode one complete frame from a buffer whose length is exactly that frame.
bool decode_frame(const uint8_t* data, std::size_t bytes, Frame& frame, std::string& err);

/// Decode one complete frame from a buffer. Any remaining bytes are left untouched and reported through consumed.
bool decode_next_frame(const uint8_t* data, std::size_t available, Frame& frame, std::size_t& consumed, std::string& err);

void append_u8(std::vector<uint8_t>& out, uint8_t value);
void append_u16(std::vector<uint8_t>& out, uint16_t value);
void append_u32(std::vector<uint8_t>& out, uint32_t value);
void append_u64(std::vector<uint8_t>& out, uint64_t value);

bool read_u8(const uint8_t* data, std::size_t available, std::size_t& at, uint8_t& value);
bool read_u16(const uint8_t* data, std::size_t available, std::size_t& at, uint16_t& value);
bool read_u32(const uint8_t* data, std::size_t available, std::size_t& at, uint32_t& value);
bool read_u64(const uint8_t* data, std::size_t available, std::size_t& at, uint64_t& value);

}  // namespace strata::rpc
