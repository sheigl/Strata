// include/strata/rpc/control.hpp - the little stateful wrapper both RPC processes use on top of protocol/transport.
//
// The supervisor completes HELLO/HELLO_ACK before it hands the socket to the controller. The channel therefore has a
// worker handshake step and a controller resume step; after that both sides use one-in-flight requests and responses.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/rpc/protocol.hpp"
#include "strata/rpc/transport.hpp"

namespace strata::rpc {

/// The hand-off region fixed for the first local slice: 8 verify rows fit with a wide margin for Qwen3.8.
inline constexpr uint64_t kDefaultBulkBytes = 16ull * 1024ull * 1024ull;

struct WorkerInit {
    uint32_t layer_begin = 0;
    uint32_t layer_end = 0;
    uint32_t handoff_floats = 0;
    uint32_t max_window = 0;
    uint64_t max_context = 0;
};

struct WindowRequest {
    uint64_t pos0 = 0;
    std::vector<int32_t> tokens;
};

struct WindowResult {
    uint32_t rows = 0;
    uint64_t float_count = 0;
};

std::vector<uint8_t> encode_worker_init(const WorkerInit& init);
bool decode_worker_init(const std::vector<uint8_t>& payload, WorkerInit& init, std::string& err);

std::vector<uint8_t> encode_window_request(const WindowRequest& window);
bool decode_window_request(const std::vector<uint8_t>& payload, WindowRequest& window, std::string& err);

std::vector<uint8_t> encode_window_result(const WindowResult& result);
bool decode_window_result(const std::vector<uint8_t>& payload, WindowResult& result, std::string& err);

std::vector<uint8_t> encode_commit_request(uint32_t keep);
bool decode_commit_request(const std::vector<uint8_t>& payload, uint32_t& keep, std::string& err);

std::vector<uint8_t> encode_commit_result(uint32_t keep);
bool decode_commit_result(const std::vector<uint8_t>& payload, uint32_t& keep, std::string& err);

class ControlChannel {
public:
    /// Worker: accept the supervisor's connection and complete the HELLO/HELLO_ACK exchange.  The bulk wait is left
    /// to `receive_bulk()` so the supervisor can start the controller after its handshake succeeds.
    static std::unique_ptr<ControlChannel> worker_listen(const std::string& socket_path, const std::string& token,
                                                         std::uint64_t bulk_bytes, std::string& err,
                                                         int timeout_ms = 30000);

    /// Controller: adopt the supervisor's authenticated socket and offer the controller's bulk region.
    static std::unique_ptr<ControlChannel> controller_adopt(int fd, std::uint64_t bulk_bytes, std::string& err,
                                                            int timeout_ms = 30000);

    /// Controller: development path with no supervisor; connect and perform the token handshake directly.
    static std::unique_ptr<ControlChannel> controller_connect(const std::string& socket_path, const std::string& token,
                                                              std::uint64_t bulk_bytes, std::string& err,
                                                              int timeout_ms = 30000);

    bool receive_bulk(std::string& err);
    bool send_request(Opcode opcode, std::vector<uint8_t> payload, std::string& err);
    bool recv(Frame& frame, std::string& err);
    bool send_response(Opcode opcode, std::vector<uint8_t> payload, std::string& err);
    bool send_error(const std::string& message, std::string& err);

    BulkRegion& bulk() { return transport_->bulk(); }
    const BulkRegion& bulk() const { return transport_->bulk(); }
    ConnectionState& state() { return state_; }
    uint64_t active_request_id() const { return state_.active_request_id(); }

private:
    ControlChannel(std::unique_ptr<LocalTransport> transport, ConnectionDirection state, uint64_t next_request_id);

    std::unique_ptr<LocalTransport> transport_;
    ConnectionState state_;
};

}  // namespace strata::rpc
