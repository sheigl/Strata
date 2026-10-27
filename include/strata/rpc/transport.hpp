// include/strata/rpc/transport.hpp - local transport for the RPC worker boundary.
//
// LocalTransport is deliberately Linux/POSIX-only in its first implementation. A future network transport can use the
// protocol layer without changing the engine-side bulk/message API. No GPU runtime memory is shared by this class.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "strata/rpc/protocol.hpp"

namespace strata::rpc {

/// A fixed-size, fd-passed bulk region. Both peers map it for CPU-visible staging only; the engine owns the copies
/// into and out of its device-visible buffers.
class BulkRegion {
public:
    BulkRegion() = default;
    ~BulkRegion();
    BulkRegion(const BulkRegion&) = delete;
    BulkRegion& operator=(const BulkRegion&) = delete;

    uint8_t* data() const { return static_cast<uint8_t*>(mapping_); }
    std::size_t size() const { return size_; }
    bool valid() const { return mapping_ != nullptr && size_ != 0; }

    void reset();
    bool attach(int fd, std::uint64_t expected_bytes, std::string& err);

private:
    int fd_ = -1;
    void* mapping_ = nullptr;
    std::size_t size_ = 0;
};

/// A Unix-domain control channel plus one memfd bulk region. The controller uses connect(); the worker uses listen().
class LocalTransport {
public:
    LocalTransport(const LocalTransport&) = delete;
    LocalTransport& operator=(const LocalTransport&) = delete;
    ~LocalTransport();

    /// Accept one local worker connection and, when bulk_bytes is non-zero, receive the controller's memfd.  The
    /// handshake (HELLO/HELLO_ACK) may run before the controller exists, so a worker can defer the bulk wait until
    /// after it has answered that handshake.
    static std::unique_ptr<LocalTransport> listen(const std::string& socket_path, const std::string& token,
                                                  std::uint64_t bulk_bytes, std::string& err, int timeout_ms = 30000,
                                                  bool defer_bulk = false);

    /// Connect to a local worker and, when bulk_bytes is non-zero, pass a newly created sealed memfd.
    static std::unique_ptr<LocalTransport> connect(const std::string& socket_path, const std::string& token,
                                                   std::uint64_t bulk_bytes, std::string& err, int timeout_ms = 30000);

    /// Adopt a connected control socket whose token handshake the supervisor already completed, then pass the
    /// controller's memfd.  The controller uses this when the supervisor hands its authenticated fd to the engine.
    static std::unique_ptr<LocalTransport> adopt(int fd, std::uint64_t bulk_bytes, std::string& err,
                                                 int timeout_ms = 30000);

    /// Worker side of a deferred listen: wait for the controller's memfd.
    bool receive_bulk(std::string& err);

    bool send(const Frame& frame, std::string& err);
    bool recv(Frame& frame, std::string& err);

    BulkRegion& bulk() { return bulk_; }
    const BulkRegion& bulk() const { return bulk_; }

private:
    explicit LocalTransport(int timeout_ms);
    bool configure_socket(int fd, std::string& err);
    bool handshake_server(const std::string& token, std::string& err);
    bool handshake_client(const std::string& token, std::string& err);
    bool offer_bulk(std::string& err);
    bool accept_bulk(std::string& err);
    bool send_fd(int fd, std::string& err);
    bool receive_fd(int& fd, std::string& err);

    int socket_ = -1;
    int listening_socket_ = -1;
    std::string socket_path_;
    int timeout_ms_ = 30000;
    std::uint64_t bulk_bytes_ = 0;
    BulkRegion bulk_;
};

}  // namespace strata::rpc
