// src/rpc/local_transport_stub.cpp - a clear refusal where no local RPC transport is compiled in.
//
// The first feature is Linux-only.  This stub keeps the backend-neutral protocol library buildable on Windows and
// other platforms while making sure neither controller nor worker can silently select an unimplemented transport.
#include "strata/rpc/transport.hpp"

#include <utility>

namespace strata::rpc {
namespace {
void reject(std::string& err) {
    err = "Strata's local RPC transport is only implemented on Linux (AF_UNIX + memfd)";
}
}  // namespace

BulkRegion::~BulkRegion() = default;

void BulkRegion::reset() {}

bool BulkRegion::attach(int fd, std::uint64_t expected_bytes, std::string& err) {
    (void)fd;
    (void)expected_bytes;
    reject(err);
    return false;
}

LocalTransport::~LocalTransport() = default;

std::unique_ptr<LocalTransport> LocalTransport::listen(const std::string& socket_path, const std::string& token,
                                                      std::uint64_t bulk_bytes, std::string& err, int timeout_ms,
                                                      bool defer_bulk) {
    (void)socket_path;
    (void)token;
    (void)bulk_bytes;
    (void)timeout_ms;
    (void)defer_bulk;
    reject(err);
    return nullptr;
}

std::unique_ptr<LocalTransport> LocalTransport::adopt(int fd, std::uint64_t bulk_bytes, std::string& err,
                                                      int timeout_ms) {
    (void)fd;
    (void)bulk_bytes;
    (void)timeout_ms;
    reject(err);
    return nullptr;
}

bool LocalTransport::receive_bulk(std::string& err) {
    reject(err);
    return false;
}

std::unique_ptr<LocalTransport> LocalTransport::connect(const std::string& socket_path, const std::string& token,
                                                       std::uint64_t bulk_bytes, std::string& err, int timeout_ms) {
    (void)socket_path;
    (void)token;
    (void)bulk_bytes;
    (void)timeout_ms;
    reject(err);
    return nullptr;
}

bool LocalTransport::send(const Frame& frame, std::string& err) {
    (void)frame;
    reject(err);
    return false;
}

bool LocalTransport::recv(Frame& frame, std::string& err) {
    (void)frame;
    reject(err);
    return false;
}

}  // namespace strata::rpc
