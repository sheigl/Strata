// src/rpc/local_transport_posix.cpp - one-worker Linux transport.
//
// Authentication is intentionally local-only: same UID plus a per-run token supplied by the supervisor. There is no
// TCP listener, token in argv, shared GPU flag, or cross-vendor pointer sharing here.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include "strata/rpc/transport.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::rpc {

namespace {

void set_err(std::string& err, const char* prefix, int saved_errno) {
    err = std::string(prefix) + ": " + std::strerror(saved_errno);
}

bool token_equal(const std::string& a, const std::string& b) {
    const std::size_t n = std::max(a.size(), b.size());
    volatile unsigned char diff = static_cast<unsigned char>(a.size() ^ b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char ac = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        const unsigned char bc = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff = static_cast<unsigned char>(diff | (ac ^ bc));
    }
    return diff == 0 && a.size() == b.size();
}

bool read_exact(int fd, void* data, std::size_t bytes, std::string& err) {
    std::size_t got = 0;
    while (got < bytes) {
        const ssize_t n = ::read(fd, static_cast<unsigned char*>(data) + got, bytes - got);
        if (n == 0) {
            err = "RPC transport peer closed the control channel";
            return false;
        }
        if (n < 0) {
            const int e = errno;
            if (e == EINTR) continue;
            set_err(err, "RPC transport read failed", e);
            return false;
        }
        got += static_cast<std::size_t>(n);
    }
    return true;
}

bool write_exact(int fd, const void* data, std::size_t bytes, std::string& err) {
    std::size_t sent = 0;
    while (sent < bytes) {
        const ssize_t n = ::send(fd, static_cast<const unsigned char*>(data) + sent, bytes - sent, MSG_NOSIGNAL);
        if (n < 0) {
            const int e = errno;
            if (e == EINTR) continue;
            set_err(err, "RPC transport write failed", e);
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

int create_memfd(std::uint64_t bytes, std::string& err) {
    const int fd = static_cast<int>(::syscall(SYS_memfd_create, "strata-rpc-bulk", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (fd < 0) {
        set_err(err, "memfd_create failed", errno);
        return -1;
    }
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        const int e = errno;
        ::close(fd);
        set_err(err, "RPC bulk resize failed", e);
        return -1;
    }
    if (::fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0) {
        const int e = errno;
        ::close(fd);
        set_err(err, "RPC bulk seal failed", e);
        return -1;
    }
    return fd;
}

bool attach_memfd(int fd, std::uint64_t expected_bytes, BulkRegion& bulk, std::string& err) {
    return bulk.attach(fd, expected_bytes, err);
}

}  // namespace

BulkRegion::~BulkRegion() { reset(); }

void BulkRegion::reset() {
    if (mapping_ != nullptr) {
        ::munmap(mapping_, size_);
        mapping_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
}

bool BulkRegion::attach(int fd, std::uint64_t expected_bytes, std::string& err) {
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        err = std::string("RPC bulk stat failed: ") + std::strerror(errno);
        return false;
    }
    if (static_cast<std::uint64_t>(st.st_size) != expected_bytes) {
        err = "RPC bulk size does not match the negotiated size";
        return false;
    }
    void* map = ::mmap(nullptr, static_cast<size_t>(expected_bytes), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        err = std::string("RPC bulk mmap failed: ") + std::strerror(errno);
        return false;
    }
    fd_ = fd;
    mapping_ = map;
    size_ = static_cast<std::size_t>(expected_bytes);
    return true;
}

LocalTransport::LocalTransport(int timeout_ms) : timeout_ms_(timeout_ms < 1 ? 1 : timeout_ms) {}

LocalTransport::~LocalTransport() {
    if (socket_ >= 0) ::close(socket_);
    if (listening_socket_ >= 0) ::close(listening_socket_);
    if (!socket_path_.empty()) ::unlink(socket_path_.c_str());
}

bool LocalTransport::configure_socket(int fd, std::string& err) {
    // The supervisor uses Python's settimeout(), which leaves the descriptor O_NONBLOCK.  An inherited controller
    // socket must be blocking again or every read returns EAGAIN before the worker has finished a window.
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0 && (flags & O_NONBLOCK) != 0 && ::fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) != 0) {
        set_err(err, "RPC transport blocking mode setup failed", errno);
        return false;
    }
    timeval tv {};
    tv.tv_sec = timeout_ms_ / 1000;
    tv.tv_usec = static_cast<long>(timeout_ms_ % 1000) * 1000;
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
        set_err(err, "RPC transport receive timeout setup failed", errno);
        return false;
    }
    if (::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
        set_err(err, "RPC transport send timeout setup failed", errno);
        return false;
    }
    return true;
}

bool LocalTransport::handshake_server(const std::string& token, std::string& err) {
    Frame request;
    if (!recv(request, err)) return false;
    if (request.header.opcode != Opcode::Hello) {
        err = "RPC worker handshake did not begin with Hello";
        return false;
    }
    const std::string received(reinterpret_cast<const char*>(request.payload.data()), request.payload.size());
    if (!token_equal(token, received)) {
        err = "RPC worker token rejected";
        return false;
    }
    Frame ack(Opcode::HelloAck, request.header.request_id, request.payload);
    return send(ack, err);
}

bool LocalTransport::handshake_client(const std::string& token, std::string& err) {
    Frame hello(Opcode::Hello, 1, std::vector<uint8_t>(token.begin(), token.end()));
    if (!send(hello, err)) return false;
    Frame ack;
    if (!recv(ack, err)) return false;
    if (ack.header.opcode != Opcode::HelloAck) {
        err = "RPC worker handshake did not return HelloAck";
        return false;
    }
    const std::string received(reinterpret_cast<const char*>(ack.payload.data()), ack.payload.size());
    if (!token_equal(token, received)) {
        err = "RPC worker HelloAck token rejected";
        return false;
    }
    return true;
}

bool LocalTransport::offer_bulk(std::string& err) {
    const int fd = create_memfd(bulk_bytes_, err);
    if (fd < 0) return false;
    if (!attach_memfd(fd, bulk_bytes_, bulk_, err)) {
        ::close(fd);
        return false;
    }
    return send_fd(fd, err);
}

bool LocalTransport::accept_bulk(std::string& err) {
    int fd = -1;
    if (!receive_fd(fd, err)) return false;
    return attach_memfd(fd, bulk_bytes_, bulk_, err);
}

bool LocalTransport::send_fd(int fd, std::string& err) {
    char data = 0;
    iovec iov {};
    iov.iov_base = &data;
    iov.iov_len = 1;
    char control[CMSG_SPACE(sizeof(fd))] {};
    msghdr msg {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    cmsghdr* hdr = CMSG_FIRSTHDR(&msg);
    if (hdr == nullptr) {
        err = "RPC bulk fd control message is unavailable";
        return false;
    }
    hdr->cmsg_level = SOL_SOCKET;
    hdr->cmsg_type = SCM_RIGHTS;
    hdr->cmsg_len = CMSG_LEN(sizeof(fd));
    std::memcpy(CMSG_DATA(hdr), &fd, sizeof(fd));

    while (::sendmsg(socket_, &msg, 0) < 0) {
        const int e = errno;
        if (e == EINTR) continue;
        set_err(err, "RPC bulk fd send failed", e);
        return false;
    }
    return true;
}

bool LocalTransport::receive_fd(int& fd, std::string& err) {
    char data = 0;
    iovec iov {};
    iov.iov_base = &data;
    iov.iov_len = 1;
    char control[CMSG_SPACE(sizeof(int))] {};
    msghdr msg {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    const ssize_t n = ::recvmsg(socket_, &msg, MSG_CMSG_CLOEXEC);
    if (n < 0) {
        set_err(err, "RPC bulk fd receive failed", errno);
        return false;
    }
    if (n != 1) {
        err = "RPC bulk fd receive message is malformed";
        return false;
    }
    if ((msg.msg_flags & MSG_CTRUNC) != 0) {
        err = "RPC bulk fd control message was truncated";
        return false;
    }
    int count = 0;
    fd = -1;
    for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) continue;
        if (c->cmsg_len < CMSG_LEN(sizeof(int))) {
            err = "RPC bulk fd control message is malformed";
            return false;
        }
        std::memcpy(&fd, CMSG_DATA(c), sizeof(int));
        ++count;
    }
    if (count != 1 || fd < 0) {
        err = "RPC bulk negotiation did not receive exactly one fd";
        return false;
    }
    return true;
}

bool LocalTransport::send(const Frame& frame, std::string& err) {
    std::vector<uint8_t> encoded;
    if (!encode_frame(frame, encoded, err)) return false;
    return write_exact(socket_, encoded.data(), encoded.size(), err);
}

bool LocalTransport::recv(Frame& frame, std::string& err) {
    uint8_t header_bytes[kFrameHeaderBytes] {};
    if (!read_exact(socket_, header_bytes, sizeof(header_bytes), err)) return false;
    FrameHeader header;
    if (!decode_header(header_bytes, sizeof(header_bytes), header, err)) return false;

    std::vector<uint8_t> payload;
    if (header.payload_bytes != 0) {
        payload.resize(static_cast<std::size_t>(header.payload_bytes));
        if (!read_exact(socket_, payload.data(), payload.size(), err)) return false;
    }
    return assemble_frame(header, payload.data(), payload.size(), frame, err);
}

std::unique_ptr<LocalTransport> LocalTransport::listen(const std::string& socket_path, const std::string& token,
                                                       std::uint64_t bulk_bytes, std::string& err, int timeout_ms,
                                                       bool defer_bulk) {
    err.clear();
    if (socket_path.empty() || socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        err = "RPC socket path is empty or too long";
        return nullptr;
    }
    auto transport = std::unique_ptr<LocalTransport>(new LocalTransport(timeout_ms));
    transport->socket_path_ = socket_path;
    transport->bulk_bytes_ = bulk_bytes;

    const int listening = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listening < 0) {
        set_err(err, "RPC listener socket failed", errno);
        return nullptr;
    }
    transport->listening_socket_ = listening;
    ::unlink(socket_path.c_str());

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, socket_path.data(), socket_path.size());
    const socklen_t addr_len = static_cast<socklen_t>(sizeof(sa_family_t) + socket_path.size());
    if (::bind(listening, reinterpret_cast<const sockaddr*>(&addr), addr_len) != 0 || ::listen(listening, 1) != 0) {
        const int e = errno;
        transport.reset();
        set_err(err, "RPC listener bind failed", e);
        return nullptr;
    }

    // accept() has no socket receive timeout of its own.  Poll it so a supervisor can tell an absent worker from
    // a live process that has not created its socket yet.
    const auto accept_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(transport->timeout_ms_);
    bool ready_for_accept = false;
    while (!ready_for_accept) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= accept_deadline) {
            transport.reset();
            err = "RPC worker connection did not arrive before the transport deadline";
            return nullptr;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(accept_deadline - now);
        pollfd pfd {listening, POLLIN, 0};
        const int ready = ::poll(&pfd, 1, static_cast<int>(remaining.count()));
        if (ready == 0) continue;
        if (ready < 0) {
            const int e = errno;
            if (e == EINTR) continue;
            transport.reset();
            set_err(err, "RPC listener poll failed", e);
            return nullptr;
        }
        ready_for_accept = true;
    }

    sockaddr_un peer {};
    socklen_t peer_len = sizeof(peer);
    const int peer_fd = ::accept(listening, reinterpret_cast<sockaddr*>(&peer), &peer_len);
    if (peer_fd < 0) {
        const int e = errno;
        transport.reset();
        set_err(err, "RPC listener accept failed", e);
        return nullptr;
    }
    transport->socket_ = peer_fd;
    if (!transport->configure_socket(peer_fd, err)) {
        transport.reset();
        return nullptr;
    }

    ucred cred {};
    socklen_t cred_len = sizeof(cred);
    if (::getsockopt(peer_fd, SOL_SOCKET, SO_PEERCRED, &cred, &cred_len) != 0) {
        const int e = errno;
        transport.reset();
        set_err(err, "RPC peer credential check failed", e);
        return nullptr;
    }
    if (cred.uid != ::getuid()) {
        transport.reset();
        err = "RPC peer UID does not match this user";
        return nullptr;
    }
    if (!transport->handshake_server(token, err)) {
        transport.reset();
        return nullptr;
    }
    if (bulk_bytes != 0 && !defer_bulk && !transport->accept_bulk(err)) {
        transport.reset();
        return nullptr;
    }
    return transport;
}

bool LocalTransport::receive_bulk(std::string& err) {
    if (bulk_bytes_ == 0 || bulk_.valid()) return true;
    return accept_bulk(err);
}

std::unique_ptr<LocalTransport> LocalTransport::adopt(int fd, std::uint64_t bulk_bytes, std::string& err,
                                                      int timeout_ms) {
    err.clear();
    if (fd < 0) {
        err = "RPC supervisor did not pass a control socket";
        return nullptr;
    }
    auto transport = std::unique_ptr<LocalTransport>(new LocalTransport(timeout_ms));
    transport->bulk_bytes_ = bulk_bytes;
    transport->socket_ = fd;
    if (!transport->configure_socket(fd, err)) {
        transport.reset();
        return nullptr;
    }
    if (bulk_bytes != 0 && !transport->offer_bulk(err)) {
        transport.reset();
        return nullptr;
    }
    return transport;
}

std::unique_ptr<LocalTransport> LocalTransport::connect(const std::string& socket_path, const std::string& token,
                                                        std::uint64_t bulk_bytes, std::string& err, int timeout_ms) {
    err.clear();
    if (socket_path.empty() || socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        err = "RPC socket path is empty or too long";
        return nullptr;
    }
    auto transport = std::unique_ptr<LocalTransport>(new LocalTransport(timeout_ms));
    transport->bulk_bytes_ = bulk_bytes;

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, socket_path.data(), socket_path.size());
    const socklen_t addr_len = static_cast<socklen_t>(sizeof(sa_family_t) + socket_path.size());
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        set_err(err, "RPC client socket failed", errno);
        return nullptr;
    }
    transport->socket_ = fd;
    if (!transport->configure_socket(fd, err)) {
        transport.reset();
        return nullptr;
    }

    // The Unix connect itself should be immediate; retry briefly only to absorb a listener-start race in tests.
    bool connected = false;
    const int effective_timeout_ms = timeout_ms < 1 ? 1 : timeout_ms;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(effective_timeout_ms);
    while (!connected) {
        if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), addr_len) == 0) {
            connected = true;
            break;
        }
        const int e = errno;
        if (e != ENOENT && e != ECONNREFUSED && e != EAGAIN && e != EWOULDBLOCK) {
            transport.reset();
            set_err(err, "RPC client connect failed", e);
            return nullptr;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            transport.reset();
            err = "RPC worker socket did not become available";
            return nullptr;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!transport->handshake_client(token, err)) {
        transport.reset();
        return nullptr;
    }
    if (bulk_bytes != 0 && !transport->offer_bulk(err)) {
        transport.reset();
        return nullptr;
    }
    return transport;
}

}  // namespace strata::rpc
