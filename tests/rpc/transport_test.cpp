// Local RPC transport: one Unix socket, one token handshake, one memfd bulk region. No GPU or model is used.
#include <dirent.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "strata/rpc/transport.hpp"

using namespace strata::rpc;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static int raw_fd_count() {
    DIR* dir = ::opendir("/proc/self/fd");
    if (dir == nullptr) return 0;
    int count = 0;
    while (dirent* entry = ::readdir(dir)) ++count;
    ::closedir(dir);
    return count;
}

static int raw_connect(const std::string& path) {
    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, path.data(), path.size());
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr),
                  static_cast<socklen_t>(sizeof(sa_family_t) + path.size())) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

int main() {
    const std::string base = "/tmp/opencode/strata_rpc_transport_" + std::to_string(static_cast<long>(::getpid()));
    const std::string token = "0123456789abcdef0123456789abcdef";
    constexpr std::uint64_t bulk_bytes = 4096;
    const int initial_fds = raw_fd_count();

    {
        const std::string path = base + "_ok.sock";
        std::unique_ptr<LocalTransport> server;
        std::string server_err;
        std::thread server_thread([&] { server = LocalTransport::listen(path, token, bulk_bytes, server_err, 5000); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        std::string client_err;
        auto client = LocalTransport::connect(path, token, bulk_bytes, client_err, 5000);
        server_thread.join();
        CHECK(client != nullptr);
        CHECK(server != nullptr);
        if (client && server) {
            CHECK(client_err.empty());
            CHECK(server_err.empty());
            CHECK(client->bulk().valid());
            CHECK(server->bulk().valid());
            CHECK(client->bulk().size() == bulk_bytes);
            CHECK(server->bulk().size() == bulk_bytes);

            client->bulk().data()[0] = 0xa5;
            client->bulk().data()[bulk_bytes - 1] = 0x5a;
            CHECK(server->bulk().data()[0] == 0xa5);
            CHECK(server->bulk().data()[bulk_bytes - 1] == 0x5a);

            std::string err;
            Frame out(Opcode::Window, 9, std::vector<uint8_t>(257, 0x7e));
            out.header.stage_id = 3;
            out.header.bulk_bytes = bulk_bytes;
            CHECK(client->send(out, err));
            Frame in;
            CHECK(server->recv(in, err));
            CHECK(in.header.opcode == Opcode::Window);
            CHECK(in.header.request_id == 9);
            CHECK(in.header.stage_id == 3);
            CHECK(in.payload == out.payload);

            out = Frame(Opcode::WindowDone, 9, std::vector<uint8_t>{1, 2, 3});
            CHECK(server->send(out, err));
            CHECK(client->recv(in, err));
            CHECK(in.header.opcode == Opcode::WindowDone);
            CHECK(in.payload == out.payload);
        }
    }

    {
        const std::string path = base + "_bad_token.sock";
        std::unique_ptr<LocalTransport> server;
        std::string server_err;
        std::thread server_thread([&] { server = LocalTransport::listen(path, token, 0, server_err, 5000); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::string client_err;
        auto client = LocalTransport::connect(path, "wrong", 0, client_err, 5000);
        server_thread.join();
        CHECK(client == nullptr);
        CHECK(server == nullptr);
        CHECK(!client_err.empty());
        CHECK(!server_err.empty());
    }

    {
        const std::string path = base + "_no_connection.sock";
        const auto started = std::chrono::steady_clock::now();
        std::string server_err;
        auto server = LocalTransport::listen(path, token, 0, server_err, 150);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started);
        CHECK(server == nullptr);
        CHECK(!server_err.empty());
        CHECK(elapsed.count() >= 100 && elapsed.count() < 1500);
        CHECK(::access(path.c_str(), F_OK) == -1);
    }

    {
        const std::string path = base + "_dead_peer.sock";
        std::unique_ptr<LocalTransport> server;
        std::string server_err;
        std::thread server_thread([&] { server = LocalTransport::listen(path, token, 0, server_err, 5000); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int peer = raw_connect(path);
        CHECK(peer >= 0);
        if (peer >= 0) ::close(peer);
        server_thread.join();
        CHECK(server == nullptr);
        CHECK(!server_err.empty());
        CHECK(::access(path.c_str(), F_OK) == -1);
    }

    {
        const std::string path = base + "_short_frame.sock";
        std::unique_ptr<LocalTransport> server;
        std::string server_err;
        std::thread server_thread([&] { server = LocalTransport::listen(path, token, 0, server_err, 5000); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int peer = raw_connect(path);
        CHECK(peer >= 0);
        if (peer >= 0) {
            std::string err;
            Frame hello(Opcode::Hello, 1, std::vector<uint8_t>(token.begin(), token.end()));
            std::vector<uint8_t> encoded;
            CHECK(encode_frame(hello, encoded, err));
            CHECK(::send(peer, encoded.data(), 20, MSG_NOSIGNAL) == 20);
            ::close(peer);
        }
        server_thread.join();
        CHECK(server == nullptr);
        CHECK(!server_err.empty());
    }

    {
        const std::string path = base + "_last.sock";
        std::unique_ptr<LocalTransport> server;
        std::string server_err;
        std::thread server_thread([&] { server = LocalTransport::listen(path, token, 0, server_err, 5000); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::string client_err;
        auto client = LocalTransport::connect(path, token, 0, client_err, 5000);
        server_thread.join();
        CHECK(client != nullptr);
        CHECK(server != nullptr);
        if (client && server) {
            client.reset();
            std::string err;
            Frame in;
            CHECK(!server->recv(in, err));
            CHECK(!err.empty());
        }
    }
    CHECK(::access((base + "_last.sock").c_str(), F_OK) == -1);

    {
        BulkRegion bad_fd;
        std::string err;
        CHECK(!bad_fd.attach(-1, 4096, err));
        CHECK(!err.empty());
        CHECK(!bad_fd.valid());
    }

    CHECK(raw_fd_count() <= initial_fds + 1);

    if (failed == 0) std::printf("rpc_local_transport_test: ok\n");
    return failed ? 1 : 0;
}
