// Protocol framing and state-machine rules for RPC workers: no GPU, no model, no sockets.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/rpc/protocol.hpp"

using namespace strata::rpc;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static Frame make(Opcode opcode, uint64_t id, std::vector<uint8_t> payload = {}) {
    Frame f(opcode, id, std::move(payload));
    return f;
}

static bool send(ConnectionState& c, Opcode opcode, uint64_t id) {
    FrameHeader h;
    h.opcode = opcode;
    h.request_id = id;
    std::string err;
    return c.accepts(h, err);
}

int main() {
    {
        Frame original(Opcode::Window, 42, std::vector<uint8_t>{1, 2, 3, 255});
        original.header.stage_id = 7;
        original.header.flags = 0xabcdef01u;
        original.header.bulk_bytes = 4096;
        std::vector<uint8_t> encoded;
        std::string err;
        CHECK(encode_frame(original, encoded, err));
        CHECK(encoded.size() == kFrameHeaderBytes + original.payload.size());

        Frame decoded;
        CHECK(decode_frame(encoded.data(), encoded.size(), decoded, err));
        CHECK(decoded.header.opcode == Opcode::Window);
        CHECK(decoded.header.request_id == 42);
        CHECK(decoded.header.stage_id == 7);
        CHECK(decoded.header.flags == 0xabcdef01u);
        CHECK(decoded.header.payload_bytes == original.payload.size());
        CHECK(decoded.header.bulk_bytes == 4096);
        CHECK(decoded.payload == original.payload);
    }

    {
        Frame original(Opcode::Stats, 1);
        std::vector<uint8_t> encoded;
        std::string err;
        CHECK(encode_frame(original, encoded, err));

        encoded[0] ^= 0xff;
        Frame ignored;
        CHECK(!decode_frame(encoded.data(), encoded.size(), ignored, err));

        encoded = {0};
        CHECK(encode_frame(original, encoded, err));
        encoded[4] = kProtocolVersion + 1;
        CHECK(!decode_frame(encoded.data(), encoded.size(), ignored, err));

        encoded[4] = 0;
        CHECK(!decode_frame(encoded.data(), encoded.size(), ignored, err));

        encoded[4] = kProtocolVersion;
        encoded[6] = 99;
        CHECK(!decode_frame(encoded.data(), encoded.size(), ignored, err));
    }

    {
        Frame original(Opcode::Window, 1, std::vector<uint8_t>(16, 9));
        std::vector<uint8_t> encoded;
        std::string err;
        CHECK(encode_frame(original, encoded, err));
        Frame ignored;
        CHECK(!decode_frame(encoded.data(), kFrameHeaderBytes - 1, ignored, err));
        CHECK(!decode_frame(encoded.data(), encoded.size() - 1, ignored, err));
        CHECK(!decode_frame(encoded.data(), encoded.size() + 1, ignored, err));
    }

    {
        std::vector<uint8_t> oversized;
        append_u32(oversized, kFrameMagic);
        append_u16(oversized, kProtocolVersion);
        append_u16(oversized, static_cast<uint16_t>(Opcode::Stats));
        append_u64(oversized, 1);
        append_u32(oversized, 0);
        append_u32(oversized, 0);
        append_u64(oversized, kMaxControlPayloadBytes + 1);
        append_u64(oversized, 0);
        Frame ignored;
        std::string err;
        CHECK(!decode_frame(oversized.data(), oversized.size(), ignored, err));
    }

    {
        ConnectionState c;
        std::string err;
        CHECK(send(c, Opcode::Hello, 1));
        CHECK(send(c, Opcode::HelloAck, 1));
        CHECK(send(c, Opcode::InitStage, 2));
        CHECK(send(c, Opcode::InitProgress, 2));
        CHECK(c.state() == ConnectionDirection::Loading);
        CHECK(send(c, Opcode::Ready, 2));
        CHECK(c.state() == ConnectionDirection::Ready);
        CHECK(send(c, Opcode::Window, 3));
        CHECK(c.state() == ConnectionDirection::Running);
        CHECK(send(c, Opcode::WindowDone, 3));
        CHECK(c.state() == ConnectionDirection::Ready);
        CHECK(send(c, Opcode::Commit, 4));
        CHECK(c.state() == ConnectionDirection::Committing);
        CHECK(send(c, Opcode::CommitDone, 4));
        CHECK(c.state() == ConnectionDirection::Ready);
    }

    {
        ConnectionState c;
        CHECK(send(c, Opcode::Hello, 1));
        CHECK(send(c, Opcode::HelloAck, 1));
        CHECK(send(c, Opcode::InitStage, 2));
        CHECK(send(c, Opcode::Ready, 2));
        CHECK(send(c, Opcode::Window, 3));
        CHECK(!send(c, Opcode::Window, 4));
        CHECK(c.active_request_id() == 3);
        CHECK(!send(c, Opcode::WindowDone, 4));
        CHECK(send(c, Opcode::WindowDone, 3));
        CHECK(c.active_request_id() == 0);
        CHECK(!send(c, Opcode::WindowDone, 3));
    }

    {
        ConnectionState c;
        CHECK(send(c, Opcode::Hello, 1));
        CHECK(send(c, Opcode::HelloAck, 1));
        CHECK(send(c, Opcode::InitStage, 2));
        CHECK(send(c, Opcode::Ready, 2));
        CHECK(send(c, Opcode::Window, 3));
        CHECK(send(c, Opcode::Cancel, 4));
        CHECK(c.state() == ConnectionDirection::CancelRequested);
        CHECK(send(c, Opcode::CancelDone, 4));
        CHECK(c.state() == ConnectionDirection::Ready);
        CHECK(send(c, Opcode::PrefillChunk, 5));
        CHECK(c.state() == ConnectionDirection::Running);
        CHECK(send(c, Opcode::PrefillChunkDone, 5));
    }

    if (failed == 0) std::printf("rpc_protocol_test: ok\n");
    return failed ? 1 : 0;
}
