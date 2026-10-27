// include/strata/program/rpc_prefix_stage.hpp - the controller-side proxy: run the worker prefix, then the local suffix.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "strata/core/decode_stage.hpp"
#include "strata/core/verify.hpp"
#include "strata/rpc/control.hpp"

namespace strata::program {

class RpcPrefixStage final : public strata::core::DecodeStage {
public:
    RpcPrefixStage(std::unique_ptr<strata::rpc::ControlChannel> control, strata::core::Verifier& suffix,
                   float* handoff, std::uint64_t handoff_bytes, std::uint64_t handoff_floats, int max_window);
    ~RpcPrefixStage() override;

    /// Receive READY after the controller's own suffix is loaded.
    bool wait_ready(std::string& err);

    bool run_decode_window(int T, const int32_t* tokens, int64_t pos0, strata::core::PoolMultiFn pool, void* user,
                           int32_t* out, std::string& err) override;
    bool commit_decode_prefix(int n_keep, std::string& err) override;
    bool supports_batch_decode() const override { return false; }

    /// Send SHUTDOWN after a clean controller stop (the supervisor terminates it if this never runs).
    bool shutdown(std::string& err);

private:
    bool receive_expected(strata::rpc::Opcode expected, strata::rpc::Frame& frame, std::string& err);

    std::unique_ptr<strata::rpc::ControlChannel> control_;
    strata::core::Verifier& suffix_;
    float* handoff_ = nullptr;
    std::uint64_t handoff_bytes_ = 0;
    std::uint64_t handoff_floats_ = 0;
    int max_window_ = 0;
    int last_rows_ = 0;
    bool ready_ = false;
};

}  // namespace strata::program
