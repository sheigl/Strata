// include/strata/program/rpc_worker.hpp - the worker's small surface over the decode boundary.
#pragma once

#include <cstdint>
#include <string>

#include "strata/rpc/control.hpp"

namespace strata::program {

/// The worker owns a Verifier for [0,K); this interface keeps the RPC loop out of core/verify.hpp.
struct RpcWorkerStage {
    virtual ~RpcWorkerStage() = default;
    virtual bool run_window(int T, const int32_t* tokens, int64_t pos0, int32_t* out, std::string& err) = 0;
    virtual bool commit_window(uint32_t keep, std::string& err) = 0;
};

/// Wait for INIT_STAGE, send READY, then serve WINDOW/COMMIT/SHUTDOWN until the peer closes or asks to stop.
/// `staging` is the pinned host buffer the local Verifier writes through `Verifier::set_stage`; the worker copies
/// it into the shared bulk region after each window.
int run_rpc_worker(strata::rpc::ControlChannel& control, const strata::rpc::WorkerInit& expected, const float* staging,
                   std::uint64_t staging_bytes, RpcWorkerStage& stage, std::string& err);

}  // namespace strata::program
