// include/strata/program/stage_runtime.hpp - the CUDA/HIP program's per-GPU stage runtime data.
//
// A layer-split stage is deliberately one object with explicit ownership: dense weights and an expert cache only for
// [lb,le), its session and verifier window for the same range, its prompt path, and (last stage) the head.  The
// hand-off buffers and Verifier::run/commit chain remain where they always were.  The RPC work reuses this boundary;
// it does not replace the existing in-process split path.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"
#include "strata/prefill/prefill.hpp"

namespace strata::program {

/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    strata::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// THE ROUTING TRACE, which is P0.S8 and a stated prerequisite of Phase 3.  `drive_pool` is called once
    /// per layer from the main loop - the workers live inside `expert_pool_dispatch` - so a single FILE* here
    /// needs no locking.  `d.layers` is the CURRENT layer on entry (the adapter increments it as it walks the
    /// blob), which is why the layer index comes from there rather than from a counter of our own.
    std::FILE* routing = nullptr;
};

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out);

/// Plan v0.3 P6: the pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer);

/// Layer split: every verify stage shares one Drive (its counters, usage and failure flags); the GPU plan, the expert
/// cache and the PCIe share the pool uses for a layer are those of the stage that runs it.
struct SplitDrive {
    static constexpr int kMax = 8;
    Drive* base = nullptr;
    int n = 0;                                    ///< stages
    int64_t end[kMax] = {};                       ///< stage i runs the layers from end[i - 1] (0) below end[i]
    strata::core::GpuPlanSink* plan[kMax] = {};
    const uint8_t* cache_base[kMax] = {};
    const uint64_t* cache_slot_off[kMax] = {};
    int pcie_num[kMax] = {};
};

void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer);

/// The stage that owns a layer for explicit boundaries.  `split_at[i]` is the first layer of stage i + 1, so
/// stage 0 runs layers below split_at[0] and the last stage runs from the final boundary to the end.
inline int stage_for_layer(int64_t layer, const std::vector<int64_t>& split_at) {
    int st = 0;
    while (st < (int) split_at.size() && layer >= split_at[(size_t) st]) ++st;
    return st;
}

/// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
/// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
/// lives on its device too).
struct GpuStage {
    int dev = 0;
    int64_t lb = 0, le = 0;
    double pcie_frac = 0.0;
    double link_gbps = 0.0;   ///< this card's probed host->device bandwidth (PcieAuto's per-stage share)
    strata::core::WeightTable wt;
    strata::core::NativeDense dense;
    strata::core::NativeHead head;
    strata::core::SessionState ss;
    cudaStream_t stream = nullptr;
    strata::core::ExpertCache cache;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< its layers' share of the profile, hottest first
    int32_t* d_res = nullptr;                            ///< the residency table on its device
    strata::core::Verifier ver;
    strata::core::Verifier ver_b;                        ///< --pipeline-windows: the odd windows' verifier
    strata::prefill::Prefill sp;
    cudaStream_t adapt_stream = nullptr;
    cudaEvent_t adapt_ev = nullptr;
    bool adapt_live = false;                             ///< swaps of this request are in flight on it
    int32_t* mrope = nullptr;                            ///< --vision: the image-position table on its device
};

}  // namespace strata::program
