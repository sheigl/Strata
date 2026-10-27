// src/program/stage_runtime.cpp - the CUDA/HIP driver's shared CPU expert-pool routing for stage ranges.
//
// The split decision, stage allocation order, and GPU work remain in generate.cpp.  This file owns only the routing
// and ownership data that every split stage uses, so the RPC prefix stage (Task 7) can use the same interface rather
// than invent a second notion of which card runs which layer.
#include "strata/program/stage_runtime.hpp"

#include <chrono>

namespace strata::program {
namespace {
using Clock = std::chrono::steady_clock;
}  // namespace

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out) {
    Drive* t = (Drive*) user;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch(&t->d, x_f, ids, weights, n_embd, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // THE ROUTING TRACE.  Written AFTER the dispatch so the layer index is still this layer's: `d.layers` is
    // advanced by the adapter as it consumes the blob, and reading it after the call is the same value the
    // dispatch used.  Record = int32 layer, int32 k, k int32 ids, k float weights.
    if (t->routing != nullptr) {
        // **`d.layers` HAS ALREADY BEEN ADVANCED BY THE TIME THIS RUNS, AND THE FIRST TRACE WAS OFF BY ONE
        // BECAUSE OF IT.**  The adapter walks the blob by incrementing `d.layers` as it consumes each layer's
        // experts, so after the dispatch it holds the NEXT layer's index.  `tools/make_profile.py` caught it
        // with a bounds check when the trace turned out to span 1..48 instead of 0..47.  The hit-rate CURVE was
        // unaffected - it is a per-layer split, and shifting every layer by one preserves both metrics - but
        // anything keyed on the layer index, which is exactly what a cache profile is, would have been wrong.
        const int32_t layer_idx = (int32_t) (t->d.layers - 1);
        if (layer_idx < 0 || layer_idx >= 48) {
            std::fprintf(stderr, "strata generate: the routing trace saw layer %d, outside 0..47\n", layer_idx);
            return;
        }
        const int32_t rec[2] = {layer_idx, (int32_t) k};
        std::fwrite(rec, sizeof rec, 1, t->routing);
        std::fwrite(ids, sizeof(int32_t), (size_t) k, t->routing);
        std::fwrite(weights, sizeof(float), (size_t) k, t->routing);
    }
}

void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // the routing trace for the serve path: the same record format drive_pool writes (layer, k, ids, weights),
    // one record per token.  The multi dispatch fuses the router weights into the kernel and does not surface
    // them, so records carry unit weights: tools/make_profile.py ranks pairs by routed frequency, which is the
    // signal that matters; a one-shot --dump-routing run records true weights if a weighted ranking is wanted.
    if (t->routing != nullptr && layer >= 0 && layer < 48) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};   // k <= 64 in a verify window; zeros read as unit weights
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
}

void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    SplitDrive* s = (SplitDrive*) user;
    int st = 0;
    while (st + 1 < s->n && layer >= s->end[st]) ++st;
    Drive& d = *s->base;
    d.d.plan = s->plan[st];
    d.d.cache_base = s->cache_base[st];
    d.d.cache_slot_off = s->cache_slot_off[st];
    d.d.pcie_num = s->pcie_num[st];
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}

}  // namespace strata::program
