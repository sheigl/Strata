// include/strata/prefill/prefill_stage.hpp - the chunk boundary between two prompt stages.
//
// The local Prefill chain remains the implementation.  This interface names only what a later stage receives today:
// one chunk's tokens and positions, the hand-off from the previous stage, pipeline drain/reserve, and the counters the
// caller already reports.  Stop callbacks and MTP draft work stay on the first/local stage and do not cross it.
#pragma once

#include <cstdint>
#include <string>

namespace strata::prefill {

class PrefillStage {
public:
    virtual ~PrefillStage() = default;

    /// Run one prompt chunk `tokens[0..n)` at positions pos0.. on this stage.  Called on the stage's own thread when
    /// a layer split pipelines two stages.
    virtual bool run_chunk(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) = 0;

    /// The previous stage's residual rows and (optional) pipeline event, plus the chunk's shape.
    virtual void set_handoff(const void* rows, const void* event, int pieces, bool single_chunk,
                             bool pipe_chunked) = 0;

    /// Finish any chunk this stage still has in flight.
    virtual bool drain_pipeline(std::string& err) = 0;

    /// Grow the pinned upload slots for a text chunk of T tokens (no-op on a stage with no local buffers).
    virtual bool reserve_uploads(int64_t T) = 0;

    virtual int stage_count() const = 0;
    virtual int64_t chain_experts_streamed() const = 0;

    /// STRATA_MMQ_DIRECT_LOG=1 only: print this stage's and the later local stages' running direct-read totals.
    virtual void log_mmq_direct_totals() const {}
};

}  // namespace strata::prefill
