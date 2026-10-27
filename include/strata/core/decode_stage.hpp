// include/strata/core/decode_stage.hpp - the recursive run/commit boundary of a decode stage.
//
// A decode stage runs one speculative window and commits the accepted prefix.  The local Verifier chain remains the
// implementation: this interface names only the boundary RPC must later cross, so unsupported batch paths fail
// explicitly instead of falling through to the ordinary single-sequence behavior.
#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
using PoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer);

class DecodeStage {
public:
    virtual ~DecodeStage() = default;

    /// One window over `tokens[0..T)` beginning at `pos0`.  `out[t]` is the pick after token t.
    virtual bool run_decode_window(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user,
                                   int32_t* out, std::string& err) = 0;

    /// Keep the first `n_keep` tokens of the window just run.
    virtual bool commit_decode_prefix(int n_keep, std::string& err) = 0;

    /// Does this stage implement the batch-window calls below?  A local Verifier does; a remote proxy does not.
    virtual bool supports_batch_decode() const { return false; }

    virtual bool run_batch_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user,
                                 int32_t* out, std::string& err) {
        (void) S;
        (void) tokens;
        (void) pos;
        (void) pool;
        (void) user;
        (void) out;
        err = "this decode stage does not support batch windows";
        return false;
    }

    virtual bool run_batch_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos,
                                     PoolMultiFn pool, void* user, int32_t* out, std::string& err) {
        (void) rows;
        (void) S;
        (void) tokens;
        (void) pos;
        (void) pool;
        (void) user;
        (void) out;
        err = "this decode stage does not support batch windows";
        return false;
    }

    virtual bool commit_batch_slots(std::string& err) {
        err = "this decode stage does not support batch windows";
        return false;
    }

    virtual bool commit_batch_slot_prefixes(const int* keep, std::string& err) {
        (void) keep;
        err = "this decode stage does not support batch windows";
        return false;
    }
};

}  // namespace strata::core
