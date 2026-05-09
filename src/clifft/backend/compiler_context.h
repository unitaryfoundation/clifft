#pragma once

// CompilerContext: Internal state for the AOT Back-End compiler.
//
// Exposed in a header (rather than anonymous namespace) so that
// test_backend.cc can directly instantiate contexts, feed them
// PauliStrings, and verify the resulting V_cum and bytecode.

#include "clifft/backend/backend.h"
#include "clifft/backend/source_map.h"
#include "clifft/frontend/hir.h"

#include "stim.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {
namespace internal {

// =============================================================================
// VirtualRegisterManager
// =============================================================================
//
// Tracks the Active/Dormant partition of virtual axes using a simple
// contiguous split: axes 0..k-1 are Active, axes k..n-1 are Dormant.
//
// The compiler does NOT maintain a software dictionary mapping virtual
// qubits to array axes. Instead, V_cum itself tracks the permutation
// via explicit SWAP instructions emitted by the compiler. This means
// the axis indices in opcodes are always literal -- no translation needed.
//
// All n axes start Dormant (k=0). activate() increments k, deactivate()
// decrements k. The caller must SWAP a qubit to the boundary before
// changing its partition.

class VirtualRegisterManager {
  public:
    explicit VirtualRegisterManager(uint32_t num_qubits) : num_qubits_(num_qubits), active_k_(0) {}

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    [[nodiscard]] uint32_t active_k() const { return active_k_; }

    [[nodiscard]] bool is_active(uint16_t axis) const { return axis < active_k_; }
    [[nodiscard]] bool is_dormant(uint16_t axis) const { return axis >= active_k_; }

    // Promote the next dormant axis (k) to active. The caller must have
    // already SWAPped the target qubit to axis k before calling this.
    void activate() {
        assert(active_k_ < num_qubits_ && "All qubits already active");
        ++active_k_;
        if (active_k_ > peak_k_) {
            peak_k_ = active_k_;
        }
    }

    // Demote axis k-1 from active to dormant. The caller must have
    // already SWAPped the target qubit to axis k-1 before calling this.
    void deactivate() {
        assert(active_k_ > 0 && "No active qubits to deactivate");
        --active_k_;
    }

    [[nodiscard]] uint32_t peak_k() const { return peak_k_; }

  private:
    uint32_t num_qubits_;
    uint32_t active_k_;
    uint32_t peak_k_ = 0;
};

/// Validate that the peak active rank fits the SVM's `1ULL << k` shift.
/// k >= 63 produces undefined behavior in the bytecode interpreter, so
/// lower() must reject before shipping the CompiledModule. Pulled out
/// of lower() so the guard can be unit-tested without spinning up a
/// 65k-qubit Stim tableau -- and so the uint32_t input type is fixed
/// at the boundary, preventing a regression where a uint16_t narrowing
/// at the call site silently wraps a 65,536-axis peak to 0.
inline void validate_peak_rank(uint32_t peak) {
    if (peak >= 63) {
        throw std::runtime_error("peak active rank (" + std::to_string(peak) +
                                 ") >= 63: would cause undefined behavior in SVM 1ULL << k shifts");
    }
}

// =============================================================================
// CompilerContext
// =============================================================================
//
// Bundle of mutable state threaded through the Back-End compilation.
// Holds the virtual frame, the register manager, and the output bytecode.

// Gate types for the pending gate queue. These identify the Clifford gate
// applied to v_cum, independent of the bytecode opcode (FRAME vs ARRAY).
enum class PendingGateType : uint8_t {
    CNOT,
    CZ,
    S,
    H,
    SWAP,
};

struct PendingGate {
    PendingGateType type;
    uint16_t axis_1;
    uint16_t axis_2;  // unused for S and H
};

class VirtualFrame {
  public:
    // flush_threshold: maximum pending gate queue depth before flushing into
    // the materialized tableau. Balances O(n^2) flush cost vs O(queue_len)
    // per-operation propagation. Empirically, 64-128 is near-optimal for
    // 454-qubit circuits; performance degrades sharply above 256.
    VirtualFrame(uint32_t num_qubits, size_t flush_threshold)
        : materialized_(num_qubits), flush_threshold_(flush_threshold) {
        pending_gates_.reserve(flush_threshold);
    }

    [[nodiscard]] bool has_pending() const { return !pending_gates_.empty(); }
    [[nodiscard]] size_t pending_size() const { return pending_gates_.size(); }

    void append_gate(PendingGate gate) { pending_gates_.push_back(gate); }

    // Flush all pending gates into the materialized tableau in a single
    // transposed scope. Call before operations that require direct tableau
    // access such as inverse computation.
    void flush() {
        if (pending_gates_.empty())
            return;
        {
            stim::TableauTransposedRaii<kStimWidth> trans(materialized_);
            for (const auto& g : pending_gates_) {
                switch (g.type) {
                    case PendingGateType::CNOT:
                        trans.append_ZCX(g.axis_1, g.axis_2);
                        break;
                    case PendingGateType::CZ:
                        trans.append_ZCZ(g.axis_1, g.axis_2);
                        break;
                    case PendingGateType::S:
                        trans.append_S(g.axis_1);
                        break;
                    case PendingGateType::H:
                        trans.append_H_XZ(g.axis_1);
                        break;
                    case PendingGateType::SWAP:
                        trans.append_ZCX(g.axis_1, g.axis_2);
                        trans.append_ZCX(g.axis_2, g.axis_1);
                        trans.append_ZCX(g.axis_1, g.axis_2);
                        break;
                }
            }
        }
        pending_gates_.clear();
    }

    [[nodiscard]] stim::PauliString<kStimWidth> map_pauli(MaskView destab_mask, MaskView stab_mask,
                                                          bool sign, uint32_t n) {
        // Note: do not eager-flush here. map_pauli is called once per
        // mask-carrying HIR op (T_GATE, MEASURE, etc.) and the queue
        // between consecutive calls is short (the gates emitted by the
        // previous localize_pauli). Replaying a short queue inline beats
        // paying a fixed tableau-transpose-scope per op; eager flushing
        // here regresses T-heavy circuits like QV-20 by ~40%.
        maybe_flush();

        stim::PauliString<kStimWidth> p(n);
        uint32_t words = (n + 63) / 64;
        for (uint32_t w = 0; w < words && w < destab_mask.num_words(); ++w) {
            p.xs.u64[w] = destab_mask.words[w];
            p.zs.u64[w] = stab_mask.words[w];
        }
        p.sign = sign;

        stim::PauliString<kStimWidth> mapped = materialized_(p);
        if (pending_gates_.empty())
            return mapped;

        // Apply pending gates directly on the stim::PauliString's u64
        // storage as runtime-width MaskViews. Restrict to the live `words`
        // prefix; stim's simd_bits may pad above that.
        bool s_out = mapped.sign;
        MutableMaskView x_view{std::span<uint64_t>(mapped.xs.u64, words)};
        MutableMaskView z_view{std::span<uint64_t>(mapped.zs.u64, words)};
        apply_pending_to_pauli(x_view, z_view, s_out);
        mapped.sign = s_out;
        return mapped;
    }

    /// Map a noise channel's (X, Z) masks through the current virtual frame
    /// and write the result into `out_x` and `out_z`. Sign is unused for
    /// noise channels.
    void map_noise_channel(MaskView in_x, MaskView in_z, MutableMaskView out_x,
                           MutableMaskView out_z, uint32_t n) {
        // Flush eagerly here. A noise op typically maps several channels
        // back to back (e.g. DEPOLARIZE2 has 15) and the lower() loop
        // visits thousands of noise sites on surface QEC circuits. Each
        // channel mapping that runs against a non-empty pending queue pays
        // O(queue_size * n_words) replay cost; flushing once amortizes
        // that into one tableau update and lets all subsequent channel
        // mappings see an empty queue (cheap early-out).
        flush();

        out_x.zero_out();
        out_z.zero_out();
        const uint32_t words = (n + 63) / 64;

        // Take the tableau row by view, not by value: materialized_.xs[q]
        // returns a PauliStringRef (non-owning), and binding it to
        // `const PauliString<W>&` would copy-construct a fresh PauliString
        // per call. On QEC circuits this lambda is invoked ~10^5 times per
        // compile, so the copies were the dominant heap allocator load.
        auto xor_row = [&](stim::PauliStringRef<kStimWidth> row) {
            for (uint32_t w = 0; w < words && w < out_x.num_words(); ++w) {
                out_x.words[w] ^= row.xs.u64[w];
                out_z.words[w] ^= row.zs.u64[w];
            }
        };

        // Iterate set X-bits via a working copy. Reuses member-owned scratch
        // storage so QEC circuits that map thousands of noise channels per
        // compile don't pay a heap round-trip per channel.
        noise_x_scratch_.assign(in_x.words.begin(), in_x.words.end());
        MutableMaskView x_iter{std::span<uint64_t>(noise_x_scratch_)};
        while (!x_iter.is_zero()) {
            uint32_t q = x_iter.lowest_bit();
            xor_row(materialized_.xs[q]);
            x_iter.clear_lowest_bit();
        }
        noise_z_scratch_.assign(in_z.words.begin(), in_z.words.end());
        MutableMaskView z_iter{std::span<uint64_t>(noise_z_scratch_)};
        while (!z_iter.is_zero()) {
            uint32_t q = z_iter.lowest_bit();
            xor_row(materialized_.zs[q]);
            z_iter.clear_lowest_bit();
        }

        if (!pending_gates_.empty()) {
            bool dummy_sign = false;
            apply_pending_to_pauli(out_x, out_z, dummy_sign);
        }
    }

    [[nodiscard]] const stim::Tableau<kStimWidth>& materialized_tableau() const {
        assert(pending_gates_.empty() && "materialized_tableau requires a flushed virtual frame");
        return materialized_;
    }

    [[nodiscard]] stim::Tableau<kStimWidth>& mutable_materialized_tableau() {
        flush();
        return materialized_;
    }

  private:
    static void apply_gate_to_pauli(const PendingGate& g, MutableMaskView x, MutableMaskView z,
                                    bool& sign) {
        bool xa = x.bit_get(g.axis_1);
        bool za = z.bit_get(g.axis_1);

        switch (g.type) {
            case PendingGateType::CNOT: {
                bool xb = x.bit_get(g.axis_2);
                bool zb = z.bit_get(g.axis_2);
                if (xa && zb && !(xb ^ za))
                    sign ^= true;
                if (xa)
                    x.bit_xor(g.axis_2);
                if (zb)
                    z.bit_xor(g.axis_1);
                break;
            }
            case PendingGateType::CZ: {
                bool xb = x.bit_get(g.axis_2);
                bool zb = z.bit_get(g.axis_2);
                if (xa && xb && (za ^ zb))
                    sign ^= true;
                if (xb)
                    z.bit_xor(g.axis_1);
                if (xa)
                    z.bit_xor(g.axis_2);
                break;
            }
            case PendingGateType::S:
                if (xa && za)
                    sign ^= true;
                if (xa)
                    z.bit_xor(g.axis_1);
                break;
            case PendingGateType::H:
                if (xa && za)
                    sign ^= true;
                x.bit_set(g.axis_1, za);
                z.bit_set(g.axis_1, xa);
                break;
            case PendingGateType::SWAP:
                x.bit_swap(g.axis_1, g.axis_2);
                z.bit_swap(g.axis_1, g.axis_2);
                break;
        }
    }

    void apply_pending_to_pauli(MutableMaskView x, MutableMaskView z, bool& sign) const {
        for (const auto& g : pending_gates_) {
            apply_gate_to_pauli(g, x, z, sign);
        }
    }

    void maybe_flush() {
        if (pending_gates_.size() >= flush_threshold_) {
            flush();
        }
    }

    stim::Tableau<kStimWidth> materialized_;
    std::vector<PendingGate> pending_gates_;
    size_t flush_threshold_;

    // Per-instance scratch buffers reused across calls to map_noise_channel.
    // QEC circuits with thousands of noise channels per compile would
    // otherwise allocate and free two std::vector<uint64_t> per call.
    std::vector<uint64_t> noise_x_scratch_;
    std::vector<uint64_t> noise_z_scratch_;
};

struct CompilerContext {
    VirtualFrame virtual_frame;
    VirtualRegisterManager reg_manager;
    std::vector<Instruction> bytecode;
    ConstantPool constant_pool;
    double noise_hazards_accum = 0.0;

    // Playground telemetry (populated by lower(), parallel to bytecode)
    SourceMap source_map;
    std::vector<uint32_t> emit_k_history;  // per-instruction k captured at emit time

    // Scratch buffers reused across calls to localize_pauli. The greedy
    // CNOT/CZ loop walks per-bit working copies of the X and Z masks; on
    // QEC circuits localize_pauli runs hundreds of times per compile.
    std::vector<uint64_t> localize_x_bits;
    std::vector<uint64_t> localize_z_bits;
    std::vector<uint64_t> localize_to_clear;

    explicit CompilerContext(uint32_t num_qubits, size_t flush_threshold = 128)
        : virtual_frame(num_qubits, flush_threshold), reg_manager(num_qubits) {}

    // Emit an instruction and record the current active dimension.
    void emit(const Instruction& instr) {
        bytecode.push_back(instr);
        emit_k_history.push_back(reg_manager.active_k());
    }
};

// =============================================================================
// Result of localize_pauli: identifies what the Pauli was localized to.
// =============================================================================

enum class LocalizedBasis : uint8_t {
    Z_BASIS,  // Localized to Z_v
    X_BASIS,  // Localized to X_v
};

struct LocalizationResult {
    uint16_t pivot;        // The virtual qubit the Pauli was localized onto
    LocalizedBasis basis;  // Whether it ended up as X_v or Z_v
    bool sign;             // Accumulated sign (true = negative)
};

// =============================================================================
// Core Localization Algorithm
// =============================================================================

/// Localize an arbitrary multi-qubit Pauli string onto a single virtual qubit.
///
/// Given a PauliString P, computes a sequence V of virtual Clifford gates
/// such that V P V^dag = (+/-)P_v where P_v in {X_v, Z_v}.
///
/// Side effects:
///   - Appends virtual gates to ctx.virtual_frame
///   - Emits corresponding VM opcodes to ctx.bytecode
///   - Does NOT modify ctx.reg_manager (activation is the caller's job)
///
/// The input PauliString must be non-identity (at least one qubit set).
[[nodiscard]] LocalizationResult localize_pauli(CompilerContext& ctx,
                                                const stim::PauliString<kStimWidth>& pauli);

}  // namespace internal
}  // namespace clifft
