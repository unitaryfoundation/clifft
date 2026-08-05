#pragma once

// SamplingPlan is a private semantic IR produced after a circuit is compiled
// to HIR and optimized. It captures the action sequence needed to sample the
// circuit, including stochastic events as Boolean symbols, changes to the
// active state, record writes, and instrument boundaries. This separates
// compile-once planning from values determined for each shot.
//
// The plan is executor-independent. CPU layout, SIMD selection, dispatch, and
// device lowering belong to execution-specific layers, which should lower it
// to packed, preallocated storage before entering per-shot hot loops.
//
// The symbolic-expression and stabilizer-coordinate split is informed by:
// SymFT: Universal Fault-Tolerant Quantum Circuit Simulation via Symbolic
// Clifford--Pauli Frames and Stabilizer Coordinates, arXiv:2607.28600.

#include "clifft/util/numeric.h"

#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace clifft::sampling {

// A symbol is a plan-local Boolean value assigned once per shot. It represents
// a presampled stochastic event, a sampled measurement branch, or a named
// parity derived from earlier symbols.
enum class SymbolId : uint32_t {};
enum class RecordSlot : uint32_t {};
enum class NoiseSiteId : uint32_t {};
enum class InstrumentSiteId : uint32_t {};

enum class SymbolKind : uint8_t {
    Presampled,  // Available before the action stream, such as sampled noise.
    Derived,     // Computed as a parity of previously available symbols.
    Branch,      // Sampled while applying a measurement action.
};

// A parity expression over plan symbols, optionally XORed with true. For
// example, s0 ^ s2 ^ true can track how sampled noise and a measurement branch
// combine into a later sign or record bit. Terms are stored in strict ascending
// order; XOR construction removes duplicates.
class AffineBool {
  public:
    AffineBool() = default;
    explicit AffineBool(bool constant);
    AffineBool(bool constant, std::vector<SymbolId> terms);

    [[nodiscard]] static AffineBool symbol(SymbolId id);
    [[nodiscard]] bool constant() const { return constant_; }
    [[nodiscard]] const std::vector<SymbolId>& terms() const { return terms_; }
    [[nodiscard]] bool is_canonical() const;

    AffineBool& operator^=(const AffineBool& other);
    AffineBool& operator^=(bool value);

    friend bool operator==(const AffineBool&, const AffineBool&) = default;

  private:
    bool constant_ = false;
    std::vector<SymbolId> terms_;
};

[[nodiscard]] AffineBool operator^(AffineBool left, const AffineBool& right);
[[nodiscard]] AffineBool operator^(AffineBool left, bool right);
[[nodiscard]] AffineBool operator^(bool left, AffineBool right);

// Operations requiring coefficient-state work are expressed over active
// stabilizer coordinates. An active coordinate is represented explicitly in
// the dense 2^k coefficient state. ActivePauli stores the unsigned Hermitian
// operator i^popcount(x & z) X^x Z^z. Each action carries a separate AffineBool
// that can change its sign for each shot.
//
// Exponential storage requirements currently restrict dense active widths to
// below kDenseActiveWidthLimit, which fits in one uint64_t. This assertion
// requires reconsidering the mask representation if that limit ever grows.
static_assert(kDenseActiveWidthLimit <= std::numeric_limits<uint64_t>::digits);

struct ActivePauli {
    uint64_t x = 0;
    uint64_t z = 0;

    [[nodiscard]] bool is_identity() const;
};

// Applies exp(-i * pi * half_turns * P / 2) to the current active state.
struct RotateActivePauli {
    ActivePauli pauli;
    double half_turns = 0.0;
    // When true for a shot, use -P, equivalently negating half_turns.
    AffineBool sign;
};

// Promotes the next active coordinate and applies the rotation that introduced
// it. The planner has already rewritten later operations into the new basis.
struct PromoteDormantRotation {
    double half_turns = 0.0;
    // When true for a shot, negate the promoted generator and half_turns.
    AffineBool sign;
};

// Measuring a Pauli supported on the active coordinates requires sampling and
// collapsing the coefficient state. The selected pivot is then removed, so the
// active width decreases by one. The branch symbol holds the sampled eigenvalue
// bit; outcome combines it with symbolic corrections to produce the record bit.
struct MeasureActivePauli {
    ActivePauli pauli;
    uint32_t active_pivot = 0;
    SymbolId branch{};
    AffineBool outcome;
    RecordSlot record{};
};

// A random Pauli measurement supported outside the active coefficient state
// samples an unbiased branch and replaces the selected dormant stabilizer. It
// leaves the active state unchanged. The outcome combines the branch with
// symbolic corrections to produce the record bit.
struct MeasureDormantRandom {
    uint32_t dormant_pivot = 0;
    SymbolId branch{};
    AffineBool outcome;
    RecordSlot record{};
};

// Writes a deterministic expression to visible or hidden circuit record
// history without touching the state. Use DefineSymbol instead when an
// internal parity needs a reusable name but no record slot.
struct RecordClassical {
    AffineBool outcome;
    RecordSlot record{};
};

// Names an affine expression for reuse by later actions without adding a
// circuit record entry.
struct DefineSymbol {
    SymbolId symbol{};
    AffineBool value;
};

// Divides ahead-of-time execution segments. A continuation must preserve the
// live coefficients, active-coordinate meaning and order, active width,
// symbol and record values, and RNG position established here. It need not
// reproduce an arbitrary prior plan prefix byte-for-byte. The marker does not
// itself change the active state; any transition belongs to a following action.
struct InstrumentBoundary {
    InstrumentSiteId site{};
};

using SamplingAction =
    std::variant<RotateActivePauli, PromoteDormantRotation, MeasureActivePauli,
                 MeasureDormantRandom, RecordClassical, DefineSymbol, InstrumentBoundary>;

struct PlannedAction {
    uint32_t active_before = 0;
    uint32_t active_after = 0;
    SamplingAction action;
};

// SamplingPlan::validate enforces the legal field combinations and verifies
// that definition metadata agrees with the referenced action.
struct SymbolInfo {
    SymbolKind kind = SymbolKind::Presampled;

    // This is nullopt for Presampled. For Branch and Derived it identifies the
    // unique action that assigns the symbol.
    std::optional<uint32_t> defining_action;

    // For a Presampled noise symbol, this identifies its stable HIR noise site.
    // Nullopt means the presampled event has no noise-site identity. Branch and
    // Derived symbols must always use nullopt.
    std::optional<NoiseSiteId> noise_site;
};

struct SamplingPlan {
    uint32_t num_qubits = 0;

    // Active width is the number of stabilizer coordinates represented in the
    // dense coefficient state, which contains 2^active_width entries. It is the
    // descriptive name for legacy active_k. The planner computes its maximum
    // while emitting actions so runtime lowering can preallocate storage.
    uint32_t initial_active_width = 0;
    uint32_t max_active_width = 0;
    uint32_t num_visible_records = 0;
    uint32_t num_hidden_records = 0;
    uint32_t num_noise_sites = 0;
    uint32_t num_instrument_sites = 0;
    std::complex<double> global_weight = {1.0, 0.0};

    std::vector<SymbolInfo> symbols;
    std::vector<PlannedAction> actions;

    // Throws std::invalid_argument when the plan is structurally inconsistent.
    // This checks active-width transitions, masks, stable ids, record slots,
    // symbol definitions, and assignment-before-use.
    void validate() const;

    // Deterministic, executor-independent inspection for tests and diagnostics.
    [[nodiscard]] std::string inspect() const;
};

// Estimates full coefficient-state traversals for a direct, unfused lowering.
// This is a planning diagnostic; an executor may fuse or specialize actions.
[[nodiscard]] uint32_t predicted_dense_passes(const SamplingAction& action);

}  // namespace clifft::sampling
