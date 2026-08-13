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
// An active coordinate is represented explicitly in the dense 2^k coefficient
// state. A dormant coordinate remains stabilizer-only until an operation makes
// it active. Affine symbols are per-shot Boolean values that carry stochastic
// dependencies without expanding the coefficient state.
//
// The symbolic-expression and stabilizer-coordinate split is informed by:
// SymFT: Universal Fault-Tolerant Quantum Circuit Simulation via Symbolic
// Clifford--Pauli Frames and Stabilizer Coordinates, arXiv:2607.28600.

#include "clifft/util/numeric.h"
#include "clifft/util/stim_mask.h"

#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace clifft::sampling {

// A symbol is a plan-local Boolean variable representing a sampled circuit
// outcome or a parity derived from earlier symbols.
enum class SymbolKind : uint8_t {
    Unused,      // Stable operation-order slot that this compiled plan does not need.
    Presampled,  // Available before the action stream, such as sampled noise.
    Derived,     // Computed as a parity of previously available symbols.
    Branch,      // Sampled while applying a measurement action.
    Readout,     // Sampled from a record-dependent readout channel.
    Instrument,  // Records an in-line computational instrument destination flip.
};

// Strongly typed plan-local indices prevent unrelated storage from being
// indexed interchangeably.
enum class SymbolId : uint32_t {};
enum class RecordSlot : uint32_t {};
enum class NoiseSiteId : uint32_t {};
enum class InstrumentSiteId : uint32_t {};
enum class DetectorSlot : uint32_t {};
enum class ObservableSlot : uint32_t {};
enum class ExpValSlot : uint32_t {};

[[nodiscard]] constexpr uint32_t index(SymbolId id) noexcept {
    return static_cast<uint32_t>(id);
}
[[nodiscard]] constexpr uint32_t index(RecordSlot slot) noexcept {
    return static_cast<uint32_t>(slot);
}
[[nodiscard]] constexpr uint32_t index(NoiseSiteId site) noexcept {
    return static_cast<uint32_t>(site);
}
[[nodiscard]] constexpr uint32_t index(InstrumentSiteId site) noexcept {
    return static_cast<uint32_t>(site);
}
[[nodiscard]] constexpr uint32_t index(DetectorSlot slot) noexcept {
    return static_cast<uint32_t>(slot);
}
[[nodiscard]] constexpr uint32_t index(ObservableSlot slot) noexcept {
    return static_cast<uint32_t>(slot);
}
[[nodiscard]] constexpr uint32_t index(ExpValSlot slot) noexcept {
    return static_cast<uint32_t>(slot);
}

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
    // Planner internals can preserve storage when they already produced the
    // canonical ordering. Debug builds validate this precondition.
    [[nodiscard]] static AffineBool from_canonical_terms(bool constant,
                                                         std::vector<SymbolId> terms);
    [[nodiscard]] bool constant() const { return constant_; }
    [[nodiscard]] const std::vector<SymbolId>& terms() const { return terms_; }
    [[nodiscard]] bool is_canonical() const;

    AffineBool& operator^=(const AffineBool& other);
    AffineBool& operator^=(AffineBool&& other);
    AffineBool& operator^=(bool value);

    friend bool operator==(const AffineBool&, const AffineBool&) = default;

  private:
    struct CanonicalTermsTag {};

    AffineBool(bool constant, std::vector<SymbolId> terms, CanonicalTermsTag);

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
    // When true for a shot, negate the promoted generator, equivalently negating half_turns.
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

// Samples whether a completed measurement record flips. The probability may
// depend on the record before the flip, so unlike Pauli noise this symbol is
// defined at its circuit position instead of before the action stream.
struct ApplyReadoutNoise {
    SymbolId flip{};
    AffineBool source;
    RecordSlot record{};
    double prob_zero_to_one = 0.0;
    double prob_one_to_zero = 0.0;
};

// Writes a detector parity after the planner has XORed its expected reference
// parity into the expression. A nonzero postselected detector rejects the shot
// immediately; later actions and outputs are irrelevant for it.
struct WriteDetector {
    AffineBool outcome;
    DetectorSlot detector{};
    bool postselected = false;
};

// Writes one fully accumulated logical observable after the planner has XORed
// its expected reference parity into the expression.
struct WriteObservable {
    AffineBool outcome;
    ObservableSlot observable{};
};

// Writes a non-destructive Pauli expectation probe. The planner retains only
// the coefficient-kernel input, not the full transformed observable: an absent
// active projection means dormant X or Y support proved the result is zero.
struct WriteExpectationValue {
    std::optional<ActivePauli> active_projection;
    AffineBool sign;
    ExpValSlot exp_val{};
};

// Instruments share one semantic action because their execution forms differ
// only in how the source observable enters the dense state. The planner fixes
// that choice; runtime never performs localization or topology discovery.
enum class InstrumentMode : uint8_t {
    Classical,    // The source is the Boolean sign; no coefficient work is needed.
    Active,       // The source Pauli is already supported on active coordinates.
    Activate,     // Add one |0> coordinate before applying the source Pauli.
    DormantTrap,  // Sample a dormant-random source and let a continuation collapse it.
};

struct ApplyInstrument {
    InstrumentSiteId site{};
    InstrumentMode mode = InstrumentMode::Classical;
    // The source is identity for Classical and DormantTrap. Activate uses the
    // post-activation width, while Active uses the unchanged width.
    ActivePauli source;
    AffineBool sign;
    // Present when a computational destination can continue in-line. The
    // symbol is true exactly when that destination differs from its source.
    std::optional<SymbolId> destination_flip;
};

// Divides ahead-of-time execution segments. A continuation must preserve the
// live coefficients, active-coordinate meaning and order, active width,
// symbol and record values, and RNG position established here. It need not
// reproduce an arbitrary prior plan prefix byte-for-byte. The instrument
// action precedes this marker, so a trap resumes at the marker and samples only
// the replacement suffix's presampled-noise segment.
struct InstrumentBoundary {
    InstrumentSiteId site{};
    uint32_t next_noise_site = 0;
    uint32_t symbol_prefix_size = 0;
};

using SamplingAction = std::variant<RotateActivePauli, PromoteDormantRotation, MeasureActivePauli,
                                    MeasureDormantRandom, RecordClassical, DefineSymbol,
                                    ApplyReadoutNoise, WriteDetector, WriteObservable,
                                    WriteExpectationValue, ApplyInstrument, InstrumentBoundary>;

struct PlannedAction {
    // Dense active-coordinate widths immediately before and after this action.
    // Validation checks that adjacent actions form one continuous width chain.
    uint32_t active_before = 0;
    uint32_t active_after = 0;
    SamplingAction action;
};

// Describes how one symbol is populated. SamplingPlan::validate checks each
// kind's legal fields and its agreement with the referenced defining action.
struct SymbolInfo {
    SymbolKind kind = SymbolKind::Unused;

    // Index of the action that assigns this symbol. Required for Derived,
    // Branch, Readout, and Instrument; absent for Presampled and Unused.
    std::optional<uint32_t> defining_action;

    // For a Presampled noise symbol, this identifies its stable HIR noise site.
    // Nullopt means the presampled event has no noise-site identity. All other
    // symbol kinds must use nullopt.
    std::optional<NoiseSiteId> noise_site;
};

// One nonidentity outcome of a mutually exclusive Pauli-noise site. The
// identity outcome has the remaining probability and no symbol.
struct PresampledNoiseOutcome {
    SymbolId symbol{};
    double probability = 0.0;
};

struct PresampledNoiseSite {
    NoiseSiteId site{};
    // Exact semantic probability copied from the HIR noise site. Execution
    // still uses the ordered outcome probabilities for channel selection.
    double total_probability = 0.0;
    std::vector<PresampledNoiseOutcome> outcomes;
};

// Plan-owned copy of one HIR instrument distribution. Source and destination
// indices use 0 for g and 1 for e. Computational entries are unconditional;
// their row sum is at most p_fire[source].
struct InstrumentDistribution {
    InstrumentSiteId site{};
    std::array<double, 2> p_fire{};
    std::array<std::array<double, 2>, 2> p_computational_dest{};
};

struct SamplingPlan {
    uint32_t num_qubits = 0;

    // Active width is the number of stabilizer coordinates represented in the
    // dense coefficient state, which contains 2^active_width entries. The
    // planner computes its maximum while emitting actions so runtime lowering
    // can preallocate storage.
    uint32_t initial_active_width = 0;
    uint32_t max_active_width = 0;
    uint32_t num_visible_records = 0;
    uint32_t num_hidden_records = 0;
    uint32_t num_noise_sites = 0;
    uint32_t num_instrument_sites = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;
    bool has_postselection = false;
    std::complex<double> global_weight = {1.0, 0.0};

    // Present only for pure-state plans eligible for exact final-state
    // queries. It maps the final stabilizer coordinates used by the action
    // stream into physical qubits and is never read by ordinary dispatch.
    std::optional<stim::Tableau<kStimWidth>> final_tableau;

    std::vector<SymbolInfo> symbols;
    std::vector<PresampledNoiseSite> presampled_noise_sites;
    std::vector<InstrumentDistribution> instrument_distributions;
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
