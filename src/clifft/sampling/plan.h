#pragma once

// Private semantic plan for the symbolic-coordinate sampling path.
//
// This representation deliberately does not reuse the legacy SVM Instruction.
// It describes sampling semantics; CPU layout, SIMD selection, dispatch, and
// device lowering belong to later execution-specific layers.
// Executors must lower it to a packed, allocation-free runtime plan rather
// than traversing these variants and vectors in per-shot hot loops.
//
// The symbolic-expression and stabilizer-coordinate action split is an
// independent Clifft implementation informed by the published SymFT design.

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace clifft::sampling {

enum class SymbolId : uint32_t {};
enum class RecordSlot : uint32_t {};
enum class NoiseSiteId : uint32_t {};
enum class InstrumentSiteId : uint32_t {};

enum class SymbolKind : uint8_t {
    Presampled,
    Derived,
    Branch,
};

// An XOR (parity) expression over plan symbols, possibly inverted. Terms are
// stored in strict ascending order; XOR construction removes duplicates.
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

// The Hermitian Pauli body for one action, expressed only in the active
// coordinates visible immediately before that action. The unsigned operator is
// i^popcount(x & z) X^x Z^z; an action's AffineBool carries its possibly
// shot-dependent sign. Dense active width is below kDenseActiveWidthLimit, so
// each mask fits in one word even when the physical circuit has many qubits.
struct ActivePauli {
    uint64_t x = 0;
    uint64_t z = 0;

    [[nodiscard]] bool is_identity() const;
};

// Applies exp(-i * pi * half_turns * P / 2) to the current active state.
struct RotateActivePauli {
    ActivePauli pauli;
    double half_turns = 0.0;
    // True means the Pauli enters the rotation with a minus sign.
    AffineBool sign;
};

// Promotes one dormant coordinate and applies the rotation that introduced it.
struct PromoteDormantRotation {
    uint32_t dormant_pivot = 0;
    double half_turns = 0.0;
    // True means the promoted Pauli enters with a minus sign.
    AffineBool sign;
};

// Samples one active Pauli branch and removes the selected active pivot. The
// branch symbol is the effective eigenvalue label used by state updates; the
// outcome expression is the user-visible record bit after symbolic signs.
struct MeasureActivePauli {
    ActivePauli pauli;
    uint32_t active_pivot = 0;
    SymbolId branch{};
    AffineBool outcome;
    RecordSlot record{};
};

// Introduces an unbiased effective eigenvalue label while replacing a dormant
// stabilizer. The outcome expression is the separately recorded bit, and the
// active vector is untouched.
struct MeasureDormantRandom {
    uint32_t dormant_pivot = 0;
    SymbolId branch{};
    AffineBool outcome;
    RecordSlot record{};
};

// Records a deterministic classical expression without touching the state.
struct RecordClassical {
    AffineBool outcome;
    RecordSlot record{};
};

// Materializes an affine expression as a reusable derived symbol.
struct DefineSymbol {
    SymbolId symbol{};
    AffineBool value;
};

// Divides ahead-of-time execution segments. A continuation must preserve the
// live coefficients, active-coordinate meaning and order, active width,
// symbol and record values, and RNG position established here. It need not
// reproduce an arbitrary prior plan prefix byte-for-byte. The boundary itself
// is width-neutral; any state transition belongs to a following action.
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

struct SymbolInfo {
    SymbolKind kind = SymbolKind::Presampled;

    // Presampled symbols have no defining action. Branch and
    // derived symbols name the action that makes them available.
    std::optional<uint32_t> defining_action;

    // Presampled symbols may identify the stable HIR noise site that supplies
    // them. Other presampled event kinds can add their own source identity
    // without changing expression semantics.
    std::optional<NoiseSiteId> noise_site;
};

struct SamplingPlan {
    uint32_t num_qubits = 0;

    // Active width is the new subsystem's name for legacy active_k. The
    // planner computes its maximum (legacy peak_rank) while emitting actions,
    // before runtime lowering uses it to preallocate coefficient storage.
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

[[nodiscard]] uint32_t predicted_dense_passes(const SamplingAction& action);

}  // namespace clifft::sampling
