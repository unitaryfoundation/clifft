#pragma once

// Flat private descriptors shared by HIP host lowering and device execution.
// SamplingPlan remains the semantic boundary; these types are not a public or
// cross-backend command format.

#include <cstdint>
#include <type_traits>

namespace clifft::sampling::hip::detail {

// ExecutablePlan::lower_action exhaustively visits SamplingAction. Its dependent
// static assertion makes ordinary CPU builds fail if a new alternative lacks
// explicit HIP lowering.
enum class ActionTag : uint8_t {
    RotateActivePauli,
    PromoteDormantRotation,
    MeasureActivePauli,
    MeasureDormantRandom,
    RecordClassical,
    DefineSymbol,
    ApplyReadoutNoise,
    WriteDetector,
    WriteObservable,
    WriteExpectationValue,
};

inline constexpr uint8_t kPostselected = 1U << 0;
inline constexpr uint8_t kAbsentActiveProjection = 1U << 1;

struct Expression {
    uint32_t term_begin = 0;
    uint32_t term_count = 0;
    uint8_t constant = 0;
    uint8_t reserved[3]{};
};

// Operands are interpreted by tag. Geometry such as the Pauli phase and
// pairing bit is prepared on the host instead of rediscovered by the kernel.
struct Action {
    ActionTag tag = ActionTag::RecordClassical;
    uint8_t flags = 0;
    int8_t phase_real = 1;
    int8_t phase_imag = 0;
    uint32_t active_before = 0;
    uint32_t expression = 0;
    uint32_t index0 = 0;
    uint32_t index1 = 0;
    uint32_t index2 = 0;
    uint64_t x = 0;
    uint64_t z = 0;
    uint64_t pair_stride = 0;
    double value0 = 0.0;
    double value1 = 0.0;
};

struct NoiseOutcome {
    uint32_t symbol = 0;
    uint32_t reserved = 0;
    double cumulative_probability = 0.0;
};

struct NoiseSite {
    uint32_t outcome_begin = 0;
    uint32_t outcome_count = 0;
    double execution_probability = 0.0;
};

struct ProgramView {
    const Action* actions = nullptr;
    const Expression* expressions = nullptr;
    const uint32_t* expression_terms = nullptr;
    const NoiseSite* noise_sites = nullptr;
    const NoiseOutcome* noise_outcomes = nullptr;
    uint32_t action_count = 0;
    uint32_t initial_active_width = 0;
    uint32_t peak_active_width = 0;
    uint32_t num_symbols = 0;
    uint32_t num_records = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;
    uint32_t noise_site_count = 0;
};

struct SeedRoot {
    uint64_t words[4]{};
};

static_assert(std::is_trivially_copyable_v<Expression>);
static_assert(std::is_trivially_copyable_v<Action>);
static_assert(std::is_trivially_copyable_v<NoiseOutcome>);
static_assert(std::is_trivially_copyable_v<NoiseSite>);
static_assert(std::is_trivially_copyable_v<ProgramView>);
static_assert(std::is_trivially_copyable_v<SeedRoot>);

}  // namespace clifft::sampling::hip::detail
