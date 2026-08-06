#pragma once

// Executes a backend-neutral SamplingPlan on the CPU without runtime topology
// work. ExecutablePlan prepares direct-Pauli kernel descriptors and packed
// affine expressions once; Executor then evaluates per-shot symbols, evolves
// the active-coordinate coefficient state, samples measurements, and writes
// records using only preallocated storage.

#include "clifft/sampling/kernels.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/state.h"
#include "clifft/util/xoshiro.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace clifft::sampling {

// Zero and one identify the selected Pauli eigenvalue branch before affine
// sign corrections turn that branch into a physical measurement record.
enum class MeasurementBranchKind : uint8_t {
    Random,
    DeterministicZero,
    DeterministicOne,
};

struct MeasurementBranchClassification {
    MeasurementBranchKind kind = MeasurementBranchKind::Random;
    bool clamped_dust = false;
};

struct ReplayResult {
    bool reachable = true;
    // Meaningful only when reachable is true.
    double log_probability = 0.0;
};

[[nodiscard]] MeasurementBranchClassification classify_measurement_branch(
    MeasurementProbabilities probabilities) noexcept;

// Owns the CPU lowering of one validated SamplingPlan. Direct-Pauli kernel
// descriptors and affine-expression term ranges are prepared once here so a
// shot only reads fixed storage.
class ExecutablePlan {
  public:
    explicit ExecutablePlan(const SamplingPlan& plan);

    [[nodiscard]] uint32_t num_visible_records() const { return num_visible_records_; }
    [[nodiscard]] uint32_t num_hidden_records() const { return num_hidden_records_; }
    [[nodiscard]] uint32_t num_symbols() const { return num_symbols_; }
    [[nodiscard]] uint32_t num_presampled_symbols() const {
        return static_cast<uint32_t>(presampled_symbols_.size());
    }
    [[nodiscard]] size_t num_actions() const { return actions_.size(); }

  private:
    friend class Executor;

    struct PreparedExpression {
        uint32_t term_begin = 0;
        uint32_t term_count = 0;
        bool constant = false;
    };

    struct ExecuteRotation {
        PreparedRotation rotation;
        PreparedExpression sign;
    };

    struct ExecutePromotion {
        PreparedPromotion promotion;
        PreparedExpression sign;
    };

    struct ExecuteActiveMeasurement {
        PreparedMeasurement measurement;
        // The physical record is branch XOR this expression. Removing the
        // newly defined branch at preparation time lets replay invert that
        // relation without runtime dependency discovery.
        PreparedExpression correction;
        uint32_t branch = 0;
        uint32_t record = 0;
    };

    struct ExecuteDormantMeasurement {
        PreparedExpression correction;
        uint32_t branch = 0;
        uint32_t record = 0;
    };

    struct ExecuteClassicalRecord {
        PreparedExpression outcome;
        uint32_t record = 0;
    };

    struct ExecuteSymbolDefinition {
        PreparedExpression value;
        uint32_t symbol = 0;
    };

    using Action =
        std::variant<ExecuteRotation, ExecutePromotion, ExecuteActiveMeasurement,
                     ExecuteDormantMeasurement, ExecuteClassicalRecord, ExecuteSymbolDefinition>;

    PreparedExpression prepare_expression(const AffineBool& expression);
    PreparedExpression prepare_measurement_correction(const AffineBool& outcome, uint32_t branch);

    uint32_t initial_active_width_ = 0;
    uint32_t max_active_width_ = 0;
    uint32_t num_visible_records_ = 0;
    uint32_t num_hidden_records_ = 0;
    uint32_t num_symbols_ = 0;
    std::complex<double> global_weight_ = {1.0, 0.0};
    std::vector<uint32_t> expression_terms_;
    // Maps each dense presampled input position to its plan-local SymbolId.
    // The constructor records those ids in ascending order.
    std::vector<uint32_t> presampled_symbols_;
    std::vector<Action> actions_;
};

// Holds the dense active-coordinate coefficients and global scalar, Boolean
// symbols carrying stochastic frame dependencies, visible and hidden records,
// measurement RNG, and numerical-dust telemetry for repeated shots. The plan
// must outlive the executor. Construction allocates all storage; run_shot only
// resets and overwrites it.
class Executor {
  public:
    explicit Executor(const ExecutablePlan& plan, uint64_t seed = 0);

    // Values correspond to presampled plan symbols in ascending SymbolId order.
    void run_shot(std::span<const uint8_t> presampled_values = {}) noexcept;

    // Forces one Boolean value for every plan record slot, ordered as visible
    // records followed by hidden records. The reported probability is
    // conditional on any supplied presampled symbols. Replay consumes no RNG.
    [[nodiscard]] ReplayResult replay_shot(
        std::span<const uint8_t> forced_records,
        std::span<const uint8_t> presampled_values = {}) noexcept;

    [[nodiscard]] std::span<const uint8_t> visible_records() const {
        return std::span<const uint8_t>(records_).first(plan_.num_visible_records_);
    }
    [[nodiscard]] std::span<const uint8_t> hidden_records() const {
        return std::span<const uint8_t>(records_).subspan(plan_.num_visible_records_);
    }
    [[nodiscard]] std::span<const uint8_t> symbols() const { return symbols_; }
    [[nodiscard]] const State& state() const { return state_; }
    // Counts positive branch probability mass classified as numerical dust.
    // This telemetry accumulates across shots.
    [[nodiscard]] uint64_t dust_clamps() const { return dust_clamps_; }

  private:
    struct ForcedBranchResult {
        bool reachable = true;
        double log_increment = 0.0;
    };

    void initialize_shot(std::span<const uint8_t> presampled_values) noexcept;

    template <bool ForceRecords>
    [[nodiscard]] ReplayResult execute_actions(std::span<const uint8_t> forced_records) noexcept;

    [[nodiscard]] bool evaluate(ExecutablePlan::PreparedExpression expression) const noexcept;
    [[nodiscard]] bool sample_active_branch(MeasurementProbabilities probabilities) noexcept;
    [[nodiscard]] ForcedBranchResult force_active_branch(MeasurementProbabilities probabilities,
                                                         bool branch) noexcept;
    [[nodiscard]] bool sample_dormant_branch() noexcept;

    const ExecutablePlan& plan_;
    State state_;
    std::vector<uint8_t> symbols_;
    std::vector<uint8_t> records_;
    Xoshiro256PlusPlus rng_;
    uint64_t dust_clamps_ = 0;
};

}  // namespace clifft::sampling
