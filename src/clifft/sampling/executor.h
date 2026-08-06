#pragma once

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

// Classifies an active measurement before ordinary sampling or forced replay
// chooses a branch.
enum class MeasurementBranchKind : uint8_t {
    Random,
    Zero,
    One,
};

struct MeasurementBranchClassification {
    MeasurementBranchKind kind = MeasurementBranchKind::Random;
    bool clamped_dust = false;
};

[[nodiscard]] MeasurementBranchClassification classify_measurement_branch(
    MeasurementProbabilities probabilities) noexcept;

// Owns the execution-specific lowering of one validated SamplingPlan. Kernel
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
        PreparedExpression outcome;
        uint32_t branch = 0;
        uint32_t record = 0;
    };

    struct ExecuteDormantMeasurement {
        PreparedExpression outcome;
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

    uint32_t initial_active_width_ = 0;
    uint32_t max_active_width_ = 0;
    uint32_t num_visible_records_ = 0;
    uint32_t num_hidden_records_ = 0;
    uint32_t num_symbols_ = 0;
    std::complex<double> global_weight_ = {1.0, 0.0};
    std::vector<uint32_t> expression_terms_;
    std::vector<uint32_t> presampled_symbols_;
    std::vector<Action> actions_;
};

// Holds every mutable value needed to execute repeated shots of one prepared
// plan. The plan must outlive the executor. Construction allocates all state,
// symbol, and record storage; run_shot only resets and overwrites it.
class Executor {
  public:
    explicit Executor(const ExecutablePlan& plan, uint64_t seed = 0);

    void run_shot(std::span<const uint8_t> presampled_values = {}) noexcept;

    [[nodiscard]] std::span<const uint8_t> visible_records() const {
        return std::span<const uint8_t>(records_).first(plan_.num_visible_records_);
    }
    [[nodiscard]] std::span<const uint8_t> hidden_records() const {
        return std::span<const uint8_t>(records_).subspan(plan_.num_visible_records_);
    }
    [[nodiscard]] std::span<const uint8_t> symbols() const { return symbols_; }
    [[nodiscard]] const State& state() const { return state_; }
    [[nodiscard]] uint64_t dust_clamps() const { return dust_clamps_; }

  private:
    [[nodiscard]] bool evaluate(ExecutablePlan::PreparedExpression expression) const noexcept;
    [[nodiscard]] bool sample_active_branch(MeasurementProbabilities probabilities) noexcept;
    [[nodiscard]] bool sample_dormant_branch() noexcept;

    const ExecutablePlan& plan_;
    State state_;
    std::vector<uint8_t> symbols_;
    std::vector<uint8_t> records_;
    Xoshiro256PlusPlus rng_;
    uint64_t dust_clamps_ = 0;
};

}  // namespace clifft::sampling
