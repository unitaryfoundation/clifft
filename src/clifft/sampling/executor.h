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
#include <optional>
#include <span>
#include <utility>
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

// Describes whether a requested record can occur and, if so, its conditional
// joint log probability. The probability is meaningful only when reachable.
struct ReplayResult {
    bool reachable = true;
    double log_probability = 0.0;
};

struct InstrumentTrap {
    InstrumentSiteId site{};
    uint8_t source = 0;
    bool destination_pending = false;
};

[[nodiscard]] MeasurementBranchClassification classify_measurement_branch(
    MeasurementProbabilities probabilities) noexcept;

// Owns the CPU lowering of one validated SamplingPlan. Direct-Pauli kernel
// descriptors and affine-expression term ranges are prepared once here so a
// shot only reads fixed storage.
class ExecutablePlan {
  public:
    explicit ExecutablePlan(const SamplingPlan& plan);

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    [[nodiscard]] uint32_t num_visible_records() const { return num_visible_records_; }
    [[nodiscard]] uint32_t num_hidden_records() const { return num_hidden_records_; }
    [[nodiscard]] uint32_t num_detectors() const { return num_detectors_; }
    [[nodiscard]] uint32_t num_observables() const { return num_observables_; }
    [[nodiscard]] bool has_postselection() const { return has_postselection_; }
    [[nodiscard]] bool has_readout_noise() const { return has_readout_noise_; }
    [[nodiscard]] bool has_instruments() const { return has_instruments_; }
    [[nodiscard]] uint32_t num_instrument_sites() const {
        return static_cast<uint32_t>(instrument_distributions_.size());
    }
    [[nodiscard]] uint32_t num_symbols() const { return num_symbols_; }
    [[nodiscard]] uint32_t num_presampled_symbols() const {
        return static_cast<uint32_t>(presampled_symbols_.size());
    }
    [[nodiscard]] size_t num_actions() const { return actions_.size(); }
    [[nodiscard]] uint32_t num_unbound_presampled_symbols() const {
        return static_cast<uint32_t>(unbound_presampled_symbols_.size());
    }

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
        // The physical record is the raw measurement branch XOR this
        // correction. The correction contains the known sign flips from
        // earlier stochastic events, so replay can recover the raw branch
        // needed to produce a requested record.
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

    struct ExecuteReadoutNoise {
        PreparedExpression source;
        uint32_t flip = 0;
        uint32_t record = 0;
        double prob_zero_to_one = 0.0;
        double prob_one_to_zero = 0.0;
    };

    struct ExecuteDetector {
        PreparedExpression outcome;
        uint32_t detector = 0;
        bool postselected = false;
    };

    struct ExecuteObservable {
        PreparedExpression outcome;
        uint32_t observable = 0;
    };

    struct ExecuteInstrument {
        InstrumentMode mode = InstrumentMode::Classical;
        PreparedExpression sign;
        std::optional<PreparedMeasurement> measurement;
        uint32_t site = 0;
        std::optional<uint32_t> destination_flip;
    };

    struct ExecuteBoundary {
        uint32_t site = 0;
        uint32_t active_width = 0;
        uint32_t noise_begin = 0;
        uint32_t noise_end = 0;
        uint32_t symbol_prefix_size = 0;
    };

    struct PreparedNoiseOutcome {
        uint32_t symbol = 0;
        double cumulative_probability = 0.0;
    };

    struct PreparedNoiseSite {
        uint32_t outcome_begin = 0;
        uint32_t outcome_count = 0;
        double total_probability = 0.0;
    };

    using Action = std::variant<ExecuteRotation, ExecutePromotion, ExecuteActiveMeasurement,
                                ExecuteDormantMeasurement, ExecuteClassicalRecord,
                                ExecuteSymbolDefinition, ExecuteReadoutNoise, ExecuteDetector,
                                ExecuteObservable, ExecuteInstrument, ExecuteBoundary>;

    PreparedExpression prepare_expression(const AffineBool& expression);
    PreparedExpression prepare_measurement_correction(const AffineBool& outcome, uint32_t branch);

    uint32_t num_qubits_ = 0;
    uint32_t initial_active_width_ = 0;
    uint32_t max_active_width_ = 0;
    uint32_t num_visible_records_ = 0;
    uint32_t num_hidden_records_ = 0;
    uint32_t num_detectors_ = 0;
    uint32_t num_observables_ = 0;
    uint32_t num_symbols_ = 0;
    bool has_postselection_ = false;
    bool has_readout_noise_ = false;
    bool has_instruments_ = false;
    uint32_t initial_noise_end_ = 0;
    std::complex<double> global_weight_ = {1.0, 0.0};
    std::vector<uint32_t> expression_terms_;
    // Maps each dense presampled input position to its plan-local SymbolId.
    // The constructor records those ids in ascending order.
    std::vector<uint32_t> presampled_symbols_;
    std::vector<uint32_t> unbound_presampled_symbols_;
    std::vector<PreparedNoiseOutcome> noise_outcomes_;
    std::vector<PreparedNoiseSite> noise_sites_;
    std::vector<double> noise_hazards_;
    std::vector<InstrumentDistribution> instrument_distributions_;
    std::vector<uint32_t> instrument_resume_offsets_;
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

    // Replace the deterministic seed with OS entropy before executing shots.
    void reseed_from_entropy() { rng_.seed_from_entropy(); }

    // Draw plan-bound quantum noise from this executor's RNG before dispatch.
    void run_shot() noexcept;

    // Use caller-supplied Boolean values for every Presampled symbol in
    // ascending SymbolId order. This path consumes no quantum-noise RNG draws.
    void run_shot(std::span<const uint8_t> presampled_values) noexcept;

    // Continue a trapped shot in a plan compiled for the selected trajectory.
    // Storage may grow here, before dispatch resumes. A forced record supplies
    // the source chosen by a trap-only dormant instrument to the continuation's
    // hidden trace-out measurement.
    void resume(const ExecutablePlan& continuation,
                std::optional<std::pair<RecordSlot, uint8_t>> forced_record = std::nullopt);

    // Replays the plan while forcing each record to a supplied Boolean value.
    // This reconstructs the corresponding branch state and computes its joint
    // log probability, enabling differential validation and exact record
    // probability queries. Records are ordered as visible followed by hidden;
    // the probability is conditional on supplied presampled symbols. Replay
    // consumes no RNG. After an unreachable result, state and record accessors
    // expose only the executed prefix, not completed-shot output.
    [[nodiscard]] ReplayResult replay_shot(
        std::span<const uint8_t> forced_records,
        std::span<const uint8_t> presampled_values = {}) noexcept;

    [[nodiscard]] std::span<const uint8_t> visible_records() const {
        return std::span<const uint8_t>(records_).first(plan_->num_visible_records_);
    }
    [[nodiscard]] std::span<const uint8_t> hidden_records() const {
        return std::span<const uint8_t>(records_).subspan(plan_->num_visible_records_,
                                                          plan_->num_hidden_records_);
    }
    [[nodiscard]] std::span<const uint8_t> symbols() const {
        return std::span<const uint8_t>(symbols_).first(plan_->num_symbols_);
    }
    [[nodiscard]] std::span<const uint8_t> detectors() const { return detectors_; }
    [[nodiscard]] std::span<const uint8_t> observables() const { return observables_; }
    [[nodiscard]] bool discarded() const { return discarded_; }
    [[nodiscard]] std::optional<InstrumentTrap> pending_trap() const { return pending_trap_; }
    [[nodiscard]] const State& state() const { return state_; }
    // Counts positive branch probability mass classified as numerical dust.
    // This telemetry accumulates across shots.
    [[nodiscard]] uint64_t dust_clamps() const { return dust_clamps_; }

  private:
    void reset_shot() noexcept;
    void assign_presampled_values(std::span<const uint8_t> presampled_values) noexcept;
    void sample_presampled_noise(uint32_t begin, uint32_t end) noexcept;

    template <bool ForceRecords, bool SampleNoise>
    [[nodiscard]] ReplayResult execute_actions(std::span<const uint8_t> forced_records,
                                               uint32_t begin = 0) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteRotation& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecutePromotion& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteClassicalRecord& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteReadoutNoise& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteDetector& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteObservable& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool ForceRecords>
    void execute_action(const ExecutablePlan::ExecuteInstrument& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <bool SampleNoise>
    void execute_action(const ExecutablePlan::ExecuteBoundary& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;

    [[nodiscard]] bool evaluate(ExecutablePlan::PreparedExpression expression) const noexcept;
    [[nodiscard]] bool sample_active_branch(MeasurementProbabilities probabilities) noexcept;
    [[nodiscard]] std::optional<double> force_active_branch(MeasurementProbabilities probabilities,
                                                            bool branch) noexcept;
    [[nodiscard]] bool sample_dormant_branch() noexcept;

    const ExecutablePlan* root_plan_;
    const ExecutablePlan* plan_;
    State state_;
    std::vector<uint8_t> symbols_;
    std::vector<uint8_t> records_;
    std::vector<uint8_t> detectors_;
    std::vector<uint8_t> observables_;
    std::vector<uint8_t> forced_record_mask_;
    std::vector<uint8_t> forced_record_values_;
    std::vector<uint32_t> previous_presampled_ones_;
    Xoshiro256PlusPlus rng_;
    bool discarded_ = false;
    std::optional<InstrumentTrap> pending_trap_;
    uint64_t dust_clamps_ = 0;
};

// Samples a fixed number of shots into row-major visible-record storage. The
// plan and executor are prepared once, and all output is allocated before the
// first shot enters hot execution. Plans with presampled symbols are rejected
// until their sampling distribution is part of the executable contract.
[[nodiscard]] std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                                  std::optional<uint64_t> seed = std::nullopt);

// Replays each row-major visible record and returns its joint log probability.
// Unreachable records map to the lowest finite double because release builds
// assume finite arithmetic. Plans with presampled symbols or hidden records
// are rejected because this API does not yet marginalize over either source
// of hidden stochastic state.
[[nodiscard]] std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                                           std::span<const uint8_t> forced_records,
                                                           size_t num_records);

struct SamplingResult {
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
};

struct SamplingSurvivorResult {
    uint32_t total_shots = 0;
    uint32_t passed_shots = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
};

[[nodiscard]] SamplingResult sample(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed = std::nullopt);

[[nodiscard]] SamplingSurvivorResult sample_survivors(const ExecutablePlan& plan, uint32_t shots,
                                                      std::optional<uint64_t> seed = std::nullopt,
                                                      bool keep_records = false);

}  // namespace clifft::sampling
