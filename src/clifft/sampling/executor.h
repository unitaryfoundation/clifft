#pragma once

#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/state.h"
#include "clifft/util/config.h"
#include "clifft/util/xoshiro.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft {
class KFaultSampler;
}

namespace clifft::sampling {

class BasisAmplitudeQuery;

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

struct ForcedTraceOut {
    RecordSlot record{};
    uint8_t source = 0;
};

// Holds the dense active-coordinate coefficients, Boolean symbols carrying
// stochastic frame dependencies, visible and hidden records, measurement RNG,
// and numerical-dust telemetry for repeated shots. The plan must outlive the
// executor. Construction allocates all storage; run_shot only resets and
// overwrites it.
class Executor {
  public:
    explicit Executor(const ExecutablePlan& plan, uint64_t seed = 0,
                      uint32_t intra_shot_workers = 1,
                      uint32_t intra_shot_min_active_width = kDefaultIntraShotMinActiveWidth);

    // Replace the deterministic seed with OS entropy before executing shots.
    void reseed_from_entropy() { rng_.seed_from_entropy(); }

    // Replace all RNG state words. The trajectory driver uses this to keep
    // executor draws in a domain separate from its own per-shot decisions.
    void reseed_full(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) noexcept {
        rng_.seed_full(s0, s1, s2, s3);
    }

    // Draw plan-bound quantum noise from this executor's RNG before dispatch.
    void run_shot() noexcept;

    // Use caller-supplied Boolean values for every Presampled symbol in
    // ascending SymbolId order. This path consumes no quantum-noise RNG draws.
    void run_shot(std::span<const uint8_t> presampled_values) noexcept;

    // Draw one exact conditioned set of fault sites, choose a Pauli channel
    // within each selected quantum site, and force selected readout flips.
    void run_shot(KFaultSampler& fault_sampler) noexcept;

    // Continue a trapped shot in a plan compiled for the selected trajectory.
    // Storage may grow here, before dispatch resumes. A forced record supplies
    // the source chosen by a trap-only dormant instrument to the continuation's
    // hidden trace-out measurement.
    void resume(const ExecutablePlan& continuation,
                std::optional<ForcedTraceOut> forced_trace_out = std::nullopt);

    // Drop the borrowed continuation reference after its completed-shot
    // outputs have been consumed, allowing the caller to destroy that plan.
    void return_to_root_plan() noexcept;

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
    [[nodiscard]] std::span<const double> exp_vals() const { return exp_vals_; }
    [[nodiscard]] bool discarded() const { return discarded_; }
    [[nodiscard]] std::optional<InstrumentTrap> pending_trap() const { return pending_trap_; }
    [[nodiscard]] const State& state() const { return state_; }
    // Counts positive branch probability mass classified as numerical dust.
    // This telemetry accumulates across shots.
    [[nodiscard]] uint64_t dust_clamps() const { return dust_clamps_; }

  private:
    friend class BasisAmplitudeQuery;

    // Terminal effects preserve small positive branch mass that ordinary
    // sampling intentionally classifies as numerical dust.
    [[nodiscard]] ReplayResult replay_effect(std::span<const uint8_t> forced_records) noexcept;

    // Selects the complete per-shot policy at compile time. Keeping the
    // supported modes named prevents unsupported combinations of record,
    // noise, and fixed-fault behavior from reaching the action loop.
    enum class ShotMode : uint8_t {
        SampleNoise,
        UsePresampledNoise,
        FixedFaultCount,
        ReplayRecords,
        ReplayEffects,
    };

    enum class IntraShotMode : uint8_t {
        Serial,
        OpenMP,
    };

    void reset_shot() noexcept;
    void assign_presampled_values(std::span<const uint8_t> presampled_values) noexcept;
    void sample_presampled_noise(uint32_t begin, uint32_t end) noexcept;
    void activate_noise_site(uint32_t site) noexcept;
    void assign_forced_quantum_faults() noexcept;
    void assign_symbol(uint32_t symbol, bool value) noexcept;
    void propagate_true_symbol(const ExecutablePlan& plan, uint32_t symbol) noexcept;
    void initialize_expression_registers(const ExecutablePlan& plan,
                                         uint32_t symbol_prefix_size) noexcept;

    template <ShotMode Mode>
    [[nodiscard]] ReplayResult execute_actions_for_backend(std::span<const uint8_t> forced_records,
                                                           uint32_t begin = 0) noexcept;
    template <ExecutorBackend Backend, ShotMode Mode, IntraShotMode IntraShot>
    [[nodiscard]] ReplayResult execute_actions(std::span<const uint8_t> forced_records,
                                               uint32_t begin = 0) noexcept;
    template <ExecutorBackend Backend, IntraShotMode IntraShot>
    void execute_action(const ExecutablePlan::ExecuteRotation& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <IntraShotMode IntraShot>
    void execute_action(const ExecutablePlan::ExecuteFusedRotation& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <IntraShotMode IntraShot>
    void execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <IntraShotMode IntraShot>
    void execute_action(const ExecutablePlan::ExecutePromotion& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <ExecutorBackend Backend, ShotMode Mode>
    void execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <ShotMode Mode>
    void execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <ShotMode Mode>
    void execute_action(const ExecutablePlan::ExecuteClassicalRecord& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    void execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <ShotMode Mode>
    void execute_action(const ExecutablePlan::ExecuteReadoutNoise& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDetector& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    void execute_action(const ExecutablePlan::ExecuteObservable& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    void execute_action(const ExecutablePlan::ExecuteExpectation& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <ExecutorBackend Backend, ShotMode Mode>
    void execute_action(const ExecutablePlan::ExecuteInstrument& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;
    template <ShotMode Mode>
    void execute_action(const ExecutablePlan::ExecuteBoundary& action,
                        std::span<const uint8_t> forced_records, ReplayResult& result) noexcept;

    [[nodiscard]] bool evaluate(ExecutablePlan::PreparedExpression expression) const noexcept;
    [[nodiscard]] bool evaluate_observable(
        const ExecutablePlan::PreparedObservableValue& value) const noexcept;
    [[nodiscard]] bool evaluate_record_parity(
        ExecutablePlan::PreparedRecordParity parity) const noexcept;
    [[nodiscard]] bool sample_active_branch(MeasurementProbabilities probabilities) noexcept;
    [[nodiscard]] std::optional<double> force_active_branch(MeasurementProbabilities probabilities,
                                                            bool branch) noexcept;
    [[nodiscard]] static std::optional<double> force_active_effect_branch(
        MeasurementProbabilities probabilities, bool branch) noexcept;
    [[nodiscard]] bool sample_dormant_branch() noexcept;
    void trap_instrument(uint32_t site, uint8_t source, bool destination_pending) noexcept;
    void finish_instrument_fire(uint32_t site, uint32_t destination_flip, uint8_t source,
                                const InstrumentDistribution& distribution) noexcept;
    void execute_instrument(const ExecutablePlan::ExecuteClassicalInstrument& action) noexcept;
    void execute_instrument(const ExecutablePlan::ExecuteDormantInstrumentTrap& action) noexcept;
    template <ExecutorBackend Backend>
    void execute_instrument(const ExecutablePlan::ExecuteActiveInstrument& action) noexcept;
    template <ExecutorBackend Backend>
    void execute_instrument(
        const ExecutablePlan::ExecuteMeasuredInstrumentActivation& action) noexcept;
    template <ExecutorBackend Backend>
    void execute_instrument(const ExecutablePlan::ExecuteNewXInstrumentActivation& action) noexcept;
    template <ExecutorBackend Backend, typename Action>
    void execute_quantum_instrument(const Action& action) noexcept;

    // The caller-owned root plan must outlive this executor. After resume(),
    // plan_ keeps borrowing that continuation for accessors and any later
    // resume validation. A successful resume replaces the borrow, while
    // return_to_root_plan releases the final one. A throwing resume does not
    // release whichever continuation was current when it threw.
    const ExecutablePlan* root_plan_;
    const ExecutablePlan* plan_;

    // Capacity is allocated before ordinary dispatch and reused across shots.
    State state_;
    std::vector<uint8_t> symbols_;
    std::vector<uint8_t> expression_registers_;
    std::vector<uint8_t> records_;
    std::vector<uint8_t> detectors_;
    std::vector<uint8_t> observables_;
    std::vector<double> exp_vals_;

    // resume() can force a hidden trace-out record in the continuation.
    std::vector<uint8_t> forced_record_mask_;
    std::vector<uint8_t> forced_record_values_;

    // Reset only the presampled symbols that were true on the previous shot.
    std::vector<uint32_t> previous_presampled_ones_;

    // Fixed-fault-count sampling supplies its selected circuit sites here.
    std::span<const uint32_t> forced_fault_sites_;
    uint32_t forced_fault_cursor_ = 0;

    // RNG position deliberately persists across shots.
    Xoshiro256PlusPlus rng_;

    // Selected with the root plan and reused by every shot and continuation.
    ExecutorBackend backend_;
    uint32_t intra_shot_workers_ = 1;
    uint32_t intra_shot_min_active_width_ = kDefaultIntraShotMinActiveWidth;

    // Per-shot control state.
    bool discarded_ = false;
    std::optional<InstrumentTrap> pending_trap_;

    // Lifetime numerical telemetry; reset_shot does not clear it.
    uint64_t dust_clamps_ = 0;
};

}  // namespace clifft::sampling
