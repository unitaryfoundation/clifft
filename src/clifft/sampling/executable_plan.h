#pragma once

#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/batch/presampled_program.h"
#include "clifft/sampling/kernel_dispatch.h"
#include "clifft/sampling/kernels.h"
#include "clifft/sampling/plan.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace clifft::sampling {

class Executor;
class BatchExecutor;
class ExecutablePlanBuilder;

// Owns one portable fused descriptor and the optional sidecar prepared by the
// plan's executor backend. The sidecar carries its matching entry point, so hot
// execution needs neither an ISA check nor an architecture-specific type.
class PreparedFusedRotationExecution {
  public:
    PreparedFusedRotationExecution(PreparedFusedRotation rotation, ExecutorBackend backend);

    void apply(State& state) const noexcept {
        if (sidecar_.storage != nullptr && sidecar_.kernel != nullptr) {
            sidecar_.kernel(state, rotation_, sidecar_.storage.get());
        } else {
            apply_fused_rotation(state, rotation_);
        }
    }

    void apply_parallel(State& state, uint32_t workers, uint32_t min_active_width) const noexcept {
        if (sidecar_.storage != nullptr && sidecar_.parallel_kernel != nullptr) {
            sidecar_.parallel_kernel(state, rotation_, sidecar_.storage.get(), workers,
                                     min_active_width);
        } else {
            apply_fused_rotation_parallel(state, rotation_, workers, min_active_width);
        }
    }

    [[nodiscard]] const PreparedFusedRotation& rotation() const noexcept { return rotation_; }

  private:
    PreparedFusedRotation rotation_;
    FusedRotationSidecar sidecar_;
};

// Owns the CPU lowering of one validated SamplingPlan. Direct-Pauli kernel
// descriptors and affine-expression register dependencies are prepared once
// here so a shot only reads fixed storage.
class ExecutablePlan {
  public:
    // Half-open SamplingPlan action range lowered into one executable action.
    struct PlanActionRange {
        uint32_t begin = 0;
        uint32_t end = 0;

        friend bool operator==(const PlanActionRange&, const PlanActionRange&) = default;
    };

    explicit ExecutablePlan(const SamplingPlan& plan);

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    [[nodiscard]] uint32_t peak_active_width() const { return peak_active_width_; }
    [[nodiscard]] uint32_t initial_active_width() const { return initial_active_width_; }
    [[nodiscard]] uint32_t num_visible_records() const { return num_visible_records_; }
    [[nodiscard]] uint32_t num_hidden_records() const { return num_hidden_records_; }
    [[nodiscard]] uint32_t num_detectors() const { return num_detectors_; }
    [[nodiscard]] uint32_t num_observables() const { return num_observables_; }
    [[nodiscard]] uint32_t num_exp_vals() const { return num_exp_vals_; }
    [[nodiscard]] bool has_postselection() const { return has_postselection_; }
    [[nodiscard]] bool has_readout_noise() const { return has_readout_noise_; }
    [[nodiscard]] bool has_instruments() const { return has_instruments_; }
    [[nodiscard]] bool supports_final_state_queries() const { return final_tableau_.has_value(); }
    // Exact final-state queries need the coordinate-to-physical map, but
    // ordinary execution must not depend on or mutate it.
    [[nodiscard]] const Tableau* final_state_tableau() const noexcept {
        return final_tableau_ ? &*final_tableau_ : nullptr;
    }
    [[nodiscard]] uint32_t num_instrument_sites() const {
        return static_cast<uint32_t>(instrument_distributions_.size());
    }
    [[nodiscard]] uint32_t num_symbols() const { return num_symbols_; }
    [[nodiscard]] uint32_t num_presampled_symbols() const {
        return static_cast<uint32_t>(presampled_symbols_.size());
    }
    [[nodiscard]] size_t num_expression_registers() const noexcept {
        return expression_register_constants_.size();
    }
    [[nodiscard]] bool output_parities_read_records() const noexcept {
        return !record_parity_terms_.empty();
    }
    [[nodiscard]] uint32_t num_readout_noise_sites() const noexcept {
        return num_readout_noise_sites_;
    }
    [[nodiscard]] uint32_t num_batch_noise_carriers() const noexcept {
        return batch_presampled_program_.has_value() ? batch_presampled_program_->num_carriers()
                                                     : 0;
    }
    [[nodiscard]] size_t num_actions() const { return actions_.size(); }
    [[nodiscard]] uint64_t estimated_batch_lane_work(BatchOutputMode output_mode) const noexcept {
        return estimated_batch_lane_work_.all_widths.for_output_mode(output_mode);
    }
    [[nodiscard]] uint64_t estimated_width_five_batch_lane_work(
        BatchOutputMode output_mode) const noexcept {
        return estimated_batch_lane_work_.width_five.for_output_mode(output_mode);
    }
    [[nodiscard]] size_t num_new_x_instrument_activations() const;
    [[nodiscard]] uint32_t num_unbound_presampled_symbols() const {
        return static_cast<uint32_t>(unbound_presampled_symbols_.size());
    }
    [[nodiscard]] std::vector<double> noise_site_probabilities() const;
    // Deterministic target-specific inspection for diagnostics and tooling.
    [[nodiscard]] std::string inspect() const;
    [[nodiscard]] std::string inspect_action(size_t action) const;
    // Present when the source SamplingPlan retained debug provenance. The
    // half-open range names every semantic action contributing to this action.
    [[nodiscard]] std::optional<PlanActionRange> action_plan_range(size_t action) const;

  private:
    friend class Executor;
    friend class BatchExecutor;
    friend class BatchPresampledProgram;
    friend class ExecutablePlanBuilder;

    // Actions retain only this index so repeated affine expressions stay in
    // shared plan storage.
    struct PreparedExpression {
        uint32_t register_id = 0;
    };

    // Reverse index used to update affine registers when a symbol becomes true.
    // Construction transposes expression terms into one contiguous range per symbol.
    class ExpressionDependencies {
      public:
        [[nodiscard]] std::span<const uint32_t> dependent_registers(
            uint32_t symbol) const noexcept {
            assert(static_cast<size_t>(symbol) + 1 < offsets_.size() &&
                   "assigned symbol must have an expression dependency range");
            const uint32_t begin = offsets_[symbol];
            const uint32_t end = offsets_[symbol + 1];
            return std::span<const uint32_t>(targets_).subspan(begin, end - begin);
        }

      private:
        friend class ExecutablePlanBuilder;

        [[nodiscard]] static ExpressionDependencies build(
            uint32_t num_symbols, std::span<const uint32_t> expression_terms,
            std::span<const uint32_t> expression_term_begins);
        void validate(uint32_t num_symbols, size_t num_registers) const noexcept;

        std::vector<uint32_t> offsets_;
        std::vector<uint32_t> targets_;
    };

    // CPU action descriptors. They contain only fixed operands and indices;
    // lowering has already selected topology and any applicable kernel shape.
    struct ExecuteRotation {
        // Geometry and trigonometric weights shared by every implementation.
        PreparedRotation rotation;
        // Register containing the branch-dependent Pauli sign for this shot.
        PreparedExpression sign;
        // Backend-selected traversal stored in the descriptor's tail padding.
        DirectRotationKernel kernel = DirectRotationKernel::Scalar;
    };

    static_assert(sizeof(ExecuteRotation) == 72,
                  "direct rotation dispatch must not expand its action descriptor");

    struct ExecuteFusedRotation {
        uint32_t rotation_index = 0;
    };

    struct ExecuteDynamicFusedRotation {
        uint32_t rotation_index = 0;
    };

    struct PreparedDynamicFusedRotationExecution {
        std::vector<PreparedExpression> sign_basis;
        std::vector<PreparedFusedRotationExecution> variants;
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
        ActiveMeasurementKernel kernel = ActiveMeasurementKernel::Scalar;
    };

    static_assert(sizeof(ExecuteActiveMeasurement) == 88,
                  "active measurement dispatch must not expand its action descriptor");

    // Measurement records and derived symbols.
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

    // Stochastic readout and derived circuit outputs.
    struct ExecuteReadoutNoise {
        PreparedExpression source;
        uint32_t flip = 0;
        uint32_t record = 0;
        uint32_t site = 0;
        double prob_zero_to_one = 0.0;
        double prob_one_to_zero = 0.0;
        double batch_symmetric_inverse_hazard = 0.0;
    };

    struct PreparedRecordParity {
        uint32_t begin = 0;
        uint32_t count = 0;
        bool constant = false;
    };

    using PreparedObservableValue = std::variant<PreparedExpression, PreparedRecordParity>;

    struct ExecuteDetector {
        PreparedRecordParity outcome;
        uint32_t detector = 0;
        bool postselected = false;
        // Planner-precomputed coefficient work remaining after this detector,
        // measured in approximate per-lane coefficient visits for each output mode.
        batch_detail::BatchLaneWork remaining_batch_lane_work;
    };

    struct ExecuteObservable {
        PreparedObservableValue outcome;
        uint32_t observable = 0;
    };

    // Exact expectation-value probe of the current state.
    struct PreparedExpectation {
        PreparedPauli projection;
        PreparedExpression sign;
    };

    struct ExecuteExpectation {
        std::optional<PreparedExpectation> active;
        uint32_t exp_val = 0;
    };

    // Each instrument mode has a separate action type so sampling does not
    // check the mode on every shot.
    struct ExecuteClassicalInstrument {
        PreparedExpression sign;
        uint32_t site = 0;
        uint32_t destination_flip = 0;
    };

    struct ExecuteDormantInstrumentTrap {
        uint32_t site = 0;
    };

    struct ExecuteActiveInstrument {
        PreparedMeasurement measurement;
        PreparedExpression sign;
        uint32_t site = 0;
        uint32_t destination_flip = 0;
    };

    // These forms intentionally store the same prepared operands but remain
    // distinct so dispatch encodes whether a clean coordinate must be added.
    // The current planner selects new-X activation, while validated plans and
    // future planners may still require the generic measured-source fallback.
    struct ExecuteMeasuredInstrumentActivation {
        PreparedMeasurement measurement;
        PreparedExpression sign;
        uint32_t site = 0;
        uint32_t destination_flip = 0;
    };

    struct ExecuteNewXInstrumentActivation {
        PreparedExpression sign;
        uint32_t site = 0;
        uint32_t destination_flip = 0;
        // Lowering identifies the exact new-coordinate X shape so execution
        // need not rediscover instrument topology inside the hot loop.
        NewXInstrumentKernel kernel = NewXInstrumentKernel::Scalar;
    };

    static_assert(sizeof(ExecuteActiveInstrument) == 88,
                  "active instrument specialization must remain compact");
    static_assert(sizeof(ExecuteMeasuredInstrumentActivation) == 88,
                  "instrument activation specialization must remain compact");

    // Keep the concrete forms behind one outer alternative so instrument-free
    // dispatch retains its existing visit table. The trivial dormant form must
    // remain first because GCC queries the nested variant's default alternative
    // before the enclosing ExecutablePlan is complete.
    using InstrumentAction =
        std::variant<ExecuteDormantInstrumentTrap, ExecuteClassicalInstrument,
                     ExecuteActiveInstrument, ExecuteMeasuredInstrumentActivation,
                     ExecuteNewXInstrumentActivation>;

    struct ExecuteInstrument {
        InstrumentAction form;
    };

    static_assert(sizeof(ExecuteInstrument) <= 96,
                  "instrument specialization must preserve the compact descriptor");

    struct ExecuteBoundary {
        uint32_t site = 0;
        uint32_t active_width = 0;
        uint32_t noise_begin = 0;
        uint32_t noise_end = 0;
        uint32_t symbol_prefix_size = 0;
    };

    // Prepared input distributions used before dispatch and at continuation
    // boundaries.
    struct PreparedNoiseOutcome {
        uint32_t symbol = 0;
        double cumulative_probability = 0.0;
    };

    struct PreparedNoiseSite {
        uint32_t outcome_begin = 0;
        uint32_t outcome_count = 0;
        // Exact HIR total exposed to fixed-k conditioning. Ordinary execution
        // retains the final cumulative outcome as its channel-draw bound.
        double conditioned_probability = 0.0;
    };

    using Action =
        std::variant<ExecuteRotation, ExecuteFusedRotation, ExecuteDynamicFusedRotation,
                     ExecutePromotion, ExecuteActiveMeasurement, ExecuteDormantMeasurement,
                     ExecuteClassicalRecord, ExecuteSymbolDefinition, ExecuteReadoutNoise,
                     ExecuteDetector, ExecuteObservable, ExecuteExpectation, ExecuteInstrument,
                     ExecuteBoundary>;

    // Immutable plan metadata and externally visible dimensions.
    uint32_t num_qubits_ = 0;
    uint32_t initial_active_width_ = 0;
    uint32_t peak_active_width_ = 0;
    uint32_t num_visible_records_ = 0;
    uint32_t num_hidden_records_ = 0;
    uint32_t num_detectors_ = 0;
    uint32_t num_observables_ = 0;
    uint32_t num_exp_vals_ = 0;
    uint32_t num_symbols_ = 0;
    bool has_postselection_ = false;
    bool has_readout_noise_ = false;
    bool has_instruments_ = false;
    ExecutorBackend backend_ = ExecutorBackend::Scalar;
    uint32_t num_readout_noise_sites_ = 0;
    uint32_t initial_noise_end_ = 0;
    batch_detail::BatchWorkEstimate estimated_batch_lane_work_;
    std::optional<Tableau> final_tableau_;

    // Affine registers. Constants are indexed by
    // PreparedExpression::register_id.
    std::vector<uint8_t> expression_register_constants_;
    ExpressionDependencies expression_dependencies_;

    // Presampled inputs and their circuit-site distributions. Symbol lists use
    // ascending plan-local ids so execution can index them without a map.
    std::vector<uint32_t> presampled_symbols_;
    std::vector<uint32_t> unbound_presampled_symbols_;
    std::vector<PreparedNoiseOutcome> noise_outcomes_;
    std::vector<PreparedNoiseSite> noise_sites_;
    std::vector<double> noise_hazards_;
    std::optional<double> uniform_noise_inverse_hazard_;
    std::optional<BatchPresampledProgram> batch_presampled_program_;

    // Packed-only output parities.
    std::vector<uint32_t> record_parity_terms_;

    // Instrument inputs and continuation offsets share the same site index.
    std::vector<InstrumentDistribution> instrument_distributions_;
    std::vector<uint32_t> instrument_resume_offsets_;

    // Hot action stream and its indexed prepared payloads. The source map is
    // an optional debug sidecar rather than parallel production storage.
    std::vector<PreparedFusedRotationExecution> fused_rotations_;
    std::vector<PreparedDynamicFusedRotationExecution> dynamic_fused_rotations_;
    std::vector<Action> actions_;
    std::vector<PlanActionRange> action_plan_ranges_;
};

}  // namespace clifft::sampling
