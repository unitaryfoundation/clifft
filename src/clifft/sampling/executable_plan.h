#pragma once

#include "clifft/sampling/active_measurement_dispatch.h"
#include "clifft/sampling/direct_rotation_dispatch.h"
#include "clifft/sampling/fused_rotation_dispatch.h"
#include "clifft/sampling/instrument_activation_dispatch.h"
#include "clifft/sampling/kernels.h"
#include "clifft/sampling/plan.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace clifft::sampling {

class Executor;
class ExecutablePlanBuilder;

// Owns the CPU lowering of one validated SamplingPlan. Direct-Pauli kernel
// descriptors and affine-expression register dependencies are prepared once
// here so a shot only reads fixed storage.
class ExecutablePlan {
  public:
    explicit ExecutablePlan(const SamplingPlan& plan);

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
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
    [[nodiscard]] const stim::Tableau<kStimWidth>* final_state_tableau() const noexcept {
        return final_tableau_ ? &*final_tableau_ : nullptr;
    }
    [[nodiscard]] uint32_t num_instrument_sites() const {
        return static_cast<uint32_t>(instrument_distributions_.size());
    }
    [[nodiscard]] uint32_t num_symbols() const { return num_symbols_; }
    [[nodiscard]] uint32_t num_presampled_symbols() const {
        return static_cast<uint32_t>(presampled_symbols_.size());
    }
    [[nodiscard]] size_t num_actions() const { return actions_.size(); }
    [[nodiscard]] size_t num_new_x_instrument_activations() const;
    [[nodiscard]] uint32_t num_unbound_presampled_symbols() const {
        return static_cast<uint32_t>(unbound_presampled_symbols_.size());
    }
    [[nodiscard]] std::vector<double> noise_site_probabilities() const;

  private:
    friend class Executor;
    friend class ExecutablePlanBuilder;

    struct BuilderTag {};
    ExecutablePlan(const SamplingPlan& plan, BuilderTag);

    struct PreparedExpression {
        uint32_t register_id = 0;
    };

    struct ExecuteRotation {
        // Geometry and trigonometric weights shared by every implementation.
        PreparedRotation rotation;
        // Register containing the branch-dependent Pauli sign for this shot.
        PreparedExpression sign;
        // Host-selected shape tag stored in the descriptor's tail padding.
        DirectRotationKernel kernel;

        void apply(State& state, bool sign_value) const noexcept {
            apply_direct_rotation(state, rotation, kernel, sign_value);
        }
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

        [[nodiscard]] MeasurementProbabilities probabilities(const State& state) const noexcept {
            if (kernel == ActiveMeasurementKernel::Scalar) {
                return measurement_probabilities(state, measurement);
            }
            return active_measurement_probabilities(state, measurement, kernel);
        }

        void collapse(State& state, bool selected_branch,
                      double branch_probability) const noexcept {
            if (kernel == ActiveMeasurementKernel::Scalar) {
                collapse_measurement(state, measurement, selected_branch, branch_probability);
                return;
            }
            collapse_active_measurement(state, measurement, kernel, selected_branch,
                                        branch_probability);
        }
    };

    static_assert(sizeof(ExecuteActiveMeasurement) == 88,
                  "active measurement dispatch must not expand its action descriptor");

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
        uint32_t site = 0;
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

    struct ExecuteExpectation {
        std::optional<PreparedPauli> active_projection;
        PreparedExpression sign;
        uint32_t exp_val = 0;
    };

    struct ExecuteInstrument {
        InstrumentMode mode = InstrumentMode::Classical;
        // Lowering identifies the exact new-coordinate X shape so execution
        // need not rediscover instrument topology inside the hot loop.
        std::optional<NewXInstrumentKernel> new_x_kernel;
        PreparedExpression sign;
        std::optional<PreparedMeasurement> measurement;
        uint32_t site = 0;
        std::optional<uint32_t> destination_flip;

        [[nodiscard]] bool activates_new_x() const noexcept { return new_x_kernel.has_value(); }
    };

    static_assert(sizeof(ExecuteInstrument) == 104,
                  "instrument specialization must not expand its action descriptor");

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

    uint32_t num_qubits_ = 0;
    uint32_t initial_active_width_ = 0;
    uint32_t max_active_width_ = 0;
    uint32_t num_visible_records_ = 0;
    uint32_t num_hidden_records_ = 0;
    uint32_t num_detectors_ = 0;
    uint32_t num_observables_ = 0;
    uint32_t num_exp_vals_ = 0;
    uint32_t num_symbols_ = 0;
    bool has_postselection_ = false;
    bool has_readout_noise_ = false;
    bool has_instruments_ = false;
    uint32_t num_readout_noise_sites_ = 0;
    uint32_t initial_noise_end_ = 0;
    std::complex<double> global_weight_ = {1.0, 0.0};
    std::optional<stim::Tableau<kStimWidth>> final_tableau_;
    // Constants are indexed by PreparedExpression::register_id. The dependency
    // vectors form CSR: targets[offsets[symbol]..offsets[symbol + 1]) lists the
    // expression registers toggled when that symbol is true.
    std::vector<uint8_t> expression_register_constants_;
    std::vector<uint32_t> expression_dependency_offsets_;
    std::vector<uint32_t> expression_dependency_targets_;
    // Maps each dense presampled input position to its plan-local SymbolId.
    // The constructor records those ids in ascending order.
    std::vector<uint32_t> presampled_symbols_;
    std::vector<uint32_t> unbound_presampled_symbols_;
    std::vector<PreparedNoiseOutcome> noise_outcomes_;
    std::vector<PreparedNoiseSite> noise_sites_;
    std::vector<double> noise_hazards_;
    std::vector<InstrumentDistribution> instrument_distributions_;
    std::vector<uint32_t> instrument_resume_offsets_;
    std::vector<PreparedFusedRotationExecution> fused_rotations_;
    std::vector<PreparedDynamicFusedRotationExecution> dynamic_fused_rotations_;
    std::vector<Action> actions_;
};

}  // namespace clifft::sampling
