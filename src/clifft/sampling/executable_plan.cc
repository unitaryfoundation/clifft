#include "clifft/sampling/executable_plan.h"

#include "clifft/sampling/executable_plan_builder.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft::sampling {

uint32_t ExecutablePlan::num_postselected_detectors() const noexcept {
    uint32_t count = 0;
    for (const auto& action : actions_) {
        if (const auto* detector = std::get_if<ExecuteDetector>(&action)) {
            count += detector->postselected;
        }
    }
    return count;
}

bool ExecutablePlan::has_reusable_unitary_prefix() const noexcept {
    uint32_t width = initial_active_width_;
    bool measured = false;
    uint64_t prefix_work = 0;
    uint64_t suffix_work = 0;
    const auto add_work = [&](uint64_t work) noexcept {
        uint64_t& target = measured ? suffix_work : prefix_work;
        target = saturating_add_u64(target, work);
    };
    for (const auto& action : actions_) {
        const bool rejected = std::visit(
            [&](const auto& typed) noexcept {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecuteActiveMeasurement>) {
                    measured = true;
                    --width;
                    return false;
                } else if constexpr (std::is_same_v<T, ExecuteDormantMeasurement>) {
                    measured = true;
                    // Conservatively reject dormant branches that feed later expressions so
                    // classical feedback cannot double groups without shrinking active width.
                    return !expression_dependencies_.dependent_registers(typed.branch).empty();
                } else if constexpr (std::is_same_v<T, ExecutePromotion>) {
                    if (measured) {
                        return true;
                    }
                    add_work(saturating_multiply_u64(uint64_t{1} << width, 4));
                    ++width;
                    return false;
                } else if constexpr (std::is_same_v<T, ExecuteRotation>) {
                    add_work(saturating_multiply_u64(uint64_t{1} << width, 4));
                    return false;
                } else if constexpr (std::is_same_v<T, ExecuteFusedRotation>) {
                    const auto& rot = fused_rotations_[typed.rotation_index].rotation();
                    add_work(saturating_multiply_u64(uint64_t{1} << rot.active_width,
                                                     uint64_t{2} << rot.orbit_rank));
                    return false;
                } else if constexpr (std::is_same_v<T, ExecuteDynamicFusedRotation>) {
                    const auto& rot =
                        dynamic_fused_rotations_[typed.rotation_index].variants.front().rotation();
                    add_work(saturating_multiply_u64(uint64_t{1} << rot.active_width,
                                                     uint64_t{2} << rot.orbit_rank));
                    return false;
                } else if constexpr (std::is_same_v<T, ExecuteClassicalRecord> ||
                                     std::is_same_v<T, ExecuteSymbolDefinition> ||
                                     std::is_same_v<T, ExecuteReadoutNoise> ||
                                     std::is_same_v<T, ExecuteDetector> ||
                                     std::is_same_v<T, ExecuteObservable> ||
                                     std::is_same_v<T, ExecuteExpectation> ||
                                     std::is_same_v<T, ExecuteInstrument> ||
                                     std::is_same_v<T, ExecuteBoundary>) {
                    return false;
                } else {
                    static_assert(!sizeof(T*), "Unhandled ExecutablePlan action alternative");
                }
            },
            action);
        if (rejected) {
            return false;
        }
    }
    return prefix_work != 0 && suffix_work <= prefix_work;
}

PreparedFusedRotationExecution::PreparedFusedRotationExecution(PreparedFusedRotation rotation,
                                                               ExecutorBackend backend)
    : rotation_(std::move(rotation)) {
    switch (backend) {
        case ExecutorBackend::Scalar:
            break;
        case ExecutorBackend::Neon:
#if defined(CLIFFT_ENABLE_APPLE_NEON)
            sidecar_ = prepare_fused_rotation_neon_sidecar(rotation_);
#endif
            break;
        case ExecutorBackend::Avx2:
#if defined(CLIFFT_ENABLE_X86_RUNTIME_DISPATCH)
            sidecar_ = prepare_fused_rotation_avx2_sidecar(rotation_);
#endif
            break;
        case ExecutorBackend::Avx512:
#if defined(CLIFFT_ENABLE_X86_RUNTIME_DISPATCH)
            sidecar_ = prepare_fused_rotation_avx512_sidecar(rotation_);
#endif
            break;
    }
    assert((sidecar_.storage == nullptr) == (sidecar_.kernel == nullptr) &&
           "fused sidecar storage and serial kernel must be set together");
    assert((sidecar_.kernel == nullptr) == (sidecar_.parallel_kernel == nullptr) &&
           "fused sidecar serial and parallel kernels must be set together");
}

ExecutablePlan::ExecutablePlan(const SamplingPlan& plan)
    : num_qubits_(plan.num_qubits),
      initial_active_width_(plan.initial_active_width),
      peak_active_width_(plan.peak_active_width),
      num_visible_records_(plan.num_visible_records),
      num_hidden_records_(plan.num_hidden_records),
      num_detectors_(plan.num_detectors),
      num_observables_(plan.num_observables),
      num_exp_vals_(plan.num_exp_vals),
      final_tableau_(plan.final_tableau),
      instrument_distributions_(plan.instrument_distributions),
      frame_width_(plan.frame_width) {
    frame_entry_peak_width_ = initial_active_width_;
    for (const PlannedAction& planned : plan.actions) {
        if (std::holds_alternative<ApplyConditionalClifford>(planned.action) ||
            std::holds_alternative<RotateFramePauli>(planned.action) ||
            std::holds_alternative<MeasureFramePauli>(planned.action)) {
            break;
        }
        frame_entry_peak_width_ = std::max(frame_entry_peak_width_, planned.active_after);
    }
    // Keep construction-only lowering state out of the immutable executable.
    ExecutablePlanBuilder::build(*this, plan);
}

ExecutablePlan ExecutablePlan::compact_outputs(SamplingOutputs outputs, bool early_postselection,
                                               bool survivor_counts) const {
    return ExecutablePlanBuilder::compact_outputs(*this, outputs, early_postselection,
                                                  survivor_counts);
}

std::vector<AffineBool> ExecutablePlan::affine_registers() const {
    std::vector<AffineBool> expressions;
    expressions.reserve(num_expression_registers());
    for (uint8_t constant : expression_register_constants_) {
        expressions.emplace_back(constant != 0);
    }
    for (uint32_t symbol = 0; symbol < num_symbols_; ++symbol) {
        for (uint32_t reg : expression_dependencies_.dependent_registers(symbol)) {
            expressions[reg] ^= AffineBool::symbol(SymbolId{symbol});
        }
    }
    return expressions;
}

void ExecutablePlan::ExpressionDependencies::validate(uint32_t num_symbols,
                                                      size_t num_registers) const noexcept {
#ifndef NDEBUG
    assert(offsets_.size() == static_cast<size_t>(num_symbols) + 1 &&
           "expression dependency offsets have the wrong size");
    assert(!offsets_.empty() && offsets_.front() == 0 && offsets_.back() == targets_.size() &&
           "expression dependency ranges are inconsistent");
    for (size_t i = 1; i < offsets_.size(); ++i) {
        assert(offsets_[i] >= offsets_[i - 1] && "expression dependency offsets are not ordered");
    }
    for (uint32_t target : targets_) {
        assert(target < num_registers && "expression dependency target is out of range");
    }
#else
    static_cast<void>(num_symbols);
    static_cast<void>(num_registers);
#endif
}

size_t ExecutablePlan::num_new_x_instrument_activations() const {
    return static_cast<size_t>(
        std::count_if(actions_.begin(), actions_.end(), [](const Action& action) {
            const auto* instrument = std::get_if<ExecuteInstrument>(&action);
            return instrument != nullptr &&
                   std::holds_alternative<ExecuteNewXInstrumentActivation>(instrument->form);
        }));
}

double ExecutablePlan::expected_noise_faults() const noexcept {
    double faults = 0.0;
    for (const PreparedNoiseSite& site : noise_sites_) {
        faults += site.conditioned_probability;
    }
    return faults;
}

std::vector<double> ExecutablePlan::noise_site_probabilities() const {
    std::vector<double> probabilities;
    probabilities.reserve(noise_sites_.size() + num_readout_noise_sites_);
    for (const PreparedNoiseSite& site : noise_sites_) {
        probabilities.push_back(site.conditioned_probability);
    }
    for (const Action& action : actions_) {
        const auto* readout = std::get_if<ExecuteReadoutNoise>(&action);
        if (readout == nullptr) {
            continue;
        }
        if (readout->prob_zero_to_one != readout->prob_one_to_zero) {
            throw std::invalid_argument(
                "k-fault conditioning does not support asymmetric readout noise; "
                "measurement record index " +
                std::to_string(readout->record) + " has probabilities (" +
                std::to_string(readout->prob_zero_to_one) + ", " +
                std::to_string(readout->prob_one_to_zero) + ")");
        }
        probabilities.push_back(readout->prob_zero_to_one);
    }
    return probabilities;
}

}  // namespace clifft::sampling
