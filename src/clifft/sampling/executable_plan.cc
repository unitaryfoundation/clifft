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
                } else if constexpr (std::is_same_v<T, ExecuteFrameClifford> ||
                                     std::is_same_v<T, ExecuteFrameRotation> ||
                                     std::is_same_v<T, ExecuteFrameMeasurement> ||
                                     std::is_same_v<T, ExecuteFrameRetirement> ||
                                     std::is_same_v<T, ExecuteFrameRelease>) {
                    // A frame action is not part of a unitary prefix.
                    return true;
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
    // Widths are exact outside a live frame and upper bounds inside one.
    frame_entry_peak_width_ = initial_active_width_;
    bool frame_live = false;
    for (const PlannedAction& planned : plan.actions) {
        if (std::visit([](const auto& typed) { return FrameAction<decltype(typed)>; },
                       planned.action)) {
            frame_live = true;
        } else if (std::holds_alternative<RetireFrame>(planned.action)) {
            frame_live = false;
        }
        if (!frame_live) {
            frame_entry_peak_width_ = std::max(frame_entry_peak_width_, planned.active_after);
        }
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

std::vector<uint8_t> ExecutablePlan::rotation_sign_registers(bool frame_signs) const {
    std::vector<uint8_t> required(num_expression_registers());
    for (const auto& action : actions_) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecuteRotation> ||
                              std::is_same_v<T, ExecutePromotion>) {
                    required[typed.sign.register_id] = 1;
                } else if constexpr (std::is_same_v<T, ExecuteFrameRotation>) {
                    required[typed.sign.register_id] |= frame_signs;
                } else if constexpr (std::is_same_v<T, ExecuteFrameClifford>) {
                    required[typed.condition.register_id] |=
                        frame_signs && typed.rotation != CliffordRotation::IDENTITY;
                } else if constexpr (std::is_same_v<T, ExecuteDynamicFusedRotation>) {
                    for (const auto& sign :
                         dynamic_fused_rotations_[typed.rotation_index].sign_basis) {
                        required[sign.register_id] = 1;
                    }
                }
            },
            action);
    }
    // A derived sign input is uniform only when its defining expression is
    // uniform. Definitions are ordered, so one reverse pass closes this set.
    for (auto it = actions_.rbegin(); it != actions_.rend(); ++it) {
        if (const auto* definition = std::get_if<ExecuteSymbolDefinition>(&*it)) {
            for (auto reg : expression_dependencies_.dependent_registers(definition->symbol)) {
                if (required[reg]) {
                    required[definition->value.register_id] = 1;
                    break;
                }
            }
        }
    }
    return required;
}

ExecutablePlan::TrajectoryDivergence ExecutablePlan::measure_trajectory_divergence() const {
    const auto sign_registers = rotation_sign_registers(/*frame_signs=*/true);
    const auto feeds_sign = [&](uint32_t symbol) {
        return std::ranges::any_of(expression_dependencies_.dependent_registers(symbol),
                                   [&](uint32_t reg) { return sign_registers[reg] != 0; });
    };
    TrajectoryDivergence divergence;
    for (const PreparedNoiseSite& site : noise_sites_) {
        double previous = 0.0;
        for (uint32_t k = 0; k < site.outcome_count; ++k) {
            const PreparedNoiseOutcome& outcome = noise_outcomes_[site.outcome_begin + k];
            if (feeds_sign(outcome.symbol)) {
                divergence.expected_sign_faults += outcome.cumulative_probability - previous;
            }
            previous = outcome.cumulative_probability;
        }
    }
    // Active measurements at or after this index precede no coefficient work.
    size_t coefficient_work_end = actions_.size();
    while (coefficient_work_end != 0 &&
           !std::visit(
               [](const auto& typed) {
                   using T = std::decay_t<decltype(typed)>;
                   return std::is_same_v<T, ExecuteRotation> ||
                          std::is_same_v<T, ExecuteFusedRotation> ||
                          std::is_same_v<T, ExecuteDynamicFusedRotation> ||
                          std::is_same_v<T, ExecutePromotion>;
               },
               actions_[coefficient_work_end - 1])) {
        --coefficient_work_end;
    }
    for (size_t index = 0; index < actions_.size(); ++index) {
        const auto& action = actions_[index];
        if (const auto* dormant = std::get_if<ExecuteDormantMeasurement>(&action)) {
            divergence.sign_dormant_measurements += feeds_sign(dormant->branch) ? 1 : 0;
        } else if (const auto* readout = std::get_if<ExecuteReadoutNoise>(&action)) {
            // The flip rate depends on the recorded bit, so take the larger one.
            if (feeds_sign(readout->flip)) {
                divergence.expected_sign_faults +=
                    std::max(readout->prob_zero_to_one, readout->prob_one_to_zero);
            }
        } else if (std::holds_alternative<ExecuteActiveMeasurement>(action) &&
                   index < coefficient_work_end) {
            ++divergence.mid_circuit_active_measurements;
        } else if (const auto* frame = std::get_if<ExecuteFrameMeasurement>(&action)) {
            divergence.mid_circuit_active_measurements += feeds_sign(frame->branch) ? 1 : 0;
        }
    }
    return divergence;
}

bool ExecutablePlan::has_bounded_trajectory_divergence(uint32_t lanes) const noexcept {
    // Narrower plans have too little coefficient work to share.
    constexpr uint32_t kMinWidth = 3;
    // At one expected sign fault per shot about 63% of a batch leaves the
    // fault-free trajectory. Sharing then breaks even with AVX-512 packed
    // batches and is ahead at narrower vector widths.
    constexpr double kMaxExpectedSignFaults = 1.0;
    // Every dormant sign measurement, and every mid-circuit active measurement
    // whose outcome is random, doubles the trajectories. With all of them
    // random and spread through the plan, sharing stays ahead through 12 on a
    // full batch. It falls behind when they all precede the coefficient work
    // or the batch is small. A plan does not determine whether an active
    // outcome is random, so this limit bounds that loss without excluding it.
    constexpr uint64_t kMaxBranchingMeasurements = 12;
    // Dormant sign measurements split a batch into equally likely
    // trajectories. Sharing breaks even at 4 to 8 lanes per trajectory.
    constexpr uint32_t kMinLanesPerDormantBranch = 16;
    const TrajectoryDivergence& divergence = trajectory_divergence_;
    const uint64_t branching =
        uint64_t{divergence.sign_dormant_measurements} + divergence.mid_circuit_active_measurements;
    return peak_active_width_ >= kMinWidth &&
           divergence.expected_sign_faults <= kMaxExpectedSignFaults &&
           branching <= kMaxBranchingMeasurements &&
           (lanes >> divergence.sign_dormant_measurements) >= kMinLanesPerDormantBranch;
}

bool ExecutablePlan::frame_trajectories_diverge() const noexcept {
    // Sharing still matches the scalar executor when every shot has its own
    // tableau, but each frame action then walks one tableau per lane, and a
    // large batch of them leaves the cache. On five factory circuits with
    // their noise scaled, a 256-lane batch is level with a full one at two
    // expected sign faults and 3 to 22% faster from three.
    constexpr double kMaxExpectedSignFaults = 3.0;
    // Each such measurement doubles the tableaus a batch holds.
    constexpr uint64_t kMaxBranchingMeasurements = 12;
    const TrajectoryDivergence& divergence = trajectory_divergence_;
    return uses_runtime_frame() && (divergence.expected_sign_faults > kMaxExpectedSignFaults ||
                                    uint64_t{divergence.sign_dormant_measurements} +
                                            divergence.mid_circuit_active_measurements >
                                        kMaxBranchingMeasurements);
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
