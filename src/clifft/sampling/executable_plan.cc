#include "clifft/sampling/executable_plan.h"

#include "clifft/sampling/executable_plan_builder.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft::sampling {

PreparedFusedRotationExecution::PreparedFusedRotationExecution(PreparedFusedRotation rotation,
                                                               ExecutorBackend backend)
    : rotation_(std::move(rotation)) {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    switch (backend) {
        case ExecutorBackend::Scalar:
            break;
        case ExecutorBackend::Avx2:
            sidecar_ = prepare_fused_rotation_avx2_sidecar(rotation_);
            break;
        case ExecutorBackend::Avx512:
            sidecar_ = prepare_fused_rotation_avx512_sidecar(rotation_);
            break;
    }
#else
    (void)backend;
#endif
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
      instrument_distributions_(plan.instrument_distributions) {
    // Keep construction-only lowering state out of the immutable executable.
    ExecutablePlanBuilder::build(*this, plan);
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
