#include "clifft/sampling/executable_plan.h"

#include "clifft/sampling/executable_plan_builder.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace clifft::sampling {

ExecutablePlan::ExecutablePlan(const SamplingPlan& plan) : ExecutablePlan(plan, BuilderTag{}) {
    ExecutablePlanBuilder::build(*this, plan);
}

ExecutablePlan::ExecutablePlan(const SamplingPlan& plan, BuilderTag)
    : num_qubits_(plan.num_qubits),
      initial_active_width_(plan.initial_active_width),
      max_active_width_(plan.max_active_width),
      num_visible_records_(plan.num_visible_records),
      num_hidden_records_(plan.num_hidden_records),
      num_detectors_(plan.num_detectors),
      num_observables_(plan.num_observables),
      num_exp_vals_(plan.num_exp_vals),
      has_postselection_(plan.has_postselection),
      global_weight_(plan.global_weight),
      final_tableau_(plan.final_tableau),
      instrument_distributions_(plan.instrument_distributions) {}

size_t ExecutablePlan::num_new_x_instrument_activations() const {
    return static_cast<size_t>(
        std::count_if(actions_.begin(), actions_.end(), [](const Action& action) {
            const auto* instrument = std::get_if<ExecuteInstrument>(&action);
            return instrument != nullptr && instrument->activates_new_x();
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
