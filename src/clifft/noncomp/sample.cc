#include "clifft/noncomp/sample.h"

#include "clifft/noncomp/trajectory_driver.h"
#include "clifft/util/shot_seed.h"

#include <stdexcept>

namespace clifft {

namespace {

void validate_noncomputational_entry(const Circuit& circuit) {
    if (circuit.num_exp_vals != 0) {
        throw std::invalid_argument(
            "sample_noncomputational: EXP_VAL probes are not supported in noncomputational "
            "sampling");
    }
}

}  // namespace

NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed,
                                               std::optional<uint32_t> max_active_width,
                                               uint32_t threads) {
    validate_noncomputational_entry(circuit);
    return run_trajectory_driver(circuit, model, shots, make_seed_root(shots, seed),
                                 max_active_width, threads);
}

}  // namespace clifft
