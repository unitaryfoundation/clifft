#include "clifft/noncomp/orchestrator.h"

#include "clifft/noncomp/exact_driver.h"
#include "clifft/util/xoshiro.h"

#include <stdexcept>

namespace clifft {

NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed,
                                               std::optional<uint32_t> max_rank) {
    if (circuit.num_exp_vals != 0) {
        throw std::invalid_argument(
            "sample_noncomputational: EXP_VAL probes are not supported in noncomputational "
            "sampling");
    }

    uint64_t global_seed = 0;
    if (seed.has_value()) {
        global_seed = *seed;
    } else if (shots > 0) {
        Xoshiro256PlusPlus entropy;
        entropy.seed_from_entropy();
        global_seed = entropy();
    }

    return sample_noncomputational_exact(circuit, model, shots, global_seed, max_rank);
}

}  // namespace clifft
