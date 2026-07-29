#include "clifft/noncomp/sample.h"

#include "clifft/noncomp/seed.h"
#include "clifft/noncomp/trajectory_driver.h"
#include "clifft/util/xoshiro.h"

#include <array>
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

    SeedRoot root{};
    if (seed.has_value()) {
        root = seed_root_from_seed(*seed);
    } else if (shots > 0) {
        const std::array<uint64_t, 4> w = entropy_seed_words();
        root.w[0] = w[0];
        root.w[1] = w[1];
        root.w[2] = w[2];
        root.w[3] = w[3];
    }

    return run_trajectory_driver(circuit, model, shots, root, max_rank);
}

}  // namespace clifft
