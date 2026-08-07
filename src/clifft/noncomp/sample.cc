#include "clifft/noncomp/sample.h"

#include "clifft/noncomp/seed.h"
#include "clifft/noncomp/trajectory_driver.h"
#include "clifft/util/xoshiro.h"

#include <array>
#include <stdexcept>

namespace clifft {

namespace {

SeedRoot make_seed_root(uint32_t shots, std::optional<uint64_t> seed) {
    if (seed.has_value()) {
        return seed_root_from_seed(*seed);
    }
    SeedRoot root{};
    if (shots > 0) {
        const std::array<uint64_t, 4> words = entropy_seed_words();
        root.w[0] = words[0];
        root.w[1] = words[1];
        root.w[2] = words[2];
        root.w[3] = words[3];
    }
    return root;
}

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
                                               std::optional<uint32_t> max_rank) {
    validate_noncomputational_entry(circuit);
    return run_trajectory_driver(circuit, model, shots, make_seed_root(shots, seed), max_rank);
}

NonComputationalSample sample_noncomputational_experimental(const Circuit& circuit,
                                                            const NonComputationalModel& model,
                                                            uint32_t shots,
                                                            std::optional<uint64_t> seed,
                                                            std::optional<uint32_t> max_rank) {
    validate_noncomputational_entry(circuit);
    return run_sampling_trajectory_driver(circuit, model, shots, make_seed_root(shots, seed),
                                          max_rank);
}

}  // namespace clifft
