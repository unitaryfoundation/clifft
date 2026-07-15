#include "clifft/noncomp/sample.h"

#include "clifft/noncomp/exact_driver.h"
#include "clifft/noncomp/seed.h"
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

    SeedRoot root{};
    if (seed.has_value()) {
        root = seed_root_from_seed(*seed);
    } else if (shots > 0) {
        Xoshiro256PlusPlus e;
        e.seed_from_entropy();
        root.w[0] = e();
        root.w[1] = e();
        root.w[2] = e();
        root.w[3] = e();
    }

    return run_exact_driver(circuit, model, shots, root, max_rank);
}

}  // namespace clifft
