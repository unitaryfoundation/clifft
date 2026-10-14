#include "clifft/noncomp/sample.h"

#include "clifft/noncomp/preparation.h"

namespace clifft {

NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed,
                                               std::optional<uint32_t> max_active_width,
                                               uint32_t threads) {
    const PreparedNonComputationalRequest request(
        circuit, model, {.final_status = true, .max_active_width = max_active_width});
    return request.sample(shots, seed, threads);
}

}  // namespace clifft
