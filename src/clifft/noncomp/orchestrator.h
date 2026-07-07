#pragma once

// Orchestrator: the top-level noncomputational sampling entry point.
//
// sample_noncomputational(circuit, model, shots, seed) validates the
// circuit, resolves the global seed, and hands the run to the exact-mode
// driver (exact_driver.h) -- the one sampling path: the annotated circuit
// compiles once as a shared main line, transition fires resolve at
// runtime against the live state, and the driver returns the user-facing
// records plus the noncomputational sidecar (final statuses and herald
// flags).
//
// Randomness is deterministic in `seed`: per-shot driver and SVM streams
// derive from domain-separated sub-seeds (seed.h). Stochastic classifier
// bits are drawn by the SVM's own stream at the rewriter's READOUT_NOISE
// sites. With no seed, a global seed is drawn from OS entropy and the
// run is non-reproducible.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/qubit_status.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace clifft {

// Aggregated result of a noncomputational sampling run.
struct NonComputationalSample {
    uint32_t shots = 0;
    uint32_t num_qubits = 0;
    uint32_t num_measurements = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;

    // User-facing records, row-major [shot, slot].
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;

    // Sidecar: each qubit's final status per shot, row-major [shot, qubit].
    // Leaked/lost statuses carry their driver-drawn per-shot level.
    // Computational statuses carry no level: fires with computational
    // destinations resolve inside the VM without reaching the driver, so
    // no final level is knowable here.
    std::vector<QubitStatus> final_status;

    // Sidecar: 1 where the classifier sampled the herald (third) symbol for
    // that visible measurement, row-major [shot, slot]. The visible record
    // stays binary -- a heralded slot carries a uniformly drawn bit -- so the
    // record layout and every rec[-k] reference are unchanged; the herald
    // rides here. All zeros for a two-symbol classifier or none at all.
    std::vector<uint8_t> heralds;
};

// Throws std::invalid_argument when the trajectory policy rejects an
// operation, when a measurement on a leaked/lost qubit needs a classifier the
// model does not provide, or when such a classifier column is not a two- or
// three-symbol stochastic column (a third symbol heralds the measurement).
// Reject (substochastic) classifier columns model a heralded abort outcome
// and are not supported by this entry point yet.
// `max_rank` caps the compiled peak rank: compilation fails with the
// first offending circuit line named, before any state is allocated,
// instead of attempting a 2^k allocation. Unlimited when unset.
NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed = std::nullopt,
                                               std::optional<uint32_t> max_rank = std::nullopt);

}  // namespace clifft
