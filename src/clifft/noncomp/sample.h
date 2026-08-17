#pragma once

// Top-level noncomputational sampling entry point.
//
// sample_noncomputational(circuit, model, shots, seed) validates the circuit,
// resolves transitions against the live quantum state, and returns the
// ordinary sample records plus final-status and herald sidecars.
//
// Randomness is deterministic in `seed`: per-shot driver and executor streams
// derive from domain-separated sub-seeds (seed.h). Stochastic classifier
// bits are drawn by the executor's own stream at the rewriter's READOUT_NOISE
// sites. With no seed, a global seed is drawn from OS entropy and the
// run is non-reproducible.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"

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
    // LeakG, LeakE, and Lost are definite levels. Computational deliberately
    // does not distinguish G from E; that state remains in the active state.
    std::vector<QubitStatus> final_status;

    // Sidecar: 1 where the classifier sampled the herald (third) symbol for
    // that visible measurement, row-major [shot, slot]. The visible record
    // stays binary -- a heralded slot carries a uniformly drawn bit -- so the
    // record layout and every rec[-k] reference are unchanged; the herald
    // rides here. All zeros when the classifier has two symbols or the model
    // has no classifier.
    std::vector<uint8_t> heralds;
};

// Throws std::invalid_argument when the trajectory policy rejects an
// operation, or when a measurement on a leaked/lost qubit needs a
// classifier the model does not provide. (Classifier shape -- two or
// three symbols, stochastic columns -- is the model's own construction
// contract, enforced before a circuit ever meets it.)
// `max_rank` caps the compiled peak rank. Compilation reports the first
// offending active width before allocating or growing the executor state.
// Unlimited when unset.
NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed = std::nullopt,
                                               std::optional<uint32_t> max_rank = std::nullopt);

}  // namespace clifft
