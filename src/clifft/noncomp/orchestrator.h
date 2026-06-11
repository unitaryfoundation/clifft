#pragma once

// Orchestrator: the top-level noncomputational sampling entry point.
//
// sample_noncomputational(circuit, model, shots, seed) runs the full per-shot
// pipeline and returns the user-facing records plus a noncomputational
// sidecar. For each shot it:
//   1. samples a structural history (initial statuses + transition outcomes);
//   2. rewrites the circuit for that history (X-prep, trace-out R, policy);
//   3. injects the model classifier's outcome for each measurement on a
//      leaked/lost qubit -- swapping M for MPAD(bit), and a measure-and-reset
//      for MPAD(bit) plus the matching reset -- so the forced bit reaches the
//      SVM's detector/observable evaluation, not just postprocessing;
//   4. compiles the result through the ordinary pipeline and samples one shot.
//
// Randomness is deterministic in `seed`: each shot draws domain-separated
// sub-seeds for the history sampler, the classifier, and the SVM, so the
// three streams never coincide. With no seed, a global seed is drawn from OS
// entropy and the run is non-reproducible.

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
NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed = std::nullopt);

}  // namespace clifft
