#pragma once

// Orchestrator: the top-level noncomputational sampling entry point.
//
// sample_noncomputational(circuit, model, shots, seed) expands the model's
// gate hooks into explicit LEVEL_TRANSITION annotations once, then runs the full
// per-shot pipeline and returns the user-facing records plus a
// noncomputational sidecar. For each shot it:
//   1. samples a structural history (initial statuses + an outcome per
//      LEVEL_TRANSITION/LOSS annotation target);
//   2. rewrites the circuit for that history (X-prep, trace-out R, policy,
//      and the classifier record write for each measurement on a leaked/lost
//      qubit: MPAD plus a READOUT_NOISE for a stochastic column, so the bit
//      is drawn at sample time inside the VM and reaches its
//      detector/observable evaluation, not just postprocessing);
//   3. for a three-symbol classifier, draws each classified slot's herald
//      flag and re-points heralded slots' record flip at one half;
//   4. compiles the result through the ordinary pipeline and samples one shot.
//
// Randomness is deterministic in `seed`: each shot draws domain-separated
// sub-seeds for the history sampler, the herald pass, and the SVM, so the
// three streams never coincide. Stochastic classifier bits are drawn by the
// SVM's own stream at the injected READOUT_NOISE sites. With no seed, a
// global seed is drawn from OS entropy and the run is non-reproducible.

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
// `max_rank` caps the compiled peak rank in exact mode
// compilation: it fails with the first
// offending circuit line named, before any state is allocated, instead
// of attempting a 2^k allocation. Unlimited when unset; ignored by the
// AOT policies, whose per-shot modules never exceed the annotated
// circuit's own rank.
NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed = std::nullopt,
                                               std::optional<uint32_t> max_rank = std::nullopt);

}  // namespace clifft
