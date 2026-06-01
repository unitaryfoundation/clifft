#pragma once

// History sampler: walks a parsed circuit once and samples a single
// noncomputational trajectory under a model.
//
// It owns step 3 of the section 5.1 pipeline (sample initial statuses,
// then per operation sample any attached transition and advance each
// qubit's status). It does not rewrite, compile, or run the SVM, and it
// never consults a measurement outcome -- the status it tracks is the
// pre-SVM-known status of section 5.2.2. Sampling is deterministic in
// the seed.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/history.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/qubit_status.h"

#include <cstdint>
#include <vector>

namespace clifft {

struct HistorySample {
    // Canonical record: initial statuses plus sampled transition outcomes.
    NonComputationalHistory history;
    // Derived convenience: each qubit's status after the full walk.
    std::vector<QubitStatus> final_status;
};

// Sample one trajectory over `circuit` under `model`, deterministic in
// `seed`. Throws std::invalid_argument on a section 4.2 source-context
// violation: a source-dependent transition firing on a
// ComputationalUnknown qubit, naming the operation, qubit, and gate.
HistorySample sample_history(const Circuit& circuit, const NonComputationalModel& model,
                             uint64_t seed);

}  // namespace clifft
