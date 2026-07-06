#pragma once

// History sampler: walks a parsed circuit once and samples a single
// noncomputational trajectory under a model.
//
// It performs the sampling pass only: sample initial statuses, walk the
// operations advancing each qubit's status, and sample an outcome at each
// LEVEL_TRANSITION or LOSS annotation -- the only transition consult points.
// A consult is positional: the source is the qubit's status at the
// annotation's place in the circuit (gate hooks are expanded to
// annotations by annotate() before sampling). It does not rewrite,
// compile, or run the SVM, and it never consults a measurement outcome --
// the status it tracks is what is classically known before the simulation
// runs (a Z-basis measurement on an unknown qubit does not pin a value
// here). Sampling is deterministic in the seed.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/history.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/qubit_status.h"
#include "clifft/util/xoshiro.h"

#include <cstdint>
#include <vector>

namespace clifft {

struct HistorySample {
    // Canonical record: initial statuses plus sampled transition outcomes.
    NonComputationalHistory history;
    // Derived convenience: each qubit's status after the full walk.
    std::vector<QubitStatus> final_status;
};

// Draw one qubit's initial level from the model's shared initial-state
// distribution, with the last positive level catching the floating-point
// tail so a draw always resolves to a level the distribution can produce.
// Shared by the AOT history sampler and the exact-mode driver so the two
// paths sample initials identically.
uint8_t draw_initial_level(const NonComputationalModel& model, Xoshiro256PlusPlus& rng);

// Sample one trajectory over `circuit` under `model`, deterministic in
// `seed`. Under UnknownSourcePolicy::Reject (the default), throws
// std::invalid_argument on a source-context violation: a source-dependent
// transition firing on a ComputationalUnknown qubit, naming the operation,
// qubit, and gate. Under EqualizeRates that case is sampled with the
// equalized-rates approximation instead (see policy.h).
HistorySample sample_history(const Circuit& circuit, const NonComputationalModel& model,
                             uint64_t seed);

}  // namespace clifft
