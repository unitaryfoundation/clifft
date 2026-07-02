#include "clifft/noncomp/orchestrator.h"

#include "clifft/backend/backend.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/noncomp/annotate.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sampler.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"
#include "clifft/util/xoshiro.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace clifft {

namespace {

// Distinct domain tags so per-shot sub-seeds for the three RNG consumers
// never coincide (seeding all from the same value would correlate them).
constexpr uint64_t kHistoryDomain = 0x1;
constexpr uint64_t kClassifierDomain = 0x2;
constexpr uint64_t kSvmDomain = 0x3;

// SplitMix64 finalizer over a mix of the global seed, shot, and domain.
uint64_t derive_seed(uint64_t global, uint64_t shot, uint64_t domain) {
    uint64_t z = global ^ (shot * 0x9E3779B97F4A7C15ULL) ^ (domain * 0xBF58476D1CE4E5B9ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Herald pass for a three-symbol classifier: draw each classified slot's
// herald flag and re-point the heralded slots' record-flip probability at one
// half -- a heralded outcome carries no preferred computational value, so a
// uniform bit keeps downstream detector statistics unbiased rather than
// silently pinning them. Non-heralded slots keep the not-heralded conditional
// probability the rewriter emitted. Herald flags land in `heralds` (pre-sized
// to the visible measurement count, zero-filled).
void apply_heralds(RewriteResult& rw, const MeasurementClassifier& classifier,
                   Xoshiro256PlusPlus& rng, std::vector<uint8_t>& heralds) {
    for (const ClassifiedMeasurement& m : rw.classified_measurements) {
        const double p_herald = classifier.prob(2, m.level);
        if (rng.next_double() < p_herald) {
            heralds[m.slot] = 1;
            // The rewriter always emits a READOUT_NOISE node for a ternary
            // classifier's slots, so there is a node to patch.
            assert(m.noise_node != SIZE_MAX);
            rw.circuit.nodes[m.noise_node].args[0] = 0.5;
        }
    }
}

CompiledModule compile_circuit(const Circuit& circuit) {
    HirModule hir = trace(circuit);
    default_hir_pass_manager().run(hir);
    CompiledModule program = lower(hir);
    default_bytecode_pass_manager().run(program);
    return program;
}

}  // namespace

NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed) {
    NonComputationalSample result;
    result.shots = shots;
    result.num_qubits = circuit.num_qubits;
    // The visible record layout is invariant under rewriting, so the record
    // widths come straight from the circuit and hold even for zero shots.
    result.num_measurements = circuit.num_measurements;
    result.num_detectors = circuit.num_detectors;
    result.num_observables = circuit.num_observables;

    if (circuit.num_exp_vals != 0) {
        throw std::invalid_argument(
            "sample_noncomputational: EXP_VAL probes are not supported in noncomputational "
            "sampling");
    }
    if (shots == 0) {
        return result;
    }

    uint64_t global_seed;
    if (seed.has_value()) {
        global_seed = *seed;
    } else {
        Xoshiro256PlusPlus entropy;
        entropy.seed_from_entropy();
        global_seed = entropy();
    }

    const MeasurementClassifier* classifier = model.classifier();

    // Expand the model's gate hooks into explicit TRANSITION annotations
    // once: the per-shot layers below consume only annotations.
    const Circuit annotated = annotate(circuit, model);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        HistorySample hs =
            sample_history(annotated, model, derive_seed(global_seed, shot, kHistoryDomain));
        RewriteResult rw = rewrite(annotated, hs.history, model);
        std::vector<uint8_t> shot_heralds(circuit.num_measurements, 0);
        if (classifier != nullptr && classifier->num_symbols() == 3 &&
            !rw.classified_measurements.empty()) {
            Xoshiro256PlusPlus classifier_rng(derive_seed(global_seed, shot, kClassifierDomain));
            apply_heralds(rw, *classifier, classifier_rng, shot_heralds);
        }

        CompiledModule program = compile_circuit(rw.circuit);
        SampleResult sr = sample(program, 1, derive_seed(global_seed, shot, kSvmDomain));

        result.measurements.insert(result.measurements.end(), sr.measurements.begin(),
                                   sr.measurements.end());
        result.detectors.insert(result.detectors.end(), sr.detectors.begin(), sr.detectors.end());
        result.observables.insert(result.observables.end(), sr.observables.begin(),
                                  sr.observables.end());
        result.final_status.insert(result.final_status.end(), hs.final_status.begin(),
                                   hs.final_status.end());
        result.heralds.insert(result.heralds.end(), shot_heralds.begin(), shot_heralds.end());
    }

    return result;
}

}  // namespace clifft
