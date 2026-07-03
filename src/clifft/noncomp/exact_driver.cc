#include "clifft/noncomp/exact_driver.h"

#include "clifft/backend/backend.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/annotate.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sampler.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"
#include "clifft/svm/svm_math.h"
#include "clifft/util/xoshiro.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {

namespace {

// Domain tags for the per-shot sub-seeds, disjoint from the AOT
// orchestrator's history/classifier/SVM tags so the two modes never
// correlate. One host stream serves every host-side draw in a shot
// (initial levels, trap destinations, classical follow-ons, herald
// flags), in a deterministic order that never depends on cache state.
constexpr uint64_t kExactHostDomain = 0x11;
constexpr uint64_t kExactSvmDomain = 0x12;

uint64_t derive_exact_seed(uint64_t global, uint64_t shot, uint64_t domain) {
    uint64_t z = global ^ (shot * 0x9E3779B97F4A7C15ULL) ^ (domain * 0xBF58476D1CE4E5B9ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// The transition consulted by one annotation node: a named instrument for
// LEVEL_TRANSITION, a uniform all-lost rate for LOSS.
struct AnnotationChannel {
    const TransitionInstrument* instrument = nullptr;  // null for LOSS
    double loss_p = 0.0;
    uint8_t lost_level = 0;
};

AnnotationChannel resolve_annotation(const AstNode& node, const NonComputationalModel& model,
                                     uint32_t op_index) {
    AnnotationChannel channel;
    if (node.gate == GateType::LOSS) {
        const std::optional<uint8_t> lost = sole_lost_level(model.levels());
        if (!lost.has_value()) {
            throw std::invalid_argument(
                "sample_noncomputational: LOSS at op " + std::to_string(op_index) +
                " requires a level table with exactly one Lost-category level");
        }
        channel.loss_p = node.args.empty() ? 0.0 : node.args[0];
        channel.lost_level = *lost;
        return channel;
    }
    channel.instrument = model.transition_named(node.tag);
    if (channel.instrument == nullptr) {
        throw std::invalid_argument("sample_noncomputational: LEVEL_TRANSITION[" + node.tag +
                                    "] at op " + std::to_string(op_index) +
                                    " does not name a transition in the model");
    }
    return channel;
}

// Draw a destination level from `column` restricted to `mass` (the sum of
// the admitted entries), with the measurement kernels' last-positive
// fallback for the floating-point tail. `admit` filters levels.
template <typename Admit>
uint8_t draw_from_column(const TransitionInstrument& instrument, uint8_t source, double mass,
                         Xoshiro256PlusPlus& rng, Admit&& admit, size_t num_levels) {
    const double u = rng.next_double() * mass;
    double acc = 0.0;
    int last_positive = -1;
    for (uint8_t to = 0; to < num_levels; ++to) {
        if (!admit(to)) {
            continue;
        }
        const double p = instrument.prob(to, source);
        if (p > 0.0) {
            last_positive = to;
        }
        acc += p;
        if (u < acc) {
            return to;
        }
    }
    assert(last_positive >= 0 && "destination draw over an empty column");
    return static_cast<uint8_t>(last_positive);
}

// Walk the annotated circuit's statuses under `events`, drawing and
// appending any classical-source consult beyond the ones already
// recorded. Returns the walk's final statuses (computed from the real,
// uncanonicalized initials). The walk is the single source of truth for
// which consults are classical, so the events stream always matches what
// rewrite_continuation will validate.
std::vector<QubitStatus> extend_classical_outcomes(const Circuit& annotated,
                                                   ExactShotEvents& events,
                                                   const NonComputationalModel& model,
                                                   Xoshiro256PlusPlus& rng) {
    const LevelSet& levels = model.levels();
    std::map<std::pair<uint32_t, uint32_t>, uint8_t> jump_dest;
    for (const ResolvedJump& jump : events.jumps) {
        jump_dest.emplace(std::make_pair(jump.op_index, jump.qubit), jump.destination_level);
    }

    std::vector<QubitStatus> status = events.initial_status;
    size_t cursor = 0;

    for (uint32_t op_index = 0; op_index < annotated.nodes.size(); ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        const GateType gate = node.gate;
        if (gate == GateType::LEVEL_TRANSITION || gate == GateType::LOSS) {
            for (const Target& target : node.targets) {
                const uint32_t qubit = target.value();
                const QubitStatus pre = status[qubit];
                const bool classical =
                    pre.kind() == QubitStatusKind::Leaked || pre.kind() == QubitStatusKind::Lost;
                if (!classical) {
                    const auto jump = jump_dest.find({op_index, qubit});
                    if (jump != jump_dest.end()) {
                        status[qubit] = levels.status_for(jump->second);
                    }
                    continue;
                }
                if (cursor < events.classical_outcomes.size()) {
                    // Already drawn when an earlier trap extended the
                    // stream; replay it.
                    const ClassicalOutcome& outcome = events.classical_outcomes[cursor++];
                    assert(outcome.op_index == op_index && outcome.qubit == qubit &&
                           "recorded classical outcomes drifted from the status walk");
                    if (outcome.jumped) {
                        status[qubit] = levels.status_for(outcome.destination_level);
                    }
                    continue;
                }
                const AnnotationChannel channel = resolve_annotation(node, model, op_index);
                ClassicalOutcome outcome{op_index, qubit, false, 0};
                if (channel.instrument != nullptr) {
                    const uint8_t source = pre.level_id();
                    const double total = channel.instrument->column_sum(source);
                    if (rng.next_double() < total) {
                        outcome.jumped = true;
                        outcome.destination_level = draw_from_column(
                            *channel.instrument, source, total, rng, [](uint8_t) { return true; },
                            levels.size());
                    }
                } else if (pre.kind() != QubitStatusKind::Lost &&
                           rng.next_double() < channel.loss_p) {
                    // LOSS on a leaked (not lost) qubit can still vacate
                    // it; an already-lost qubit records a no-op outcome
                    // without spending a draw.
                    outcome.jumped = true;
                    outcome.destination_level = channel.lost_level;
                }
                ++cursor;
                events.classical_outcomes.push_back(outcome);
                if (outcome.jumped) {
                    status[qubit] = levels.status_for(outcome.destination_level);
                }
            }
            continue;
        }
        // Ordinary operations advance statuses exactly as the rewrite
        // does; drops keep entry statuses, which the stepper handles via
        // the shared policy scan semantics.
        bool drop_op = false;
        for (const QubitOperand& operand : qubit_operands(node)) {
            if (operand_action(gate, status[operand.qubit].kind(), model.policy()) ==
                OperandAction::Drop) {
                drop_op = true;
            }
        }
        for (const QubitOperand& operand : qubit_operands(node)) {
            const QubitStatus pre = status[operand.qubit];
            status[operand.qubit] =
                drop_op ? pre
                        : normal_post_op_status(pre, gate, operand.role, model.policy(), levels);
        }
    }
    return status;
}

// Cache key: the canonicalized status-outcome delta. Computational
// initial levels canonicalize to the zero level -- the rewrite provably
// depends on initial statuses only through their kinds (plus levels for
// noncomputational initials) -- so shots differing only in |1> preloads
// share one module.
std::string cache_key(const ExactShotEvents& events, const LevelSet& levels) {
    std::string key;
    key.reserve(events.initial_status.size() + events.jumps.size() * 9 +
                events.classical_outcomes.size() * 10);
    for (const QubitStatus& s : events.initial_status) {
        key.push_back(static_cast<char>(s.kind()));
        const bool classical =
            s.kind() == QubitStatusKind::Leaked || s.kind() == QubitStatusKind::Lost;
        key.push_back(classical ? static_cast<char>(s.level_id())
                                : static_cast<char>(levels.computational_zero_id()));
    }
    auto push32 = [&key](uint32_t v) {
        for (int b = 0; b < 4; ++b) {
            key.push_back(static_cast<char>((v >> (8 * b)) & 0xFF));
        }
    };
    key.push_back('J');
    for (const ResolvedJump& jump : events.jumps) {
        push32(jump.op_index);
        push32(jump.qubit);
        key.push_back(static_cast<char>(jump.destination_level));
    }
    key.push_back('C');
    for (const ClassicalOutcome& outcome : events.classical_outcomes) {
        push32(outcome.op_index);
        push32(outcome.qubit);
        key.push_back(outcome.jumped ? 1 : 0);
        key.push_back(static_cast<char>(outcome.destination_level));
    }
    return key;
}

ExactShotEvents canonicalize(const ExactShotEvents& events, const LevelSet& levels) {
    ExactShotEvents canonical = events;
    for (QubitStatus& s : canonical.initial_status) {
        if (s.kind() == QubitStatusKind::ComputationalKnown) {
            s = levels.status_for(levels.computational_zero_id());
        }
    }
    return canonical;
}

// One rewritten continuation plus its per-herald-flag compiled modules.
struct ContinuationEntry {
    ContinuationRewrite rw;
    std::map<std::vector<uint8_t>, CompiledModule> modules;
};

void check_max_rank(const CompiledModule& module, std::optional<uint32_t> max_rank) {
    if (!max_rank.has_value() || module.peak_rank <= *max_rank) {
        return;
    }
    // Name the first instruction whose active dimension exceeded the cap,
    // through the compile's per-instruction k history.
    std::string site;
    for (size_t i = 0; i < module.source_map.size(); ++i) {
        if (module.source_map.active_k_at(i) > *max_rank) {
            const auto lines = module.source_map.lines_for(i);
            if (!lines.empty()) {
                site = " (first exceeded at circuit line " + std::to_string(lines[0]) + ")";
            }
            break;
        }
    }
    throw std::invalid_argument(
        "sample_noncomputational: compiled peak rank " + std::to_string(module.peak_rank) +
        " exceeds max_rank " + std::to_string(*max_rank) + site +
        "; consider damping=\"neglect\" for high-rate sites or a larger max_rank");
}

}  // namespace

NonComputationalSample sample_noncomputational_exact(const Circuit& circuit,
                                                     const NonComputationalModel& model,
                                                     uint32_t shots, uint64_t global_seed,
                                                     std::optional<uint32_t> max_rank) {
    NonComputationalSample result;
    result.shots = shots;
    result.num_qubits = circuit.num_qubits;
    result.num_measurements = circuit.num_measurements;
    result.num_detectors = circuit.num_detectors;
    result.num_observables = circuit.num_observables;
    if (shots == 0) {
        return result;
    }

    const LevelSet& levels = model.levels();
    const MeasurementClassifier* classifier = model.classifier();
    const bool ternary = classifier != nullptr && classifier->num_symbols() == 3;

    const Circuit annotated = annotate(circuit, model);
    const InstrumentTraceOptions instrument_options = instrument_trace_options(model);

    std::map<std::string, ContinuationEntry> cache;

    // Level one: the rewrite for `events`, computed once per delta. It is
    // deterministic in the events and consumes no randomness, so a fetch
    // never perturbs sampling.
    auto get_entry = [&](const ExactShotEvents& events, bool force_last) -> ContinuationEntry& {
        const std::string key =
            cache_key(events, levels) + static_cast<char>(force_last ? 'F' : 'f');
        auto [it, inserted] = cache.try_emplace(key);
        ContinuationEntry& entry = it->second;
        if (inserted) {
            entry.rw =
                rewrite_continuation(annotated, canonicalize(events, levels), force_last, model);
        }
        return entry;
    };

    // Level two: the compiled module for one herald-flag assignment (one
    // flag per classified slot, in slot order). Every host draw feeding
    // this call happens before it, so cache hits and misses sample
    // identically.
    auto get_module = [&](ContinuationEntry& entry, const std::vector<uint8_t>& flags,
                          const CompiledModule* executed_prefix_module,
                          uint32_t prefix_end) -> CompiledModule* {
        auto [mit, module_inserted] = entry.modules.try_emplace(flags);
        if (module_inserted) {
            Circuit patched = entry.rw.circuit;
            assert(flags.size() == entry.rw.classified_measurements.size());
            for (size_t i = 0; i < flags.size(); ++i) {
                if (flags[i] != 0) {
                    const ClassifiedMeasurement& m = entry.rw.classified_measurements[i];
                    assert(m.noise_node != SIZE_MAX);
                    patched.nodes[m.noise_node].args[0] = 0.5;
                }
            }
            HirModule hir = trace(patched, &instrument_options);
            default_hir_pass_manager().run(hir);
            CompiledModule module = lower(hir);
            default_bytecode_pass_manager().run(module);
            check_max_rank(module, max_rank);
#ifndef NDEBUG
            // Re-entry contract: the continuation's prefix must be
            // bit-identical to the code the shot already executed. A
            // determinism regression shows up here, loudly, rather than
            // as state corruption.
            if (executed_prefix_module != nullptr) {
                assert(prefix_end <= module.bytecode.size() &&
                       prefix_end <= executed_prefix_module->bytecode.size());
                for (uint32_t i = 0; i < prefix_end; ++i) {
                    assert(std::memcmp(&module.bytecode[i], &executed_prefix_module->bytecode[i],
                                       sizeof(Instruction)) == 0 &&
                           "continuation prefix diverged from the executed module");
                }
            }
#else
            (void)executed_prefix_module;
            (void)prefix_end;
#endif
            mit->second = std::move(module);
        }
        return &mit->second;
    };

    // The shared main line: the continuation of "nothing happened".
    ExactShotEvents no_events;
    no_events.initial_status.assign(circuit.num_qubits,
                                    levels.status_for(levels.computational_zero_id()));
    ContinuationEntry& main_entry = get_entry(no_events, false);
    CompiledModule* main_module = get_module(
        main_entry, std::vector<uint8_t>(main_entry.rw.classified_measurements.size(), 0), nullptr,
        0);

    // The state is reused across shots (growth from trap continuations
    // amortizes to the chain maximum); a starting module that outgrows it
    // -- a rare noncomputational-initial shot -- rebuilds it instead,
    // since grow_for_continuation is trap-gated by design.
    auto make_state = [&](const CompiledModule& module) {
        return SchrodingerState(StateConfig{.peak_rank = module.peak_rank,
                                            .num_measurements = module.total_meas_slots,
                                            .num_qubits = module.num_qubits,
                                            .num_detectors = module.num_detectors,
                                            .num_observables = module.num_observables,
                                            .seed = 0});
    };
    SchrodingerState state = make_state(*main_module);
    uint32_t state_rank = main_module->peak_rank;
    uint32_t state_slots = main_module->total_meas_slots;

    for (uint32_t shot = 0; shot < shots; ++shot) {
        Xoshiro256PlusPlus host_rng(derive_exact_seed(global_seed, shot, kExactHostDomain));

        ExactShotEvents events;
        events.initial_status.reserve(circuit.num_qubits);
        bool any_noncomp_initial = false;
        for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
            const uint8_t level = draw_initial_level(model, host_rng);
            events.initial_status.push_back(levels.status_for(level));
            const QubitStatusKind kind = events.initial_status.back().kind();
            any_noncomp_initial |= kind == QubitStatusKind::Leaked || kind == QubitStatusKind::Lost;
        }

        // Herald flags accumulate per visible slot across the shot's
        // trap chain: a slot's flag is drawn once, when its classified
        // measurement first appears in a continuation.
        std::map<uint32_t, uint8_t> herald_flags;
        auto flags_for = [&](const ContinuationRewrite& rw) {
            std::vector<uint8_t> flags;
            flags.reserve(rw.classified_measurements.size());
            for (const ClassifiedMeasurement& m : rw.classified_measurements) {
                auto [it, inserted] = herald_flags.try_emplace(m.slot, 0);
                if (inserted && ternary) {
                    const double p_herald = classifier->prob(2, m.level);
                    it->second = host_rng.next_double() < p_herald ? 1 : 0;
                }
                flags.push_back(it->second);
            }
            return flags;
        };

        // Resolve the shot's starting module. A noncomputational initial
        // (rare) compiles its own continuation-from-the-top; classical
        // consults over the whole circuit are pre-drawn for it. The
        // rewrite fetch consumes no randomness, so drawing the herald
        // flags after it keeps every host draw ahead of module lookup.
        ContinuationEntry* entry = &main_entry;
        CompiledModule* module = main_module;
        std::vector<QubitStatus> final_status =
            extend_classical_outcomes(annotated, events, model, host_rng);
        if (any_noncomp_initial) {
            entry = &get_entry(events, false);
            module = get_module(*entry, flags_for(entry->rw), nullptr, 0);
        }

        if (shot > 0) {
            state.reset();
        }
        if (module->peak_rank > state_rank || module->total_meas_slots > state_slots) {
            state_rank = std::max(state_rank, module->peak_rank);
            state_slots = std::max(state_slots, module->total_meas_slots);
            state = make_state(*module);
        }
        state.reseed(derive_exact_seed(global_seed, shot, kExactSvmDomain));
        if (state.meas_record.size() < module->total_meas_slots) {
            state.meas_record.resize(module->total_meas_slots, 0);
        }
        // Known |1> initial levels are an X at time zero: a Pauli, so a
        // per-shot frame preload rather than a distinct module.
        for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
            const QubitStatus& s = events.initial_status[q];
            if (s.kind() == QubitStatusKind::ComputationalKnown &&
                s.level_id() == levels.computational_one_id()) {
                bit_set(state.p_x, q, true);
            }
        }
        state.next_noise_idx = 0;
        state.draw_next_noise(module->constant_pool.noise_hazards);
        execute(*module, state);

        while (state.pending_trap.has_value()) {
            const SchrodingerState::InstrumentTrap trap = *state.pending_trap;
            if (trap.site_id >= entry->rw.site_targets.size()) {
                throw std::logic_error(
                    "sample_noncomputational: trap site id is out of range of the executing "
                    "module's site table");
            }
            const auto [op_index, qubit] = entry->rw.site_targets[trap.site_id];
            const AstNode& node = annotated.nodes[op_index];
            const AnnotationChannel channel = resolve_annotation(node, model, op_index);

            // The correlated continuation for a neglect-form trap (the
            // forced-to-source trace-out) is not wired yet; the guard
            // keeps the decorrelated behavior from running silently.
            if (trap.destination_pending) {
                throw std::invalid_argument(
                    "sample_noncomputational: a neglect-form site fired; the correlated "
                    "continuation for damping=\"neglect\" is not wired yet -- use "
                    "damping=\"exact\" (the default)");
            }

            // Destination: the class is already drawn as leaked/lost, so
            // only the level within the trap remainder remains. (A
            // pending destination would draw over the full column.)
            uint8_t dest;
            if (channel.instrument == nullptr) {
                dest = channel.lost_level;
            } else {
                double remainder = 0.0;
                for (uint8_t to = 0; to < levels.size(); ++to) {
                    if (levels.at(to).category != LevelCategory::Computational) {
                        remainder += channel.instrument->prob(to, trap.source);
                    }
                }
                dest = draw_from_column(
                    *channel.instrument, trap.source, remainder, host_rng,
                    [&](uint8_t to) {
                        return levels.at(to).category != LevelCategory::Computational;
                    },
                    levels.size());
            }

            events.jumps.push_back({op_index, qubit, dest});
            final_status = extend_classical_outcomes(annotated, events, model, host_rng);

            const uint32_t prefix_end = module->instrument_offsets[trap.site_id] + 1;
            ContinuationEntry& next_entry = get_entry(events, false);
            CompiledModule* next_module =
                get_module(next_entry, flags_for(next_entry.rw), module, prefix_end);

            entry = &next_entry;
            module = next_module;
            resume(*module, state, module->instrument_offsets[trap.site_id] + 1);
        }

        // Emit the visible records. The noncomputational path never sets
        // expected_observables, so no reference normalization applies.
        result.measurements.insert(result.measurements.end(), state.meas_record.begin(),
                                   state.meas_record.begin() + circuit.num_measurements);
        result.detectors.insert(result.detectors.end(), state.det_record.begin(),
                                state.det_record.end());
        result.observables.insert(result.observables.end(), state.obs_record.begin(),
                                  state.obs_record.end());
        result.final_status.insert(result.final_status.end(), final_status.begin(),
                                   final_status.end());
        std::vector<uint8_t> shot_heralds(circuit.num_measurements, 0);
        for (const auto& [slot, flag] : herald_flags) {
            shot_heralds[slot] = flag;
        }
        result.heralds.insert(result.heralds.end(), shot_heralds.begin(), shot_heralds.end());
    }

    return result;
}

}  // namespace clifft
