#include "clifft/sampling/planner.h"

#include "clifft/sampling/planner_frame.h"
#include "clifft/util/hir_introspection.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clifft::sampling {

namespace {

using Pauli = internal::PlannerPauli;
using Tableau = internal::PlannerTableau;
using internal::CoordinateFrame;
using internal::SymbolicPauliFrame;

Pauli pauli_from_hir(const HirModule& hir, const HeisenbergOp& op) {
    Pauli result(hir.num_qubits);
    result.mut_x().xor_with(hir.destab_mask(op));
    result.mut_z().xor_with(hir.stab_mask(op));
    result.set_sign(false);
    return result;
}

Pauli noise_pauli_from_hir(const HirModule& hir, PauliMaskHandle handle) {
    Pauli result(hir.num_qubits);
    const PauliMaskView mask = hir.noise_channel_masks.at(handle);
    result.mut_x().xor_with(mask.x());
    result.mut_z().xor_with(mask.z());
    result.set_sign(false);
    return result;
}

Pauli pauli_from_mask(const HirModule& hir, PauliMaskHandle handle) {
    Pauli result(hir.num_qubits);
    const PauliMaskView mask = hir.pauli_masks.at(handle);
    result.mut_x().xor_with(mask.x());
    result.mut_z().xor_with(mask.z());
    result.set_sign(false);
    return result;
}

Pauli single_x(uint32_t num_qubits, uint32_t q) {
    Pauli result(num_qubits);
    result.set_pauli(q, true, false);
    return result;
}

struct ResolvedPauli {
    Pauli body;
    AffineBool sign;
};

ResolvedPauli resolve_pauli(const Pauli& initial_body, const AffineBool& initial_sign,
                            CoordinateFrame& coordinates, SymbolicPauliFrame& symbolic_frame) {
    AffineBool sign = initial_sign;
    sign ^= symbolic_frame.sign_for(initial_body);
    Pauli body = coordinates.to_current(initial_body);
    sign ^= body.sign();
    body.set_sign(false);
    return ResolvedPauli{std::move(body), std::move(sign)};
}

std::optional<uint32_t> first_x_at_or_above(const Pauli& pauli, uint32_t begin) {
    for (uint32_t q = begin; q < pauli.num_qubits(); ++q) {
        if (pauli.x().bit_get(q)) {
            return q;
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> last_x_below(const Pauli& pauli, uint32_t end) {
    for (uint32_t q = end; q > 0; --q) {
        if (pauli.x().bit_get(q - 1)) {
            return q - 1;
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> first_z_below(const Pauli& pauli, uint32_t end) {
    for (uint32_t q = 0; q < end; ++q) {
        if (pauli.z().bit_get(q)) {
            return q;
        }
    }
    return std::nullopt;
}

ActivePauli active_projection(const Pauli& pauli, uint32_t active_width) {
    ActivePauli result;
    for (uint32_t q = 0; q < active_width; ++q) {
        result.x |= static_cast<uint64_t>(pauli.x().bit_get(q)) << q;
        result.z |= static_cast<uint64_t>(pauli.z().bit_get(q)) << q;
    }
    return result;
}

SymbolId reserve_symbol(SamplingPlan& plan) {
    const SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
    plan.symbols.emplace_back();
    return symbol;
}

std::span<const uint32_t> source_lines_for(const HirModule& hir, size_t operation_index) {
    assert(hir.source_map.size() == hir.ops.size() &&
           "retained HIR provenance must remain parallel to operations");
    return hir.source_map[operation_index];
}

void append_action(SamplingPlan& plan, PlannedAction action,
                   std::span<const uint32_t> source_lines) {
    plan.actions.push_back(std::move(action));
    if (plan.source_map.has_value()) {
        plan.source_map->append(source_lines);
    }
}

void define_symbol(SamplingPlan& plan, SymbolId symbol, SymbolKind kind, uint32_t action) {
    SymbolInfo& info = plan.symbols.at(index(symbol));
    if (info.kind != SymbolKind::Unused || info.defining_action.has_value() ||
        info.noise_site.has_value()) {
        throw std::logic_error("sampling planner attempted to redefine a reserved symbol");
    }
    info = SymbolInfo{kind, action, std::nullopt};
}

bool option_bit(std::span<const uint8_t> values, uint32_t index) {
    return index < values.size() && values[index] != 0;
}

struct PlanningRequirements {
    uint32_t symbol_count = 0;
    bool supports_final_state_queries = true;
};

bool operation_supports_final_state_queries(OpType type) {
    switch (type) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::EXP_VAL:
            return true;
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
        case OpType::NOISE:
        case OpType::READOUT_NOISE:
        case OpType::INSTRUMENT:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::NUM_OP_TYPES:
            return false;
    }
    return false;
}

PlanningRequirements inspect_planning_requirements(const HirModule& hir) {
    PlanningRequirements result;
    auto add = [&](size_t amount) {
        if (amount > std::numeric_limits<uint32_t>::max() - result.symbol_count) {
            throw std::length_error("sampling planner symbol count exceeds uint32 range");
        }
        result.symbol_count += static_cast<uint32_t>(amount);
    };

    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const HeisenbergOp& op = hir.ops[i];
        result.supports_final_state_queries &= operation_supports_final_state_queries(op.op_type());
        switch (op.op_type()) {
            case OpType::MEASURE:
            case OpType::READOUT_NOISE:
            case OpType::INSTRUMENT:
                add(1);
                break;
            case OpType::NOISE: {
                const uint32_t site_index = static_cast<uint32_t>(op.noise_site_idx());
                if (site_index >= hir.noise_sites.size()) {
                    throw std::invalid_argument("sampling planner noise site is out of range");
                }
                const NoiseSite& site = hir.noise_sites[site_index];
                add(std::ranges::count_if(site.channels, [](const NoiseChannel& channel) {
                    return channel.prob != 0.0;
                }));
                break;
            }
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
            case OpType::EXP_VAL:
                break;
            case OpType::CONDITIONAL_PAULI:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
                break;
            case OpType::NUM_OP_TYPES:
                throw std::invalid_argument("sampling planner does not support HIR operation " +
                                            op_type_to_str(op.op_type()) + " at index " +
                                            std::to_string(i));
        }
    }
    return result;
}

void initialize_site_metadata(const HirModule& hir, SamplingPlan& plan) {
    plan.presampled_noise_sites.resize(hir.noise_sites.size());
    for (uint32_t site = 0; site < hir.noise_sites.size(); ++site) {
        plan.presampled_noise_sites[site].site = NoiseSiteId{site};
        plan.presampled_noise_sites[site].total_probability =
            hir.noise_sites[site].total_probability;
    }
    plan.instrument_distributions.reserve(hir.instrument_sites.size());
    for (uint32_t site = 0; site < hir.instrument_sites.size(); ++site) {
        const InstrumentProbabilities& probabilities = hir.instrument_sites[site].probabilities;
        InstrumentDistribution distribution;
        distribution.site = InstrumentSiteId{site};
        for (uint8_t source = 0; source < 2; ++source) {
            distribution.p_fire[source] = probabilities.p_fire[source];
            for (uint8_t destination = 0; destination < 2; ++destination) {
                distribution.p_computational_dest[source][destination] =
                    probabilities.p_computational_dest[source][destination];
            }
        }
        plan.instrument_distributions.push_back(distribution);
    }
}

bool process_rotation(const Pauli& body, double half_turns, const AffineBool& sign,
                      SamplingPlan& plan, uint32_t& active_width, CoordinateFrame& coordinates,
                      SymbolicPauliFrame& symbolic_frame, std::span<const uint32_t> source_lines) {
    ResolvedPauli resolved = resolve_pauli(body, sign, coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);
    if (!dormant_pivot.has_value()) {
        const ActivePauli active = active_projection(resolved.body, active_width);
        if (active.is_identity()) {
            return false;
        }
        append_action(
            plan,
            PlannedAction{active_width, active_width,
                          RotateActivePauli{active, half_turns, std::move(resolved.sign)}},
            source_lines);
        return false;
    }

    if (active_width + 1 >= kDenseActiveWidthLimit) {
        throw std::overflow_error(
            "sampling planner active width would reach " + std::to_string(active_width + 1) +
            ", but the dense-state limit is " + std::to_string(kDenseActiveWidthLimit));
    }

    coordinates.promote_dormant(resolved.body, active_width, *dormant_pivot);
    append_action(plan,
                  PlannedAction{active_width, active_width + 1,
                                PromoteDormantRotation{half_turns, std::move(resolved.sign)}},
                  source_lines);
    ++active_width;
    plan.peak_active_width = std::max(plan.peak_active_width, active_width);
    return true;
}

AffineBool process_measurement(const Pauli& body, const AffineBool& sign, RecordSlot record,
                               SymbolId branch, SamplingPlan& plan, uint32_t& active_width,
                               CoordinateFrame& coordinates, SymbolicPauliFrame& symbolic_frame,
                               std::span<const uint32_t> source_lines) {
    ResolvedPauli resolved = resolve_pauli(body, sign, coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);
    if (dormant_pivot.has_value()) {
        coordinates.measure_dormant(resolved.body, *dormant_pivot);

        const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
        define_symbol(plan, branch, SymbolKind::Branch, action_index);
        Pauli correction = coordinates.to_initial(single_x(plan.num_qubits, *dormant_pivot));
        correction.set_sign(false);
        symbolic_frame.apply(correction, AffineBool::symbol(branch));
        const AffineBool outcome = resolved.sign ^ AffineBool::symbol(branch);
        append_action(plan,
                      PlannedAction{active_width, active_width,
                                    MeasureDormantRandom{*dormant_pivot, branch, outcome, record}},
                      source_lines);
        return outcome;
    }

    const ActivePauli active = active_projection(resolved.body, active_width);
    if (active.is_identity()) {
        append_action(
            plan, PlannedAction{active_width, active_width, RecordClassical{resolved.sign, record}},
            source_lines);
        return resolved.sign;
    }

    std::optional<uint32_t> pivot = last_x_below(resolved.body, active_width);
    if (!pivot.has_value()) {
        pivot = first_z_below(resolved.body, active_width);
    }
    if (!pivot.has_value()) {
        throw std::logic_error("sampling planner could not select an active measurement pivot");
    }

    Pauli active_body(resolved.body.num_qubits());
    for (uint32_t q = 0; q < active_width; ++q) {
        active_body.set_pauli(q, resolved.body.x().bit_get(q), resolved.body.z().bit_get(q));
    }
    active_body.set_sign(false);
    coordinates.measure_active(active_body, active_width, *pivot);

    const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
    define_symbol(plan, branch, SymbolKind::Branch, action_index);
    Pauli correction = coordinates.to_initial(single_x(plan.num_qubits, active_width - 1));
    correction.set_sign(false);
    symbolic_frame.apply(correction, AffineBool::symbol(branch));
    const AffineBool outcome = resolved.sign ^ AffineBool::symbol(branch);
    append_action(plan,
                  PlannedAction{active_width, active_width - 1,
                                MeasureActivePauli{active, *pivot, branch, outcome, record}},
                  source_lines);
    --active_width;
    return outcome;
}

void process_instrument(const HirModule& hir, const HeisenbergOp& op, uint32_t next_noise_site,
                        SamplingPlan& plan, uint32_t& active_width, CoordinateFrame& coordinates,
                        SymbolicPauliFrame& symbolic_frame,
                        std::span<const uint32_t> source_lines) {
    const InstrumentSiteId site{static_cast<uint32_t>(op.instrument_site_idx())};
    const InstrumentSite& hir_site = hir.instrument_sites.at(index(site));
    const SymbolId destination_flip_symbol = reserve_symbol(plan);
    const uint32_t symbol_prefix_size = static_cast<uint32_t>(plan.symbols.size());
    const Pauli body = pauli_from_hir(hir, op);
    const Pauli destination_flip = pauli_from_mask(hir, hir_site.destination_flip_mask);
    const InstrumentDistribution& distribution = plan.instrument_distributions.at(index(site));
    ResolvedPauli resolved =
        resolve_pauli(body, AffineBool(hir.sign(op)), coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);

    InstrumentMode mode = InstrumentMode::Classical;
    ActivePauli source;
    AffineBool sign = std::move(resolved.sign);
    uint32_t active_after = active_width;

    if (dormant_pivot.has_value()) {
        // Equal no-fire factors normalize away. Otherwise exact damping needs
        // the dormant-coherent source represented in the dense state.
        if (hir.neglect_instrument_damping || distribution.p_fire[0] == distribution.p_fire[1]) {
            mode = InstrumentMode::DormantTrap;
            sign = AffineBool{};
        } else {
            if (active_width + 1 >= kDenseActiveWidthLimit) {
                throw std::overflow_error("sampling planner active width would reach " +
                                          std::to_string(active_width + 1) +
                                          ", but the dense-state limit is " +
                                          std::to_string(kDenseActiveWidthLimit));
            }
            coordinates.promote_dormant(resolved.body, active_width, *dormant_pivot);
            ++active_after;
            mode = InstrumentMode::Activate;
            // The promotion frame installs this observable as the next X
            // generator, so rediscovering it through a local inverse is wasteful.
            source.x = uint64_t{1} << active_width;
        }
    } else {
        source = active_projection(resolved.body, active_width);
        // An identity projection means the source is already determined by
        // the symbolic sign; otherwise its active Pauli needs coefficient work.
        mode = source.is_identity() ? InstrumentMode::Classical : InstrumentMode::Active;
    }

    const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
    std::optional<SymbolId> destination_symbol;
    if (mode != InstrumentMode::DormantTrap) {
        // Only an in-line computational destination needs the reserved flip;
        // a trapped continuation resolves its destination instead.
        destination_symbol = destination_flip_symbol;
        define_symbol(plan, destination_flip_symbol, SymbolKind::Instrument, action_index);
    }
    append_action(plan,
                  PlannedAction{active_width, active_after,
                                ApplyInstrument{site, mode, source, sign, destination_symbol}},
                  source_lines);

    if (destination_symbol.has_value()) {
        symbolic_frame.apply(destination_flip, AffineBool::symbol(*destination_symbol));
    }
    active_width = active_after;
    plan.peak_active_width = std::max(plan.peak_active_width, active_width);
    // A trapped shot resumes here so it cannot execute the instrument twice.
    append_action(plan,
                  PlannedAction{active_width, active_width,
                                InstrumentBoundary{site, next_noise_site, symbol_prefix_size}},
                  source_lines);
}

}  // namespace

SamplingPlan plan_sampling(const HirModule& hir, SamplingPlanOptions options) {
    SamplingPlan plan;
    plan.num_qubits = hir.num_qubits;
    plan.num_visible_records = hir.num_measurements;
    plan.num_hidden_records = hir.num_hidden_measurements;
    plan.num_noise_sites = static_cast<uint32_t>(hir.noise_sites.size());
    plan.num_instrument_sites = static_cast<uint32_t>(hir.instrument_sites.size());
    plan.num_detectors = hir.num_detectors;
    plan.num_observables = hir.num_observables;
    plan.num_exp_vals = hir.num_exp_vals;
    if (options.retain_source_map) {
        if (hir.source_map.size() != hir.ops.size()) {
            throw std::invalid_argument(
                "sampling source provenance requires a complete HIR source map");
        }
        plan.source_map.emplace();
        plan.source_map->reserve(hir.ops.size() + hir.instrument_sites.size() +
                                 hir.num_observables);
    }

    // The packed symbolic frame needs its row count up front. Count only;
    // operations and their Paulis are consumed directly from HIR below.
    const PlanningRequirements requirements = inspect_planning_requirements(hir);
    plan.symbols.reserve(requirements.symbol_count);
    initialize_site_metadata(hir, plan);
    if (requirements.supports_final_state_queries) {
        plan.final_tableau = hir.final_tableau;
    }
    CoordinateFrame coordinates(hir.num_qubits);
    SymbolicPauliFrame symbolic_frame(hir.num_qubits, requirements.symbol_count);
    std::vector<std::optional<AffineBool>> record_values(static_cast<size_t>(hir.num_measurements) +
                                                         hir.num_hidden_measurements);
    std::vector<AffineBool> observable_values(hir.num_observables);
    std::vector<std::vector<uint32_t>> observable_source_lines;
    if (plan.source_map.has_value()) {
        observable_source_lines.resize(hir.num_observables);
    }

    auto require_record = [&](RecordSlot record, size_t operation_index) -> const AffineBool& {
        const uint32_t record_index = index(record);
        if (record_index >= record_values.size() || !record_values[record_index].has_value()) {
            throw std::invalid_argument("sampling planner operation " +
                                        std::to_string(operation_index) + " reads record " +
                                        std::to_string(record_index) + " before assignment");
        }
        return *record_values[record_index];
    };

    auto assign_record = [&](RecordSlot record, AffineBool value, size_t operation_index) {
        const uint32_t record_index = index(record);
        if (record_index >= record_values.size()) {
            throw std::invalid_argument("sampling planner operation " +
                                        std::to_string(operation_index) + " writes record " +
                                        std::to_string(record_index) + " out of range");
        }
        record_values[record_index] = std::move(value);
    };

    auto record_parity = [&](const std::vector<uint32_t>& records,
                             size_t operation_index) -> AffineBool {
        AffineBool result;
        for (uint32_t record : records) {
            result ^= require_record(RecordSlot{record}, operation_index);
        }
        return result;
    };

    uint32_t active_width = plan.initial_active_width;
    uint32_t detector_index = 0;
    uint32_t next_noise_site = 0;
    uint32_t next_instrument_site = 0;
    bool supports_final_state_queries = true;
    bool final_coordinates_changed = false;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const HeisenbergOp& op = hir.ops[i];
        const std::span<const uint32_t> source_lines =
            plan.source_map.has_value() ? source_lines_for(hir, i) : std::span<const uint32_t>{};
        supports_final_state_queries &= operation_supports_final_state_queries(op.op_type());
        switch (op.op_type()) {
            case OpType::T_GATE: {
                const double half_turns = op.is_dagger() ? -0.25 : 0.25;
                const Pauli body = pauli_from_hir(hir, op);
                final_coordinates_changed |=
                    process_rotation(body, half_turns, AffineBool(hir.sign(op)), plan, active_width,
                                     coordinates, symbolic_frame, source_lines);
                break;
            }
            case OpType::PHASE_ROTATION: {
                const Pauli body = pauli_from_hir(hir, op);
                final_coordinates_changed |=
                    process_rotation(body, op.alpha(), AffineBool(hir.sign(op)), plan, active_width,
                                     coordinates, symbolic_frame, source_lines);
                break;
            }
            case OpType::MEASURE: {
                const RecordSlot record{static_cast<uint32_t>(op.meas_record_idx())};
                const SymbolId branch = reserve_symbol(plan);
                const Pauli body = pauli_from_hir(hir, op);
                const AffineBool outcome =
                    process_measurement(body, AffineBool(hir.sign(op)), record, branch, plan,
                                        active_width, coordinates, symbolic_frame, source_lines);
                assign_record(record, outcome, i);
                break;
            }
            case OpType::CONDITIONAL_PAULI: {
                const RecordSlot controller{static_cast<uint32_t>(op.controlling_meas())};
                const Pauli body = pauli_from_hir(hir, op);
                symbolic_frame.apply(body, require_record(controller, i));
                break;
            }
            case OpType::NOISE: {
                const uint32_t site_index = static_cast<uint32_t>(op.noise_site_idx());
                if (site_index >= hir.noise_sites.size()) {
                    throw std::invalid_argument("sampling planner noise site is out of range");
                }
                if (site_index != next_noise_site) {
                    throw std::invalid_argument(
                        "sampling planner noise sites are not in circuit order");
                }
                ++next_noise_site;
                const NoiseSite& hir_site = hir.noise_sites[site_index];
                PresampledNoiseSite& plan_site = plan.presampled_noise_sites[site_index];
                plan_site.outcomes.reserve(hir_site.channels.size());
                for (const NoiseChannel& channel : hir_site.channels) {
                    if (channel.prob == 0.0) {
                        continue;
                    }
                    const SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
                    plan.symbols.push_back(
                        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{site_index}});
                    plan_site.outcomes.push_back(PresampledNoiseOutcome{symbol, channel.prob});
                    const Pauli body = noise_pauli_from_hir(hir, channel.mask);
                    symbolic_frame.apply(body, AffineBool::symbol(symbol));
                }
                break;
            }
            case OpType::READOUT_NOISE: {
                const uint32_t entry_index = static_cast<uint32_t>(op.readout_noise_idx());
                if (entry_index >= hir.readout_noise.size()) {
                    throw std::invalid_argument("sampling planner readout entry is out of range");
                }
                const ReadoutNoiseEntry& entry = hir.readout_noise[entry_index];
                const RecordSlot record{entry.meas_idx};
                const SymbolId flip = reserve_symbol(plan);
                const AffineBool source = require_record(record, i);
                if (entry.prob_zero_to_one == 0.0 && entry.prob_one_to_zero == 0.0) {
                    break;
                }
                const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
                define_symbol(plan, flip, SymbolKind::Readout, action_index);
                append_action(
                    plan,
                    PlannedAction{active_width, active_width,
                                  ApplyReadoutNoise{flip, source, record, entry.prob_zero_to_one,
                                                    entry.prob_one_to_zero}},
                    source_lines);
                record_values[index(record)] = source ^ AffineBool::symbol(flip);
                break;
            }
            case OpType::INSTRUMENT: {
                const uint32_t site_index = static_cast<uint32_t>(op.instrument_site_idx());
                if (site_index >= hir.instrument_sites.size() ||
                    site_index != next_instrument_site) {
                    throw std::invalid_argument(
                        "sampling planner instrument site is out of range or circuit order");
                }
                const InstrumentSite& site = hir.instrument_sites[site_index];
                if (site.destination_flip_mask == kNoMask) {
                    throw std::invalid_argument(
                        "sampling planner instrument omits its destination flip");
                }
                process_instrument(hir, op, next_noise_site, plan, active_width, coordinates,
                                   symbolic_frame, source_lines);
                ++next_instrument_site;
                break;
            }
            case OpType::DETECTOR: {
                const uint32_t targets_index = static_cast<uint32_t>(op.detector_idx());
                if (targets_index >= hir.detector_targets.size() ||
                    detector_index >= hir.num_detectors) {
                    throw std::invalid_argument("sampling planner detector is out of range");
                }
                AffineBool outcome = record_parity(hir.detector_targets[targets_index], i);
                outcome ^= option_bit(options.expected_detectors, detector_index);
                const bool postselected = option_bit(options.postselection_mask, detector_index);
                append_action(plan,
                              PlannedAction{active_width, active_width,
                                            WriteDetector{outcome, DetectorSlot{detector_index},
                                                          postselected}},
                              source_lines);
                plan.has_postselection |= postselected;
                ++detector_index;
                break;
            }
            case OpType::OBSERVABLE: {
                const uint32_t targets_index = op.observable_target_list_idx();
                const uint32_t observable_index = static_cast<uint32_t>(op.observable_idx());
                if (targets_index >= hir.observable_targets.size() ||
                    observable_index >= hir.num_observables) {
                    throw std::invalid_argument("sampling planner observable is out of range");
                }
                observable_values[observable_index] ^=
                    record_parity(hir.observable_targets[targets_index], i);
                if (plan.source_map.has_value()) {
                    observable_source_lines[observable_index].insert(
                        observable_source_lines[observable_index].end(), source_lines.begin(),
                        source_lines.end());
                }
                break;
            }
            case OpType::EXP_VAL: {
                const uint32_t exp_val_index = static_cast<uint32_t>(op.exp_val_idx());
                if (exp_val_index >= hir.num_exp_vals) {
                    throw std::invalid_argument(
                        "sampling planner expectation slot is out of range");
                }
                const Pauli body = pauli_from_hir(hir, op);
                ResolvedPauli resolved =
                    resolve_pauli(body, AffineBool(hir.sign(op)), coordinates, symbolic_frame);
                const bool is_zero = first_x_at_or_above(resolved.body, active_width).has_value();
                append_action(
                    plan,
                    PlannedAction{active_width, active_width,
                                  WriteExpectationValue{
                                      is_zero ? std::nullopt
                                              : std::optional<ActivePauli>{active_projection(
                                                    resolved.body, active_width)},
                                      is_zero ? AffineBool{} : std::move(resolved.sign),
                                      ExpValSlot{exp_val_index}}},
                    source_lines);
                break;
            }
            case OpType::NUM_OP_TYPES:
                throw std::invalid_argument("sampling planner does not support HIR operation " +
                                            op_type_to_str(op.op_type()) + " at index " +
                                            std::to_string(i));
        }
    }

    if (plan.symbols.size() != requirements.symbol_count) {
        throw std::logic_error("sampling planner symbol prepass disagrees with HIR lowering");
    }
    if (supports_final_state_queries != requirements.supports_final_state_queries) {
        throw std::logic_error("sampling planner final-state prepass disagrees with HIR lowering");
    }
    if (detector_index != hir.num_detectors) {
        throw std::invalid_argument("sampling planner detector count is inconsistent with HIR");
    }
    if (next_noise_site != hir.noise_sites.size()) {
        throw std::invalid_argument("sampling planner noise-site table is inconsistent with HIR");
    }
    if (next_instrument_site != hir.instrument_sites.size()) {
        throw std::invalid_argument(
            "sampling planner instrument-site table is inconsistent with HIR");
    }
    for (uint32_t observable = 0; observable < observable_values.size(); ++observable) {
        observable_values[observable] ^= option_bit(options.expected_observables, observable);
        const std::span<const uint32_t> source_lines =
            plan.source_map.has_value()
                ? std::span<const uint32_t>(observable_source_lines[observable])
                : std::span<const uint32_t>{};
        append_action(plan,
                      PlannedAction{active_width, active_width,
                                    WriteObservable{observable_values[observable],
                                                    ObservableSlot{observable}}},
                      source_lines);
    }

    if (plan.final_tableau.has_value() && final_coordinates_changed) {
        const Tableau& coordinates_to_physical = coordinates.current_to_initial();
        plan.final_tableau = coordinates_to_physical.then(*plan.final_tableau);
    }

    plan.validate();
    return plan;
}

}  // namespace clifft::sampling
