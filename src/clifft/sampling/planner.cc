#include "clifft/sampling/planner.h"

#include "clifft/sampling/planner_frame.h"
#include "clifft/util/hir_introspection.h"
#include "clifft/util/numeric.h"
#include "clifft/util/stim_mask.h"

#include "stim.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace clifft::sampling {

namespace {

using Pauli = internal::PlannerPauli;
using Tableau = internal::PlannerTableau;
using internal::active_measurement_frame;
using internal::CoordinateFrame;
using internal::dormant_measurement_frame;
using internal::dormant_promotion_frame;
using internal::SymbolicPauliFrame;

template <typename>
inline constexpr bool kAlwaysFalse = false;

void compose_final_tableau(SamplingPlan& plan, const Tableau& new_basis_in_old_coordinates) {
    if (plan.final_tableau.has_value()) {
        // A.then(B) represents B * A. The frame maps new coordinates into
        // the old basis, so prepend it on the right of the physical map.
        plan.final_tableau = new_basis_in_old_coordinates.then(*plan.final_tableau);
    }
}

struct PendingRotation {
    Pauli body;
    double half_turns = 0.0;
    AffineBool sign;
};

struct PendingMeasurement {
    Pauli body;
    AffineBool sign;
    RecordSlot record{};
    SymbolId branch{};
};

struct PendingExpectation {
    Pauli body;
    AffineBool sign;
    ExpValSlot exp_val{};
};

struct PendingConditionalPauli {
    Pauli body;
    RecordSlot controller{};
};

struct PendingNoiseChannel {
    Pauli body;
    SymbolId symbol{};
};

struct PendingNoise {
    std::vector<PendingNoiseChannel> channels;
};

struct PendingReadoutNoise {
    RecordSlot record{};
    double prob_zero_to_one = 0.0;
    double prob_one_to_zero = 0.0;
    SymbolId flip{};
};

struct PendingInstrument {
    // Source observable and computational destination correction stay in the
    // initial HIR coordinates used by the cumulative symbolic Pauli frame.
    Pauli body;
    Pauli destination_flip;
    // Maps the source observable's eigenspaces to physical G and E; earlier
    // stochastic outcomes can make this mapping symbolic.
    AffineBool sign;
    // Indexes both the plan-owned distribution and its continuation boundary.
    InstrumentSiteId site{};
    // Reserved in HIR order so coordinate evolution cannot renumber the prefix.
    SymbolId destination_flip_symbol{};
    // Continuations retain only noise and symbols preceding this HIR site.
    uint32_t next_noise_site = 0;
    uint32_t symbol_prefix_size = 0;
    bool neglect_damping = false;
};

struct PendingDetector {
    std::vector<RecordSlot> records;
    DetectorSlot detector{};
    bool expected = false;
    bool postselected = false;
};

struct PendingObservable {
    std::vector<RecordSlot> records;
    ObservableSlot observable{};
};

using PendingOperation = std::variant<PendingRotation, PendingMeasurement, PendingExpectation,
                                      PendingConditionalPauli, PendingNoise, PendingReadoutNoise,
                                      PendingInstrument, PendingDetector, PendingObservable>;

Pauli pauli_from_hir(const HirModule& hir, const HeisenbergOp& op) {
    Pauli result(hir.num_qubits);
    mask_view_to_stim(hir.destab_mask(op), hir.num_qubits, result.xs);
    mask_view_to_stim(hir.stab_mask(op), hir.num_qubits, result.zs);
    return result;
}

Pauli noise_pauli_from_hir(const HirModule& hir, PauliMaskHandle handle) {
    Pauli result(hir.num_qubits);
    const PauliMaskView mask = hir.noise_channel_masks.at(handle);
    mask_view_to_stim(mask.x(), hir.num_qubits, result.xs);
    mask_view_to_stim(mask.z(), hir.num_qubits, result.zs);
    return result;
}

Pauli pauli_from_mask(const HirModule& hir, PauliMaskHandle handle) {
    Pauli result(hir.num_qubits);
    const PauliMaskView mask = hir.pauli_masks.at(handle);
    mask_view_to_stim(mask.x(), hir.num_qubits, result.xs);
    mask_view_to_stim(mask.z(), hir.num_qubits, result.zs);
    result.sign = false;
    return result;
}

Pauli single_x(uint32_t num_qubits, uint32_t q) {
    Pauli result(num_qubits);
    result.xs[q] = true;
    return result;
}

struct ResolvedPauli {
    Pauli body;
    AffineBool sign;
};

ResolvedPauli resolve_pauli(const Pauli& initial_body, const AffineBool& initial_sign,
                            const CoordinateFrame& coordinates,
                            SymbolicPauliFrame& symbolic_frame) {
    AffineBool sign = initial_sign;
    sign ^= symbolic_frame.sign_for(initial_body);
    Pauli body = coordinates.to_current(initial_body);
    sign ^= static_cast<bool>(body.sign);
    body.sign = false;
    return ResolvedPauli{std::move(body), std::move(sign)};
}

std::optional<uint32_t> first_x_at_or_above(const Pauli& pauli, uint32_t begin) {
    for (uint32_t q = begin; q < pauli.num_qubits; ++q) {
        if (pauli.xs[q]) {
            return q;
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> first_x_below(const Pauli& pauli, uint32_t end) {
    for (uint32_t q = 0; q < end; ++q) {
        if (pauli.xs[q]) {
            return q;
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> first_z_below(const Pauli& pauli, uint32_t end) {
    for (uint32_t q = 0; q < end; ++q) {
        if (pauli.zs[q]) {
            return q;
        }
    }
    return std::nullopt;
}

ActivePauli active_projection(const Pauli& pauli, uint32_t active_width) {
    ActivePauli result;
    for (uint32_t q = 0; q < active_width; ++q) {
        result.x |= static_cast<uint64_t>(pauli.xs[q]) << q;
        result.z |= static_cast<uint64_t>(pauli.zs[q]) << q;
    }
    return result;
}

SymbolId reserve_symbol(SamplingPlan& plan) {
    const SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
    plan.symbols.emplace_back();
    return symbol;
}

void define_symbol(SamplingPlan& plan, SymbolId symbol, SymbolKind kind, uint32_t action) {
    SymbolInfo& info = plan.symbols.at(index(symbol));
    if (info.kind != SymbolKind::Unused || info.defining_action.has_value() ||
        info.noise_site.has_value()) {
        throw std::logic_error("sampling planner attempted to redefine a reserved symbol");
    }
    info = SymbolInfo{kind, action, std::nullopt};
}

void multiply_phase(std::complex<double>& weight, double angle) {
    weight *= std::complex<double>(std::cos(angle), std::sin(angle));
}

std::vector<RecordSlot> record_slots(const std::vector<uint32_t>& records) {
    std::vector<RecordSlot> result;
    result.reserve(records.size());
    for (uint32_t record : records) {
        result.push_back(RecordSlot{record});
    }
    return result;
}

bool option_bit(std::span<const uint8_t> values, uint32_t index) {
    return index < values.size() && values[index] != 0;
}

std::vector<PendingOperation> queue_supported_operations(const HirModule& hir, SamplingPlan& plan,
                                                         SamplingPlanOptions options) {
    std::vector<PendingOperation> pending;
    pending.reserve(hir.ops.size());
    plan.presampled_noise_sites.resize(hir.noise_sites.size());
    for (uint32_t site = 0; site < hir.noise_sites.size(); ++site) {
        plan.presampled_noise_sites[site].site = NoiseSiteId{site};
        plan.presampled_noise_sites[site].total_probability =
            hir.noise_sites[site].total_probability;
    }
    std::vector<bool> seen_noise_sites(hir.noise_sites.size(), false);
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

    uint32_t detector_index = 0;
    uint32_t next_noise_site = 0;
    uint32_t next_instrument_site = 0;
    bool final_state_queries_supported = true;

    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const HeisenbergOp& op = hir.ops[i];
        switch (op.op_type()) {
            case OpType::T_GATE: {
                const double half_turns = op.is_dagger() ? -0.25 : 0.25;
                pending.emplace_back(
                    PendingRotation{pauli_from_hir(hir, op), half_turns, AffineBool(hir.sign(op))});
                multiply_phase(plan.global_weight,
                               (op.is_dagger() ? -1.0 : 1.0) * std::numbers::pi / 8.0);
                break;
            }
            case OpType::PHASE_ROTATION: {
                pending.emplace_back(
                    PendingRotation{pauli_from_hir(hir, op), op.alpha(), AffineBool(hir.sign(op))});
                const double signed_alpha = hir.sign(op) ? -op.alpha() : op.alpha();
                multiply_phase(plan.global_weight, signed_alpha * std::numbers::pi / 2.0);
                break;
            }
            case OpType::MEASURE:
                final_state_queries_supported = false;
                pending.emplace_back(PendingMeasurement{
                    pauli_from_hir(hir, op), AffineBool(hir.sign(op)),
                    RecordSlot{static_cast<uint32_t>(op.meas_record_idx())}, reserve_symbol(plan)});
                break;
            case OpType::CONDITIONAL_PAULI:
                final_state_queries_supported = false;
                pending.emplace_back(PendingConditionalPauli{
                    pauli_from_hir(hir, op),
                    RecordSlot{static_cast<uint32_t>(op.controlling_meas())}});
                break;
            case OpType::NOISE: {
                final_state_queries_supported = false;
                const uint32_t site_index = static_cast<uint32_t>(op.noise_site_idx());
                if (site_index >= hir.noise_sites.size()) {
                    throw std::invalid_argument("sampling planner noise site is out of range");
                }
                if (seen_noise_sites[site_index]) {
                    throw std::invalid_argument(
                        "sampling planner encountered a duplicate noise site");
                }
                if (site_index != next_noise_site) {
                    throw std::invalid_argument(
                        "sampling planner noise sites are not in circuit order");
                }
                ++next_noise_site;
                seen_noise_sites[site_index] = true;
                const NoiseSite& hir_site = hir.noise_sites[site_index];
                PresampledNoiseSite& plan_site = plan.presampled_noise_sites[site_index];
                PendingNoise noise;
                noise.channels.reserve(hir_site.channels.size());
                plan_site.outcomes.reserve(hir_site.channels.size());
                for (const NoiseChannel& channel : hir_site.channels) {
                    if (channel.prob == 0.0) {
                        continue;
                    }
                    const SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
                    plan.symbols.push_back(
                        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{site_index}});
                    plan_site.outcomes.push_back(PresampledNoiseOutcome{symbol, channel.prob});
                    noise.channels.push_back(
                        PendingNoiseChannel{noise_pauli_from_hir(hir, channel.mask), symbol});
                }
                pending.emplace_back(std::move(noise));
                break;
            }
            case OpType::READOUT_NOISE: {
                final_state_queries_supported = false;
                const uint32_t entry_index = static_cast<uint32_t>(op.readout_noise_idx());
                if (entry_index >= hir.readout_noise.size()) {
                    throw std::invalid_argument("sampling planner readout entry is out of range");
                }
                const ReadoutNoiseEntry& entry = hir.readout_noise[entry_index];
                pending.emplace_back(
                    PendingReadoutNoise{RecordSlot{entry.meas_idx}, entry.prob_zero_to_one,
                                        entry.prob_one_to_zero, reserve_symbol(plan)});
                break;
            }
            case OpType::INSTRUMENT: {
                final_state_queries_supported = false;
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
                const SymbolId flip = reserve_symbol(plan);
                pending.emplace_back(PendingInstrument{
                    pauli_from_hir(hir, op), pauli_from_mask(hir, site.destination_flip_mask),
                    AffineBool(hir.sign(op)), InstrumentSiteId{site_index}, flip, next_noise_site,
                    static_cast<uint32_t>(plan.symbols.size()), hir.neglect_instrument_damping});
                ++next_instrument_site;
                break;
            }
            case OpType::DETECTOR: {
                final_state_queries_supported = false;
                const uint32_t targets_index = static_cast<uint32_t>(op.detector_idx());
                if (targets_index >= hir.detector_targets.size() ||
                    detector_index >= hir.num_detectors) {
                    throw std::invalid_argument("sampling planner detector is out of range");
                }
                pending.emplace_back(PendingDetector{
                    record_slots(hir.detector_targets[targets_index]), DetectorSlot{detector_index},
                    option_bit(options.expected_detectors, detector_index),
                    option_bit(options.postselection_mask, detector_index)});
                ++detector_index;
                break;
            }
            case OpType::OBSERVABLE: {
                final_state_queries_supported = false;
                const uint32_t targets_index = op.observable_target_list_idx();
                const uint32_t observable_index = static_cast<uint32_t>(op.observable_idx());
                if (targets_index >= hir.observable_targets.size() ||
                    observable_index >= hir.num_observables) {
                    throw std::invalid_argument("sampling planner observable is out of range");
                }
                pending.emplace_back(
                    PendingObservable{record_slots(hir.observable_targets[targets_index]),
                                      ObservableSlot{observable_index}});
                break;
            }
            case OpType::EXP_VAL: {
                const uint32_t exp_val_index = static_cast<uint32_t>(op.exp_val_idx());
                if (exp_val_index >= hir.num_exp_vals) {
                    throw std::invalid_argument(
                        "sampling planner expectation slot is out of range");
                }
                pending.emplace_back(PendingExpectation{
                    pauli_from_hir(hir, op), AffineBool(hir.sign(op)), ExpValSlot{exp_val_index}});
                break;
            }
            case OpType::NUM_OP_TYPES:
                throw std::invalid_argument("sampling planner does not support HIR operation " +
                                            op_type_to_str(op.op_type()) + " at index " +
                                            std::to_string(i));
        }
    }
    if (detector_index != hir.num_detectors) {
        throw std::invalid_argument("sampling planner detector count is inconsistent with HIR");
    }
    if (!std::ranges::all_of(seen_noise_sites, [](bool seen) { return seen; })) {
        throw std::invalid_argument("sampling planner noise-site table is inconsistent with HIR");
    }
    if (next_instrument_site != hir.instrument_sites.size()) {
        throw std::invalid_argument(
            "sampling planner instrument-site table is inconsistent with HIR");
    }
    if (final_state_queries_supported) {
        plan.final_tableau = hir.final_tableau;
    }
    return pending;
}

void process_rotation(const PendingRotation& rotation, SamplingPlan& plan, uint32_t& active_width,
                      CoordinateFrame& coordinates, SymbolicPauliFrame& symbolic_frame) {
    ResolvedPauli resolved =
        resolve_pauli(rotation.body, rotation.sign, coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);
    if (!dormant_pivot.has_value()) {
        plan.actions.push_back(
            PlannedAction{active_width, active_width,
                          RotateActivePauli{active_projection(resolved.body, active_width),
                                            rotation.half_turns, std::move(resolved.sign)}});
        return;
    }

    if (active_width + 1 >= kDenseActiveWidthLimit) {
        throw std::overflow_error(
            "sampling planner active width would reach " + std::to_string(active_width + 1) +
            ", but the dense-state limit is " + std::to_string(kDenseActiveWidthLimit));
    }

    const Tableau frame = dormant_promotion_frame(resolved.body, active_width, *dormant_pivot);
    coordinates.change_basis(frame);
    compose_final_tableau(plan, frame);
    plan.actions.push_back(
        PlannedAction{active_width, active_width + 1,
                      PromoteDormantRotation{rotation.half_turns, std::move(resolved.sign)}});
    ++active_width;
    plan.max_active_width = std::max(plan.max_active_width, active_width);
}

AffineBool process_measurement(const PendingMeasurement& measurement, SamplingPlan& plan,
                               uint32_t& active_width, CoordinateFrame& coordinates,
                               SymbolicPauliFrame& symbolic_frame) {
    ResolvedPauli resolved =
        resolve_pauli(measurement.body, measurement.sign, coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);
    if (dormant_pivot.has_value()) {
        const Tableau frame = dormant_measurement_frame(resolved.body, *dormant_pivot);
        coordinates.change_basis(frame);

        const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
        const SymbolId branch = measurement.branch;
        define_symbol(plan, branch, SymbolKind::Branch, action_index);
        Pauli correction = coordinates.to_initial(single_x(plan.num_qubits, *dormant_pivot));
        correction.sign = false;
        symbolic_frame.apply(correction, AffineBool::symbol(branch));
        const AffineBool outcome = resolved.sign ^ AffineBool::symbol(branch);
        plan.actions.push_back(PlannedAction{
            active_width, active_width,
            MeasureDormantRandom{*dormant_pivot, branch, outcome, measurement.record}});
        return outcome;
    }

    const ActivePauli active = active_projection(resolved.body, active_width);
    if (active.is_identity()) {
        plan.actions.push_back(PlannedAction{active_width, active_width,
                                             RecordClassical{resolved.sign, measurement.record}});
        return resolved.sign;
    }

    std::optional<uint32_t> pivot = first_x_below(resolved.body, active_width);
    if (!pivot.has_value()) {
        pivot = first_z_below(resolved.body, active_width);
    }
    if (!pivot.has_value()) {
        throw std::logic_error("sampling planner could not select an active measurement pivot");
    }

    Pauli active_body(resolved.body.num_qubits);
    for (uint32_t q = 0; q < active_width; ++q) {
        active_body.xs[q] = resolved.body.xs[q];
        active_body.zs[q] = resolved.body.zs[q];
    }
    const Tableau frame = active_measurement_frame(active_body, active_width, *pivot);
    coordinates.change_basis(frame);

    const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
    const SymbolId branch = measurement.branch;
    define_symbol(plan, branch, SymbolKind::Branch, action_index);
    Pauli correction = coordinates.to_initial(single_x(plan.num_qubits, active_width - 1));
    correction.sign = false;
    symbolic_frame.apply(correction, AffineBool::symbol(branch));
    const AffineBool outcome = resolved.sign ^ AffineBool::symbol(branch);
    plan.actions.push_back(
        PlannedAction{active_width, active_width - 1,
                      MeasureActivePauli{active, *pivot, branch, outcome, measurement.record}});
    --active_width;
    return outcome;
}

void process_instrument(const PendingInstrument& instrument, SamplingPlan& plan,
                        uint32_t& active_width, CoordinateFrame& coordinates,
                        SymbolicPauliFrame& symbolic_frame) {
    const InstrumentDistribution& distribution =
        plan.instrument_distributions.at(index(instrument.site));
    ResolvedPauli resolved =
        resolve_pauli(instrument.body, instrument.sign, coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);

    InstrumentMode mode = InstrumentMode::Classical;
    ActivePauli source;
    AffineBool sign = std::move(resolved.sign);
    uint32_t active_after = active_width;

    if (dormant_pivot.has_value()) {
        // Equal no-fire factors normalize away. Otherwise exact damping needs
        // the dormant-coherent source represented in the dense state.
        if (instrument.neglect_damping || distribution.p_fire[0] == distribution.p_fire[1]) {
            mode = InstrumentMode::DormantTrap;
            sign = AffineBool{};
        } else {
            if (active_width + 1 >= kDenseActiveWidthLimit) {
                throw std::overflow_error("sampling planner active width would reach " +
                                          std::to_string(active_width + 1) +
                                          ", but the dense-state limit is " +
                                          std::to_string(kDenseActiveWidthLimit));
            }
            const Tableau frame =
                dormant_promotion_frame(resolved.body, active_width, *dormant_pivot);
            coordinates.change_basis(frame);
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
        destination_symbol = instrument.destination_flip_symbol;
        define_symbol(plan, instrument.destination_flip_symbol, SymbolKind::Instrument,
                      action_index);
    }
    plan.actions.push_back(
        PlannedAction{active_width, active_after,
                      ApplyInstrument{instrument.site, mode, source, sign, destination_symbol}});

    if (destination_symbol.has_value()) {
        symbolic_frame.apply(instrument.destination_flip, AffineBool::symbol(*destination_symbol));
    }
    active_width = active_after;
    plan.max_active_width = std::max(plan.max_active_width, active_width);
    // A trapped shot resumes here so it cannot execute the instrument twice.
    plan.actions.push_back(
        PlannedAction{active_width, active_width,
                      InstrumentBoundary{instrument.site, instrument.next_noise_site,
                                         instrument.symbol_prefix_size}});
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
    plan.global_weight = hir.global_weight;

    const std::vector<PendingOperation> pending = queue_supported_operations(hir, plan, options);
    if (plan.symbols.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling planner symbol count exceeds uint32 range");
    }
    CoordinateFrame coordinates(hir.num_qubits);
    SymbolicPauliFrame symbolic_frame(hir.num_qubits, static_cast<uint32_t>(plan.symbols.size()));
    std::vector<std::optional<AffineBool>> record_values(static_cast<size_t>(hir.num_measurements) +
                                                         hir.num_hidden_measurements);
    std::vector<AffineBool> observable_values(hir.num_observables);

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

    auto record_parity = [&](const std::vector<RecordSlot>& records,
                             size_t operation_index) -> AffineBool {
        AffineBool result;
        for (RecordSlot record : records) {
            result ^= require_record(record, operation_index);
        }
        return result;
    };

    uint32_t active_width = plan.initial_active_width;
    for (size_t i = 0; i < pending.size(); ++i) {
        std::visit(
            [&](const auto& operation) {
                using T = std::decay_t<decltype(operation)>;
                if constexpr (std::is_same_v<T, PendingRotation>) {
                    process_rotation(operation, plan, active_width, coordinates, symbolic_frame);
                } else if constexpr (std::is_same_v<T, PendingMeasurement>) {
                    const AffineBool outcome = process_measurement(operation, plan, active_width,
                                                                   coordinates, symbolic_frame);
                    assign_record(operation.record, outcome, i);
                } else if constexpr (std::is_same_v<T, PendingConditionalPauli>) {
                    symbolic_frame.apply(operation.body, require_record(operation.controller, i));
                } else if constexpr (std::is_same_v<T, PendingNoise>) {
                    for (const PendingNoiseChannel& channel : operation.channels) {
                        symbolic_frame.apply(channel.body, AffineBool::symbol(channel.symbol));
                    }
                } else if constexpr (std::is_same_v<T, PendingReadoutNoise>) {
                    const AffineBool source = require_record(operation.record, i);
                    if (operation.prob_zero_to_one == 0.0 && operation.prob_one_to_zero == 0.0) {
                        return;
                    }
                    const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
                    const SymbolId flip = operation.flip;
                    define_symbol(plan, flip, SymbolKind::Readout, action_index);
                    plan.actions.push_back(PlannedAction{
                        active_width, active_width,
                        ApplyReadoutNoise{flip, source, operation.record,
                                          operation.prob_zero_to_one, operation.prob_one_to_zero}});
                    record_values[index(operation.record)] = source ^ AffineBool::symbol(flip);
                } else if constexpr (std::is_same_v<T, PendingExpectation>) {
                    ResolvedPauli resolved =
                        resolve_pauli(operation.body, operation.sign, coordinates, symbolic_frame);
                    const bool is_zero =
                        first_x_at_or_above(resolved.body, active_width).has_value();
                    plan.actions.push_back(PlannedAction{
                        active_width, active_width,
                        WriteExpectationValue{
                            is_zero ? std::nullopt
                                    : std::optional<ActivePauli>{active_projection(resolved.body,
                                                                                   active_width)},
                            is_zero ? AffineBool{} : std::move(resolved.sign), operation.exp_val}});
                } else if constexpr (std::is_same_v<T, PendingInstrument>) {
                    process_instrument(operation, plan, active_width, coordinates, symbolic_frame);
                } else if constexpr (std::is_same_v<T, PendingDetector>) {
                    AffineBool outcome = record_parity(operation.records, i);
                    outcome ^= operation.expected;
                    plan.actions.push_back(PlannedAction{
                        active_width, active_width,
                        WriteDetector{outcome, operation.detector, operation.postselected}});
                    plan.has_postselection |= operation.postselected;
                } else if constexpr (std::is_same_v<T, PendingObservable>) {
                    observable_values[index(operation.observable)] ^=
                        record_parity(operation.records, i);
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled pending operation alternative");
                }
            },
            pending[i]);
    }

    for (uint32_t observable = 0; observable < observable_values.size(); ++observable) {
        observable_values[observable] ^= option_bit(options.expected_observables, observable);
        plan.actions.push_back(PlannedAction{
            active_width, active_width,
            WriteObservable{observable_values[observable], ObservableSlot{observable}}});
    }

    plan.validate();
    return plan;
}

}  // namespace clifft::sampling
