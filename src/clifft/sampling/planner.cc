#include "clifft/sampling/planner.h"

#include "clifft/sampling/planner_frame.h"
#include "clifft/util/hir_introspection.h"
#include "clifft/util/numeric.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <bit>
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

struct ResolvedPauli {
    Pauli body;
    AffineBool sign;
};

ResolvedPauli resolve_pauli(MaskView initial_x, MaskView initial_z, const AffineBool& initial_sign,
                            CoordinateFrame& coordinates, SymbolicPauliFrame& symbolic_frame) {
    AffineBool sign = initial_sign;
    sign ^= symbolic_frame.sign_for(initial_x, initial_z);
    Pauli body = coordinates.to_current(initial_x, initial_z);
    sign ^= body.sign();
    body.set_sign(false);
    return ResolvedPauli{std::move(body), std::move(sign)};
}

std::optional<uint32_t> first_x_at_or_above(const Pauli& pauli, uint32_t begin) {
    const uint32_t n = pauli.num_qubits();
    if (begin >= n) {
        return std::nullopt;
    }
    const MaskView x = pauli.x();
    const uint32_t num_words = x.num_words();
    const uint32_t last_bits = n % 64;
    const uint64_t last_mask = last_bits == 0 ? ~uint64_t{0} : (uint64_t{1} << last_bits) - 1;
    for (uint32_t w = begin / 64; w < num_words; ++w) {
        uint64_t word = x.words[w];
        if (w == begin / 64) {
            word &= ~uint64_t{0} << (begin % 64);
        }
        if (w + 1 == num_words) {
            word &= last_mask;
        }
        if (word != 0) {
            return 64 * w + std::countr_zero(word);
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> last_x_below(const Pauli& pauli, uint32_t end) {
    assert(end <= pauli.num_qubits());
    if (end == 0) {
        return std::nullopt;
    }
    const MaskView x = pauli.x();
    for (uint32_t w = (end - 1) / 64 + 1; w-- > 0;) {
        uint64_t word = x.words[w];
        if (64 * (w + 1) > end && end % 64 != 0) {
            word &= (uint64_t{1} << (end % 64)) - 1;
        }
        if (word != 0) {
            return 64 * w + 63 - std::countl_zero(word);
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> first_z_below(const Pauli& pauli, uint32_t end) {
    assert(end <= pauli.num_qubits());
    if (end == 0) {
        return std::nullopt;
    }
    const MaskView z = pauli.z();
    for (uint32_t w = 0; 64 * w < end; ++w) {
        uint64_t word = z.words[w];
        if (64 * (w + 1) > end && end % 64 != 0) {
            word &= (uint64_t{1} << (end % 64)) - 1;
        }
        if (word != 0) {
            return 64 * w + std::countr_zero(word);
        }
    }
    return std::nullopt;
}

ActivePauli active_projection(const Pauli& pauli, uint32_t active_width) {
    assert(active_width < kDenseActiveWidthLimit);
    if (active_width == 0) {
        return {};
    }
    const uint64_t mask = (uint64_t{1} << active_width) - 1;
    return ActivePauli{pauli.x().words[0] & mask, pauli.z().words[0] & mask};
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

void define_symbol(SamplingPlan& plan, SymbolId symbol, SymbolKind kind) {
    SymbolKind& stored_kind = plan.symbols.at(index(symbol));
    if (stored_kind != SymbolKind::Unused) {
        throw std::logic_error("sampling planner attempted to redefine a reserved symbol");
    }
    stored_kind = kind;
}

bool option_bit(std::span<const uint8_t> values, uint32_t index) {
    return index < values.size() && values[index] != 0;
}

RecordParity make_record_parity(std::span<const uint32_t> records, bool constant) {
    std::vector<RecordSlot> slots;
    slots.reserve(records.size());
    for (uint32_t record : records) {
        slots.push_back(RecordSlot{record});
    }
    return RecordParity{constant, std::move(slots)};
}

struct ObservableRecordReference {
    uint32_t record = 0;
    uint32_t generation = 0;
};

void canonicalize_record_snapshots(std::vector<ObservableRecordReference>& references) {
    std::ranges::sort(references, [](const auto& left, const auto& right) {
        if (left.record != right.record) {
            return left.record < right.record;
        }
        return left.generation < right.generation;
    });
    size_t output = 0;
    for (size_t begin = 0; begin < references.size();) {
        size_t end = begin + 1;
        while (end < references.size() && references[end].record == references[begin].record &&
               references[end].generation == references[begin].generation) {
            ++end;
        }
        if ((end - begin) % 2 != 0) {
            references[output++] = references[begin];
        }
        begin = end;
    }
    references.resize(output);
}

struct PlanningRecord {
    std::optional<AffineBool> value;
    uint32_t generation = 0;
};

struct PendingObservable {
    AffineBool historical_value;
    std::vector<ObservableRecordReference> record_snapshots;
    std::vector<uint32_t> source_lines;
};

// Only rotations and measurements may change their position relative to noise.
void validate_logical_noise_prefix(const HirModule& hir) {
    if (!hir.has_logical_noise_prefix()) {
        return;
    }
    const auto num_noise_sites = static_cast<uint32_t>(hir.noise_sites.size());
    uint32_t schedule_count = 0;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const HeisenbergOp& op = hir.ops[i];
        const uint32_t entry = hir.logical_noise_prefix[i];
        if (entry > num_noise_sites) {
            throw std::invalid_argument("sampling planner logical noise prefix at operation " +
                                        std::to_string(i) + " exceeds the noise site count");
        }
        const OpType type = op.op_type();
        const bool may_cross_noise =
            type == OpType::T_GATE || type == OpType::PHASE_ROTATION || type == OpType::MEASURE;
        if (!may_cross_noise && entry != schedule_count) {
            throw std::invalid_argument(
                "sampling planner logical noise prefix at operation " + std::to_string(i) +
                " must equal its schedule position for an operation that cannot move across "
                "noise");
        }
        if (type == OpType::NOISE) {
            ++schedule_count;
        }
    }
}

// Resolve noise-dependent signs without exposing optional HIR metadata to
// the planning loop. Symbol bases also allow references to later noise sites.
class NoiseSigns {
  public:
    explicit NoiseSigns(const HirModule& hir) : hir_(hir) {
        validate_logical_noise_prefix(hir);
        if (hir.has_logical_noise_prefix()) {
            symbol_bases_.resize(hir.noise_sites.size());
        }
    }

    void record_symbols(uint32_t site, uint32_t first_symbol) {
        if (!symbol_bases_.empty()) {
            symbol_bases_[site] = first_symbol;
        }
    }

    void check_symbols([[maybe_unused]] uint32_t site, [[maybe_unused]] size_t first_symbol) const {
        assert((symbol_bases_.empty() || symbol_bases_[site] == first_symbol) &&
               "noise symbol allocation disagrees with the planning prepass");
    }

    // Moving a Z rotation across an X error reverses its angle if the error
    // fires. XOR the crossed anticommuting channels to correct either direction.
    [[nodiscard]] AffineBool sign_for(size_t operation_index, uint32_t noise_count) {
        const HeisenbergOp& op = hir_.ops[operation_index];
        AffineBool sign(hir_.sign(op));
        if (symbol_bases_.empty()) {
            return sign;
        }
        const uint32_t original_count = hir_.logical_noise_prefix[operation_index];
        if (noise_count == original_count) {
            return sign;
        }
        ensure_site_index();
        const MaskView x = hir_.destab_mask(op);
        const MaskView z = hir_.stab_mask(op);
        const uint32_t begin = std::min(noise_count, original_count);
        const uint32_t end = std::max(noise_count, original_count);
        candidate_sites_.clear();
        collect_sites_in_range(x, sites_with_z_, begin, end);
        collect_sites_in_range(z, sites_with_x_, begin, end);
        if (candidate_sites_.empty()) {
            return sign;
        }
        std::ranges::sort(candidate_sites_);
        const auto unique_end = std::ranges::unique(candidate_sites_).begin();
        candidate_sites_.erase(unique_end, candidate_sites_.end());

        const PauliCommutation commutation(x, z);
        std::vector<SymbolId> crossed_symbols;
        for (uint32_t site : candidate_sites_) {
            uint32_t local_index = 0;
            for (const NoiseChannel& channel : hir_.noise_sites[site].channels) {
                if (channel.prob == 0.0) {
                    continue;
                }
                const PauliMaskView channel_view = hir_.noise_channel_masks.at(channel.mask);
                if (commutation.anti_commutes(channel_view.x(), channel_view.z())) {
                    crossed_symbols.push_back(SymbolId{symbol_bases_[site] + local_index});
                }
                ++local_index;
            }
        }
        if (!crossed_symbols.empty()) {
            sign ^= AffineBool::from_canonical_terms(false, std::move(crossed_symbols));
        }
        return sign;
    }

  private:
    void ensure_site_index() {
        if (site_index_built_) {
            return;
        }
        site_index_built_ = true;
        sites_with_x_.resize(hir_.num_qubits);
        sites_with_z_.resize(hir_.num_qubits);
        const uint32_t num_words = (hir_.num_qubits + 63) / 64;
        const uint32_t tail_bits = hir_.num_qubits % 64;
        const uint64_t last_mask = tail_bits == 0 ? ~uint64_t{0} : (uint64_t{1} << tail_bits) - 1;
        auto record_mask = [&](MaskView mask, uint32_t site,
                               std::vector<std::vector<uint32_t>>& lists) {
            for (uint32_t w = 0; w < num_words; ++w) {
                uint64_t pending = mask.words[w] & (w + 1 == num_words ? last_mask : ~uint64_t{0});
                while (pending != 0) {
                    const uint32_t q = 64 * w + std::countr_zero(pending);
                    auto& list = lists[q];
                    if (list.empty() || list.back() != site) {
                        list.push_back(site);
                    }
                    pending &= pending - 1;
                }
            }
        };
        for (uint32_t site = 0; site < hir_.noise_sites.size(); ++site) {
            for (const NoiseChannel& channel : hir_.noise_sites[site].channels) {
                if (channel.prob == 0.0) {
                    continue;
                }
                const PauliMaskView view = hir_.noise_channel_masks.at(channel.mask);
                record_mask(view.x(), site, sites_with_x_);
                record_mask(view.z(), site, sites_with_z_);
            }
        }
    }

    void collect_sites_in_range(MaskView mask, const std::vector<std::vector<uint32_t>>& lists,
                                uint32_t begin, uint32_t end) {
        const uint32_t num_words = (hir_.num_qubits + 63) / 64;
        const uint32_t tail_bits = hir_.num_qubits % 64;
        const uint64_t last_mask = tail_bits == 0 ? ~uint64_t{0} : (uint64_t{1} << tail_bits) - 1;
        for (uint32_t w = 0; w < num_words; ++w) {
            uint64_t pending = mask.words[w] & (w + 1 == num_words ? last_mask : ~uint64_t{0});
            while (pending != 0) {
                const uint32_t q = 64 * w + std::countr_zero(pending);
                const auto& list = lists[q];
                auto it = std::ranges::lower_bound(list, begin);
                while (it != list.end() && *it < end) {
                    candidate_sites_.push_back(*it);
                    ++it;
                }
                pending &= pending - 1;
            }
        }
    }

    const HirModule& hir_;
    // First symbol for each noise site's nonzero channels. Unneeded without
    // recorded positions, so ordinary planning does not allocate this table.
    std::vector<uint32_t> symbol_bases_;
    bool site_index_built_ = false;
    std::vector<std::vector<uint32_t>> sites_with_x_;
    std::vector<std::vector<uint32_t>> sites_with_z_;
    std::vector<uint32_t> candidate_sites_;
};

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

PlanningRequirements inspect_planning_requirements(const HirModule& hir, NoiseSigns& noise_signs) {
    PlanningRequirements result;
    auto add = [&](size_t amount) {
        if (amount > std::numeric_limits<uint32_t>::max() - result.symbol_count) {
            throw std::length_error("sampling planner symbol count exceeds uint32 range");
        }
        result.symbol_count += static_cast<uint32_t>(amount);
    };

    uint32_t expected_noise_site = 0;
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
                if (site_index != expected_noise_site) {
                    throw std::invalid_argument(
                        "sampling planner noise sites are not in circuit order");
                }
                ++expected_noise_site;
                noise_signs.record_symbols(site_index, result.symbol_count);
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
        plan.presampled_noise_sites[site].source_site = site;
        plan.presampled_noise_sites[site].total_probability =
            hir.noise_sites[site].total_probability;
    }
    plan.instrument_distributions.reserve(hir.instrument_sites.size());
    for (uint32_t site = 0; site < hir.instrument_sites.size(); ++site) {
        const InstrumentProbabilities& probabilities = hir.instrument_sites[site].probabilities;
        InstrumentDistribution distribution;
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

bool process_rotation(MaskView body_x, MaskView body_z, double half_turns, const AffineBool& sign,
                      SamplingPlan& plan, uint32_t& active_width, CoordinateFrame& coordinates,
                      SymbolicPauliFrame& symbolic_frame, std::span<const uint32_t> source_lines) {
    ResolvedPauli resolved = resolve_pauli(body_x, body_z, sign, coordinates, symbolic_frame);
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

AffineBool process_measurement(MaskView body_x, MaskView body_z, const AffineBool& sign,
                               RecordSlot record, SymbolId branch, SamplingPlan& plan,
                               uint32_t& active_width, CoordinateFrame& coordinates,
                               SymbolicPauliFrame& symbolic_frame,
                               std::span<const uint32_t> source_lines) {
    ResolvedPauli resolved = resolve_pauli(body_x, body_z, sign, coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);
    if (dormant_pivot.has_value()) {
        coordinates.measure_dormant(resolved.body, *dormant_pivot);

        define_symbol(plan, branch, SymbolKind::Branch);
        const PauliStringView correction =
            coordinates.current_to_initial().x_output(*dormant_pivot);
        symbolic_frame.apply(correction.x(), correction.z(), AffineBool::symbol(branch));
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
    active_body.mut_x().words[0] = active.x;
    active_body.mut_z().words[0] = active.z;
    active_body.set_sign(false);
    coordinates.measure_active(active_body, active_width, *pivot);

    define_symbol(plan, branch, SymbolKind::Branch);
    const PauliStringView correction = coordinates.current_to_initial().x_output(active_width - 1);
    symbolic_frame.apply(correction.x(), correction.z(), AffineBool::symbol(branch));
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
    const PauliMaskView destination_flip = hir.pauli_masks.at(hir_site.destination_flip_mask);
    const InstrumentDistribution& distribution = plan.instrument_distributions.at(index(site));
    ResolvedPauli resolved = resolve_pauli(hir.destab_mask(op), hir.stab_mask(op),
                                           AffineBool(hir.sign(op)), coordinates, symbolic_frame);
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

    std::optional<SymbolId> destination_symbol;
    if (mode != InstrumentMode::DormantTrap) {
        // Only an in-line computational destination needs the reserved flip;
        // a trapped continuation resolves its destination instead.
        destination_symbol = destination_flip_symbol;
        define_symbol(plan, destination_flip_symbol, SymbolKind::Instrument);
    }
    append_action(plan,
                  PlannedAction{active_width, active_after,
                                ApplyInstrument{site, mode, source, sign, destination_symbol}},
                  source_lines);

    if (destination_symbol.has_value()) {
        symbolic_frame.apply(destination_flip.x(), destination_flip.z(),
                             AffineBool::symbol(*destination_symbol));
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
    if (options.postselection_mask.empty()) {
        options.postselection_mask = hir.postselection.mask;
    }
    if (options.expected_detectors.empty()) {
        options.expected_detectors = hir.postselection.expected_detectors;
    }
    if (options.expected_observables.empty()) {
        options.expected_observables = hir.postselection.expected_observables;
    }
    const auto validate_assumptions = [&](std::span<const uint8_t> assumed_mask,
                                          std::span<const uint8_t> assumed_expected) {
        if (assumed_mask.empty()) {
            return;
        }
        for (uint32_t d = 0; d < hir.num_detectors; ++d) {
            if (!option_bit(assumed_mask, d)) {
                continue;
            }
            if (!option_bit(options.postselection_mask, d)) {
                throw std::invalid_argument(
                    "sampling planner postselection_mask omits detectors assumed by HIR");
            }
            if (option_bit(assumed_expected, d) != option_bit(options.expected_detectors, d)) {
                throw std::invalid_argument(
                    "sampling planner expected_detectors conflicts with postselected HIR");
            }
        }
    };
    validate_assumptions(hir.postselection.assumed_mask,
                         hir.postselection.assumed_expected_detectors);
    validate_assumptions(hir.postselection.mask, hir.postselection.expected_detectors);

    NoiseSigns noise_signs(hir);

    SamplingPlan plan;
    plan.num_qubits = hir.num_qubits;
    plan.num_visible_records = hir.num_measurements;
    plan.num_hidden_records = hir.num_hidden_measurements;
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
    const PlanningRequirements requirements = inspect_planning_requirements(hir, noise_signs);
    plan.symbols.reserve(requirements.symbol_count);
    initialize_site_metadata(hir, plan);
    if (requirements.supports_final_state_queries) {
        plan.final_tableau = hir.final_tableau;
    }
    CoordinateFrame coordinates(hir.num_qubits);
    SymbolicPauliFrame symbolic_frame(hir.num_qubits, requirements.symbol_count);
    std::vector<PlanningRecord> records(static_cast<size_t>(hir.num_measurements) +
                                        hir.num_hidden_measurements);
    std::vector<PendingObservable> observables(hir.num_observables);

    auto require_record = [&](RecordSlot record, size_t operation_index) -> const AffineBool& {
        const uint32_t record_index = index(record);
        if (record_index >= records.size() || !records[record_index].value.has_value()) {
            throw std::invalid_argument("sampling planner operation " +
                                        std::to_string(operation_index) + " reads record " +
                                        std::to_string(record_index) + " before assignment");
        }
        return *records[record_index].value;
    };

    auto assign_record = [&](RecordSlot record, AffineBool value, size_t operation_index) {
        const uint32_t record_index = index(record);
        if (record_index >= records.size()) {
            throw std::invalid_argument("sampling planner operation " +
                                        std::to_string(operation_index) + " writes record " +
                                        std::to_string(record_index) + " out of range");
        }
        records[record_index].value = std::move(value);
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
                final_coordinates_changed |=
                    process_rotation(hir.destab_mask(op), hir.stab_mask(op), half_turns,
                                     noise_signs.sign_for(i, next_noise_site), plan, active_width,
                                     coordinates, symbolic_frame, source_lines);
                break;
            }
            case OpType::PHASE_ROTATION: {
                final_coordinates_changed |=
                    process_rotation(hir.destab_mask(op), hir.stab_mask(op), op.alpha(),
                                     noise_signs.sign_for(i, next_noise_site), plan, active_width,
                                     coordinates, symbolic_frame, source_lines);
                break;
            }
            case OpType::MEASURE: {
                const RecordSlot record{static_cast<uint32_t>(op.meas_record_idx())};
                const SymbolId branch = reserve_symbol(plan);
                const AffineBool outcome = process_measurement(
                    hir.destab_mask(op), hir.stab_mask(op),
                    noise_signs.sign_for(i, next_noise_site), record, branch, plan, active_width,
                    coordinates, symbolic_frame, source_lines);
                assign_record(record, outcome, i);
                break;
            }
            case OpType::CONDITIONAL_PAULI: {
                const RecordSlot controller{static_cast<uint32_t>(op.controlling_meas())};
                symbolic_frame.apply(hir.destab_mask(op), hir.stab_mask(op),
                                     require_record(controller, i));
                break;
            }
            case OpType::NOISE: {
                const uint32_t site_index = static_cast<uint32_t>(op.noise_site_idx());
                assert(site_index == next_noise_site && site_index < hir.noise_sites.size());
                ++next_noise_site;
                const NoiseSite& hir_site = hir.noise_sites[site_index];
                noise_signs.check_symbols(site_index, plan.symbols.size());
                PresampledNoiseSite& plan_site = plan.presampled_noise_sites[site_index];
                plan_site.outcomes.reserve(hir_site.channels.size());
                for (const NoiseChannel& channel : hir_site.channels) {
                    if (channel.prob == 0.0) {
                        continue;
                    }
                    const SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
                    plan.symbols.push_back(SymbolKind::Presampled);
                    plan_site.outcomes.push_back(PresampledNoiseOutcome{symbol, channel.prob});
                    const PauliMaskView mask = hir.noise_channel_masks.at(channel.mask);
                    symbolic_frame.apply(mask.x(), mask.z(), AffineBool::symbol(symbol));
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
                define_symbol(plan, flip, SymbolKind::Readout);
                append_action(
                    plan,
                    PlannedAction{active_width, active_width,
                                  ApplyReadoutNoise{flip, source, record, entry.prob_zero_to_one,
                                                    entry.prob_one_to_zero, entry_index}},
                    source_lines);
                PlanningRecord& updated = records[index(record)];
                updated.value = source ^ AffineBool::symbol(flip);
                ++updated.generation;
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
                const std::vector<uint32_t>& targets = hir.detector_targets[targets_index];
                for (uint32_t record : targets) {
                    (void)require_record(RecordSlot{record}, i);
                }
                const bool expected = option_bit(options.expected_detectors, detector_index);
                const bool postselected = option_bit(options.postselection_mask, detector_index);
                append_action(
                    plan,
                    PlannedAction{active_width, active_width,
                                  WriteDetector{make_record_parity(targets, expected),
                                                DetectorSlot{detector_index}, postselected}},
                    source_lines);
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
                PendingObservable& pending = observables[observable_index];
                pending.historical_value ^= record_parity(hir.observable_targets[targets_index], i);
                for (uint32_t record : hir.observable_targets[targets_index]) {
                    pending.record_snapshots.push_back({record, records.at(record).generation});
                }
                if (plan.source_map.has_value()) {
                    pending.source_lines.insert(pending.source_lines.end(), source_lines.begin(),
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
                ResolvedPauli resolved =
                    resolve_pauli(hir.destab_mask(op), hir.stab_mask(op), AffineBool(hir.sign(op)),
                                  coordinates, symbolic_frame);
                const bool is_zero = first_x_at_or_above(resolved.body, active_width).has_value();
                append_action(
                    plan,
                    PlannedAction{active_width, active_width,
                                  WriteExpectationValue{
                                      is_zero ? std::nullopt
                                              : std::optional<ActiveExpectation>{ActiveExpectation{
                                                    active_projection(resolved.body, active_width),
                                                    std::move(resolved.sign)}},
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
    for (uint32_t observable = 0; observable < observables.size(); ++observable) {
        PendingObservable& pending = observables[observable];
        pending.historical_value ^= option_bit(options.expected_observables, observable);
        canonicalize_record_snapshots(pending.record_snapshots);
        const bool records_are_current =
            std::ranges::all_of(pending.record_snapshots, [&](const auto& reference) {
                return reference.generation == records[reference.record].generation;
            });
        ObservableValue outcome;
        if (records_are_current) {
            std::vector<uint32_t> snapshot_records;
            snapshot_records.reserve(pending.record_snapshots.size());
            for (const ObservableRecordReference& reference : pending.record_snapshots) {
                snapshot_records.push_back(reference.record);
            }
            outcome = make_record_parity(snapshot_records,
                                         option_bit(options.expected_observables, observable));
        } else {
            outcome = std::move(pending.historical_value);
        }
        const std::span<const uint32_t> source_lines =
            plan.source_map.has_value() ? std::span<const uint32_t>(pending.source_lines)
                                        : std::span<const uint32_t>{};
        append_action(
            plan,
            PlannedAction{active_width, active_width,
                          WriteObservable{std::move(outcome), ObservableSlot{observable}}},
            source_lines);
    }

    if (plan.final_tableau.has_value() && final_coordinates_changed) {
        const Tableau& coordinates_to_physical = coordinates.current_to_initial();
        plan.final_tableau = coordinates_to_physical.then(*plan.final_tableau);
    }

    // CPU and HIP executable construction validate both compiler-produced and
    // externally assembled plans before execution. Repeating that full scan
    // here makes the normal compile path validate the same plan twice.
    return plan;
}

}  // namespace clifft::sampling
