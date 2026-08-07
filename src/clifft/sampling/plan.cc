#include "clifft/sampling/plan.h"

#include "clifft/util/numeric.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace clifft::sampling {

namespace {

template <typename>
inline constexpr bool kAlwaysFalse = false;

std::vector<SymbolId> xor_terms(const std::vector<SymbolId>& left,
                                const std::vector<SymbolId>& right) {
    // Canonical inputs are sorted and unique, so XOR is their symmetric
    // difference: a term present in both inputs cancels.
    std::vector<SymbolId> result;
    result.reserve(left.size() + right.size());
    size_t i = 0;
    size_t j = 0;
    while (i < left.size() || j < right.size()) {
        if (j == right.size() || (i < left.size() && index(left[i]) < index(right[j]))) {
            result.push_back(left[i++]);
        } else if (i == left.size() || index(right[j]) < index(left[i])) {
            result.push_back(right[j++]);
        } else {
            ++i;
            ++j;
        }
    }
    return result;
}

std::vector<SymbolId> canonicalize_terms(std::vector<SymbolId> terms) {
    // XOR retains exactly the terms with odd multiplicity. Sorting also gives
    // expressions one deterministic representation.
    std::sort(terms.begin(), terms.end(),
              [](SymbolId left, SymbolId right) { return index(left) < index(right); });
    std::vector<SymbolId> result;
    for (size_t begin = 0; begin < terms.size();) {
        size_t end = begin + 1;
        while (end < terms.size() && terms[end] == terms[begin]) {
            ++end;
        }
        if ((end - begin) % 2 != 0) {
            result.push_back(terms[begin]);
        }
        begin = end;
    }
    return result;
}

std::string format_expression(const AffineBool& expression) {
    std::ostringstream out;
    bool wrote = false;
    if (expression.constant()) {
        out << '1';
        wrote = true;
    }
    for (SymbolId term : expression.terms()) {
        if (wrote) {
            out << " ^ ";
        }
        out << 's' << index(term);
        wrote = true;
    }
    if (!wrote) {
        out << '0';
    }
    return out.str();
}

std::string_view symbol_kind_name(SymbolKind kind) {
    switch (kind) {
        case SymbolKind::Unused:
            return "unused";
        case SymbolKind::Presampled:
            return "presampled";
        case SymbolKind::Derived:
            return "derived";
        case SymbolKind::Branch:
            return "branch";
        case SymbolKind::Readout:
            return "readout";
        case SymbolKind::Instrument:
            return "instrument";
    }
    return "unknown";
}

std::string_view instrument_mode_name(InstrumentMode mode) {
    switch (mode) {
        case InstrumentMode::Classical:
            return "classical";
        case InstrumentMode::Active:
            return "active";
        case InstrumentMode::Activate:
            return "activate";
        case InstrumentMode::DormantTrap:
            return "dormant_trap";
    }
    return "unknown";
}

std::string format_mask(uint64_t mask) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << mask;
    return out.str();
}

std::string format_pauli(const ActivePauli& pauli) {
    std::ostringstream out;
    out << "x=" << format_mask(pauli.x) << " z=" << format_mask(pauli.z);
    return out.str();
}

[[noreturn]] void invalid_plan(std::string message) {
    throw std::invalid_argument("invalid SamplingPlan: " + std::move(message));
}

void validate_pauli(const ActivePauli& pauli, uint32_t expected_width, uint32_t action_index) {
    const uint64_t width_mask =
        expected_width == 0 ? uint64_t{0} : (uint64_t{1} << expected_width) - 1;
    if ((pauli.x & ~width_mask) != 0 || (pauli.z & ~width_mask) != 0) {
        invalid_plan("action " + std::to_string(action_index) +
                     " Pauli has bits outside its active width");
    }
}

void validate_measurement_pivot(const MeasureActivePauli& measurement, uint32_t action_index) {
    // Pair updates need an X-support pivot; diagonal updates remove a Z-support coordinate.
    const uint64_t pivot_bit = uint64_t{1} << measurement.active_pivot;
    const bool valid = measurement.pauli.x != 0 ? (measurement.pauli.x & pivot_bit) != 0
                                                : (measurement.pauli.z & pivot_bit) != 0;
    if (!valid) {
        invalid_plan("action " + std::to_string(action_index) +
                     " measurement pivot is outside the relevant Pauli support");
    }
}

std::optional<SymbolId> defined_symbol(const SamplingAction& action) {
    return std::visit(
        [](const auto& typed) -> std::optional<SymbolId> {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, MeasureActivePauli> ||
                          std::is_same_v<T, MeasureDormantRandom>) {
                return typed.branch;
            } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                return typed.symbol;
            } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                return typed.flip;
            } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                return typed.destination_flip;
            } else if constexpr (std::is_same_v<T, RotateActivePauli> ||
                                 std::is_same_v<T, PromoteDormantRotation> ||
                                 std::is_same_v<T, RecordClassical> ||
                                 std::is_same_v<T, WriteDetector> ||
                                 std::is_same_v<T, WriteObservable> ||
                                 std::is_same_v<T, WriteExpectationValue> ||
                                 std::is_same_v<T, InstrumentBoundary>) {
                return std::nullopt;
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
            }
        },
        action);
}

void validate_expression(const SamplingPlan& plan, const AffineBool& expression,
                         uint32_t action_index, std::optional<SymbolId> current_definition,
                         bool allow_current_definition) {
    if (!expression.is_canonical()) {
        invalid_plan("action " + std::to_string(action_index) +
                     " has a noncanonical affine expression");
    }
    for (SymbolId term : expression.terms()) {
        const uint32_t symbol_index = index(term);
        if (symbol_index >= plan.symbols.size()) {
            invalid_plan("action " + std::to_string(action_index) + " references symbol s" +
                         std::to_string(symbol_index) + " out of range");
        }
        const SymbolInfo& info = plan.symbols[symbol_index];
        if (info.kind == SymbolKind::Presampled) {
            continue;
        }
        if (!info.defining_action.has_value()) {
            invalid_plan("symbol s" + std::to_string(symbol_index) + " has no defining action");
        }
        const bool current_is_allowed = allow_current_definition && current_definition == term &&
                                        *info.defining_action == action_index;
        if (*info.defining_action >= action_index && !current_is_allowed) {
            invalid_plan("action " + std::to_string(action_index) + " references symbol s" +
                         std::to_string(symbol_index) + " before assignment");
        }
    }
}

void validate_measurement_outcome(const SamplingPlan& plan, const AffineBool& outcome,
                                  SymbolId branch, uint32_t action_index) {
    validate_expression(plan, outcome, action_index, branch, true);
    if (std::ranges::find(outcome.terms(), branch) == outcome.terms().end()) {
        invalid_plan("action " + std::to_string(action_index) +
                     " measurement outcome omits branch s" + std::to_string(index(branch)));
    }
}

void validate_record(const SamplingPlan& plan, RecordSlot record, uint32_t action_index,
                     std::unordered_set<uint32_t>& written_records) {
    const uint64_t total = static_cast<uint64_t>(plan.num_visible_records) +
                           static_cast<uint64_t>(plan.num_hidden_records);
    if (index(record) >= total) {
        invalid_plan("action " + std::to_string(action_index) + " record slot " +
                     std::to_string(index(record)) + " out of range");
    }
    if (!written_records.insert(index(record)).second) {
        invalid_plan("action " + std::to_string(action_index) + " writes record slot " +
                     std::to_string(index(record)) + " more than once");
    }
}

void validate_written_record(const SamplingPlan& plan, RecordSlot record, uint32_t action_index,
                             const std::unordered_set<uint32_t>& written_records) {
    const uint64_t total = static_cast<uint64_t>(plan.num_visible_records) +
                           static_cast<uint64_t>(plan.num_hidden_records);
    if (index(record) >= total || !written_records.contains(index(record))) {
        invalid_plan("action " + std::to_string(action_index) + " reads record slot " +
                     std::to_string(index(record)) + " before assignment");
    }
}

}  // namespace

AffineBool::AffineBool(bool constant) : constant_(constant) {}

AffineBool::AffineBool(bool constant, std::vector<SymbolId> terms)
    : constant_(constant), terms_(canonicalize_terms(std::move(terms))) {}

AffineBool AffineBool::symbol(SymbolId id) {
    return AffineBool(false, {id});
}

bool AffineBool::is_canonical() const {
    for (size_t i = 1; i < terms_.size(); ++i) {
        if (index(terms_[i - 1]) >= index(terms_[i])) {
            return false;
        }
    }
    return true;
}

AffineBool& AffineBool::operator^=(const AffineBool& other) {
    constant_ ^= other.constant_;
    terms_ = xor_terms(terms_, other.terms_);
    return *this;
}

AffineBool& AffineBool::operator^=(bool value) {
    constant_ ^= value;
    return *this;
}

AffineBool operator^(AffineBool left, const AffineBool& right) {
    left ^= right;
    return left;
}

AffineBool operator^(AffineBool left, bool right) {
    left ^= right;
    return left;
}

AffineBool operator^(bool left, AffineBool right) {
    right ^= left;
    return right;
}

bool ActivePauli::is_identity() const {
    return x == 0 && z == 0;
}

uint32_t predicted_dense_passes(const SamplingAction& action) {
    return std::visit(
        [](const auto& typed) -> uint32_t {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, RotateActivePauli>) {
                return typed.pauli.is_identity() ? 0 : 1;
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                return 1;
            } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                // Sampling requires one reduction before the branch is known,
                // then a second traversal to collapse and compact that branch.
                return 2;
            } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                switch (typed.mode) {
                    case InstrumentMode::Classical:
                    case InstrumentMode::DormantTrap:
                        return 0;
                    case InstrumentMode::Active:
                    case InstrumentMode::Activate:
                        // One population reduction followed by a filter or collapse pass.
                        return 2;
                }
                return 0;
            } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                return typed.active_projection.has_value() &&
                               !typed.active_projection->is_identity()
                           ? 1
                           : 0;
            } else if constexpr (std::is_same_v<T, MeasureDormantRandom> ||
                                 std::is_same_v<T, RecordClassical> ||
                                 std::is_same_v<T, DefineSymbol> ||
                                 std::is_same_v<T, ApplyReadoutNoise> ||
                                 std::is_same_v<T, WriteDetector> ||
                                 std::is_same_v<T, WriteObservable> ||
                                 std::is_same_v<T, InstrumentBoundary>) {
                return 0;
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
            }
        },
        action);
}

void SamplingPlan::validate() const {
    if (initial_active_width > num_qubits) {
        invalid_plan("initial active width exceeds qubit count");
    }
    if (max_active_width > num_qubits || max_active_width < initial_active_width) {
        invalid_plan("maximum active width is inconsistent with plan dimensions");
    }
    if (max_active_width >= kDenseActiveWidthLimit) {
        invalid_plan("maximum active width must be below " +
                     std::to_string(kDenseActiveWidthLimit) + " for dense coefficient storage");
    }
    const uint64_t total_records = static_cast<uint64_t>(num_visible_records) + num_hidden_records;
    if (total_records > std::numeric_limits<uint32_t>::max()) {
        invalid_plan("record count exceeds uint32 range");
    }
    if (!is_finite_robust(global_weight.real()) || !is_finite_robust(global_weight.imag())) {
        invalid_plan("global weight is not finite");
    }
    if (final_tableau.has_value() && final_tableau->num_qubits != num_qubits) {
        invalid_plan("final tableau width does not match the qubit count");
    }

    if (presampled_noise_sites.size() != num_noise_sites) {
        invalid_plan("presampled noise-site table does not match declared count");
    }
    if (instrument_distributions.size() != num_instrument_sites) {
        invalid_plan("instrument distribution table does not match declared count");
    }

    for (uint32_t site_index = 0; site_index < instrument_distributions.size(); ++site_index) {
        const InstrumentDistribution& distribution = instrument_distributions[site_index];
        if (index(distribution.site) != site_index) {
            invalid_plan("instrument distributions are not in stable id order");
        }
        for (uint8_t source = 0; source < 2; ++source) {
            if (!is_probability(distribution.p_fire[source])) {
                invalid_plan("instrument site " + std::to_string(site_index) +
                             " has an invalid fire probability");
            }
            double computational = 0.0;
            for (uint8_t destination = 0; destination < 2; ++destination) {
                const double probability = distribution.p_computational_dest[source][destination];
                if (!is_probability(probability)) {
                    invalid_plan("instrument site " + std::to_string(site_index) +
                                 " has an invalid destination probability");
                }
                computational += probability;
            }
            if (computational > distribution.p_fire[source] + 1e-12) {
                invalid_plan("instrument site " + std::to_string(site_index) +
                             " computational destinations exceed its fire probability");
            }
        }
    }

    std::vector<bool> bound_noise_symbols(symbols.size(), false);
    for (uint32_t site_index = 0; site_index < presampled_noise_sites.size(); ++site_index) {
        const PresampledNoiseSite& site = presampled_noise_sites[site_index];
        if (index(site.site) != site_index) {
            invalid_plan("presampled noise sites are not in stable id order");
        }
        double total_probability = 0.0;
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            const uint32_t symbol_index = index(outcome.symbol);
            if (symbol_index >= symbols.size() ||
                symbols[symbol_index].kind != SymbolKind::Presampled ||
                symbols[symbol_index].noise_site != site.site) {
                invalid_plan("noise site " + std::to_string(site_index) +
                             " references an incompatible symbol");
            }
            if (bound_noise_symbols[symbol_index]) {
                invalid_plan("presampled noise symbol s" + std::to_string(symbol_index) +
                             " is bound more than once");
            }
            if (!is_probability(outcome.probability) || outcome.probability == 0.0) {
                invalid_plan("noise site " + std::to_string(site_index) +
                             " has a nonpositive outcome probability");
            }
            bound_noise_symbols[symbol_index] = true;
            total_probability += outcome.probability;
        }
        if (!is_finite_robust(total_probability) || total_probability > 1.0 + 1e-12) {
            invalid_plan("noise site " + std::to_string(site_index) +
                         " outcome probabilities exceed one");
        }
    }

    std::vector<std::optional<uint32_t>> actual_definitions(symbols.size());
    for (uint32_t action_index = 0; action_index < actions.size(); ++action_index) {
        const auto symbol = defined_symbol(actions[action_index].action);
        if (!symbol.has_value()) {
            continue;
        }
        const uint32_t symbol_index = index(*symbol);
        if (symbol_index >= symbols.size()) {
            invalid_plan("action " + std::to_string(action_index) + " defines symbol s" +
                         std::to_string(symbol_index) + " out of range");
        }
        if (actual_definitions[symbol_index].has_value()) {
            invalid_plan("symbol s" + std::to_string(symbol_index) + " is defined more than once");
        }
        actual_definitions[symbol_index] = action_index;
    }

    for (uint32_t symbol_index = 0; symbol_index < symbols.size(); ++symbol_index) {
        const SymbolInfo& info = symbols[symbol_index];
        if (info.noise_site.has_value() &&
            (info.kind != SymbolKind::Presampled || index(*info.noise_site) >= num_noise_sites)) {
            invalid_plan("symbol s" + std::to_string(symbol_index) +
                         " has an invalid noise-site identity");
        }
        if (info.kind == SymbolKind::Unused) {
            if (info.defining_action.has_value() || info.noise_site.has_value() ||
                actual_definitions[symbol_index].has_value()) {
                invalid_plan("unused symbol s" + std::to_string(symbol_index) +
                             " must not have a definition or noise identity");
            }
            continue;
        }
        if (info.kind == SymbolKind::Presampled) {
            if (info.defining_action.has_value() || actual_definitions[symbol_index].has_value()) {
                invalid_plan("presampled symbol s" + std::to_string(symbol_index) +
                             " must be presampled");
            }
            if (info.noise_site.has_value() && !bound_noise_symbols[symbol_index]) {
                invalid_plan("noise symbol s" + std::to_string(symbol_index) +
                             " is absent from its site distribution");
            }
            continue;
        }
        if (!info.defining_action.has_value() ||
            info.defining_action != actual_definitions[symbol_index]) {
            invalid_plan("symbol s" + std::to_string(symbol_index) +
                         " definition metadata does not match its action");
        }
        const SamplingAction& action = actions[*info.defining_action].action;
        if (info.kind == SymbolKind::Branch &&
            !std::holds_alternative<MeasureActivePauli>(action) &&
            !std::holds_alternative<MeasureDormantRandom>(action)) {
            invalid_plan("branch symbol s" + std::to_string(symbol_index) +
                         " must be defined by a measurement");
        }
        if (info.kind == SymbolKind::Derived && !std::holds_alternative<DefineSymbol>(action)) {
            invalid_plan("derived symbol s" + std::to_string(symbol_index) +
                         " must be defined by DefineSymbol");
        }
        if (info.kind == SymbolKind::Readout &&
            !std::holds_alternative<ApplyReadoutNoise>(action)) {
            invalid_plan("readout symbol s" + std::to_string(symbol_index) +
                         " must be defined by ApplyReadoutNoise");
        }
        if (info.kind == SymbolKind::Instrument &&
            !std::holds_alternative<ApplyInstrument>(action)) {
            invalid_plan("instrument symbol s" + std::to_string(symbol_index) +
                         " must be defined by ApplyInstrument");
        }
    }

    uint32_t active_width = initial_active_width;
    uint32_t observed_max = active_width;
    std::unordered_set<uint32_t> written_records;
    std::unordered_set<uint32_t> written_detectors;
    std::unordered_set<uint32_t> written_observables;
    std::unordered_set<uint32_t> written_exp_vals;
    bool observed_postselection = false;
    uint32_t observed_instruments = 0;
    uint32_t observed_instrument_boundaries = 0;
    uint32_t previous_noise_boundary = 0;
    uint32_t previous_symbol_boundary = 0;
    written_records.reserve(std::min<size_t>(
        actions.size(), static_cast<uint64_t>(num_visible_records) + num_hidden_records));
    for (uint32_t action_index = 0; action_index < actions.size(); ++action_index) {
        const PlannedAction& planned = actions[action_index];
        if (planned.active_before != active_width || planned.active_before > num_qubits ||
            planned.active_after > num_qubits) {
            invalid_plan("action " + std::to_string(action_index) +
                         " breaks the active-width chain");
        }
        const uint32_t action_max_width = std::max(planned.active_before, planned.active_after);
        if (action_max_width >= kDenseActiveWidthLimit) {
            invalid_plan("action " + std::to_string(action_index) + " reaches active width " +
                         std::to_string(action_max_width) +
                         ", but dense storage requires widths below " +
                         std::to_string(kDenseActiveWidthLimit));
        }

        const std::optional<SymbolId> definition = defined_symbol(planned.action);
        if (final_tableau.has_value() &&
            !std::holds_alternative<RotateActivePauli>(planned.action) &&
            !std::holds_alternative<PromoteDormantRotation>(planned.action) &&
            !std::holds_alternative<WriteExpectationValue>(planned.action)) {
            invalid_plan("final tableau is retained for a nonunitary action stream");
        }
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, RotateActivePauli>) {
                    if (planned.active_after != planned.active_before) {
                        invalid_plan("active rotation changes active width");
                    }
                    validate_pauli(typed.pauli, planned.active_before, action_index);
                    if (!is_finite_robust(typed.half_turns)) {
                        invalid_plan("active rotation angle is not finite");
                    }
                    validate_expression(*this, typed.sign, action_index, definition, false);
                } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                    if (planned.active_after != planned.active_before + 1) {
                        invalid_plan("dormant promotion has an invalid width");
                    }
                    if (!is_finite_robust(typed.half_turns)) {
                        invalid_plan("dormant promotion angle is not finite");
                    }
                    validate_expression(*this, typed.sign, action_index, definition, false);
                } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                    if (planned.active_before == 0 ||
                        planned.active_after + 1 != planned.active_before ||
                        typed.active_pivot >= planned.active_before) {
                        invalid_plan("active measurement has an invalid width or pivot");
                    }
                    validate_pauli(typed.pauli, planned.active_before, action_index);
                    if (typed.pauli.is_identity()) {
                        invalid_plan("active measurement Pauli is identity");
                    }
                    validate_measurement_pivot(typed, action_index);
                    validate_measurement_outcome(*this, typed.outcome, typed.branch, action_index);
                    validate_record(*this, typed.record, action_index, written_records);
                } else if constexpr (std::is_same_v<T, MeasureDormantRandom>) {
                    if (planned.active_after != planned.active_before ||
                        typed.dormant_pivot < planned.active_before ||
                        typed.dormant_pivot >= num_qubits) {
                        invalid_plan("dormant measurement has an invalid width or pivot");
                    }
                    validate_measurement_outcome(*this, typed.outcome, typed.branch, action_index);
                    validate_record(*this, typed.record, action_index, written_records);
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    if (planned.active_after != planned.active_before) {
                        invalid_plan("classical record changes active width");
                    }
                    validate_expression(*this, typed.outcome, action_index, definition, false);
                    validate_record(*this, typed.record, action_index, written_records);
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    if (planned.active_after != planned.active_before) {
                        invalid_plan("symbol definition changes active width");
                    }
                    validate_expression(*this, typed.value, action_index, definition, false);
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    if (planned.active_after != planned.active_before ||
                        !is_probability(typed.prob_zero_to_one) ||
                        !is_probability(typed.prob_one_to_zero)) {
                        invalid_plan("readout noise has invalid width or probabilities");
                    }
                    validate_expression(*this, typed.source, action_index, definition, false);
                    validate_written_record(*this, typed.record, action_index, written_records);
                } else if constexpr (std::is_same_v<T, WriteDetector>) {
                    if (planned.active_after != planned.active_before ||
                        index(typed.detector) >= num_detectors ||
                        !written_detectors.insert(index(typed.detector)).second) {
                        invalid_plan("detector write has invalid width or slot");
                    }
                    validate_expression(*this, typed.outcome, action_index, definition, false);
                    observed_postselection |= typed.postselected;
                } else if constexpr (std::is_same_v<T, WriteObservable>) {
                    if (planned.active_after != planned.active_before ||
                        index(typed.observable) >= num_observables ||
                        !written_observables.insert(index(typed.observable)).second) {
                        invalid_plan("observable write has invalid width or slot");
                    }
                    validate_expression(*this, typed.outcome, action_index, definition, false);
                } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                    if (planned.active_after != planned.active_before ||
                        index(typed.exp_val) >= num_exp_vals ||
                        !written_exp_vals.insert(index(typed.exp_val)).second) {
                        invalid_plan("expectation write has invalid width or slot");
                    }
                    if (typed.active_projection.has_value()) {
                        validate_pauli(*typed.active_projection, planned.active_before,
                                       action_index);
                    } else if (typed.sign != AffineBool{}) {
                        invalid_plan("zero expectation write has an irrelevant symbolic sign");
                    }
                    validate_expression(*this, typed.sign, action_index, definition, false);
                } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                    if (index(typed.site) != observed_instruments ||
                        index(typed.site) >= num_instrument_sites) {
                        invalid_plan("instrument action has an invalid or out-of-order site id");
                    }
                    const bool width_unchanged = planned.active_after == planned.active_before;
                    switch (typed.mode) {
                        case InstrumentMode::Classical:
                            if (!width_unchanged || !typed.source.is_identity()) {
                                invalid_plan("classical instrument has invalid width or source");
                            }
                            break;
                        case InstrumentMode::Active:
                            if (!width_unchanged || typed.source.is_identity()) {
                                invalid_plan("active instrument has invalid width or source");
                            }
                            validate_pauli(typed.source, planned.active_before, action_index);
                            break;
                        case InstrumentMode::Activate:
                            if (planned.active_after != planned.active_before + 1 ||
                                typed.source.is_identity()) {
                                invalid_plan("activating instrument has invalid width or source");
                            }
                            validate_pauli(typed.source, planned.active_after, action_index);
                            break;
                        case InstrumentMode::DormantTrap:
                            if (!width_unchanged || !typed.source.is_identity() ||
                                typed.destination_flip.has_value() || typed.sign != AffineBool{}) {
                                invalid_plan("dormant trap instrument has incompatible fields");
                            }
                            break;
                    }
                    if (typed.mode != InstrumentMode::DormantTrap &&
                        !typed.destination_flip.has_value()) {
                        invalid_plan("in-line instrument omits its destination-flip symbol");
                    }
                    validate_expression(*this, typed.sign, action_index, definition, false);
                    ++observed_instruments;
                } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                    if (planned.active_after != planned.active_before ||
                        index(typed.site) >= num_instrument_sites ||
                        typed.next_noise_site > num_noise_sites ||
                        typed.next_noise_site < previous_noise_boundary || action_index == 0 ||
                        index(typed.site) != observed_instrument_boundaries ||
                        typed.symbol_prefix_size > symbols.size() ||
                        typed.symbol_prefix_size < previous_symbol_boundary) {
                        invalid_plan("instrument boundary has an invalid width or site id");
                    }
                    const auto* instrument =
                        std::get_if<ApplyInstrument>(&actions[action_index - 1].action);
                    if (instrument == nullptr || instrument->site != typed.site) {
                        invalid_plan("instrument boundary must immediately follow its action");
                    }
                    previous_noise_boundary = typed.next_noise_site;
                    previous_symbol_boundary = typed.symbol_prefix_size;
                    ++observed_instrument_boundaries;
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
                }
            },
            planned.action);

        active_width = planned.active_after;
        observed_max = std::max(observed_max, active_width);
    }
    if (observed_max != max_active_width) {
        invalid_plan("declared maximum active width does not match the action stream");
    }
    if (written_records.size() != total_records) {
        invalid_plan("declared record count does not match the action stream");
    }
    if (written_detectors.size() != num_detectors ||
        written_observables.size() != num_observables || written_exp_vals.size() != num_exp_vals) {
        invalid_plan(
            "declared detector, observable, or expectation count does not match the action stream");
    }
    if (observed_postselection != has_postselection) {
        invalid_plan("declared postselection flag does not match detector actions");
    }
    if (observed_instruments != num_instrument_sites ||
        observed_instrument_boundaries != num_instrument_sites) {
        invalid_plan("declared instrument count does not match the action stream");
    }
}

std::string SamplingPlan::inspect() const {
    validate();

    std::ostringstream out;
    out << std::setprecision(17);
    out << "sampling_plan qubits=" << num_qubits << " initial_width=" << initial_active_width
        << " max_width=" << max_active_width << " visible_records=" << num_visible_records
        << " hidden_records=" << num_hidden_records << " noise_sites=" << num_noise_sites
        << " instrument_sites=" << num_instrument_sites << " detectors=" << num_detectors
        << " observables=" << num_observables << " exp_vals=" << num_exp_vals
        << " postselection=" << has_postselection
        << " final_state_queries=" << final_tableau.has_value()
        << " dust_epsilon=" << kMeasurementDustEpsilon << '\n';
    out << "global_weight=" << global_weight.real() << ',' << global_weight.imag() << '\n';
    out << "symbols=" << symbols.size() << '\n';
    for (uint32_t i = 0; i < symbols.size(); ++i) {
        out << "  s" << i << " kind=" << symbol_kind_name(symbols[i].kind);
        if (symbols[i].defining_action.has_value()) {
            out << " action=" << *symbols[i].defining_action;
        }
        if (symbols[i].noise_site.has_value()) {
            out << " noise_site=" << index(*symbols[i].noise_site);
        }
        out << '\n';
    }
    for (const PresampledNoiseSite& site : presampled_noise_sites) {
        out << "  noise_site " << index(site.site) << " outcomes=" << site.outcomes.size();
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            out << " s" << index(outcome.symbol) << ':' << outcome.probability;
        }
        out << '\n';
    }
    out << "actions=" << actions.size() << '\n';
    for (uint32_t i = 0; i < actions.size(); ++i) {
        const PlannedAction& planned = actions[i];
        out << "  " << i << " active_width=" << planned.active_before << "->"
            << planned.active_after << " dense_passes=" << predicted_dense_passes(planned.action)
            << ' ';
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, RotateActivePauli>) {
                    out << "rotate_active " << format_pauli(typed.pauli)
                        << " half_turns=" << typed.half_turns
                        << " sign=" << format_expression(typed.sign);
                } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                    out << "promote_dormant half_turns=" << typed.half_turns
                        << " sign=" << format_expression(typed.sign);
                } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                    out << "measure_active " << format_pauli(typed.pauli)
                        << " pivot=" << typed.active_pivot << " branch=s" << index(typed.branch)
                        << " outcome=" << format_expression(typed.outcome)
                        << " record=" << index(typed.record);
                } else if constexpr (std::is_same_v<T, MeasureDormantRandom>) {
                    out << "measure_dormant pivot=" << typed.dormant_pivot << " branch=s"
                        << index(typed.branch) << " outcome=" << format_expression(typed.outcome)
                        << " record=" << index(typed.record);
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    out << "record_classical outcome=" << format_expression(typed.outcome)
                        << " record=" << index(typed.record);
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    out << "define_symbol s" << index(typed.symbol)
                        << " value=" << format_expression(typed.value);
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    out << "readout_noise s" << index(typed.flip)
                        << " source=" << format_expression(typed.source)
                        << " record=" << index(typed.record) << " p01=" << typed.prob_zero_to_one
                        << " p10=" << typed.prob_one_to_zero;
                } else if constexpr (std::is_same_v<T, WriteDetector>) {
                    out << "write_detector outcome=" << format_expression(typed.outcome)
                        << " detector=" << index(typed.detector)
                        << " postselected=" << typed.postselected;
                } else if constexpr (std::is_same_v<T, WriteObservable>) {
                    out << "write_observable outcome=" << format_expression(typed.outcome)
                        << " observable=" << index(typed.observable);
                } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                    out << "write_expectation ";
                    if (typed.active_projection.has_value()) {
                        out << format_pauli(*typed.active_projection)
                            << " sign=" << format_expression(typed.sign);
                    } else {
                        out << "zero";
                    }
                    out << " exp_val=" << index(typed.exp_val);
                } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                    out << "apply_instrument site=" << index(typed.site)
                        << " mode=" << instrument_mode_name(typed.mode) << ' '
                        << format_pauli(typed.source) << " sign=" << format_expression(typed.sign);
                    if (typed.destination_flip.has_value()) {
                        out << " flip=s" << index(*typed.destination_flip);
                    }
                } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                    out << "instrument_boundary site=" << index(typed.site)
                        << " next_noise_site=" << typed.next_noise_site
                        << " symbol_prefix_size=" << typed.symbol_prefix_size;
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
                }
            },
            planned.action);
        out << '\n';
    }
    return out.str();
}

}  // namespace clifft::sampling
