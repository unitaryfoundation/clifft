#include "clifft/sampling/inspection_format.h"
#include "clifft/sampling/plan.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <type_traits>

namespace clifft::sampling {

namespace {

constexpr size_t kCompactInspectionExpressionTerms = 4;

template <typename>
inline constexpr bool kAlwaysFalse = false;

std::string format_expression(const AffineBool& expression,
                              std::optional<size_t> max_terms = std::nullopt) {
    // Unspaced XOR notation: the leading constant (if any) does not count
    // against max_terms, which bounds only the number of symbol terms shown.
    std::string out;
    bool wrote = false;
    if (expression.constant()) {
        out += '1';
        wrote = true;
    }
    const size_t terms_to_write =
        std::min(expression.terms().size(), max_terms.value_or(expression.terms().size()));
    for (SymbolId term : std::span(expression.terms()).first(terms_to_write)) {
        if (wrote) {
            out += '^';
        }
        out += 's';
        out += std::to_string(index(term));
        wrote = true;
    }
    const size_t omitted = expression.terms().size() - terms_to_write;
    if (omitted > 0) {
        if (wrote) {
            out += '^';
        }
        out += "...(+" + std::to_string(omitted) + ")";
        wrote = true;
    }
    if (!wrote) {
        out += '0';
    }
    return out;
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

// Writes the mnemonic, operand, and key=value fields for one action, without
// the leading active-width prefix or dense-pass count. Both the full and
// compact inspection forms share this body and differ only in how they wrap
// it (see write_action_inspection_full / write_action_inspection_compact).
void write_action_body(std::ostream& out, const SamplingAction& action,
                       std::optional<size_t> max_expression_terms) {
    std::visit(
        [&](const auto& typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, RotateActivePauli>) {
                out << "ROTATE_ACTIVE " << format_pauli_product(typed.pauli.x, typed.pauli.z)
                    << " half_turns=" << format_double_roundtrip(typed.half_turns)
                    << " sign=" << format_expression(typed.sign, max_expression_terms);
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                out << "PROMOTE_DORMANT half_turns=" << format_double_roundtrip(typed.half_turns)
                    << " sign=" << format_expression(typed.sign, max_expression_terms);
            } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                out << "MEASURE_ACTIVE " << format_pauli_product(typed.pauli.x, typed.pauli.z)
                    << " pivot=" << typed.active_pivot << " branch=s" << index(typed.branch)
                    << " outcome=" << format_expression(typed.outcome, max_expression_terms)
                    << " record=r" << index(typed.record);
            } else if constexpr (std::is_same_v<T, MeasureDormantRandom>) {
                out << "MEASURE_DORMANT pivot=" << typed.dormant_pivot << " branch=s"
                    << index(typed.branch)
                    << " outcome=" << format_expression(typed.outcome, max_expression_terms)
                    << " record=r" << index(typed.record);
            } else if constexpr (std::is_same_v<T, RecordClassical>) {
                out << "RECORD_CLASSICAL outcome="
                    << format_expression(typed.outcome, max_expression_terms) << " record=r"
                    << index(typed.record);
            } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                out << "DEFINE_SYMBOL s" << index(typed.symbol)
                    << " value=" << format_expression(typed.value, max_expression_terms);
            } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                out << "READOUT_NOISE flip=s" << index(typed.flip)
                    << " source=" << format_expression(typed.source, max_expression_terms)
                    << " record=r" << index(typed.record)
                    << " p01=" << format_double_roundtrip(typed.prob_zero_to_one)
                    << " p10=" << format_double_roundtrip(typed.prob_one_to_zero);
            } else if constexpr (std::is_same_v<T, WriteDetector>) {
                out << "WRITE_DETECTOR outcome="
                    << format_expression(typed.outcome, max_expression_terms) << " detector=d"
                    << index(typed.detector);
                if (typed.postselected) {
                    out << " postselect";
                }
            } else if constexpr (std::is_same_v<T, WriteObservable>) {
                out << "WRITE_OBSERVABLE outcome="
                    << format_expression(typed.outcome, max_expression_terms) << " observable=o"
                    << index(typed.observable);
            } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                out << "WRITE_EXPECTATION ";
                if (typed.active_projection.has_value()) {
                    out << format_pauli_product(typed.active_projection->x,
                                                typed.active_projection->z)
                        << " sign=" << format_expression(typed.sign, max_expression_terms);
                } else {
                    out << "zero";
                }
                out << " exp_val=v" << index(typed.exp_val);
            } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                out << "APPLY_INSTRUMENT ";
                if (!typed.source.is_identity()) {
                    out << format_pauli_product(typed.source.x, typed.source.z) << ' ';
                }
                out << "site=" << index(typed.site) << " mode=" << instrument_mode_name(typed.mode)
                    << " sign=" << format_expression(typed.sign, max_expression_terms);
                if (typed.destination_flip.has_value()) {
                    out << " flip=s" << index(*typed.destination_flip);
                }
            } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                out << "INSTRUMENT_BOUNDARY site=" << index(typed.site)
                    << " next_noise_site=" << typed.next_noise_site
                    << " symbol_prefix_size=" << typed.symbol_prefix_size;
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
            }
        },
        action);
}

// Full form: <width> dense_passes=<n> <MNEMONIC> ... . Affine expressions are
// never truncated.
void write_action_inspection_full(std::ostream& out, const PlannedAction& planned) {
    out << format_width_prefix(planned.active_before, planned.active_after)
        << " dense_passes=" << predicted_dense_passes(planned.action) << ' ';
    write_action_body(out, planned.action, std::nullopt);
}

// Compact form: <width> <MNEMONIC> ... [passes=<n>]. Affine expressions are
// bounded to kCompactInspectionExpressionTerms symbol terms, and the trailing
// pass count is written only for actions that touch the dense coefficient
// state.
void write_action_inspection_compact(std::ostream& out, const PlannedAction& planned) {
    out << format_width_prefix(planned.active_before, planned.active_after) << ' ';
    write_action_body(out, planned.action, kCompactInspectionExpressionTerms);
    const uint32_t passes = predicted_dense_passes(planned.action);
    if (passes > 0) {
        out << " passes=" << passes;
    }
}

}  // namespace

std::string SamplingPlan::inspect() const {
    validate();

    std::ostringstream out;
    out << "sampling_plan qubits=" << num_qubits << " initial_width=" << initial_active_width
        << " peak_width=" << peak_active_width << " visible_records=" << num_visible_records
        << " hidden_records=" << num_hidden_records << " noise_sites=" << num_noise_sites
        << " instrument_sites=" << num_instrument_sites << " detectors=" << num_detectors
        << " observables=" << num_observables << " exp_vals=" << num_exp_vals
        << " postselection=" << has_postselection
        << " final_state_queries=" << final_tableau.has_value()
        << " dust_epsilon=" << format_double_roundtrip(kMeasurementDustEpsilon) << '\n';
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
        out << "  noise_site " << index(site.site)
            << " probability=" << format_double_roundtrip(site.total_probability)
            << " outcomes=" << site.outcomes.size();
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            out << " s" << index(outcome.symbol) << ':'
                << format_double_roundtrip(outcome.probability);
        }
        out << '\n';
    }
    out << "actions=" << actions.size() << '\n';
    for (uint32_t i = 0; i < actions.size(); ++i) {
        out << "  " << i << ' ' << inspect_action(i) << '\n';
    }
    return out.str();
}

std::string SamplingPlan::inspect_action(size_t action) const {
    const PlannedAction& planned = actions.at(action);
    std::ostringstream out;
    write_action_inspection_full(out, planned);
    return out.str();
}

std::string SamplingPlan::inspect_action_compact(size_t action) const {
    const PlannedAction& planned = actions.at(action);
    std::ostringstream out;
    write_action_inspection_compact(out, planned);
    return out.str();
}

}  // namespace clifft::sampling
