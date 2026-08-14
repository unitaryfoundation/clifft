#include "clifft/sampling/executable_plan.h"

#include "clifft/sampling/executable_plan_builder.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace clifft::sampling {

namespace {

template <typename>
inline constexpr bool kAlwaysFalse = false;

std::string_view backend_name(ExecutorBackend backend) {
    switch (backend) {
        case ExecutorBackend::Scalar:
            return "scalar";
        case ExecutorBackend::Avx2:
            return "avx2";
        case ExecutorBackend::Avx512:
            return "avx512";
    }
    return "unknown";
}

std::string_view direct_rotation_kernel_name(DirectRotationKernel kernel) {
    switch (kernel) {
        case DirectRotationKernel::Scalar:
            return "scalar";
        case DirectRotationKernel::Diagonal:
            return "diagonal";
        case DirectRotationKernel::HighPivot:
            return "high_pivot";
        case DirectRotationKernel::LanePaired:
            return "lane_paired";
    }
    return "unknown";
}

std::string_view active_measurement_kernel_name(ActiveMeasurementKernel kernel) {
    switch (kernel) {
        case ActiveMeasurementKernel::Scalar:
            return "scalar";
        case ActiveMeasurementKernel::LanePaired:
            return "lane_paired";
    }
    return "unknown";
}

std::string_view new_x_instrument_kernel_name(NewXInstrumentKernel kernel) {
    switch (kernel) {
        case NewXInstrumentKernel::Scalar:
            return "scalar";
        case NewXInstrumentKernel::Vectorized:
            return "vectorized";
    }
    return "unknown";
}

void write_prepared_pauli(std::ostream& out, const PreparedPauli& pauli) {
    out << "active_width=" << pauli.active_width << " x=0x" << std::hex << pauli.x << " z=0x"
        << pauli.z << std::dec << " pairing_bit=" << pauli.pairing_bit;
}

}  // namespace

PreparedFusedRotationExecution::PreparedFusedRotationExecution(PreparedFusedRotation rotation,
                                                               ExecutorBackend backend)
    : rotation_(std::move(rotation)) {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    switch (backend) {
        case ExecutorBackend::Scalar:
            break;
        case ExecutorBackend::Avx2:
            sidecar_ = prepare_fused_rotation_avx2_sidecar(rotation_);
            break;
        case ExecutorBackend::Avx512:
            sidecar_ = prepare_fused_rotation_avx512_sidecar(rotation_);
            break;
    }
#else
    (void)backend;
#endif
}

ExecutablePlan::ExecutablePlan(const SamplingPlan& plan)
    : num_qubits_(plan.num_qubits),
      initial_active_width_(plan.initial_active_width),
      max_active_width_(plan.max_active_width),
      num_visible_records_(plan.num_visible_records),
      num_hidden_records_(plan.num_hidden_records),
      num_detectors_(plan.num_detectors),
      num_observables_(plan.num_observables),
      num_exp_vals_(plan.num_exp_vals),
      has_postselection_(plan.has_postselection),
      global_weight_(plan.global_weight),
      final_tableau_(plan.final_tableau),
      instrument_distributions_(plan.instrument_distributions) {
    // Keep construction-only lowering state out of the immutable executable.
    ExecutablePlanBuilder::build(*this, plan);
}

void ExecutablePlan::ExpressionDependencies::validate(uint32_t num_symbols,
                                                      size_t num_registers) const noexcept {
#ifndef NDEBUG
    assert(offsets_.size() == static_cast<size_t>(num_symbols) + 1 &&
           "expression dependency offsets have the wrong size");
    assert(!offsets_.empty() && offsets_.front() == 0 && offsets_.back() == targets_.size() &&
           "expression dependency ranges are inconsistent");
    for (size_t i = 1; i < offsets_.size(); ++i) {
        assert(offsets_[i] >= offsets_[i - 1] && "expression dependency offsets are not ordered");
    }
    for (uint32_t target : targets_) {
        assert(target < num_registers && "expression dependency target is out of range");
    }
#else
    static_cast<void>(num_symbols);
    static_cast<void>(num_registers);
#endif
}

size_t ExecutablePlan::num_new_x_instrument_activations() const {
    return static_cast<size_t>(
        std::count_if(actions_.begin(), actions_.end(), [](const Action& action) {
            const auto* instrument = std::get_if<ExecuteInstrument>(&action);
            return instrument != nullptr &&
                   std::holds_alternative<ExecuteNewXInstrumentActivation>(instrument->form);
        }));
}

std::vector<double> ExecutablePlan::noise_site_probabilities() const {
    std::vector<double> probabilities;
    probabilities.reserve(noise_sites_.size() + num_readout_noise_sites_);
    for (const PreparedNoiseSite& site : noise_sites_) {
        probabilities.push_back(site.conditioned_probability);
    }
    for (const Action& action : actions_) {
        const auto* readout = std::get_if<ExecuteReadoutNoise>(&action);
        if (readout == nullptr) {
            continue;
        }
        if (readout->prob_zero_to_one != readout->prob_one_to_zero) {
            throw std::invalid_argument(
                "k-fault conditioning does not support asymmetric readout noise; "
                "measurement record index " +
                std::to_string(readout->record) + " has probabilities (" +
                std::to_string(readout->prob_zero_to_one) + ", " +
                std::to_string(readout->prob_one_to_zero) + ")");
        }
        probabilities.push_back(readout->prob_zero_to_one);
    }
    return probabilities;
}

std::string ExecutablePlan::inspect() const {
    std::ostringstream out;
    out << "executable_plan backend=" << backend_name(backend_) << " actions=" << actions_.size()
        << " expression_registers=" << expression_register_constants_.size()
        << " fused_rotations=" << fused_rotations_.size()
        << " dynamic_fused_rotations=" << dynamic_fused_rotations_.size() << '\n';
    for (size_t i = 0; i < actions_.size(); ++i) {
        out << "  " << i;
        if (const auto range = action_plan_range(i)) {
            out << " plans=" << range->begin << ".." << (range->end - 1);
        }
        out << ' ' << inspect_action(i) << '\n';
    }
    return out.str();
}

std::string ExecutablePlan::inspect_action(size_t action) const {
    const Action& selected = actions_.at(action);
    std::ostringstream out;
    out << std::setprecision(17);
    std::visit(
        [&](const auto& typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, ExecuteRotation>) {
                out << "rotate ";
                write_prepared_pauli(out, typed.rotation.pauli);
                out << " sign=e" << typed.sign.register_id
                    << " kernel=" << direct_rotation_kernel_name(typed.kernel);
            } else if constexpr (std::is_same_v<T, ExecuteFusedRotation>) {
                out << "fused_rotation descriptor=" << typed.rotation_index;
            } else if constexpr (std::is_same_v<T, ExecuteDynamicFusedRotation>) {
                out << "dynamic_fused_rotation descriptor=" << typed.rotation_index;
            } else if constexpr (std::is_same_v<T, ExecutePromotion>) {
                out << "promote cosine=" << typed.promotion.cosine
                    << " sine=" << typed.promotion.sine << " sign=e" << typed.sign.register_id;
            } else if constexpr (std::is_same_v<T, ExecuteActiveMeasurement>) {
                out << "measure_active ";
                write_prepared_pauli(out, typed.measurement.pauli);
                out << " pivot=" << typed.measurement.pivot << " branch=s" << typed.branch
                    << " record=r" << typed.record << " correction=e"
                    << typed.correction.register_id
                    << " kernel=" << active_measurement_kernel_name(typed.kernel);
            } else if constexpr (std::is_same_v<T, ExecuteDormantMeasurement>) {
                out << "measure_dormant branch=s" << typed.branch << " record=r" << typed.record
                    << " correction=e" << typed.correction.register_id;
            } else if constexpr (std::is_same_v<T, ExecuteClassicalRecord>) {
                out << "record_classical record=r" << typed.record << " outcome=e"
                    << typed.outcome.register_id;
            } else if constexpr (std::is_same_v<T, ExecuteSymbolDefinition>) {
                out << "define_symbol s" << typed.symbol << " value=e" << typed.value.register_id;
            } else if constexpr (std::is_same_v<T, ExecuteReadoutNoise>) {
                out << "readout_noise site=" << typed.site << " flip=s" << typed.flip << " record=r"
                    << typed.record << " source=e" << typed.source.register_id
                    << " p01=" << typed.prob_zero_to_one << " p10=" << typed.prob_one_to_zero;
            } else if constexpr (std::is_same_v<T, ExecuteDetector>) {
                out << "write_detector detector=d" << typed.detector << " outcome=e"
                    << typed.outcome.register_id << " postselected=" << typed.postselected;
            } else if constexpr (std::is_same_v<T, ExecuteObservable>) {
                out << "write_observable observable=l" << typed.observable << " outcome=e"
                    << typed.outcome.register_id;
            } else if constexpr (std::is_same_v<T, ExecuteExpectation>) {
                out << "write_expectation exp_val=" << typed.exp_val << ' ';
                if (typed.active_projection.has_value()) {
                    write_prepared_pauli(out, *typed.active_projection);
                    out << " sign=e" << typed.sign.register_id;
                } else {
                    out << "zero";
                }
            } else if constexpr (std::is_same_v<T, ExecuteInstrument>) {
                std::visit(
                    [&](const auto& instrument) {
                        using Instrument = std::decay_t<decltype(instrument)>;
                        if constexpr (std::is_same_v<Instrument, ExecuteDormantInstrumentTrap>) {
                            out << "instrument_dormant_trap site=" << instrument.site;
                        } else if constexpr (std::is_same_v<Instrument,
                                                            ExecuteClassicalInstrument>) {
                            out << "instrument_classical site=" << instrument.site << " sign=e"
                                << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        } else if constexpr (std::is_same_v<Instrument, ExecuteActiveInstrument>) {
                            out << "instrument_active site=" << instrument.site << ' ';
                            write_prepared_pauli(out, instrument.measurement.pauli);
                            out << " sign=e" << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        } else if constexpr (std::is_same_v<Instrument,
                                                            ExecuteMeasuredInstrumentActivation>) {
                            out << "instrument_activate_measured site=" << instrument.site << ' ';
                            write_prepared_pauli(out, instrument.measurement.pauli);
                            out << " sign=e" << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        } else if constexpr (std::is_same_v<Instrument,
                                                            ExecuteNewXInstrumentActivation>) {
                            out << "instrument_activate_new_x site=" << instrument.site << " sign=e"
                                << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip
                                << " kernel=" << new_x_instrument_kernel_name(instrument.kernel);
                        } else {
                            static_assert(kAlwaysFalse<Instrument>,
                                          "Unhandled executable instrument alternative");
                        }
                    },
                    typed.form);
            } else if constexpr (std::is_same_v<T, ExecuteBoundary>) {
                out << "instrument_boundary site=" << typed.site
                    << " active_width=" << typed.active_width << " noise=" << typed.noise_begin
                    << ".." << typed.noise_end
                    << " symbol_prefix_size=" << typed.symbol_prefix_size;
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled executable action alternative");
            }
        },
        selected);
    return out.str();
}

std::optional<ExecutablePlan::PlanActionRange> ExecutablePlan::action_plan_range(
    size_t action) const {
    if (action >= actions_.size()) {
        throw std::out_of_range("executable action is out of range");
    }
    if (action_plan_ranges_.empty()) {
        return std::nullopt;
    }
    assert(action_plan_ranges_.size() == actions_.size() &&
           "executable action provenance must remain parallel");
    return action_plan_ranges_[action];
}

}  // namespace clifft::sampling
