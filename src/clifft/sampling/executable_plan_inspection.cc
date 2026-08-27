#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/inspection_format.h"

#include <bit>
#include <cassert>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace clifft::sampling {

namespace {

template <class... Visitors>
struct Overloaded : Visitors... {
    using Visitors::operator()...;
};

template <class... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

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
        case ActiveMeasurementKernel::Diagonal:
            return "diagonal";
        case ActiveMeasurementKernel::HighPivot:
            return "high_pivot";
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
    out << 'w' << pauli.active_width << ' ' << format_pauli_product(pauli.x, pauli.z);
    if (pauli.pairing_bit != 0) {
        out << " pair=" << std::countr_zero(pauli.pairing_bit);
    }
}

}  // namespace

std::string ExecutablePlan::inspect() const {
    std::ostringstream out;
    out << "executable_plan backend=" << backend_name(backend_) << " actions=" << actions_.size()
        << " expression_registers=" << expression_register_constants_.size()
        << " fused_rotations=" << fused_rotations_.size()
        << " dynamic_fused_rotations=" << dynamic_fused_rotations_.size() << '\n';
    for (size_t i = 0; i < actions_.size(); ++i) {
        out << "  " << i;
        if (const auto range = action_plan_range(i)) {
            out << " plans=[" << range->begin << ',' << range->end << ')';
        }
        out << ' ' << inspect_action(i) << '\n';
    }
    return out.str();
}

std::string ExecutablePlan::inspect_action(size_t action) const {
    const Action& selected = actions_.at(action);
    std::ostringstream out;
    const auto write_record_parity = [&](PreparedRecordParity parity) {
        const size_t end = static_cast<size_t>(parity.begin) + parity.count;
        assert(end <= record_parity_terms_.size() &&
               "inspected record parity must stay in its term tape");
        bool wrote = false;
        if (parity.constant) {
            out << '1';
            wrote = true;
        }
        for (size_t term = parity.begin; term < end; ++term) {
            if (wrote) {
                out << '^';
            }
            out << 'r' << record_parity_terms_[term];
            wrote = true;
        }
        if (!wrote) {
            out << '0';
        }
    };
    const auto write_observable_value = [&](const PreparedObservableValue& value) {
        if (const auto* expression = std::get_if<PreparedExpression>(&value)) {
            out << 'e' << expression->register_id;
        } else {
            write_record_parity(std::get<PreparedRecordParity>(value));
        }
    };
    std::visit(
        Overloaded{
            [&](const ExecuteRotation& typed) {
                out << "ROTATE ";
                write_prepared_pauli(out, typed.rotation.pauli);
                out << " cos=" << format_double_roundtrip(typed.rotation.cosine)
                    << " sin=" << format_double_roundtrip(typed.rotation.sine) << " sign=e"
                    << typed.sign.register_id
                    << " kernel=" << direct_rotation_kernel_name(typed.kernel);
            },
            [&](const ExecuteFusedRotation& typed) {
                out << "FUSED_ROTATION descriptor=" << typed.rotation_index;
            },
            [&](const ExecuteDynamicFusedRotation& typed) {
                out << "DYNAMIC_FUSED_ROTATION descriptor=" << typed.rotation_index;
            },
            [&](const ExecutePromotion& typed) {
                out << "PROMOTE cos=" << format_double_roundtrip(typed.promotion.cosine)
                    << " sin=" << format_double_roundtrip(typed.promotion.sine) << " sign=e"
                    << typed.sign.register_id;
            },
            [&](const ExecuteActiveMeasurement& typed) {
                out << "MEASURE_ACTIVE ";
                write_prepared_pauli(out, typed.measurement.pauli);
                out << " pivot=" << typed.measurement.pivot << " branch=s" << typed.branch
                    << " record=r" << typed.record << " correction=e"
                    << typed.correction.register_id
                    << " kernel=" << active_measurement_kernel_name(typed.kernel);
            },
            [&](const ExecuteDormantMeasurement& typed) {
                out << "MEASURE_DORMANT branch=s" << typed.branch << " record=r" << typed.record
                    << " correction=e" << typed.correction.register_id;
            },
            [&](const ExecuteClassicalRecord& typed) {
                out << "RECORD_CLASSICAL record=r" << typed.record << " outcome=e"
                    << typed.outcome.register_id;
            },
            [&](const ExecuteSymbolDefinition& typed) {
                out << "DEFINE_SYMBOL s" << typed.symbol << " value=e" << typed.value.register_id;
            },
            [&](const ExecuteReadoutNoise& typed) {
                out << "READOUT_NOISE site=" << typed.site << " flip=s" << typed.flip << " record=r"
                    << typed.record << " source=e" << typed.source.register_id
                    << " p01=" << format_double_roundtrip(typed.prob_zero_to_one)
                    << " p10=" << format_double_roundtrip(typed.prob_one_to_zero);
            },
            [&](const ExecuteDetector& typed) {
                out << "WRITE_DETECTOR detector=d" << typed.detector << " outcome=";
                write_record_parity(typed.outcome);
                if (typed.postselected) {
                    out << " postselect";
                }
            },
            [&](const ExecuteObservable& typed) {
                out << "WRITE_OBSERVABLE observable=o" << typed.observable << " outcome=";
                write_observable_value(typed.outcome);
            },
            [&](const ExecuteExpectation& typed) {
                out << "WRITE_EXPECTATION exp_val=v" << typed.exp_val << ' ';
                if (typed.active.has_value()) {
                    write_prepared_pauli(out, typed.active->projection);
                    out << " sign=e" << typed.active->sign.register_id;
                } else {
                    out << "zero";
                }
            },
            [&](const ExecuteInstrument& typed) {
                std::visit(
                    Overloaded{
                        [&](const ExecuteDormantInstrumentTrap& instrument) {
                            out << "INSTRUMENT_DORMANT_TRAP site=" << instrument.site;
                        },
                        [&](const ExecuteClassicalInstrument& instrument) {
                            out << "INSTRUMENT_CLASSICAL site=" << instrument.site << " sign=e"
                                << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        },
                        [&](const ExecuteActiveInstrument& instrument) {
                            out << "INSTRUMENT_ACTIVE site=" << instrument.site << ' ';
                            write_prepared_pauli(out, instrument.measurement.pauli);
                            out << " sign=e" << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        },
                        [&](const ExecuteMeasuredInstrumentActivation& instrument) {
                            out << "INSTRUMENT_ACTIVATE_MEASURED site=" << instrument.site << ' ';
                            write_prepared_pauli(out, instrument.measurement.pauli);
                            out << " sign=e" << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        },
                        [&](const ExecuteNewXInstrumentActivation& instrument) {
                            out << "INSTRUMENT_ACTIVATE_NEW_X site=" << instrument.site << " sign=e"
                                << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip
                                << " kernel=" << new_x_instrument_kernel_name(instrument.kernel);
                        },
                    },
                    typed.form);
            },
            [&](const ExecuteBoundary& typed) {
                out << "INSTRUMENT_BOUNDARY site=" << typed.site
                    << " active_width=" << typed.active_width << " noise=[" << typed.noise_begin
                    << ',' << typed.noise_end << ')'
                    << " symbol_prefix_size=" << typed.symbol_prefix_size;
            },
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
