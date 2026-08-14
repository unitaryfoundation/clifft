#include "clifft/sampling/executable_plan.h"

#include <cassert>
#include <iomanip>
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
        << pauli.z << " pairing_bit=0x" << pauli.pairing_bit << std::dec;
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
    out << std::setprecision(17);
    std::visit(
        Overloaded{
            [&](const ExecuteRotation& typed) {
                out << "rotate ";
                write_prepared_pauli(out, typed.rotation.pauli);
                out << " cosine=" << typed.rotation.cosine << " sine=" << typed.rotation.sine
                    << " sign=e" << typed.sign.register_id
                    << " kernel=" << direct_rotation_kernel_name(typed.kernel);
            },
            [&](const ExecuteFusedRotation& typed) {
                out << "fused_rotation descriptor=" << typed.rotation_index;
            },
            [&](const ExecuteDynamicFusedRotation& typed) {
                out << "dynamic_fused_rotation descriptor=" << typed.rotation_index;
            },
            [&](const ExecutePromotion& typed) {
                out << "promote cosine=" << typed.promotion.cosine
                    << " sine=" << typed.promotion.sine << " sign=e" << typed.sign.register_id;
            },
            [&](const ExecuteActiveMeasurement& typed) {
                out << "measure_active ";
                write_prepared_pauli(out, typed.measurement.pauli);
                out << " pivot=" << typed.measurement.pivot << " branch=s" << typed.branch
                    << " record=r" << typed.record << " correction=e"
                    << typed.correction.register_id
                    << " kernel=" << active_measurement_kernel_name(typed.kernel);
            },
            [&](const ExecuteDormantMeasurement& typed) {
                out << "measure_dormant branch=s" << typed.branch << " record=r" << typed.record
                    << " correction=e" << typed.correction.register_id;
            },
            [&](const ExecuteClassicalRecord& typed) {
                out << "record_classical record=r" << typed.record << " outcome=e"
                    << typed.outcome.register_id;
            },
            [&](const ExecuteSymbolDefinition& typed) {
                out << "define_symbol s" << typed.symbol << " value=e" << typed.value.register_id;
            },
            [&](const ExecuteReadoutNoise& typed) {
                out << "readout_noise site=" << typed.site << " flip=s" << typed.flip << " record=r"
                    << typed.record << " source=e" << typed.source.register_id
                    << " p01=" << typed.prob_zero_to_one << " p10=" << typed.prob_one_to_zero;
            },
            [&](const ExecuteDetector& typed) {
                out << "write_detector detector=d" << typed.detector << " outcome=e"
                    << typed.outcome.register_id << " postselected=" << typed.postselected;
            },
            [&](const ExecuteObservable& typed) {
                out << "write_observable observable=l" << typed.observable << " outcome=e"
                    << typed.outcome.register_id;
            },
            [&](const ExecuteExpectation& typed) {
                out << "write_expectation exp_val=" << typed.exp_val << ' ';
                if (typed.active_projection.has_value()) {
                    write_prepared_pauli(out, *typed.active_projection);
                    out << " sign=e" << typed.sign.register_id;
                } else {
                    out << "zero";
                }
            },
            [&](const ExecuteInstrument& typed) {
                std::visit(
                    Overloaded{
                        [&](const ExecuteDormantInstrumentTrap& instrument) {
                            out << "instrument_dormant_trap site=" << instrument.site;
                        },
                        [&](const ExecuteClassicalInstrument& instrument) {
                            out << "instrument_classical site=" << instrument.site << " sign=e"
                                << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        },
                        [&](const ExecuteActiveInstrument& instrument) {
                            out << "instrument_active site=" << instrument.site << ' ';
                            write_prepared_pauli(out, instrument.measurement.pauli);
                            out << " sign=e" << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        },
                        [&](const ExecuteMeasuredInstrumentActivation& instrument) {
                            out << "instrument_activate_measured site=" << instrument.site << ' ';
                            write_prepared_pauli(out, instrument.measurement.pauli);
                            out << " sign=e" << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip;
                        },
                        [&](const ExecuteNewXInstrumentActivation& instrument) {
                            out << "instrument_activate_new_x site=" << instrument.site << " sign=e"
                                << instrument.sign.register_id << " flip=s"
                                << instrument.destination_flip
                                << " kernel=" << new_x_instrument_kernel_name(instrument.kernel);
                        },
                    },
                    typed.form);
            },
            [&](const ExecuteBoundary& typed) {
                out << "instrument_boundary site=" << typed.site
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
