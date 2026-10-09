#pragma once

// Research diagnostics only; these snapshots never enter native execution.

#include "clifft/frontend/hir.h"
#include "clifft/sampling/plan.h"
#include "clifft/util/hir_introspection.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace planning_reuse_audit {
using namespace clifft;

inline bool diagonal_boundary_fixes_prefix_axes(const HirModule& prefix) {
    if (!prefix.final_tableau)
        return false;
    for (const auto& op : prefix.ops) {
        if (op.op_type() != OpType::T_GATE && op.op_type() != OpType::PHASE_ROTATION)
            return false;
        PauliString axis(prefix.num_qubits);
        axis.mut_x().xor_with(prefix.destab_mask(op));
        axis.mut_z().xor_with(prefix.stab_mask(op));
        axis.set_sign(prefix.sign(op));
        if (!prefix.final_tableau->apply(axis.view()).x().is_zero())
            return false;
    }
    return true;
}

inline std::string fingerprint(const std::string& text) {
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << '"' << std::hex << std::setw(16) << std::setfill('0') << hash << '"';
    return out.str();
}

inline std::string frame_key(const std::optional<Tableau>& frame) {
    std::ostringstream out;
    if (frame) {
        for (uint32_t q = 0; q < frame->num_qubits(); ++q) {
            for (const auto row : {frame->x_output(q), frame->z_output(q)}) {
                out << static_cast<unsigned>(row.phase()) << ':';
                for (const auto word : row.x().words)
                    out << word << ',';
                out << ':';
                for (const auto word : row.z().words)
                    out << word << ',';
                out << ';';
            }
        }
    }
    return fingerprint(out.str());
}

inline std::string hir_snapshot(const HirModule& hir) {
    auto unsigned_hir = hir;
    for (const auto& op : unsigned_hir.ops)
        if (op.has_mask())
            unsigned_hir.set_sign(op, false);
    std::ostringstream out;
    out << "{\"ops\":" << hir.ops.size() << ",\"t_count\":" << hir.num_t_gates()
        << ",\"frame\":" << frame_key(hir.final_tableau);
    for (bool signs : {true, false}) {
        out << (signs ? ",\"signed_rows\":[" : ",\"unsigned_rows\":[");
        const auto& view = signs ? hir : unsigned_hir;
        for (size_t i = 0; i < view.ops.size(); ++i) {
            const auto& op = view.ops[i];
            std::ostringstream row;
            row << format_hir_op(op,
                                 op.has_mask() ? std::optional{view.mask_view(op)} : std::nullopt)
                << " flags=" << static_cast<unsigned>(op.flags());
            if (op.op_type() == OpType::PHASE_ROTATION)
                row << " angle=" << std::setprecision(17) << op.alpha();
            out << (i ? "," : "") << fingerprint(row.str());
        }
        out << ']';
    }
    out << ",\"origins\":[";
    for (size_t i = 0; i < hir.source_map.size(); ++i) {
        out << (i ? ",[" : "[");
        for (size_t j = 0; j < hir.source_map[i].size(); ++j)
            out << (j ? "," : "") << hir.source_map[i][j];
        out << ']';
    }
    out << "]}";
    return out.str();
}

inline void clear_constant(sampling::AffineBool& expression) {
    expression ^= expression.constant();
}
inline void clear_constant(sampling::RecordParity& expression) {
    expression = sampling::RecordParity(false, expression.records());
}
inline void clear_constant(sampling::ObservableValue& expression) {
    std::visit([](auto& value) { clear_constant(value); }, expression);
}

inline std::string plan_snapshot(const sampling::SamplingPlan& plan) {
    if (!plan.presampled_noise_sites.empty() || !plan.instrument_distributions.empty())
        throw std::invalid_argument("Plan audit requires materialized Pauli faults");
    auto normalized = plan;
    for (auto& planned : normalized.actions) {
        std::visit(
            [](auto& action) {
                using T = std::decay_t<decltype(action)>;
                if constexpr (std::is_same_v<T, sampling::RotateActivePauli> ||
                              std::is_same_v<T, sampling::PromoteDormantRotation>)
                    clear_constant(action.sign);
                else if constexpr (std::is_same_v<T, sampling::MeasureActivePauli> ||
                                   std::is_same_v<T, sampling::MeasureDormantRandom> ||
                                   std::is_same_v<T, sampling::RecordClassical> ||
                                   std::is_same_v<T, sampling::WriteDetector> ||
                                   std::is_same_v<T, sampling::WriteObservable>)
                    clear_constant(action.outcome);
                else if constexpr (std::is_same_v<T, sampling::DefineSymbol>)
                    clear_constant(action.value);
                else if constexpr (std::is_same_v<T, sampling::WriteExpectationValue>) {
                    if (action.active)
                        clear_constant(action.active->sign);
                } else
                    throw std::invalid_argument("Unsupported audit action");
            },
            planned.action);
    }
    std::ostringstream out;
    out << "{\"actions\":" << plan.actions.size() << ",\"frame\":" << frame_key(plan.final_tableau)
        << ",\"symbols\":[";
    for (size_t i = 0; i < plan.symbols.size(); ++i)
        out << (i ? "," : "") << static_cast<unsigned>(plan.symbols[i]);
    out << ']';
    for (bool constants : {true, false}) {
        out << (constants ? ",\"full_actions\":[" : ",\"without_constants\":[");
        const auto& view = constants ? plan : normalized;
        for (size_t i = 0; i < view.actions.size(); ++i)
            out << (i ? "," : "") << std::quoted(view.inspect_action(i));
        out << ']';
    }
    out << ",\"kind_width\":[";
    for (size_t i = 0; i < plan.actions.size(); ++i) {
        const auto& a = plan.actions[i];
        out << (i ? ",[" : "[") << a.action.index() << ',' << a.active_before << ','
            << a.active_after << ']';
    }
    out << "]}";
    return out.str();
}
}  // namespace planning_reuse_audit
