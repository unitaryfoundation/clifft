#include "clifft/sampling/planner.h"

#include "clifft/util/hir_introspection.h"
#include "clifft/util/numeric.h"
#include "clifft/util/stim_mask.h"

#include "stim.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace clifft::sampling {

namespace {

using Pauli = stim::PauliString<kStimWidth>;
using Tableau = stim::Tableau<kStimWidth>;

template <typename>
inline constexpr bool kAlwaysFalse = false;

struct PendingRotation {
    Pauli body;
    double half_turns = 0.0;
    AffineBool sign;
};

struct PendingMeasurement {
    Pauli body;
    AffineBool sign;
    RecordSlot record{};
};

using PendingOperation = std::variant<PendingRotation, PendingMeasurement>;

Pauli pauli_from_hir(const HirModule& hir, const HeisenbergOp& op) {
    Pauli result(hir.num_qubits);
    mask_view_to_stim(hir.destab_mask(op), hir.num_qubits, result.xs);
    mask_view_to_stim(hir.stab_mask(op), hir.num_qubits, result.zs);
    return result;
}

Pauli single_x(uint32_t num_qubits, uint32_t q) {
    Pauli result(num_qubits);
    result.xs[q] = true;
    return result;
}

Pauli single_z(uint32_t num_qubits, uint32_t q) {
    Pauli result(num_qubits);
    result.zs[q] = true;
    return result;
}

Pauli positive_body_xor(const Pauli& left, const Pauli& right) {
    Pauli result(left);
    result.xs ^= right.xs;
    result.zs ^= right.zs;
    result.sign = false;
    return result;
}

bool anticommutes(const Pauli& left, const Pauli& right) {
    return !left.ref().commutes(right.ref());
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

Tableau dormant_promotion_frame(const Pauli& promoted, uint32_t active_width,
                                uint32_t dormant_pivot) {
    const uint32_t n = static_cast<uint32_t>(promoted.num_qubits);
    Tableau frame(n);
    const Pauli old_stabilizer = single_z(n, dormant_pivot);

    // The rows express the new coordinate generators in the old basis. Making
    // the promoted Pauli the next X generator turns its rotation into a
    // single-coordinate expansion; the stabilizer products keep every other
    // generator in a canonical symplectic pair.
    for (uint32_t q = 0; q < active_width; ++q) {
        Pauli x = single_x(n, q);
        Pauli z = single_z(n, q);
        if (anticommutes(x, promoted)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, promoted)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.xs[q] = x;
        frame.zs[q] = z;
    }

    frame.xs[active_width] = promoted;
    frame.zs[active_width] = old_stabilizer;

    uint32_t new_q = active_width + 1;
    for (uint32_t old_q = active_width; old_q < n; ++old_q) {
        if (old_q == dormant_pivot) {
            continue;
        }
        Pauli x = single_x(n, old_q);
        Pauli z = single_z(n, old_q);
        if (anticommutes(x, promoted)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, promoted)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.xs[new_q] = x;
        frame.zs[new_q] = z;
        ++new_q;
    }

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid promotion frame");
    }
    return frame;
}

Tableau dormant_measurement_frame(const Pauli& measured, uint32_t dormant_pivot) {
    const uint32_t n = static_cast<uint32_t>(measured.num_qubits);
    Tableau frame(n);
    const Pauli old_stabilizer = single_z(n, dormant_pivot);

    // Replacing one dormant Z generator with the measured Pauli represents the
    // collapsed eigenspace without changing the dense coefficient state.
    for (uint32_t q = 0; q < n; ++q) {
        if (q == dormant_pivot) {
            continue;
        }
        Pauli x = single_x(n, q);
        Pauli z = single_z(n, q);
        if (anticommutes(x, measured)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, measured)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.xs[q] = x;
        frame.zs[q] = z;
    }

    frame.xs[dormant_pivot] = old_stabilizer;
    frame.zs[dormant_pivot] = measured;

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid dormant measurement frame");
    }
    return frame;
}

Tableau active_measurement_frame(const Pauli& measured, uint32_t active_width, uint32_t pivot) {
    const uint32_t n = static_cast<uint32_t>(measured.num_qubits);
    Tableau frame(n);
    const bool diagonal = !first_x_below(measured, active_width).has_value();
    const Pauli pivot_x = single_x(n, pivot);
    const Pauli pivot_z = single_z(n, pivot);

    // Moving the measured Pauli to the last active Z generator gives the
    // direct measurement kernel one coordinate to remove. Future operations
    // are rewritten into the remaining packed coordinates below.
    frame.zs[active_width - 1] = measured;
    frame.xs[active_width - 1] = diagonal ? pivot_x : pivot_z;

    uint32_t new_q = 0;
    for (uint32_t old_q = 0; old_q < active_width; ++old_q) {
        if (old_q == pivot) {
            continue;
        }
        Pauli x = single_x(n, old_q);
        Pauli z = single_z(n, old_q);
        if (diagonal) {
            if (measured.zs[old_q]) {
                x = positive_body_xor(x, pivot_x);
            }
        } else {
            if (measured.xs[old_q]) {
                z = positive_body_xor(z, pivot_z);
            }
            if (measured.zs[old_q]) {
                x = positive_body_xor(x, pivot_z);
            }
        }
        frame.xs[new_q] = x;
        frame.zs[new_q] = z;
        ++new_q;
    }

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid active measurement frame");
    }
    return frame;
}

template <typename Function>
void visit_pending_pauli(PendingOperation& operation, Function&& function) {
    std::visit(
        [&](auto& typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, PendingRotation> ||
                          std::is_same_v<T, PendingMeasurement>) {
                function(typed.body, typed.sign);
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled pending operation alternative");
            }
        },
        operation);
}

void transform_future_operations(std::vector<PendingOperation>& pending, size_t begin,
                                 const Tableau& new_basis_in_old_coordinates) {
    // Planning owns coordinate evolution. Runtime actions therefore receive
    // already-transformed Paulis and never need to discover dependencies or
    // perform tableau evolution in the dispatch loop.
    const Tableau old_to_new = new_basis_in_old_coordinates.inverse();
    for (size_t i = begin; i < pending.size(); ++i) {
        visit_pending_pauli(pending[i], [&](Pauli& body, AffineBool& sign) {
            Pauli transformed = old_to_new(body);
            sign ^= static_cast<bool>(transformed.sign);
            transformed.sign = false;
            body = std::move(transformed);
        });
    }
}

void propagate_branch(std::vector<PendingOperation>& pending, size_t begin,
                      uint32_t branch_coordinate, SymbolId branch) {
    // The sampled minus eigenspace differs by X on the replaced Z coordinate.
    // Anticommuting future Paulis therefore acquire this branch in their sign.
    for (size_t i = begin; i < pending.size(); ++i) {
        visit_pending_pauli(pending[i], [&](const Pauli& body, AffineBool& sign) {
            if (body.zs[branch_coordinate]) {
                sign ^= AffineBool::symbol(branch);
            }
        });
    }
}

SymbolId append_branch(SamplingPlan& plan, uint32_t defining_action) {
    const SymbolId branch{static_cast<uint32_t>(plan.symbols.size())};
    plan.symbols.push_back(SymbolInfo{SymbolKind::Branch, defining_action, std::nullopt});
    return branch;
}

void multiply_phase(std::complex<double>& weight, double angle) {
    weight *= std::complex<double>(std::cos(angle), std::sin(angle));
}

std::vector<PendingOperation> queue_supported_operations(const HirModule& hir, SamplingPlan& plan) {
    std::vector<PendingOperation> pending;
    pending.reserve(hir.ops.size());

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
                pending.emplace_back(
                    PendingMeasurement{pauli_from_hir(hir, op), AffineBool(hir.sign(op)),
                                       RecordSlot{static_cast<uint32_t>(op.meas_record_idx())}});
                break;
            case OpType::CONDITIONAL_PAULI:
            case OpType::NOISE:
            case OpType::READOUT_NOISE:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
            case OpType::EXP_VAL:
            case OpType::INSTRUMENT:
            case OpType::NUM_OP_TYPES:
                throw std::invalid_argument("sampling planner does not support HIR operation " +
                                            op_type_to_str(op.op_type()) + " at index " +
                                            std::to_string(i));
        }
    }
    return pending;
}

void process_rotation(std::vector<PendingOperation>& pending, size_t index,
                      const PendingRotation& rotation, SamplingPlan& plan, uint32_t& active_width) {
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(rotation.body, active_width);
    if (!dormant_pivot.has_value()) {
        plan.actions.push_back(
            PlannedAction{active_width, active_width,
                          RotateActivePauli{active_projection(rotation.body, active_width),
                                            rotation.half_turns, rotation.sign}});
        return;
    }

    if (active_width + 1 >= kDenseActiveWidthLimit) {
        throw std::overflow_error(
            "sampling planner active width would reach " + std::to_string(active_width + 1) +
            ", but the dense-state limit is " + std::to_string(kDenseActiveWidthLimit));
    }

    const Tableau frame = dormant_promotion_frame(rotation.body, active_width, *dormant_pivot);
    transform_future_operations(pending, index + 1, frame);
    plan.actions.push_back(
        PlannedAction{active_width, active_width + 1,
                      PromoteDormantRotation{active_width, rotation.half_turns, rotation.sign}});
    ++active_width;
    plan.max_active_width = std::max(plan.max_active_width, active_width);
}

void process_measurement(std::vector<PendingOperation>& pending, size_t index,
                         const PendingMeasurement& measurement, SamplingPlan& plan,
                         uint32_t& active_width) {
    const std::optional<uint32_t> dormant_pivot =
        first_x_at_or_above(measurement.body, active_width);
    if (dormant_pivot.has_value()) {
        const Tableau frame = dormant_measurement_frame(measurement.body, *dormant_pivot);
        transform_future_operations(pending, index + 1, frame);

        const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
        const SymbolId branch = append_branch(plan, action_index);
        propagate_branch(pending, index + 1, *dormant_pivot, branch);
        plan.actions.push_back(
            PlannedAction{active_width, active_width,
                          MeasureDormantRandom{*dormant_pivot, branch,
                                               measurement.sign ^ AffineBool::symbol(branch),
                                               measurement.record}});
        return;
    }

    const ActivePauli active = active_projection(measurement.body, active_width);
    if (active.is_identity()) {
        plan.actions.push_back(PlannedAction{
            active_width, active_width, RecordClassical{measurement.sign, measurement.record}});
        return;
    }

    std::optional<uint32_t> pivot = first_x_below(measurement.body, active_width);
    if (!pivot.has_value()) {
        pivot = first_z_below(measurement.body, active_width);
    }
    if (!pivot.has_value()) {
        throw std::logic_error("sampling planner could not select an active measurement pivot");
    }

    Pauli active_body(measurement.body.num_qubits);
    for (uint32_t q = 0; q < active_width; ++q) {
        active_body.xs[q] = measurement.body.xs[q];
        active_body.zs[q] = measurement.body.zs[q];
    }
    const Tableau frame = active_measurement_frame(active_body, active_width, *pivot);
    transform_future_operations(pending, index + 1, frame);

    const uint32_t action_index = static_cast<uint32_t>(plan.actions.size());
    const SymbolId branch = append_branch(plan, action_index);
    propagate_branch(pending, index + 1, active_width - 1, branch);
    plan.actions.push_back(PlannedAction{
        active_width, active_width - 1,
        MeasureActivePauli{active, *pivot, branch, measurement.sign ^ AffineBool::symbol(branch),
                           measurement.record}});
    --active_width;
}

}  // namespace

SamplingPlan plan_sampling(const HirModule& hir) {
    SamplingPlan plan;
    plan.num_qubits = hir.num_qubits;
    plan.num_visible_records = hir.num_measurements;
    plan.num_hidden_records = hir.num_hidden_measurements;
    plan.num_noise_sites = static_cast<uint32_t>(hir.noise_sites.size());
    plan.num_instrument_sites = static_cast<uint32_t>(hir.instrument_sites.size());
    plan.global_weight = hir.global_weight;

    std::vector<PendingOperation> pending = queue_supported_operations(hir, plan);
    uint32_t active_width = plan.initial_active_width;
    for (size_t i = 0; i < pending.size(); ++i) {
        std::visit(
            [&](const auto& operation) {
                using T = std::decay_t<decltype(operation)>;
                if constexpr (std::is_same_v<T, PendingRotation>) {
                    process_rotation(pending, i, operation, plan, active_width);
                } else if constexpr (std::is_same_v<T, PendingMeasurement>) {
                    process_measurement(pending, i, operation, plan, active_width);
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled pending operation alternative");
                }
            },
            pending[i]);
    }

    plan.validate();
    return plan;
}

}  // namespace clifft::sampling
