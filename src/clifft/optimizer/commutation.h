#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/symplectic.h"

#include <cassert>
#include <cstdint>

namespace clifft {

/// True when an operation fixes an exact program point and must never be
/// reordered with neighbors even when their Pauli supports commute.
[[nodiscard]] constexpr bool is_positional_barrier(OpType type) noexcept {
    switch (type) {
        case OpType::EXP_VAL:
        case OpType::INSTRUMENT:
            return true;
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
        case OpType::NOISE:
        case OpType::READOUT_NOISE:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::NUM_OP_TYPES:
            return false;
    }
    return false;
}

/// Visits every classical measurement record index read or written by `op`.
template <typename Callback>
void for_each_classical_record(const HeisenbergOp& op, const HirModule& hir, Callback&& callback) {
    switch (op.op_type()) {
        case OpType::MEASURE:
            callback(static_cast<uint32_t>(op.meas_record_idx()));
            return;
        case OpType::CONDITIONAL_PAULI:
            callback(static_cast<uint32_t>(op.controlling_meas()));
            return;
        case OpType::READOUT_NOISE:
            callback(hir.readout_noise[static_cast<uint32_t>(op.readout_noise_idx())].meas_idx);
            return;
        case OpType::DETECTOR:
            for (uint32_t idx : hir.detector_targets[static_cast<uint32_t>(op.detector_idx())]) {
                callback(idx);
            }
            return;
        case OpType::OBSERVABLE:
            for (uint32_t idx : hir.observable_targets[op.observable_target_list_idx()]) {
                callback(idx);
            }
            return;
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::NOISE:
        case OpType::EXP_VAL:
        case OpType::INSTRUMENT:
        case OpType::NUM_OP_TYPES:
            return;
    }
}

/// Returns true if the two HIR operations can be safely swapped in the
/// ops vector without changing program semantics or PRNG trajectory.
/// Symmetric in its operands: left may be whichever operand has a cached axis.
/// left_axis optionally caches the unchanged inline Pauli mask of left.
bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir,
              const PauliCommutation* left_axis = nullptr);

}  // namespace clifft
