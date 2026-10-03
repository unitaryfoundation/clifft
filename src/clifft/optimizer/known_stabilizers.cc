#include "clifft/optimizer/known_stabilizers.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace clifft {
namespace {

uint32_t pivot_of(const PauliString& axis) {
    const uint32_t domain = axis.x().num_words() * 64;
    const uint32_t x = axis.x().lowest_bit();
    return x < domain ? x : domain + axis.z().lowest_bit();
}

bool body_bit(const PauliString& axis, uint32_t pivot) {
    const uint32_t domain = axis.x().num_words() * 64;
    return pivot < domain ? axis.x().bit_get(pivot) : axis.z().bit_get(pivot - domain);
}

bool identity(const PauliString& axis) {
    return axis.x().is_zero() && axis.z().is_zero();
}

PauliString copy_axis(PauliMaskView mask, uint32_t width) {
    PauliString axis(width);
    axis.mut_x().xor_with(mask.x());
    axis.mut_z().xor_with(mask.z());
    axis.set_sign(mask.sign());
    return axis;
}

}  // namespace

KnownStabilizers::KnownStabilizers(uint32_t num_qubits) {
    for (uint32_t q = 0; q < num_qubits; ++q) {
        PauliString axis(num_qubits);
        axis.set_pauli(q, false, true);
        rows_.emplace(pivot_of(axis), std::move(axis));
    }
}

PauliString KnownStabilizers::reduce_body(PauliString axis) const {
    for (auto it = rows_.lower_bound(pivot_of(axis)); it != rows_.end(); ++it) {
        if (body_bit(axis, it->first)) {
            axis.mut_x().xor_with(it->second.x());
            axis.mut_z().xor_with(it->second.z());
            if (identity(axis)) {
                break;
            }
        }
    }
    return axis;
}

std::optional<bool> KnownStabilizers::eigenvalue(PauliString axis) const {
    for (auto it = rows_.lower_bound(pivot_of(axis)); it != rows_.end(); ++it) {
        if (body_bit(axis, it->first)) {
            axis.right_multiply(it->second.view());
            if (identity(axis)) {
                break;
            }
        }
    }
    if (!identity(axis)) {
        return std::nullopt;
    }
    assert(axis.is_hermitian());
    return axis.sign();
}

bool KnownStabilizers::commutes(PauliStringView axis) const {
    return std::ranges::all_of(rows_,
                               [&](const auto& row) { return axis.commutes(row.second.view()); });
}

void KnownStabilizers::insert(PauliString axis) {
    assert(axis.is_hermitian());
    for (auto it = rows_.lower_bound(pivot_of(axis)); it != rows_.end(); ++it) {
        if (body_bit(axis, it->first)) {
            axis.right_multiply(it->second.view());
            if (identity(axis)) {
                break;
            }
        }
    }
    if (identity(axis)) {
        assert(!axis.sign());
    } else {
        rows_.emplace(pivot_of(axis), std::move(axis));
    }
}

void KnownStabilizers::intersect(PauliStringView axis) {
    auto pivot = rows_.end();
    for (auto it = rows_.begin(); it != rows_.end(); ++it) {
        if (!axis.commutes(it->second.view())) {
            pivot = it;
        }
    }
    if (pivot == rows_.end()) {
        return;
    }
    // The largest pivot preserves the earlier rows' leading bits.
    for (auto it = rows_.begin(); it != pivot; ++it) {
        if (!axis.commutes(it->second.view())) {
            it->second.right_multiply(pivot->second.view());
        }
    }
    rows_.erase(pivot);
}

void KnownStabilizers::advance(const HirModule& hir, const HeisenbergOp& op) {
    // Advancing only removes or changes the signs of existing facts.
    if (rows_.empty()) {
        return;
    }
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
            // Keep relations common to both outcomes; future postselection
            // cannot justify a fact at the current program point.
            intersect(copy_axis(hir.mask_view(op), hir.num_qubits).view());
            break;
        case OpType::NOISE: {
            const auto& channels =
                hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
            for (const auto& channel : channels) {
                if (channel.prob == 0) {
                    continue;
                }
                const auto axis =
                    copy_axis(hir.noise_channel_masks.at(channel.mask), hir.num_qubits);
                if (channel.prob == 1) {
                    for (auto& [pivot, row] : rows_) {
                        (void)pivot;
                        if (!axis.view().commutes(row.view())) {
                            row.negate();
                        }
                    }
                } else {
                    intersect(axis.view());
                }
            }
            break;
        }
        case OpType::INSTRUMENT:
            rows_.clear();
            break;
        case OpType::EXP_VAL:
        case OpType::READOUT_NOISE:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::NUM_OP_TYPES:
            break;
    }
}

}  // namespace clifft
