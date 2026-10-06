#pragma once

#include "clifft/optimizer/pauli_axis.h"

#include <cassert>
#include <optional>

namespace clifft::optimizer_detail {

// Accumulate Clifford rotations removed during optimization to avoid repeated
// scans of the remaining circuit. Transform subsequent Pauli axes as they are
// visited, then compose the accumulated Clifford into the final tableau.
class CliffordAbsorption {
  public:
    bool empty() const { return !forward_; }

    PauliString read(PauliMaskView mask, uint32_t width) const {
        auto axis = copy_axis(mask, width);
        return inverse_ ? inverse_->apply(axis.view()) : axis;
    }

    void absorb(const PauliString& axis, uint8_t coefficient) {
        assert(coefficient == 2 || coefficient == 4 || coefficient == 6);
        if (!forward_) {
            forward_.emplace(axis.num_qubits());
            inverse_.emplace(axis.num_qubits());
        }
        const auto original_axis = forward_->apply(axis.view());
        if (coefficient == 4) {
            forward_->prepend_pauli(axis.view());
            inverse_->prepend_pauli(original_axis.view());
        } else {
            forward_->prepend_pauli_rotation(axis.view(), coefficient == 6);
            inverse_->prepend_pauli_rotation(original_axis.view(), coefficient == 2);
        }
    }

    void transform(HirModule& hir, const HeisenbergOp& op) const {
        if (empty()) {
            return;
        }
        const auto transform_mask = [&](MutablePauliMaskView mask) {
            write_axis(mask, read(mask, hir.num_qubits));
        };
        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
            case OpType::MEASURE:
            case OpType::CONDITIONAL_PAULI:
            case OpType::EXP_VAL:
                transform_mask(hir.mask_at(op));
                break;
            case OpType::INSTRUMENT: {
                transform_mask(hir.mask_at(op));
                const auto& site =
                    hir.instrument_sites[static_cast<uint32_t>(op.instrument_site_idx())];
                transform_mask(hir.pauli_masks.mut_at(site.destination_flip_mask));
                break;
            }
            case OpType::NOISE:
                for (const auto& channel :
                     hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels) {
                    auto mask = hir.noise_channel_masks.mut_at(channel.mask);
                    transform_mask(mask);
                    mask.set_sign(false);
                }
                break;
            case OpType::READOUT_NOISE:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
            case OpType::NUM_OP_TYPES:
                break;
        }
    }

    void finish(HirModule& hir) const {
        if (forward_ && hir.final_tableau) {
            hir.final_tableau = forward_->then(*hir.final_tableau);
        }
    }

  private:
    std::optional<Tableau> forward_;
    std::optional<Tableau> inverse_;
};

}  // namespace clifft::optimizer_detail
