#include "clifft/optimizer/known_stabilizers.h"

#include <cassert>
#include <utility>

namespace clifft {

KnownStabilizers::KnownStabilizers(uint32_t num_qubits) : basis_(num_qubits) {}

PauliString KnownStabilizers::reduce_body(PauliString axis) const {
    return basis_.reduce_body(std::move(axis));
}

std::optional<bool> KnownStabilizers::eigenvalue(PauliString axis) const {
    Basis::Row target{0, std::move(axis), {}};
    size_t products = 0;
    basis_.reduce(target, products);
    if (!Basis::identity_axis(target.axis)) {
        return std::nullopt;
    }
    assert(target.axis.is_hermitian());
    return target.axis.sign();
}

bool KnownStabilizers::commutes(PauliStringView axis) const {
    return basis_.commutes(axis.x(), axis.z());
}

void KnownStabilizers::insert(PauliString axis) {
    assert(axis.is_hermitian());
    basis_.ensure_width(axis.num_qubits());
    Basis::Row row{0, std::move(axis), {}};
    size_t products = 0;
    basis_.reduce(row, products);
    if (Basis::identity_axis(row.axis)) {
        assert(!row.axis.sign());
        return;
    }
    basis_.insert_reduced(std::move(row));
}

bool KnownStabilizers::reduce_and_insert_bounded(PauliString axis, size_t& products,
                                                 size_t max_products) {
    assert(axis.is_hermitian());
    basis_.ensure_width(axis.num_qubits());
    Basis::Row row{0, std::move(axis), {}};
    if (!basis_.reduce(row, products, max_products)) {
        return false;
    }
    if (Basis::identity_axis(row.axis)) {
        assert(!row.axis.sign());
        return true;
    }
    basis_.insert_reduced(std::move(row));
    return true;
}

void KnownStabilizers::intersect(PauliStringView axis) {
    basis_.intersect(axis.x(), axis.z());
}

void KnownStabilizers::advance(const HirModule& hir, const HeisenbergOp& op) {
    // Advancing only removes known stabilizers.
    if (empty()) {
        return;
    }
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI: {
            // Keep relations common to both outcomes; future postselection
            // cannot justify a stabilizer constraint at the current program point.
            const auto mask = hir.mask_view(op);
            basis_.intersect(mask.x(), mask.z());
            break;
        }
        case OpType::NOISE: {
            const auto& channels =
                hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
            for (const auto& channel : channels) {
                if (empty()) {
                    break;
                }
                if (channel.prob == 0) {
                    continue;
                }
                const auto mask = hir.noise_channel_masks.at(channel.mask);
                // Reference-syndrome computation strips even probability-one
                // noise after optimization, so constraints must also hold without it.
                basis_.intersect(mask.x(), mask.z());
            }
            break;
        }
        case OpType::INSTRUMENT:
            clear();
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
