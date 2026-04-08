#include "clifft/optimizer/commutation.h"

#include <optional>

namespace clifft {

namespace {

/// Returns the classical measurement index written by this operation, if any.
std::optional<uint32_t> get_written_meas_idx(const HeisenbergOp& op, const HirModule& hir) {
    if (op.op_type() == OpType::MEASURE) {
        return static_cast<uint32_t>(op.meas_record_idx());
    }
    if (op.op_type() == OpType::READOUT_NOISE) {
        return hir.readout_noise[static_cast<uint32_t>(op.readout_noise_idx())].meas_idx;
    }
    return std::nullopt;
}

/// Returns true if the operation accesses (reads or writes) the given
/// classical measurement index.
bool accesses_classical_index(const HeisenbergOp& op, uint32_t target_idx, const HirModule& hir) {
    switch (op.op_type()) {
        case OpType::MEASURE:
            return static_cast<uint32_t>(op.meas_record_idx()) == target_idx;
        case OpType::CONDITIONAL_PAULI:
            return static_cast<uint32_t>(op.controlling_meas()) == target_idx;
        case OpType::READOUT_NOISE:
            return hir.readout_noise[static_cast<uint32_t>(op.readout_noise_idx())].meas_idx ==
                   target_idx;
        case OpType::DETECTOR:
            for (uint32_t idx : hir.detector_targets[static_cast<uint32_t>(op.detector_idx())]) {
                if (idx == target_idx)
                    return true;
            }
            return false;
        case OpType::OBSERVABLE:
            for (uint32_t idx : hir.observable_targets[op.observable_target_list_idx()]) {
                if (idx == target_idx)
                    return true;
            }
            return false;
        default:
            return false;
    }
}

/// Check Pauli commutativity between an op's masks and a noise site's channels.
bool anti_commutes_with_noise(const HeisenbergOp& op, const NoiseSite& site) {
    for (const auto& ch : site.channels) {
        if (anti_commute(op.destab_mask(), op.stab_mask(), ch.destab_mask, ch.stab_mask)) {
            return true;
        }
    }
    return false;
}

/// Check Pauli anti-commutativity between any channel pair of two noise sites.
bool noise_sites_anti_commute(const NoiseSite& a, const NoiseSite& b) {
    for (const auto& ch_a : a.channels) {
        for (const auto& ch_b : b.channels) {
            if (anti_commute(ch_a.destab_mask, ch_a.stab_mask, ch_b.destab_mask, ch_b.stab_mask)) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir) {
    auto lt = left.op_type();
    auto rt = right.op_type();

    // Precise classical dataflow barrier: prevent swapping if one op writes
    // to a classical measurement index that the other accesses.
    auto left_write = get_written_meas_idx(left, hir);
    if (left_write.has_value() && accesses_classical_index(right, *left_write, hir)) {
        return false;
    }
    auto right_write = get_written_meas_idx(right, hir);
    if (right_write.has_value() && accesses_classical_index(left, *right_write, hir)) {
        return false;
    }

    // EXP_VAL is a positional probe: the user expects the expectation value
    // at an exact circuit point. Never reorder anything across it.
    if (lt == OpType::EXP_VAL || rt == OpType::EXP_VAL) {
        return false;
    }

    // Quantum commutativity via symplectic inner product.
    // Both ops carry inline Pauli masks:
    bool left_is_noise = (lt == OpType::NOISE);
    bool right_is_noise = (rt == OpType::NOISE);

    // NOISE ops carry zero inline Pauli masks; the actual channel content
    // lives in the NoiseSite side-table. Two NOISE ops must be checked
    // via noise_sites_anti_commute (channel-vs-channel), not via inline masks.
    if (left_is_noise && right_is_noise) {
        auto li = static_cast<uint32_t>(left.noise_site_idx());
        auto ri = static_cast<uint32_t>(right.noise_site_idx());
        return !noise_sites_anti_commute(hir.noise_sites[li], hir.noise_sites[ri]);
    }

    if (left_is_noise) {
        auto li = static_cast<uint32_t>(left.noise_site_idx());
        return !anti_commutes_with_noise(right, hir.noise_sites[li]);
    }

    if (right_is_noise) {
        auto ri = static_cast<uint32_t>(right.noise_site_idx());
        return !anti_commutes_with_noise(left, hir.noise_sites[ri]);
    }

    // DETECTOR, OBSERVABLE, READOUT_NOISE have no quantum Pauli footprint
    // (they only read classical data), so they commute with everything
    // that passes the classical/PRNG checks above.
    bool left_classical =
        (lt == OpType::DETECTOR || lt == OpType::OBSERVABLE || lt == OpType::READOUT_NOISE);
    bool right_classical =
        (rt == OpType::DETECTOR || rt == OpType::OBSERVABLE || rt == OpType::READOUT_NOISE);
    if (left_classical || right_classical) {
        return true;
    }

    // Standard Pauli anti-commutation check
    return !anti_commute(left.destab_mask(), left.stab_mask(), right.destab_mask(),
                         right.stab_mask());
}

}  // namespace clifft
