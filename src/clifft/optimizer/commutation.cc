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
bool anti_commutes_with_noise(const HeisenbergOp& op, const NoiseSite& site, const HirModule& hir) {
    for (const auto& ch : site.channels) {
        auto ch_view = hir.noise_channel_masks.at(ch.mask);
        if (anti_commute(hir.destab_mask(op), hir.stab_mask(op), ch_view.x(), ch_view.z())) {
            return true;
        }
    }
    return false;
}

/// Check Pauli anti-commutativity between any channel pair of two noise sites.
bool noise_sites_anti_commute(const NoiseSite& a, const NoiseSite& b, const HirModule& hir) {
    for (const auto& ch_a : a.channels) {
        auto va = hir.noise_channel_masks.at(ch_a.mask);
        for (const auto& ch_b : b.channels) {
            auto vb = hir.noise_channel_masks.at(ch_b.mask);
            if (anti_commute(va.x(), va.z(), vb.x(), vb.z())) {
                return true;
            }
        }
    }
    return false;
}

// FNV-1a, matching the mixing constants clifft::sampling's expression_hash
// uses. Not a cryptographic hash: a fingerprint only has to make a
// commutation_fingerprint mismatch between what a relation was built from
// and its apply_schedule target overwhelmingly likely to be caught, not
// resist a deliberate collision.
void fnv_mix(uint64_t& hash, uint64_t word) {
    hash ^= word;
    hash *= 0x100000001b3ULL;
}

void fnv_mix_words(uint64_t& hash, MaskView view) {
    for (uint64_t word : view.words) {
        fnv_mix(hash, word);
    }
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
    // INSTRUMENT is a positional barrier for a stronger reason: a trap at
    // the site defines the remaining work as "the circuit's operations
    // after it", and re-entry requires prefix compilation to be a function
    // of the prefix alone. Both carry Pauli masks, so without these
    // clauses the symplectic test below would let commuting ops cross.
    if (lt == OpType::EXP_VAL || rt == OpType::EXP_VAL || lt == OpType::INSTRUMENT ||
        rt == OpType::INSTRUMENT) {
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
        return !noise_sites_anti_commute(hir.noise_sites[li], hir.noise_sites[ri], hir);
    }

    if (left_is_noise) {
        auto li = static_cast<uint32_t>(left.noise_site_idx());
        return !anti_commutes_with_noise(right, hir.noise_sites[li], hir);
    }

    if (right_is_noise) {
        auto ri = static_cast<uint32_t>(right.noise_site_idx());
        return !anti_commutes_with_noise(left, hir.noise_sites[ri], hir);
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
    return !anti_commute(hir.destab_mask(left), hir.stab_mask(left), hir.destab_mask(right),
                         hir.stab_mask(right));
}

CommutationFingerprint commutation_fingerprint(const HirModule& hir) {
    CommutationFingerprint fingerprint;
    fingerprint.op_count = hir.ops.size();
    fingerprint.num_qubits = hir.num_qubits;

    uint64_t hash = 0xcbf29ce484222325ULL;  // FNV offset basis.
    fnv_mix(hash, fingerprint.op_count);
    fnv_mix(hash, fingerprint.num_qubits);

    // T_GATE, MEASURE, CONDITIONAL_PAULI, and PHASE_ROTATION are exactly the
    // op types can_swap's final Pauli anti-commutation check (and, when
    // paired against a NOISE op, anti_commutes_with_noise) reads an inline
    // mask from.
    auto mix_inline_mask = [&](const HeisenbergOp& op) {
        fnv_mix_words(hash, hir.destab_mask(op));
        fnv_mix_words(hash, hir.stab_mask(op));
    };

    for (const HeisenbergOp& op : hir.ops) {
        fnv_mix(hash, static_cast<uint64_t>(op.op_type()));
        switch (op.op_type()) {
            case OpType::T_GATE:
                mix_inline_mask(op);
                break;
            case OpType::MEASURE:
                mix_inline_mask(op);
                fnv_mix(hash, static_cast<uint64_t>(op.meas_record_idx()));
                break;
            case OpType::CONDITIONAL_PAULI:
                mix_inline_mask(op);
                fnv_mix(hash, static_cast<uint64_t>(op.controlling_meas()));
                break;
            case OpType::PHASE_ROTATION:
                mix_inline_mask(op);
                break;
            case OpType::NOISE: {
                // can_swap follows noise_site_idx into NoiseSite::channels
                // and reads every channel's mask regardless of probability
                // (anti_commutes_with_noise / noise_sites_anti_commute loop
                // over all of them unconditionally), so the fingerprint
                // must too. The site index itself is never compared by
                // can_swap, only dereferenced, so it is not hashed.
                const NoiseSite& site = hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())];
                fnv_mix(hash, static_cast<uint64_t>(site.channels.size()));
                for (const NoiseChannel& channel : site.channels) {
                    const PauliMaskView view = hir.noise_channel_masks.at(channel.mask);
                    fnv_mix_words(hash, view.x());
                    fnv_mix_words(hash, view.z());
                }
                break;
            }
            case OpType::READOUT_NOISE: {
                // Only the referenced entry's meas_idx feeds
                // accesses_classical_index / get_written_meas_idx; the raw
                // readout_noise_idx is otherwise just an array offset.
                const auto entry_idx = static_cast<uint32_t>(op.readout_noise_idx());
                fnv_mix(hash, static_cast<uint64_t>(hir.readout_noise[entry_idx].meas_idx));
                break;
            }
            case OpType::DETECTOR: {
                // Length is hashed explicitly (not just implied by the
                // following elements) so two different-length target lists
                // whose flattened elements happen to concatenate the same
                // way cannot hash equal.
                const auto& targets =
                    hir.detector_targets[static_cast<uint32_t>(op.detector_idx())];
                fnv_mix(hash, static_cast<uint64_t>(targets.size()));
                for (uint32_t target : targets) {
                    fnv_mix(hash, target);
                }
                break;
            }
            case OpType::OBSERVABLE: {
                const auto& targets = hir.observable_targets[op.observable_target_list_idx()];
                fnv_mix(hash, static_cast<uint64_t>(targets.size()));
                for (uint32_t target : targets) {
                    fnv_mix(hash, target);
                }
                break;
            }
            case OpType::EXP_VAL:
            case OpType::INSTRUMENT:
                // can_swap refuses to reorder across either unconditionally,
                // from the type check alone, without reading their sites --
                // so only the op type (already mixed above) distinguishes
                // them.
                break;
            case OpType::NUM_OP_TYPES:
                break;
        }
    }

    fingerprint.hash = hash;
    return fingerprint;
}

}  // namespace clifft
