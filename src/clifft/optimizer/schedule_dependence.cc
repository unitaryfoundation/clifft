#include "clifft/optimizer/schedule_dependence.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>

namespace clifft::detail {

namespace {

bool is_movable_op(OpType type) {
    return type == OpType::T_GATE || type == OpType::PHASE_ROTATION || type == OpType::MEASURE;
}

// FNV-1a, matching the mixing constants clifft::sampling's expression_hash
// uses. Not a cryptographic hash: a fingerprint only has to make a build()
// versus apply_schedule() mismatch overwhelmingly likely to be caught, not
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

uint64_t double_bits(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Folds every op's type, flags, payload fields, and Pauli mask words into
// one hash, plus the words of each NOISE op's channel masks (NOISE carries
// no inline Pauli mask of its own) and logical_noise_prefix when
// materialized. See HirFingerprint's own comment for why this exists.
HirFingerprint compute_hir_fingerprint(const HirModule& hir) {
    HirFingerprint fingerprint;
    fingerprint.op_count = hir.ops.size();
    fingerprint.qubit_count = hir.num_qubits;

    uint64_t hash = 0xcbf29ce484222325ULL;  // FNV offset basis.
    fnv_mix(hash, fingerprint.op_count);
    fnv_mix(hash, fingerprint.qubit_count);

    for (const HeisenbergOp& op : hir.ops) {
        fnv_mix(hash, static_cast<uint64_t>(op.op_type()));
        fnv_mix(hash, op.flags());
        if (op.has_mask()) {
            fnv_mix_words(hash, hir.destab_mask(op));
            fnv_mix_words(hash, hir.stab_mask(op));
        }
        switch (op.op_type()) {
            case OpType::MEASURE:
                fnv_mix(hash, static_cast<uint64_t>(op.meas_record_idx()));
                break;
            case OpType::CONDITIONAL_PAULI:
                fnv_mix(hash, static_cast<uint64_t>(op.controlling_meas()));
                break;
            case OpType::NOISE: {
                const auto site_idx = static_cast<uint32_t>(op.noise_site_idx());
                fnv_mix(hash, site_idx);
                const NoiseSite& site = hir.noise_sites[site_idx];
                for (const NoiseChannel& channel : site.channels) {
                    const PauliMaskView channel_view = hir.noise_channel_masks.at(channel.mask);
                    fnv_mix_words(hash, channel_view.x());
                    fnv_mix_words(hash, channel_view.z());
                }
                break;
            }
            case OpType::READOUT_NOISE:
                fnv_mix(hash, static_cast<uint64_t>(op.readout_noise_idx()));
                break;
            case OpType::DETECTOR:
                fnv_mix(hash, static_cast<uint64_t>(op.detector_idx()));
                break;
            case OpType::OBSERVABLE:
                fnv_mix(hash, static_cast<uint64_t>(op.observable_idx()));
                fnv_mix(hash, op.observable_target_list_idx());
                break;
            case OpType::EXP_VAL:
                fnv_mix(hash, static_cast<uint64_t>(op.exp_val_idx()));
                break;
            case OpType::INSTRUMENT:
                fnv_mix(hash, static_cast<uint64_t>(op.instrument_site_idx()));
                break;
            case OpType::PHASE_ROTATION:
                fnv_mix(hash, double_bits(op.alpha()));
                break;
            case OpType::T_GATE:
            case OpType::NUM_OP_TYPES:
                break;
        }
    }

    // Distinguishes a materialized-but-empty vector (an empty-ops HIR) from
    // an unmaterialized one, and from disagreement on materialization
    // itself between build() and apply_schedule()'s target.
    fnv_mix(hash, hir.has_logical_noise_prefix() ? uint64_t{1} : uint64_t{0});
    if (hir.has_logical_noise_prefix()) {
        for (uint32_t prefix : hir.logical_noise_prefix) {
            fnv_mix(hash, prefix);
        }
    }

    fingerprint.content_hash = hash;
    return fingerprint;
}

// True when no edge is needed between the ops at i < j: either they pass
// the ordinary commutation test, or noise transparency specifically waives
// it for a NOISE-versus-movable pair. See the file comment for why the
// second clause is sound.
bool allowed(const HirModule& hir, const HeisenbergOp& left, const HeisenbergOp& right,
             bool noise_transparent) {
    if (can_swap(left, right, hir)) {
        return true;
    }
    if (!noise_transparent) {
        return false;
    }
    const bool left_noise = left.op_type() == OpType::NOISE;
    const bool right_noise = right.op_type() == OpType::NOISE;
    const bool left_movable = is_movable_op(left.op_type());
    const bool right_movable = is_movable_op(right.op_type());
    return (left_noise && right_movable) || (right_noise && left_movable);
}

// Groups (key, value) pairs into CSR form: offsets[k]..offsets[k + 1]
// indexes into `indices` for every value recorded under key k. std::pair's
// default order is lexicographic, so sorting by (key, value) both groups
// equal keys together and, within each group, leaves values ascending.
void group_into_csr(size_t num_keys, std::vector<std::pair<uint32_t, uint32_t>> pairs,
                    std::vector<uint32_t>& offsets, std::vector<uint32_t>& indices) {
    std::ranges::sort(pairs);
    offsets.assign(num_keys + 1, 0);
    for (const auto& entry : pairs) {
        ++offsets[entry.first + 1];
    }
    for (size_t i = 0; i < num_keys; ++i) {
        offsets[i + 1] += offsets[i];
    }
    indices.clear();
    indices.reserve(pairs.size());
    for (const auto& entry : pairs) {
        indices.push_back(entry.second);
    }
}

}  // namespace

ScheduleDependence ScheduleDependence::build(const HirModule& hir,
                                             ScheduleDependenceOptions options) {
    const size_t n = hir.ops.size();
    ScheduleDependence dep;
    dep.noise_transparent_ = options.noise_transparent;
    dep.fingerprint_ = compute_hir_fingerprint(hir);
    dep.movable_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        dep.movable_[i] = is_movable_op(hir.ops[i].op_type());
    }

    // (from, to) edges, gathered in two structurally different passes below
    // and only grouped into per-op adjacency (by group_into_csr) once both
    // are collected, since neither pass alone produces edges in an order
    // that is already sorted the way the other direction needs.
    std::vector<std::pair<uint32_t, uint32_t>> edges;

    // Chain consecutive fixed ops so every fixed op keeps its original
    // relative order, regardless of what can_swap would say about any one
    // pair of them.
    std::optional<uint32_t> previous_fixed;
    for (uint32_t j = 0; j < n; ++j) {
        if (dep.movable_[j]) {
            continue;
        }
        if (previous_fixed.has_value()) {
            edges.emplace_back(*previous_fixed, j);
        }
        previous_fixed = j;
    }

    // O(N^2) can_swap calls in the worst case: acceptable for this
    // prototype (see the file comment).
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i + 1; j < n; ++j) {
            if (!dep.movable_[i] && !dep.movable_[j]) {
                continue;  // both fixed: handled by the chain above
            }
            if (allowed(hir, hir.ops[i], hir.ops[j], dep.noise_transparent_)) {
                continue;
            }
            edges.emplace_back(i, j);
        }
    }

    // Successors group by "from", ascending "to": exactly (from, to)
    // lexicographic order, which is what `edges` already sorts to.
    group_into_csr(n, edges, dep.succ_offsets_, dep.succ_indices_);

    // Predecessors group by "to", ascending "from": swap each pair first so
    // the same helper's (key, value) = (first, second) grouping applies.
    std::vector<std::pair<uint32_t, uint32_t>> reversed_edges;
    reversed_edges.reserve(edges.size());
    for (const auto& [from, to] : edges) {
        reversed_edges.emplace_back(to, from);
    }
    group_into_csr(n, std::move(reversed_edges), dep.pred_offsets_, dep.pred_indices_);

    return dep;
}

bool ScheduleDependence::is_movable(size_t op) const {
    assert(op < movable_.size());
    return movable_[op];
}

std::span<const uint32_t> ScheduleDependence::predecessors(size_t op) const {
    assert(op < num_ops());
    return std::span<const uint32_t>(pred_indices_)
        .subspan(pred_offsets_[op], pred_offsets_[op + 1] - pred_offsets_[op]);
}

std::span<const uint32_t> ScheduleDependence::successors(size_t op) const {
    assert(op < num_ops());
    return std::span<const uint32_t>(succ_indices_)
        .subspan(succ_offsets_[op], succ_offsets_[op + 1] - succ_offsets_[op]);
}

bool ScheduleDependence::is_linear_extension(std::span<const uint32_t> order) const {
    const size_t n = num_ops();
    if (order.size() != n) {
        return false;
    }
    // position[op] == n means "not seen yet"; every valid position is < n,
    // so this sentinel also catches an out-of-range entry sharing the slot
    // an in-range duplicate would otherwise collide on.
    std::vector<uint32_t> position(n, static_cast<uint32_t>(n));
    for (size_t pos = 0; pos < order.size(); ++pos) {
        const uint32_t op = order[pos];
        if (op >= n || position[op] != n) {
            return false;
        }
        position[op] = static_cast<uint32_t>(pos);
    }
    for (size_t op = 0; op < n; ++op) {
        for (uint32_t succ : successors(op)) {
            if (position[op] >= position[succ]) {
                return false;
            }
        }
    }
    return true;
}

void apply_schedule(HirModule& hir, const ScheduleDependence& dependence,
                    std::span<const uint32_t> order) {
    if (!(compute_hir_fingerprint(hir) == dependence.fingerprint())) {
        throw std::invalid_argument(
            "apply_schedule: target HIR's fingerprint does not match the HIR the dependence "
            "relation was built from");
    }
    if (!dependence.is_linear_extension(order)) {
        throw std::invalid_argument(
            "apply_schedule: order is not a linear extension of the dependence relation");
    }

    if (dependence.noise_transparent()) {
        hir.materialize_logical_noise_prefix();
    }

    const bool has_source_map = hir.source_map.size() == hir.ops.size();
    const bool has_lnp = hir.has_logical_noise_prefix();

    std::vector<HeisenbergOp> new_ops;
    new_ops.reserve(hir.ops.size());
    std::vector<std::vector<uint32_t>> new_source_map;
    std::vector<uint32_t> new_lnp;
    if (has_source_map) {
        new_source_map.reserve(hir.ops.size());
    }
    if (has_lnp) {
        new_lnp.reserve(hir.ops.size());
    }

    for (uint32_t idx : order) {
        new_ops.push_back(hir.ops[idx]);
        if (has_source_map) {
            new_source_map.push_back(std::move(hir.source_map[idx]));
        }
        if (has_lnp) {
            new_lnp.push_back(hir.logical_noise_prefix[idx]);
        }
    }

    hir.ops = std::move(new_ops);
    if (has_source_map) {
        hir.source_map = std::move(new_source_map);
    }
    if (has_lnp) {
        hir.logical_noise_prefix = std::move(new_lnp);
    }
}

}  // namespace clifft::detail
