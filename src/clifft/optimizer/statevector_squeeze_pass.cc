#include "clifft/optimizer/statevector_squeeze_pass.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace clifft {

namespace {

constexpr size_t kMaxBucketSize = 512;
constexpr uint32_t kNoOp = std::numeric_limits<uint32_t>::max();

bool is_expansion(const HeisenbergOp& op) {
    return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
}

bool has_inline_pauli_mask(const HeisenbergOp& op) {
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
            return true;
        case OpType::NOISE:
        case OpType::READOUT_NOISE:
        case OpType::EXP_VAL:
        case OpType::INSTRUMENT:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::NUM_OP_TYPES:
            return false;
    }
    return false;
}

// Per-word union of X and Z channel bits for each noise site, populated on
// first query. If a Pauli has zero bitwise overlap with `(ux & az) | (uz & ax)`
// on every word it stores, it commutes with every channel of the site without
// reading individual channel masks. Single-word registers store flat 64-bit
// X/Z unions directly instead of building sparse word lists.
class NoiseSupport {
  public:
    explicit NoiseSupport(const HirModule& hir)
        : hir_(hir), single_word_(hir.num_qubits <= 64), known_(hir.noise_sites.size(), 0) {
        if (single_word_) {
            single_ux_.assign(hir.noise_sites.size(), 0);
            single_uz_.assign(hir.noise_sites.size(), 0);
        } else {
            offsets_.assign(hir.noise_sites.size(), 0);
            counts_.assign(hir.noise_sites.size(), 0);
            entries_.reserve(hir.noise_sites.size() * 2);
        }
    }

    [[nodiscard]] bool misses(uint32_t site, PauliMaskView axis) {
        if (axis.is_identity()) {
            return true;
        }
        ensure(site);
        if (single_word_) {
            const PauliWord& word = axis.words().front();
            return ((single_ux_[site] & word.z) | (single_uz_[site] & word.x)) == 0;
        }
        bool hit = false;
        for_each_common_word(support(site), axis, [&](const PauliWord& u, const PauliWord& a) {
            hit |= ((u.x & a.z) | (u.z & a.x)) != 0;
        });
        return !hit;
    }

    void add_to_union(uint32_t site, std::vector<uint64_t>& ux, std::vector<uint64_t>& uz) {
        ensure(site);
        if (single_word_) {
            if (!ux.empty()) {
                ux[0] |= single_ux_[site];
                uz[0] |= single_uz_[site];
            }
            return;
        }
        for (const PauliWord& e : support(site)) {
            ux[e.index] |= e.x;
            uz[e.index] |= e.z;
        }
    }

  private:
    // The union of a site's channels, stored like a Pauli mask.
    [[nodiscard]] PauliMaskView support(uint32_t site) const {
        return PauliMaskView{{entries_.data() + offsets_[site], counts_[site]}, false};
    }

    void ensure(uint32_t site) {
        if (!known_[site]) {
            populate(site);
        }
    }

    void populate(uint32_t site) {
        if (single_word_) {
            uint64_t ux = 0;
            uint64_t uz = 0;
            for (const auto& ch : hir_.noise_sites[site].channels) {
                for (const PauliWord& word : hir_.noise_channel_masks.at(ch.mask)) {
                    ux |= word.x;
                    uz |= word.z;
                }
            }
            single_ux_[site] = ux;
            single_uz_[site] = uz;
            known_[site] = 1;
            return;
        }

        scratch_.clear();
        for (const auto& ch : hir_.noise_sites[site].channels) {
            const auto words = hir_.noise_channel_masks.at(ch.mask).words();
            scratch_.insert(scratch_.end(), words.begin(), words.end());
        }
        merge_union(scratch_);
        const uint32_t start = static_cast<uint32_t>(entries_.size());
        entries_.insert(entries_.end(), scratch_.begin(), scratch_.end());
        offsets_[site] = start;
        counts_[site] = static_cast<uint32_t>(entries_.size() - start);
        known_[site] = 1;
    }

    const HirModule& hir_;
    bool single_word_ = false;
    std::vector<uint8_t> known_;
    std::vector<uint64_t> single_ux_;
    std::vector<uint64_t> single_uz_;
    std::vector<uint32_t> offsets_;
    std::vector<uint32_t> counts_;
    std::vector<PauliWord> entries_;
    std::vector<PauliWord> scratch_;
};

// Equivalent to can_swap(moving, other, hir) for a measurement or
// expansion whose mask is `axis`: neither has classical or positional barriers
// against a NOISE op, so disjoint channel-support unions short-circuit before
// scanning channels, and all other commutation and hazard rules stay in can_swap().
bool can_cross(const HirModule& hir, NoiseSupport& noise, const HeisenbergOp& moving,
               const HeisenbergOp& other, PauliMaskView axis) {
    assert(moving.op_type() == OpType::MEASURE || is_expansion(moving));
    if (other.op_type() == OpType::NOISE &&
        noise.misses(static_cast<uint32_t>(other.noise_site_idx()), axis)) {
        return true;
    }
    return can_swap(moving, other, hir);
}

bool can_bypass_to(const HirModule& hir, NoiseSupport& noise, size_t moving, size_t target,
                   PauliMaskView axis) {
    assert(moving < target && target < hir.ops.size());
    assert(!is_expansion(hir.ops[target]));
    if (!can_cross(hir, noise, hir.ops[moving], hir.ops[target], axis)) {
        return false;
    }
    for (size_t next = target; next > moving + 1; --next) {
        if (!can_cross(hir, noise, hir.ops[moving], hir.ops[next - 1], axis)) {
            return false;
        }
    }
    return true;
}

}  // namespace

void StatevectorSqueezePass::run(HirModule& hir) {
    const size_t num_ops = hir.ops.size();
    if (num_ops < 2) {
        return;
    }

    const bool has_sm = hir.source_map.size() == num_ops;
    const bool has_lnp = hir.has_logical_noise_prefix();
    const uint32_t num_words = (hir.num_qubits + 63) / 64;

    size_t last_movable_measure = 0;
    bool has_non_expansion = false;
    for (size_t i = 0; i < num_ops; ++i) {
        const HeisenbergOp& op = hir.ops[i];
        if (i > 0 && op.op_type() == OpType::MEASURE) {
            last_movable_measure = i;
        }
        if (!is_expansion(op)) {
            has_non_expansion = true;
        }
    }
    if (!has_non_expansion) {
        return;
    }

    NoiseSupport noise(hir);

    // Positional barriers are enforced via is_positional_barrier() and
    // can_swap(), so neither sweep reorders operations across them.
    //
    // Sweep 1: Eager Compaction (leftward bubble of MEASUREs).
    // Chunking the processed prefix into bounded-size buckets with per-bucket
    // X/Z unions skips commuting blocks in O(active words) per bucket and avoids
    // quadratic shifts of hir.ops.
    if (last_movable_measure > 0) {
        const size_t sweep1_end = last_movable_measure + 1;
        struct Bucket {
            std::vector<uint32_t> ids;
            std::vector<uint64_t> ux;
            std::vector<uint64_t> uz;
            bool has_barrier = false;

            explicit Bucket(uint32_t words) : ux(words, 0), uz(words, 0) {
                ids.reserve(kMaxBucketSize);
            }
        };

        auto add_op_summary = [&](Bucket& bucket, uint32_t id) {
            const HeisenbergOp& op = hir.ops[id];
            if (is_positional_barrier(op.op_type())) {
                bucket.has_barrier = true;
            }
            if (op.op_type() == OpType::NOISE) {
                noise.add_to_union(static_cast<uint32_t>(op.noise_site_idx()), bucket.ux,
                                   bucket.uz);
            } else if (has_inline_pauli_mask(op)) {
                for (const PauliWord& word : hir.mask_view(op)) {
                    bucket.ux[word.index] |= word.x;
                    bucket.uz[word.index] |= word.z;
                }
            }
        };

        auto rebuild_summary = [&](Bucket& bucket) {
            std::ranges::fill(bucket.ux, 0);
            std::ranges::fill(bucket.uz, 0);
            bucket.has_barrier = false;
            for (uint32_t id : bucket.ids) {
                add_op_summary(bucket, id);
            }
        };

        std::vector<Bucket> buckets;
        buckets.reserve((sweep1_end / (kMaxBucketSize / 2)) + 2);
        buckets.emplace_back(num_words);

        std::vector<uint32_t> bucket_of(sweep1_end, 0);
        std::vector<uint32_t> last_classical_op(
            static_cast<size_t>(hir.num_measurements) + hir.num_hidden_measurements, kNoOp);
        auto record_classical = [&](const HeisenbergOp& op, uint32_t op_id) {
            for_each_classical_record(op, hir, [&](uint32_t rec) {
                if (rec >= last_classical_op.size()) {
                    last_classical_op.resize(static_cast<size_t>(rec) + 1, kNoOp);
                }
                last_classical_op[rec] = op_id;
            });
        };

        auto split_if_full = [&](size_t b) {
            if (buckets[b].ids.size() < kMaxBucketSize) {
                return;
            }
            const size_t mid = buckets[b].ids.size() / 2;
            Bucket right(num_words);
            right.ids.assign(buckets[b].ids.begin() + static_cast<std::ptrdiff_t>(mid),
                             buckets[b].ids.end());
            buckets[b].ids.resize(mid);
            buckets.insert(buckets.begin() + static_cast<std::ptrdiff_t>(b + 1), std::move(right));
            rebuild_summary(buckets[b]);
            rebuild_summary(buckets[b + 1]);
            for (size_t jb = b + 1; jb < buckets.size(); ++jb) {
                for (uint32_t id : buckets[jb].ids) {
                    bucket_of[id] = static_cast<uint32_t>(jb);
                }
            }
        };

        bool moved_in_sweep1 = false;
        for (uint32_t i = 0; i < static_cast<uint32_t>(sweep1_end); ++i) {
            const HeisenbergOp& op = hir.ops[i];
            if (op.op_type() != OpType::MEASURE || i == 0) {
                const size_t b = buckets.size() - 1;
                buckets[b].ids.push_back(i);
                bucket_of[i] = static_cast<uint32_t>(b);
                add_op_summary(buckets[b], i);
                record_classical(op, i);
                split_if_full(b);
                continue;
            }

            const PauliMaskView axis = hir.mask_view(op);
            const uint32_t rec = static_cast<uint32_t>(op.meas_record_idx());
            const size_t min_bucket =
                (rec < last_classical_op.size() && last_classical_op[rec] != kNoOp)
                    ? static_cast<size_t>(bucket_of[last_classical_op[rec]])
                    : 0;

            auto bucket_may_block = [&](size_t b) {
                if (b == min_bucket && rec < last_classical_op.size() &&
                    last_classical_op[rec] != kNoOp) {
                    return true;
                }
                const Bucket& bucket = buckets[b];
                if (bucket.has_barrier) {
                    return true;
                }
                return std::ranges::any_of(axis.words(), [&](const PauliWord& word) {
                    return ((bucket.ux[word.index] & word.z) | (bucket.uz[word.index] & word.x)) !=
                           0;
                });
            };

            size_t dest_bucket = 0;
            size_t dest_pos = 0;
            bool placed = false;
            for (size_t b = buckets.size(); b-- > min_bucket;) {
                if (!bucket_may_block(b)) {
                    continue;
                }
                const auto& ids = buckets[b].ids;
                for (size_t p = ids.size(); p-- > 0;) {
                    if (!can_cross(hir, noise, op, hir.ops[ids[p]], axis)) {
                        dest_bucket = b;
                        dest_pos = p + 1;
                        placed = true;
                        break;
                    }
                }
                if (placed) {
                    break;
                }
            }

            if (!(dest_bucket == buckets.size() - 1 && dest_pos == buckets.back().ids.size())) {
                moved_in_sweep1 = true;
            }
            buckets[dest_bucket].ids.insert(
                buckets[dest_bucket].ids.begin() + static_cast<std::ptrdiff_t>(dest_pos), i);
            bucket_of[i] = static_cast<uint32_t>(dest_bucket);
            add_op_summary(buckets[dest_bucket], i);
            record_classical(op, i);
            split_if_full(dest_bucket);
        }

        if (moved_in_sweep1) {
            std::vector<HeisenbergOp> prefix_ops;
            prefix_ops.reserve(sweep1_end);
            std::vector<std::vector<uint32_t>> prefix_sm;
            if (has_sm) {
                prefix_sm.reserve(sweep1_end);
            }
            std::vector<uint32_t> prefix_lnp;
            if (has_lnp) {
                prefix_lnp.reserve(sweep1_end);
            }
            for (const Bucket& bucket : buckets) {
                for (uint32_t id : bucket.ids) {
                    prefix_ops.push_back(hir.ops[id]);
                    if (has_sm) {
                        prefix_sm.push_back(std::move(hir.source_map[id]));
                    }
                    if (has_lnp) {
                        prefix_lnp.push_back(hir.logical_noise_prefix[id]);
                    }
                }
            }
            std::ranges::copy(prefix_ops, hir.ops.begin());
            if (has_sm) {
                std::ranges::move(prefix_sm, hir.source_map.begin());
            }
            if (has_lnp) {
                std::ranges::copy(prefix_lnp, hir.logical_noise_prefix.begin());
            }
        }
    }

    // Sweep 2: Lazy Expansion (rightward bubble of non-Clifford gates).
    // For example, T0 T1 H1 M1 can become T1 H1 M1 T0: T1 is blocked by
    // H1, but T0 commutes with every operation on qubit 1.
    std::vector<size_t> non_expansions;
    non_expansions.reserve(num_ops);
    for (size_t i = 0; i < num_ops; ++i) {
        if (!is_expansion(hir.ops[i])) {
            non_expansions.push_back(i);
        }
    }

    auto rotate_right = [&](size_t first, size_t last_exclusive) {
        std::rotate(hir.ops.begin() + static_cast<std::ptrdiff_t>(first),
                    hir.ops.begin() + static_cast<std::ptrdiff_t>(first + 1),
                    hir.ops.begin() + static_cast<std::ptrdiff_t>(last_exclusive));
        if (has_sm) {
            std::rotate(hir.source_map.begin() + static_cast<std::ptrdiff_t>(first),
                        hir.source_map.begin() + static_cast<std::ptrdiff_t>(first + 1),
                        hir.source_map.begin() + static_cast<std::ptrdiff_t>(last_exclusive));
        }
        if (has_lnp) {
            std::rotate(
                hir.logical_noise_prefix.begin() + static_cast<std::ptrdiff_t>(first),
                hir.logical_noise_prefix.begin() + static_cast<std::ptrdiff_t>(first + 1),
                hir.logical_noise_prefix.begin() + static_cast<std::ptrdiff_t>(last_exclusive));
        }
    };

    for (size_t i = num_ops - 2;; --i) {
        if (is_expansion(hir.ops[i]) && !non_expansions.empty() && non_expansions.back() > i) {
            const PauliMaskView axis = hir.mask_view(hir.ops[i]);
            size_t curr = i;
            while (curr < num_ops - 1) {
                // Crossing another expansion is useful only when it lets
                // this one move past a later non-expanding operation.
                if (is_expansion(hir.ops[curr + 1])) {
                    auto target_it =
                        std::upper_bound(non_expansions.begin(), non_expansions.end(), curr);
                    if (target_it == non_expansions.end()) {
                        break;
                    }
                    const size_t target = *target_it;
                    if (!can_bypass_to(hir, noise, curr, target, axis)) {
                        break;
                    }
                    rotate_right(curr, target + 1);
                    --*target_it;
                    curr = target;
                    continue;
                }

                size_t step = curr + 1;
                while (step < num_ops && !is_expansion(hir.ops[step]) &&
                       can_cross(hir, noise, hir.ops[curr], hir.ops[step], axis)) {
                    ++step;
                }
                if (step == curr + 1) {
                    break;
                }
                const size_t count = step - curr - 1;
                auto crossed = std::upper_bound(non_expansions.begin(), non_expansions.end(), curr);
                for (size_t k = 0; k < count; ++k) {
                    assert(crossed != non_expansions.end() && *crossed == curr + 1 + k);
                    --*crossed;
                    ++crossed;
                }
                rotate_right(curr, step);
                curr = step - 1;
            }
        }
        if (i == 0) {
            break;
        }
    }
}

}  // namespace clifft
