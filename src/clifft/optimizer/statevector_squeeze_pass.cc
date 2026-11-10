#include "clifft/optimizer/statevector_squeeze_pass.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace clifft {

namespace {

bool is_expansion(const HeisenbergOp& op) {
    return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
}

// Mask words touched by the channels of each noise site, kept as one bitmap per
// site and filled on first use. A Pauli whose nonzero word range avoids every
// touched word commutes with the whole site, so its channel masks need not be
// read. Single-word registers gain nothing, so the table stays empty for them.
class NoiseSupport {
  public:
    explicit NoiseSupport(const HirModule& hir)
        : hir_(hir),
          mask_words_(std::max(hir.pauli_masks.num_words(), hir.noise_channel_masks.num_words())),
          stride_((mask_words_ + 63) / 64) {
        if (mask_words_ > 1) {
            known_.assign(hir.noise_sites.size(), 0);
            touched_.assign(hir.noise_sites.size() * stride_, 0);
        }
    }

    [[nodiscard]] bool misses(uint32_t site, const PauliCommutation& axis) {
        if (mask_words_ <= 1) {
            return false;
        }
        uint64_t* touched = touched_.data() + size_t{site} * stride_;
        if (!known_[site]) {
            // First query for this site: record every word its channels touch.
            for (const auto& ch : hir_.noise_sites[site].channels) {
                const auto view = hir_.noise_channel_masks.at(ch.mask);
                for (uint32_t w = 0; w < view.x().num_words(); ++w) {
                    if ((view.x().words[w] | view.z().words[w]) != 0) {
                        touched[w / 64] |= uint64_t{1} << (w % 64);
                    }
                }
            }
            known_[site] = 1;
        }
        for (uint32_t w = axis.begin_word(); w < axis.end_word(); ++w) {
            if ((touched[w / 64] >> (w % 64)) & 1U) {
                return false;
            }
        }
        return true;
    }

  private:
    const HirModule& hir_;
    uint32_t mask_words_;
    uint32_t stride_;
    std::vector<uint8_t> known_;
    std::vector<uint64_t> touched_;
};

// can_swap(moving, other) for a measurement or expansion with the given axis.
// Neither has a classical or positional barrier against a noise site, so
// commutation alone decides that swap and NoiseSupport can answer it when the
// axis avoids the words the site touches.
bool can_cross(const HirModule& hir, NoiseSupport& noise, const HeisenbergOp& moving,
               const HeisenbergOp& other, const PauliCommutation& axis) {
    assert(moving.op_type() == OpType::MEASURE || is_expansion(moving));
    if (other.op_type() == OpType::NOISE &&
        noise.misses(static_cast<uint32_t>(other.noise_site_idx()), axis)) {
        return true;
    }
    return can_swap(moving, other, hir, &axis);
}

bool can_bypass_to(const HirModule& hir, NoiseSupport& noise, size_t moving, size_t target,
                   const PauliCommutation& axis) {
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
    bool has_sm = hir.source_map.size() == hir.ops.size();
    bool has_lnp = hir.has_logical_noise_prefix();
    NoiseSupport noise(hir);

    // EXP_VAL acts as a hard barrier via can_swap(), so neither sweep
    // will move operations across an expectation value probe.

    // Sweep 1: Eager Compaction (leftward bubble of MEASUREs)
    for (size_t i = 1; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() != OpType::MEASURE) {
            continue;
        }
        const PauliCommutation axis(hir.destab_mask(hir.ops[i]), hir.stab_mask(hir.ops[i]));
        size_t curr = i;
        while (curr > 0 && can_cross(hir, noise, hir.ops[curr], hir.ops[curr - 1], axis)) {
            std::swap(hir.ops[curr - 1], hir.ops[curr]);
            if (has_sm) {
                std::swap(hir.source_map[curr - 1], hir.source_map[curr]);
            }
            if (has_lnp) {
                std::swap(hir.logical_noise_prefix[curr - 1], hir.logical_noise_prefix[curr]);
            }
            --curr;
        }
    }

    // Sweep 2: Lazy Expansion (rightward bubble of non-Clifford gates).
    // For example, T0 T1 H1 M1 can become T1 H1 M1 T0: T1 is blocked by
    // H1, but T0 commutes with every operation on qubit 1.
    if (hir.ops.size() >= 2) {
        // Avoid full-mask commutation scans when an expansion-only suffix has
        // no useful destination. Crossings move one cached position left and
        // never change the relative order of non-expansions.
        std::vector<size_t> non_expansions;
        for (size_t i = 0; i < hir.ops.size(); ++i) {
            if (!is_expansion(hir.ops[i])) {
                non_expansions.push_back(i);
            }
        }

        for (size_t i = hir.ops.size() - 2;; --i) {
            if (is_expansion(hir.ops[i]) && !non_expansions.empty() && non_expansions.back() > i) {
                const PauliCommutation axis(hir.destab_mask(hir.ops[i]), hir.stab_mask(hir.ops[i]));
                size_t curr = i;
                while (curr < hir.ops.size() - 1) {
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
                        std::rotate(hir.ops.begin() + static_cast<std::ptrdiff_t>(curr),
                                    hir.ops.begin() + static_cast<std::ptrdiff_t>(curr + 1),
                                    hir.ops.begin() + static_cast<std::ptrdiff_t>(target + 1));
                        if (has_sm) {
                            std::rotate(
                                hir.source_map.begin() + static_cast<std::ptrdiff_t>(curr),
                                hir.source_map.begin() + static_cast<std::ptrdiff_t>(curr + 1),
                                hir.source_map.begin() + static_cast<std::ptrdiff_t>(target + 1));
                        }
                        if (has_lnp) {
                            std::rotate(hir.logical_noise_prefix.begin() +
                                            static_cast<std::ptrdiff_t>(curr),
                                        hir.logical_noise_prefix.begin() +
                                            static_cast<std::ptrdiff_t>(curr + 1),
                                        hir.logical_noise_prefix.begin() +
                                            static_cast<std::ptrdiff_t>(target + 1));
                        }
                        --*target_it;
                        curr = target;
                        continue;
                    }

                    if (!can_cross(hir, noise, hir.ops[curr], hir.ops[curr + 1], axis)) {
                        break;
                    }
                    std::swap(hir.ops[curr], hir.ops[curr + 1]);
                    if (has_sm) {
                        std::swap(hir.source_map[curr], hir.source_map[curr + 1]);
                    }
                    if (has_lnp) {
                        std::swap(hir.logical_noise_prefix[curr],
                                  hir.logical_noise_prefix[curr + 1]);
                    }
                    auto crossed =
                        std::upper_bound(non_expansions.begin(), non_expansions.end(), curr);
                    assert(crossed != non_expansions.end() && *crossed == curr + 1);
                    --*crossed;
                    ++curr;
                }
            }
            if (i == 0)
                break;
        }
    }
}

}  // namespace clifft
