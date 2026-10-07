#include "clifft/optimizer/rotation_simplification_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/symbolic_stabilizers.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {
namespace {

bool is_rotation(const HeisenbergOp& op) {
    return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
}

std::vector<uint64_t> axis_key(const PauliString& axis) {
    std::vector<uint64_t> key(axis.x().words.begin(), axis.x().words.end());
    key.insert(key.end(), axis.z().words.begin(), axis.z().words.end());
    return key;
}

struct Term {
    PauliString axis;
    double angle;
    std::vector<size_t> origins;
};

class Rewriter {
  public:
    Rewriter(const HirModule& input, RotationSimplificationOptions options)
        : input_(input), options_(options), known_(input.num_qubits) {}

    std::optional<HirModule> run() {
        for (size_t start = 0; start < input_.ops.size();) {
            if (!is_rotation(input_.ops[start]) || known_.fixed_constraints().empty()) {
                emit_original(start++);
                continue;
            }
            terms_.clear();
            size_t end = collect(start);
            ++regions_examined;
            bool changed = false;
            uint32_t retries = 0;
            while (true) {
                const size_t before = terms_.size();
                changed |= simplify();
                const size_t extended = collect(end);
                if (extended != end) {
                    // Consuming fresh input guarantees forward progress. The
                    // retry budget applies only to revisiting retained terms.
                    end = extended;
                    retries = 0;
                    continue;
                }
                if (terms_.empty() || terms_.size() == before) {
                    break;
                }
                if (++retries == options_.max_region_passes) {
                    ++regions_capped;
                    break;
                }
            }
            if (changed) {
                start_candidate(start);
                ++regions_reduced;
                rotations_removed += end - start - terms_.size();
            }
            if (candidate_) {
                for (const auto& term : terms_) {
                    auto op = candidate_->ops[term.origins.front()];
                    if (std::abs(std::abs(term.angle) - 0.25) <
                        kRotationCanonicalizationTolerance) {
                        candidate_->demote_to_tgate(op, term.angle < 0);
                    } else {
                        candidate_->demote_to_phase_rotation(op, term.angle);
                    }
                    optimizer_detail::write_axis(candidate_->mask_at(op), term.axis);
                    known_.advance(*candidate_, op);
                    output_.push_back(op);
                    if (!input_.source_map.empty()) {
                        std::vector<uint32_t> lines;
                        for (size_t index : term.origins) {
                            const auto& original = input_.source_map[index];
                            lines.insert(lines.end(), original.begin(), original.end());
                        }
                        std::ranges::sort(lines);
                        lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
                        sources_.push_back(std::move(lines));
                    }
                }
            } else {
                for (size_t i = start; i < end; ++i) {
                    known_.advance(input_, input_.ops[i]);
                }
            }
            start = end;
        }
        if (candidate_) {
            candidate_->ops = std::move(output_);
            candidate_->source_map = std::move(sources_);
            cliffords_.finish(*candidate_);
        }
        return std::move(candidate_);
    }

    size_t regions_examined = 0;
    size_t regions_reduced = 0;
    size_t regions_capped = 0;
    size_t rotations_removed = 0;
    bool needs_cost_guard = false;

  private:
    size_t collect(size_t end) {
        while (end < input_.ops.size() && terms_.size() < options_.max_region_ops &&
               is_rotation(input_.ops[end])) {
            const auto& op = input_.ops[end];
            auto axis = cliffords_.read(input_.mask_view(op), input_.num_qubits);
            if (std::ranges::any_of(terms_, [&](const Term& earlier) {
                    return !axis.view().commutes(earlier.axis.view());
                })) {
                break;
            }
            double angle =
                op.op_type() == OpType::T_GATE ? (op.is_dagger() ? -0.25 : 0.25) : op.alpha();
            if (axis.sign()) {
                angle = -angle;
            }
            axis.set_sign(false);
            terms_.push_back({std::move(axis), std::remainder(angle, 2.0), {end}});
            ++end;
        }
        return end;
    }

    bool simplify() {
        auto preserved = known_.fixed_constraints();
        for (const auto& term : terms_) {
            preserved.intersect(term.axis.view());
        }
        std::map<std::vector<uint64_t>, size_t> indices;
        std::vector<Term> reduced_terms;
        reduced_terms.reserve(terms_.size());
        bool changed = false;
        for (auto& term : terms_) {
            auto reduced = preserved.reduce_body(term.axis);
            reduced.set_sign(false);
            if (reduced.x().is_zero() && reduced.z().is_zero()) {
                // A rotation about a known stabilizer is a common global phase.
                // Deleting it cannot alter structural active-state transitions.
                changed = true;
                continue;
            }
            auto residual = reduced;
            residual.right_multiply(term.axis.view());
            const auto negative = preserved.eigenvalue(residual);
            assert(negative.has_value());
            if (*negative) {
                term.angle = -term.angle;
            }
            if (!(reduced == term.axis)) {
                needs_cost_guard = true;
                changed = true;
            }
            term.axis = std::move(reduced);
            const auto [it, inserted] = indices.emplace(axis_key(term.axis), reduced_terms.size());
            if (inserted) {
                reduced_terms.push_back(std::move(term));
            } else {
                auto& earlier = reduced_terms[it->second];
                earlier.angle = std::remainder(earlier.angle + term.angle, 2.0);
                earlier.origins.insert(earlier.origins.end(), term.origins.begin(),
                                       term.origins.end());
                needs_cost_guard = true;
                changed = true;
            }
        }
        terms_.clear();
        for (auto& term : reduced_terms) {
            const auto clifford = classify_clifford_rotation(term.angle);
            if (!clifford) {
                terms_.push_back(std::move(term));
                continue;
            }
            needs_cost_guard = true;
            changed = true;
            if (*clifford != CliffordRotation::IDENTITY) {
                const uint8_t coefficient = *clifford == CliffordRotation::SQRT    ? 2
                                            : *clifford == CliffordRotation::PAULI ? 4
                                                                                   : 6;
                // Every retained axis commutes with this factor, so absorbing
                // it leaves the other terms in the region unchanged.
                cliffords_.absorb(term.axis, coefficient);
            }
        }
        return changed;
    }

    void start_candidate(size_t start) {
        if (candidate_) {
            return;
        }
        candidate_.emplace(input_);
        candidate_->logical_noise_prefix.clear();
        output_.reserve(input_.ops.size());
        output_.assign(input_.ops.begin(), input_.ops.begin() + start);
        if (!input_.source_map.empty()) {
            sources_.reserve(input_.source_map.size());
            sources_.assign(input_.source_map.begin(), input_.source_map.begin() + start);
        }
    }

    void emit_original(size_t index) {
        if (!candidate_) {
            known_.advance(input_, input_.ops[index]);
            return;
        }
        auto& op = candidate_->ops[index];
        cliffords_.transform(*candidate_, op);
        known_.advance(*candidate_, op);
        output_.push_back(op);
        if (!input_.source_map.empty()) {
            sources_.push_back(input_.source_map[index]);
        }
    }

    const HirModule& input_;
    RotationSimplificationOptions options_;
    SymbolicStabilizers known_;
    optimizer_detail::CliffordAbsorption cliffords_;
    std::optional<HirModule> candidate_;
    std::vector<Term> terms_;
    std::vector<HeisenbergOp> output_;
    std::vector<std::vector<uint32_t>> sources_;
};

}  // namespace

RotationSimplificationPass::RotationSimplificationPass(RotationSimplificationOptions options)
    : options_(options) {
    if (options.max_region_ops > 4096) {
        throw std::invalid_argument("max_region_ops must be between zero and 4096");
    }
    if (options.max_region_passes == 0 || options.max_region_passes > 64) {
        throw std::invalid_argument("max_region_passes must be between one and 64");
    }
}

void RotationSimplificationPass::run(HirModule& hir) {
    regions_examined_ = regions_reduced_ = regions_capped_ = rotations_removed_ = 0;
    if (options_.max_region_ops == 0 || !std::ranges::any_of(hir.ops, is_rotation) ||
        !hir.logical_noise_prefix_matches_schedule()) {
        return;
    }
    if (!hir.source_map.empty() && hir.source_map.size() != hir.ops.size()) {
        throw std::invalid_argument("HIR source map size does not match the operation count");
    }
    Rewriter rewriter(hir, options_);
    auto candidate = rewriter.run();
    regions_examined_ = rewriter.regions_examined;
    regions_capped_ = rewriter.regions_capped;
    if (!candidate) {
        return;
    }
    if (rewriter.needs_cost_guard) {
        const auto after = analyze_active_width(*candidate);
        if (after.peak_width != 0) {
            const auto before = analyze_active_width(hir);
            if (after.peak_width > before.peak_width ||
                estimate_dense_work(after) > estimate_dense_work(before)) {
                return;
            }
        }
    }
    regions_reduced_ = rewriter.regions_reduced;
    rotations_removed_ = rewriter.rotations_removed;
    hir = std::move(*candidate);
}

}  // namespace clifft
