#include "clifft/optimizer/phase_polynomial_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/phase_region.h"
#include "clifft/optimizer/symbolic_stabilizers.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {
namespace {

size_t last_t_end(const HirModule& hir) {
    const auto last = std::find_if(hir.ops.rbegin(), hir.ops.rend(),
                                   [](const auto& op) { return op.op_type() == OpType::T_GATE; });
    return static_cast<size_t>(last.base() - hir.ops.begin());
}

bool is_quantum_or_observer(OpType type) {
    return type != OpType::DETECTOR && type != OpType::OBSERVABLE && type != OpType::READOUT_NOISE;
}

bool operation_supports_final_state_queries(OpType type) {
    return type == OpType::T_GATE || type == OpType::PHASE_ROTATION || type == OpType::EXP_VAL;
}

std::optional<size_t> final_observer_end(const HirModule& hir) {
    for (size_t i = hir.ops.size(); i > 0; --i) {
        const OpType type = hir.ops[i - 1].op_type();
        if (!is_quantum_or_observer(type)) {
            continue;
        }
        if (type == OpType::MEASURE || type == OpType::INSTRUMENT) {
            return i;
        }
        if (type == OpType::EXP_VAL) {
            for (size_t j = 0; j < i; ++j) {
                if (!operation_supports_final_state_queries(hir.ops[j].op_type())) {
                    return i;
                }
            }
        }
        return std::nullopt;
    }
    return std::nullopt;
}

constexpr size_t kMaxPostselectionLookaheadOps = 256;

std::vector<size_t> collect_postselected_measure_indices(
    const HirModule& hir, const std::map<uint32_t, bool>& postselected_records) {
    std::vector<size_t> indices;
    if (postselected_records.empty()) {
        return indices;
    }
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const HeisenbergOp& op = hir.ops[i];
        if (op.op_type() == OpType::MEASURE &&
            postselected_records.contains(static_cast<uint32_t>(op.meas_record_idx()))) {
            indices.push_back(i);
        }
    }
    return indices;
}

bool commutes_with_prefix(const HirModule& hir, size_t start, size_t end, PauliMaskView target) {
    for (size_t i = start; i < end; ++i) {
        const HeisenbergOp& op = hir.ops[i];
        const OpType type = op.op_type();
        if (type == OpType::INSTRUMENT) {
            return false;
        }
        if (op.has_mask()) {
            const PauliMaskView other = hir.mask_view(op);
            if (anti_commute(target, other)) {
                return false;
            }
        } else if (type == OpType::NOISE) {
            const uint32_t site_idx = static_cast<uint32_t>(op.noise_site_idx());
            if (site_idx >= hir.noise_sites.size()) {
                return false;
            }
            for (const NoiseChannel& channel : hir.noise_sites[site_idx].channels) {
                if (channel.prob <= 0.0) {
                    continue;
                }
                const PauliMaskView other = hir.noise_channel_masks.at(channel.mask);
                if (anti_commute(target, other)) {
                    return false;
                }
            }
        }
    }
    return true;
}

using optimizer_detail::CliffordAbsorption;
using phase_detail::Observer;
using Block = phase_detail::PhaseRegion<64>;

class Rewriter {
  public:
    Rewriter(const HirModule& input, PhasePolynomialOptions options,
             bool elide_unreduced_post_observer = false)
        : input_(input),
          max_variables_(options.max_variables),
          wide_budget_{options.max_wide_work, options.max_wide_terms},
          analysis_end_(last_t_end(input)),
          final_observer_end_(final_observer_end(input)),
          elide_unreduced_post_observer_(elide_unreduced_post_observer),
          known_(input, analysis_end_),
          postselected_measure_indices_(
              collect_postselected_measure_indices(input, known_.postselected_records())) {}

    std::optional<HirModule> run() {
        for (size_t i = 0; i < hir().ops.size();) {
            if (hir().ops[i].op_type() == OpType::T_GATE) {
                const KnownStabilizers* postselected = nullptr;
                const KnownStabilizers& entry = prepare_entry(i, postselected);
                const auto fallback_limit = std::min(max_variables_, uint32_t{32});
                const auto one_word_limit =
                    allow_expansion_ ? std::min(max_variables_, uint32_t{64}) : fallback_limit;
                auto block = collect(i, one_word_limit, entry, postselected);
                if (block.axes.size() > fallback_limit) {
                    ++expansion_attempts;
                    if (block.capped) {
                        allow_expansion_ = false;
                        if (max_variables_ > 64 && !wide_budget_.work_exhausted &&
                            wide_budget_.max_updates && wide_budget_.max_terms) {
                            ++wide_attempts;
                            auto wide = phase_detail::try_wide_region(
                                hir(), i, entry, cliffords_, max_variables_, wide_budget_,
                                postselected, final_observer_end_);
                            if (wide) {
                                emit_wide(i, *wide);
                                ++blocks_examined;
                                ++blocks_expanded;
                                i = wide->end;
                                allow_expansion_ = true;
                                continue;
                            }
                        }
                        // Only rebuild the smaller prefix when a larger region actually fails.
                        block = collect(i, fallback_limit, entry, postselected);
                    } else {
                        ++blocks_expanded;
                    }
                }
                ++blocks_examined;
                blocks_capped += block.capped;
                assert(block.end > i);
                if (!rewrite(i, block)) {
                    for (size_t j = i; j < block.end; ++j) {
                        transform(hir().ops[j]);
                        emit(j);
                    }
                }
                i = block.end;
                if (!block.capped) {
                    allow_expansion_ = true;
                }
            } else {
                allow_expansion_ = true;
                transform(hir().ops[i]);
                emit(i++);
            }
        }
        if (candidate_) {
            candidate_->ops = std::move(output_);
            candidate_->source_map = std::move(sources_);
            cliffords_.finish(*candidate_);
        }
        return std::move(candidate_);
    }

    size_t wide_attempts = 0;
    size_t wide_rewrites = 0;
    size_t wide_work() const { return wide_budget_.updates; }
    bool wide_budget_exhausted() const {
        return wide_budget_.work_exhausted || wide_budget_.term_limit_reached;
    }
    size_t blocks_reduced = 0;
    size_t pauli_pullbacks = 0;
    size_t blocks_examined = 0;
    size_t blocks_capped = 0;
    size_t expansion_attempts = 0;
    size_t blocks_expanded = 0;

    bool has_postselected_records() const { return !known_.postselected_records().empty(); }

  private:
    const HirModule& hir() const { return candidate_ ? *candidate_ : input_; }

    const KnownStabilizers& prepare_entry(size_t start, const KnownStabilizers*& postselected) {
        postselected = nullptr;
        const KnownStabilizers& fixed = known_.fixed_constraints();
        while (next_postselected_idx_ < postselected_measure_indices_.size() &&
               postselected_measure_indices_[next_postselected_idx_] < start) {
            ++next_postselected_idx_;
        }
        if (next_postselected_idx_ == postselected_measure_indices_.size()) {
            return fixed;
        }
        const size_t lookahead_end = std::min(postselected_measure_indices_.back() + 1,
                                              start + kMaxPostselectionLookaheadOps);
        if (postselected_measure_indices_[next_postselected_idx_] >= lookahead_end) {
            return fixed;
        }
        const auto& postselected_records = known_.postselected_records();
        bool copied = false;
        const bool has_exp_vals = hir().num_exp_vals > 0;
        bool prefix_measurements_commute = true;
        for (size_t m = start; m < lookahead_end; ++m) {
            const HeisenbergOp& op = hir().ops[m];
            if (op.op_type() == OpType::INSTRUMENT) {
                break;
            }
            if (op.op_type() == OpType::EXP_VAL) {
                if (!prefix_measurements_commute ||
                    !commutes_with_prefix(hir(), start, m, hir().mask_view(op))) {
                    break;
                }
                continue;
            }
            if (op.op_type() != OpType::MEASURE) {
                continue;
            }
            const auto it = postselected_records.find(static_cast<uint32_t>(op.meas_record_idx()));
            if (it == postselected_records.end()) {
                if (has_exp_vals && prefix_measurements_commute) {
                    prefix_measurements_commute =
                        commutes_with_prefix(hir(), start, m, hir().mask_view(op));
                }
                continue;
            }
            const PauliMaskView mask = hir().mask_view(op);
            if (!commutes_with_prefix(hir(), start, m, mask)) {
                prefix_measurements_commute = false;
                continue;
            }
            auto axis = cliffords_.read(mask, hir().num_qubits);
            axis.set_sign(axis.sign() ^ it->second);
            const KnownStabilizers& current_fixed = copied ? lookahead_fixed_ : fixed;
            if (!current_fixed.commutes(axis.view())) {
                continue;
            }
            if (!copied) {
                lookahead_fixed_ = fixed;
                lookahead_postselected_ = KnownStabilizers{};
                copied = true;
            }
            if (!lookahead_postselected_.eigenvalue(axis).has_value()) {
                lookahead_postselected_.insert(axis);
            }
            if (!lookahead_fixed_.eigenvalue(axis).has_value()) {
                lookahead_fixed_.insert(std::move(axis));
            }
        }
        if (!copied) {
            return fixed;
        }
        postselected = &lookahead_postselected_;
        return lookahead_fixed_;
    }

    void start_candidate(size_t start) {
        if (candidate_) {
            return;
        }
        // Until a region improves, collection is read-only and the frame is
        // empty. Keep the input intact for the final cost guard and exceptions.
        candidate_.emplace(input_);
        candidate_->logical_noise_prefix.clear();
        output_.reserve(input_.ops.size());
        output_.assign(input_.ops.begin(), input_.ops.begin() + start);
        if (!input_.source_map.empty()) {
            sources_.reserve(input_.ops.size());
            for (size_t i = 0; i < start; ++i) {
                sources_.push_back(std::move(candidate_->source_map[i]));
            }
        }
    }

    Block collect(size_t start, uint32_t variable_limit, const KnownStabilizers& entry,
                  const KnownStabilizers* postselected) {
        Block block(start, entry);
        while (block.end < hir().ops.size()) {
            if (!block.extend(hir(), entry, cliffords_, postselected, variable_limit,
                              ordinary_work_))
                break;
        }
        return block;
    }

    void emit_pulled_prefix(size_t start, size_t end, std::span<const Observer> observers) {
        start_candidate(start);
        for (const auto& observer : observers) {
            candidate_->set_mask(candidate_->ops[observer.index], observer.pulled_axis);
        }
        for (size_t i = start; i < end; ++i) {
            if (hir().ops[i].op_type() != OpType::T_GATE) {
                if (hir().ops[i].op_type() == OpType::NOISE) {
                    transform(hir().ops[i]);
                }
                emit(i);
            }
        }
    }

    void elide_after_observers(size_t start, size_t end, std::span<const Observer> observers) {
        emit_pulled_prefix(start, end, observers);
        candidate_->final_tableau.reset();
        ++blocks_reduced;
        pauli_pullbacks += observers.size();
    }

    std::vector<uint32_t> collect_sources(std::span<const size_t> rotations) const {
        std::vector<uint32_t> sources;
        if (!hir().source_map.empty()) {
            for (size_t index : rotations) {
                sources.insert(sources.end(), hir().source_map[index].begin(),
                               hir().source_map[index].end());
            }
            std::ranges::sort(sources);
            sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        }
        return sources;
    }

    void emit_synthesized_phase(const PauliString& axis, int coefficient,
                                std::span<const size_t> rotations, size_t& written,
                                const std::vector<uint32_t>& sources) {
        int correction = coefficient;
        if (coefficient & 1) {
            const bool dagger = coefficient >= 5;
            auto& op = candidate_->ops[rotations[written++]];
            candidate_->demote_to_tgate(op, dagger);
            candidate_->set_mask(op, axis);
            output_.push_back(op);
            known_.advance(hir(), op);
            if (!hir().source_map.empty()) {
                sources_.push_back(sources);
            }
            correction = (coefficient - (dagger ? -1 : 1)) & 7;
        }
        if (correction) {
            cliffords_.absorb(axis, static_cast<uint8_t>(correction));
        }
    }

    bool rewrite(size_t start, Block& block) {
        const bool reaches_final_observer =
            final_observer_end_ && block.end >= *final_observer_end_ && !block.observers.empty();
        if (reaches_final_observer) {
            if (elide_unreduced_post_observer_ ||
                block.polynomial.odd_terms() < block.rotations.size() ||
                block.polynomial.core_axes(block.axes.size(), ordinary_work_).size() <
                    block.axes.size()) {
                elide_after_observers(start, block.end, block.observers);
                return true;
            }
        }
        const auto synthesis = block.polynomial.synthesize(block.axes.size(), ordinary_work_);
        const size_t synthesized_count = synthesis.t_count;
        const bool reduced_polynomial = synthesized_count < block.rotations.size() ||
                                        (synthesized_count == block.rotations.size() &&
                                         synthesis.core_width < block.axes.size());
        const bool elide_post_observer =
            (reduced_polynomial || elide_unreduced_post_observer_) && reaches_final_observer;
        if (!reduced_polynomial && !elide_post_observer) {
            return false;
        }
        if (elide_post_observer) {
            elide_after_observers(start, block.end, block.observers);
            return true;
        }
        emit_pulled_prefix(start, block.end, block.observers);
        const auto sources = collect_sources(block.rotations);
        size_t written = 0;
        for (const auto& [parity, coefficient] : synthesis.phases) {
            emit_synthesized_phase(phase_detail::axis_product(block.axes, parity, hir().num_qubits),
                                   coefficient, block.rotations, written, sources);
        }
        assert(written == synthesized_count);
        static_cast<void>(written);
        ++blocks_reduced;
        pauli_pullbacks += block.observers.size();
        return true;
    }

    void emit_wide(size_t start, const phase_detail::WideRewrite& rewrite) {
        if (rewrite.elide_post_observer) {
            elide_after_observers(start, rewrite.end, rewrite.observers);
            ++wide_rewrites;
            return;
        }
        emit_pulled_prefix(start, rewrite.end, rewrite.observers);
        const auto sources = collect_sources(rewrite.rotations);
        size_t written = 0;
        for (const auto& [axis, coefficient] : rewrite.phases) {
            emit_synthesized_phase(axis, coefficient, rewrite.rotations, written, sources);
        }
        assert(written == rewrite.t_count);
        static_cast<void>(written);
        ++blocks_reduced;
        ++wide_rewrites;
        pauli_pullbacks += rewrite.observers.size();
    }

    void transform(const HeisenbergOp& op) {
        if (candidate_) {
            cliffords_.transform(*candidate_, op);
        }
    }

    void emit(size_t index) {
        if (index < analysis_end_) {
            known_.advance(hir(), hir().ops[index]);
        }
        if (candidate_) {
            output_.push_back(candidate_->ops[index]);
            if (!candidate_->source_map.empty()) {
                sources_.push_back(std::move(candidate_->source_map[index]));
            }
        }
    }

    const HirModule& input_;
    std::optional<HirModule> candidate_;
    uint32_t max_variables_;
    phase_detail::OrdinaryWork ordinary_work_;
    phase_detail::WideBudget wide_budget_;
    size_t analysis_end_;
    std::optional<size_t> final_observer_end_;
    bool elide_unreduced_post_observer_ = false;
    bool allow_expansion_ = true;
    CliffordAbsorption cliffords_;
    // Follow emitted operations: absorbed Cliffords live in cliffords_, so their
    // effects reach this analysis through the transformed subsequent operands.
    SymbolicStabilizers known_;
    std::vector<size_t> postselected_measure_indices_;
    size_t next_postselected_idx_ = 0;
    KnownStabilizers lookahead_fixed_;
    KnownStabilizers lookahead_postselected_;
    std::vector<HeisenbergOp> output_;
    std::vector<std::vector<uint32_t>> sources_;
};

}  // namespace

PhasePolynomialPass::PhasePolynomialPass(PhasePolynomialOptions options) : options_(options) {
    if (options.max_variables > 128) {
        throw std::invalid_argument("max_variables must be between zero and 128");
    }
}

void PhasePolynomialPass::run(HirModule& hir) {
    wide_attempts_ = wide_rewrites_ = wide_work_ = 0;
    wide_budget_exhausted_ = false;
    blocks_reduced_ = pauli_pullbacks_ = 0;
    blocks_examined_ = blocks_capped_ = expansion_attempts_ = blocks_expanded_ = 0;
    input_t_count_ = output_t_count_ = hir.num_t_gates();
    hir.postselection.validate_assumptions(hir.num_detectors);
    if (options_.max_variables == 0 || input_t_count_ == 0 ||
        !hir.logical_noise_prefix_matches_schedule()) {
        return;
    }
    if (!hir.source_map.empty() && hir.source_map.size() != hir.ops.size()) {
        throw std::invalid_argument("HIR source map size does not match the operation count");
    }
    Rewriter rewriter(hir, options_);
    auto candidate = rewriter.run();
    wide_attempts_ = rewriter.wide_attempts;
    wide_work_ = rewriter.wide_work();
    wide_budget_exhausted_ = rewriter.wide_budget_exhausted();
    blocks_examined_ = rewriter.blocks_examined;
    blocks_capped_ = rewriter.blocks_capped;
    expansion_attempts_ = rewriter.expansion_attempts;
    blocks_expanded_ = rewriter.blocks_expanded;
    if (!candidate) {
        return;
    }
    size_t follow_up_reduced = 0;
    size_t follow_up_pullbacks = 0;
    if (rewriter.wide_rewrites && candidate->num_t_gates() > 0) {
        // Removing a wide region can bring canceling rotations on its two sides together.
        Rewriter follow_up(*candidate, PhasePolynomialOptions{64, 0, 0},
                           /*elide_unreduced_post_observer=*/true);
        if (auto refined = follow_up.run()) {
            candidate = std::move(refined);
            follow_up_reduced = follow_up.blocks_reduced;
            follow_up_pullbacks = follow_up.pauli_pullbacks;
            blocks_examined_ += follow_up.blocks_examined;
            blocks_capped_ += follow_up.blocks_capped;
            expansion_attempts_ += follow_up.expansion_attempts;
            blocks_expanded_ += follow_up.blocks_expanded;
        }
    }
    const ActiveWidthCost after = summarize_active_width(*candidate);
    // This is a local structural guard, not a prediction of later scheduling
    // or hardware throughput.
    // Zero width has no dense work and is already optimal for both criteria.
    if (after.peak_width != 0) {
        const ActiveWidthCost before = summarize_active_width(hir);
        if (after.peak_width > before.peak_width ||
            (after.peak_width == before.peak_width && after.dense_work > before.dense_work)) {
            // A rejected optional candidate must not displace ordinary rewrites.
            if (rewriter.wide_rewrites) {
                auto narrow_options = options_;
                narrow_options.max_variables = 64;
                narrow_options.max_wide_work = 0;
                narrow_options.max_wide_terms = 0;
                PhasePolynomialPass narrow(narrow_options);
                narrow.run(hir);
                blocks_examined_ += narrow.blocks_examined();
                blocks_capped_ += narrow.blocks_capped();
                expansion_attempts_ += narrow.expansion_attempts();
                blocks_expanded_ += narrow.blocks_expanded();
                blocks_reduced_ = narrow.blocks_reduced();
                pauli_pullbacks_ = narrow.pauli_pullbacks();
                output_t_count_ = hir.num_t_gates();
            }
            return;
        }
    }
    wide_rewrites_ = rewriter.wide_rewrites;
    blocks_reduced_ = rewriter.blocks_reduced + follow_up_reduced;
    pauli_pullbacks_ = rewriter.pauli_pullbacks + follow_up_pullbacks;
    output_t_count_ = candidate->num_t_gates();
    if (rewriter.has_postselected_records()) {
        candidate->postselection.record_specialization(std::move(hir));
    }
    hir = std::move(*candidate);
}

}  // namespace clifft
