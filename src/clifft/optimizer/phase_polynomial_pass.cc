#include "clifft/optimizer/phase_polynomial_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/optimizer/phase_polynomial.h"
#include "clifft/optimizer/symbolic_stabilizers.h"
#include "clifft/optimizer/wide_phase_region.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {
namespace {

using optimizer_detail::body_bit;
using optimizer_detail::pivot_of;
using optimizer_detail::write_axis;

PauliString product_axis(const std::vector<PauliString>& generators, uint64_t parity,
                         uint32_t width) {
    PauliString axis(width);
    while (parity) {
        axis.right_multiply(generators[std::countr_zero(parity)].view());
        parity &= parity - 1;
    }
    assert(axis.is_hermitian());
    return axis;
}

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
            if (anti_commute(target.x(), target.z(), other.x(), other.z())) {
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
                if (anti_commute(target.x(), target.z(), other.x(), other.z())) {
                    return false;
                }
            }
        }
    }
    return true;
}

using optimizer_detail::CliffordAbsorption;

struct Observer {
    size_t index;
    PauliString pulled_axis;
};

struct Block {
    size_t end;
    std::vector<size_t> rotations;
    std::vector<PauliString> generators;
    phase_detail::Polynomial polynomial;
    std::vector<Observer> observers;
    KnownStabilizers constraints;
    phase_detail::Polynomial original_terms;
    bool capped = false;
};

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
                const auto initial_limit = std::min(max_variables_, uint32_t{32});
                auto block = collect(i, initial_limit, entry, postselected);
                if (block.capped && initial_limit < max_variables_ && allow_expansion_) {
                    ++expansion_attempts;
                    auto expanded =
                        collect(i, std::min(max_variables_, uint32_t{64}), entry, postselected);
                    if (!expanded.capped) {
                        block = std::move(expanded);
                        ++blocks_expanded;
                    } else {
                        // Keep bounded synthesis for an oversized region. A
                        // later split is not evidence that expanding will help.
                        allow_expansion_ = false;
                        if (max_variables_ > 64 && !wide_budget_.exhausted &&
                            wide_budget_.max_updates && wide_budget_.max_terms) {
                            ++wide_attempts;
                            auto wide = phase_detail::wide::try_region(
                                hir(), i, entry, cliffords_, max_variables_, wide_budget_);
                            if (wide) {
                                emit_wide(i, *wide);
                                ++blocks_examined;
                                ++blocks_expanded;
                                i = wide->end;
                                allow_expansion_ = true;
                                continue;
                            }
                        }
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
    bool wide_budget_exhausted() const { return wide_budget_.exhausted; }
    size_t blocks_reduced = 0;
    size_t pauli_pullbacks = 0;
    size_t blocks_examined = 0;
    size_t blocks_capped = 0;
    size_t expansion_attempts = 0;
    size_t blocks_expanded = 0;

    bool has_postselected_records() const { return !known_.postselected_records().empty(); }

  private:
    struct Row {
        PauliString body;
        uint64_t coordinates;
    };

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
        Block block{start, {}, {}, {}, {}, {}, {}};
        // Entry coordinates stay fixed while barriers remove relations that
        // cannot hold after moving those operations ahead of the phase block.
        auto available = entry;
        std::map<uint32_t, Row> rows;
        for (size_t i = start; i < hir().ops.size(); ++i) {
            const auto& op = hir().ops[i];
            const auto type = op.op_type();
            if (type == OpType::T_GATE) {
                auto axis = cliffords_.read(hir().mask_view(op), hir().num_qubits);
                auto reduced = entry.reduce_body(axis);
                // An entry stabilizer commutes with every constraint derived
                // from that same group, without a second basis scan.
                const bool entry_stabilizer = reduced.x().is_zero() && reduced.z().is_zero();
                if ((postselected && !postselected->commutes(axis.view())) ||
                    std::ranges::any_of(block.generators,
                                        [&](const PauliString& generator) {
                                            return !axis.view().commutes(generator.view());
                                        }) ||
                    (!entry_stabilizer && !block.constraints.commutes(axis.view()))) {
                    break;
                }
                const int coefficient = op.is_dagger() != axis.sign() ? -1 : 1;
                uint64_t coordinates = 0;
                for (const auto& [pivot, row] : rows) {
                    if (body_bit(reduced, pivot)) {
                        reduced.mut_x().xor_with(row.body.x());
                        reduced.mut_z().xor_with(row.body.z());
                        coordinates ^= row.coordinates;
                    }
                }
                if (!reduced.x().is_zero() || !reduced.z().is_zero()) {
                    if (block.generators.size() == variable_limit) {
                        block.capped = true;
                        break;
                    }
                    const uint64_t bit = uint64_t{1} << block.generators.size();
                    const uint32_t pivot = pivot_of(reduced);
                    rows.emplace(pivot, Row{std::move(reduced), coordinates ^ bit});
                    block.generators.push_back(axis);
                    block.generators.back().set_sign(false);
                    coordinates = bit;
                }
                auto residual = product_axis(block.generators, coordinates, hir().num_qubits);
                auto unsigned_axis = axis;
                unsigned_axis.set_sign(false);
                residual.right_multiply(unsigned_axis.view());
                const auto negative = available.eigenvalue(residual);
                if (!negative) {
                    break;
                }
                if (!residual.x().is_zero() || !residual.z().is_zero()) {
                    residual.set_sign(residual.sign() ^ *negative);
                    block.constraints.insert(std::move(residual));
                }
                phase_detail::add_parity(block.polynomial, coordinates,
                                         *negative ? -coefficient : coefficient);
                auto& term = block.original_terms[coordinates];
                term = static_cast<uint8_t>((term + (*negative ? -coefficient : coefficient)) & 7);
                block.rotations.push_back(i);
            } else if (type == OpType::NOISE) {
                // Moving this site before the phase prefix must preserve the
                // relations already used. Later relations are checked against
                // the entry constraints after applying this channel.
                const auto& channels =
                    hir().noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
                for (const auto& channel : channels) {
                    auto axis = cliffords_.read(hir().noise_channel_masks.at(channel.mask),
                                                hir().num_qubits);
                    if (channel.prob > 0 &&
                        (!block.constraints.commutes(axis.view()) ||
                         std::ranges::any_of(block.generators, [&](const PauliString& generator) {
                             return !axis.view().commutes(generator.view());
                         }))) {
                        return block;
                    }
                    if (channel.prob > 0) {
                        available.intersect(axis.view());
                    }
                }
            } else if (type == OpType::MEASURE || type == OpType::CONDITIONAL_PAULI ||
                       type == OpType::EXP_VAL) {
                const auto axis = cliffords_.read(hir().mask_view(op), hir().num_qubits);
                if (!block.constraints.commutes(axis.view())) {
                    break;
                }
                uint64_t flip = 0;
                for (size_t j = 0; j < block.generators.size(); ++j) {
                    if (!axis.view().commutes(block.generators[j].view())) {
                        flip |= uint64_t{1} << j;
                    }
                }
                const auto difference = phase_detail::pauli_derivative(block.polynomial, flip);
                if (!difference) {
                    break;
                }
                auto pulled = axis;
                const auto correction =
                    product_axis(block.generators, difference->parity, hir().num_qubits);
                pulled.right_multiply(correction.view());
                // U^dag M U = M * omega^(-difference). The scalar also repairs
                // the imaginary phase when M anticommutes with its correction.
                pulled.add_phase(static_cast<uint8_t>((-int(difference->constant) / 2) & 3));
                assert(pulled.is_hermitian());
                block.observers.push_back({i, std::move(pulled)});
                available.intersect(axis.view());
            } else if (type != OpType::DETECTOR && type != OpType::OBSERVABLE &&
                       type != OpType::READOUT_NOISE) {
                break;
            }
            block.end = i + 1;
        }
        return block;
    }

    bool rewrite(size_t start, Block& block) {
        const auto basis = phase_detail::reduce_core(
            block.polynomial, static_cast<uint32_t>(block.generators.size()));
        const auto synthesis =
            phase_detail::synthesize_parities(block.original_terms, block.polynomial, basis);
        const size_t synthesized_count = static_cast<size_t>(std::ranges::count_if(
            synthesis, [](const auto& term) { return (term.second & 1) != 0; }));
        const bool reduced_polynomial = synthesized_count < block.rotations.size() ||
                                        (synthesized_count == block.rotations.size() &&
                                         basis.core_width < block.generators.size());
        const bool elide_post_observer = (reduced_polynomial || elide_unreduced_post_observer_) &&
                                         final_observer_end_ && block.end >= *final_observer_end_ &&
                                         !block.observers.empty();
        if (!reduced_polynomial && !elide_post_observer) {
            return false;
        }
        start_candidate(start);
        std::vector<PauliString> generators;
        for (uint64_t parity : basis.parities) {
            generators.push_back(product_axis(block.generators, parity, hir().num_qubits));
        }
        for (const auto& observer : block.observers) {
            write_axis(candidate_->mask_at(candidate_->ops[observer.index]), observer.pulled_axis);
        }
        for (size_t i = start; i < block.end; ++i) {
            if (hir().ops[i].op_type() != OpType::T_GATE) {
                if (hir().ops[i].op_type() == OpType::NOISE) {
                    transform(hir().ops[i]);
                }
                emit(i);
            }
        }
        std::vector<uint32_t> sources;
        if (!hir().source_map.empty() && !elide_post_observer) {
            for (size_t index : block.rotations) {
                const auto& original = hir().source_map[index];
                sources.insert(sources.end(), original.begin(), original.end());
            }
            std::ranges::sort(sources);
            sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        }
        size_t written = 0;
        for (const auto& [parity, coefficient] : synthesis) {
            const auto axis = product_axis(generators, parity, hir().num_qubits);
            int clifford = coefficient;
            if (coefficient & 1) {
                const bool dagger = coefficient >= 5;
                if (!elide_post_observer) {
                    auto& op = candidate_->ops[block.rotations[written++]];
                    candidate_->demote_to_tgate(op, dagger);
                    write_axis(candidate_->mask_at(op), axis);
                    output_.push_back(op);
                    known_.advance(hir(), op);
                    if (!hir().source_map.empty()) {
                        sources_.push_back(sources);
                    }
                }
                clifford = (coefficient - (dagger ? -1 : 1)) & 7;
            }
            if (clifford) {
                cliffords_.absorb(axis, static_cast<uint8_t>(clifford));
            }
        }
        assert(written == (elide_post_observer ? 0 : synthesized_count));
        static_cast<void>(written);
        ++blocks_reduced;
        pauli_pullbacks += block.observers.size();
        return true;
    }

    void emit_wide(size_t start, const phase_detail::wide::Rewrite& rewrite) {
        start_candidate(start);
        std::vector<uint32_t> sources;
        if (!hir().source_map.empty()) {
            for (size_t i = start; i < rewrite.end; ++i) {
                sources.insert(sources.end(), hir().source_map[i].begin(),
                               hir().source_map[i].end());
            }
            std::ranges::sort(sources);
            sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        }
        size_t written = 0;
        for (const auto& [axis, coefficient] : rewrite.phases) {
            int correction = coefficient;
            if (coefficient & 1) {
                const bool dagger = coefficient >= 5;
                auto& op = candidate_->ops[start + written++];
                candidate_->demote_to_tgate(op, dagger);
                write_axis(candidate_->mask_at(op), axis);
                output_.push_back(op);
                known_.advance(hir(), op);
                if (!hir().source_map.empty())
                    sources_.push_back(sources);
                correction = (coefficient - (dagger ? -1 : 1)) & 7;
            }
            if (correction)
                cliffords_.absorb(axis, static_cast<uint8_t>(correction));
        }
        assert(written == rewrite.t_count);
        ++blocks_reduced;
        ++wide_rewrites;
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
    phase_detail::wide::Budget wide_budget_;
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
    const auto after = analyze_active_width(*candidate);
    // This is a local structural guard, not a prediction of later scheduling
    // or hardware throughput.
    // Zero width has no dense work and is already optimal for both criteria.
    if (after.peak_width != 0) {
        const auto before = analyze_active_width(hir);
        if (after.peak_width > before.peak_width ||
            (after.peak_width == before.peak_width &&
             estimate_dense_work(after) > estimate_dense_work(before))) {
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
