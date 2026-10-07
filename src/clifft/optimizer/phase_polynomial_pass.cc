#include "clifft/optimizer/phase_polynomial_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/optimizer/phase_polynomial.h"
#include "clifft/optimizer/symbolic_stabilizers.h"

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

using optimizer_detail::write_axis;

uint32_t pivot_of(const PauliString& axis) {
    const uint32_t domain = axis.x().num_words() * 64;
    const uint32_t x = axis.x().lowest_bit();
    return x < domain ? x : domain + axis.z().lowest_bit();
}

bool body_bit(const PauliString& axis, uint32_t pivot) {
    const uint32_t domain = axis.x().num_words() * 64;
    return pivot < domain ? axis.x().bit_get(pivot) : axis.z().bit_get(pivot - domain);
}

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
    Rewriter(const HirModule& input, uint32_t max_variables)
        : input_(input),
          max_variables_(max_variables),
          analysis_end_(last_t_end(input)),
          known_(input, analysis_end_) {}

    std::optional<HirModule> run() {
        for (size_t i = 0; i < hir().ops.size();) {
            if (hir().ops[i].op_type() == OpType::T_GATE) {
                const auto initial_limit = std::min(max_variables_, uint32_t{32});
                auto block = collect(i, initial_limit);
                if (block.capped && initial_limit < max_variables_ && allow_expansion_) {
                    ++expansion_attempts;
                    auto expanded = collect(i, max_variables_);
                    if (!expanded.capped) {
                        block = std::move(expanded);
                        ++blocks_expanded;
                    } else {
                        // Keep bounded synthesis for an oversized region. A
                        // later split is not evidence that expanding will help.
                        allow_expansion_ = false;
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

    size_t blocks_reduced = 0;
    size_t pauli_pullbacks = 0;
    size_t blocks_examined = 0;
    size_t blocks_capped = 0;
    size_t expansion_attempts = 0;
    size_t blocks_expanded = 0;

  private:
    struct Row {
        PauliString body;
        uint64_t coordinates;
    };

    const HirModule& hir() const { return candidate_ ? *candidate_ : input_; }

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

    Block collect(size_t start, uint32_t variable_limit) {
        Block block{start, {}, {}, {}, {}, {}, {}};
        // Entry coordinates stay fixed while barriers remove relations that
        // cannot hold after moving those operations ahead of the phase block.
        auto available = known_.fixed_constraints();
        std::map<uint32_t, Row> rows;
        for (size_t i = start; i < hir().ops.size(); ++i) {
            const auto& op = hir().ops[i];
            const auto type = op.op_type();
            if (type == OpType::T_GATE) {
                auto axis = cliffords_.read(hir().mask_view(op), hir().num_qubits);
                auto reduced = known_.fixed_constraints().reduce_body(axis);
                // An entry stabilizer commutes with every constraint derived
                // from that same group, without a second basis scan.
                const bool entry_stabilizer = reduced.x().is_zero() && reduced.z().is_zero();
                if (std::ranges::any_of(block.generators,
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
                // the entry knowledge after applying this channel.
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
        const size_t output_count = std::ranges::count_if(
            synthesis, [](const auto& term) { return (term.second & 1) != 0; });
        if (output_count > block.rotations.size() ||
            (output_count == block.rotations.size() &&
             basis.core_width == block.generators.size())) {
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
        if (!hir().source_map.empty()) {
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
                auto& op = candidate_->ops[block.rotations[written++]];
                candidate_->demote_to_tgate(op, dagger);
                write_axis(candidate_->mask_at(op), axis);
                output_.push_back(op);
                known_.advance(hir(), op);
                if (!hir().source_map.empty()) {
                    sources_.push_back(sources);
                }
                clifford = (coefficient - (dagger ? -1 : 1)) & 7;
            }
            if (clifford) {
                cliffords_.absorb(axis, static_cast<uint8_t>(clifford));
            }
        }
        assert(written == output_count);
        ++blocks_reduced;
        pauli_pullbacks += block.observers.size();
        return true;
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
    size_t analysis_end_;
    bool allow_expansion_ = true;
    CliffordAbsorption cliffords_;
    // Follow emitted operations: absorbed Cliffords live in cliffords_, so their
    // effects reach this analysis through the transformed subsequent operands.
    SymbolicStabilizers known_;
    std::vector<HeisenbergOp> output_;
    std::vector<std::vector<uint32_t>> sources_;
};

}  // namespace

PhasePolynomialPass::PhasePolynomialPass(PhasePolynomialOptions options) : options_(options) {
    if (options.max_variables > 64) {
        throw std::invalid_argument("max_variables must be between zero and 64");
    }
}

void PhasePolynomialPass::run(HirModule& hir) {
    blocks_reduced_ = pauli_pullbacks_ = 0;
    blocks_examined_ = blocks_capped_ = expansion_attempts_ = blocks_expanded_ = 0;
    input_t_count_ = output_t_count_ = hir.num_t_gates();
    if (options_.max_variables == 0 || input_t_count_ == 0 ||
        !hir.logical_noise_prefix_matches_schedule()) {
        return;
    }
    if (!hir.source_map.empty() && hir.source_map.size() != hir.ops.size()) {
        throw std::invalid_argument("HIR source map size does not match the operation count");
    }
    Rewriter rewriter(hir, options_.max_variables);
    auto candidate = rewriter.run();
    blocks_examined_ = rewriter.blocks_examined;
    blocks_capped_ = rewriter.blocks_capped;
    expansion_attempts_ = rewriter.expansion_attempts;
    blocks_expanded_ = rewriter.blocks_expanded;
    if (!candidate) {
        return;
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
            return;
        }
    }
    blocks_reduced_ = rewriter.blocks_reduced;
    pauli_pullbacks_ = rewriter.pauli_pullbacks;
    output_t_count_ = candidate->num_t_gates();
    hir = std::move(*candidate);
}

}  // namespace clifft
