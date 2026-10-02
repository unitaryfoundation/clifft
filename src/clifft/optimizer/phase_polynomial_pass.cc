#include "clifft/optimizer/phase_polynomial_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/known_stabilizers.h"
#include "clifft/optimizer/phase_polynomial.h"

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

PauliString copy_axis(PauliMaskView mask, uint32_t width) {
    PauliString axis(width);
    axis.mut_x().xor_with(mask.x());
    axis.mut_z().xor_with(mask.z());
    axis.set_sign(mask.sign());
    return axis;
}

void write_axis(MutablePauliMaskView mask, const PauliString& axis) {
    std::ranges::copy(axis.x().words, mask.x().words.begin());
    std::ranges::copy(axis.z().words, mask.z().words.begin());
    mask.set_sign(axis.sign());
}

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

// Removed Clifford factors form C. Read each subsequent input axis as C^dag P C,
// and compose C into the final tableau once. Updating both directions avoids
// rescanning the circuit suffix or repeatedly inverting a dense tableau.
class CliffordFrame {
  public:
    PauliString read(PauliMaskView mask, uint32_t width) const {
        auto axis = copy_axis(mask, width);
        return inverse_ ? inverse_->apply(axis.view()) : axis;
    }

    void absorb(const PauliString& axis, uint8_t coefficient) {
        if (!forward_) {
            forward_.emplace(axis.num_qubits());
            inverse_.emplace(axis.num_qubits());
        }
        const auto original_axis = forward_->apply(axis.view());
        if (coefficient == 4) {
            forward_->prepend_pauli(axis.view());
            inverse_->prepend_pauli(original_axis.view());
        } else {
            assert(coefficient == 2 || coefficient == 6);
            forward_->prepend_pauli_rotation(axis.view(), coefficient == 6);
            inverse_->prepend_pauli_rotation(original_axis.view(), coefficient == 2);
        }
    }

    void finish(HirModule& hir) const {
        if (forward_ && hir.final_tableau) {
            hir.final_tableau = forward_->then(*hir.final_tableau);
        }
    }

  private:
    std::optional<Tableau> forward_;
    std::optional<Tableau> inverse_;
};

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
};

class Rewriter {
  public:
    Rewriter(HirModule& hir, PhasePolynomialOptions options)
        : hir_(hir), max_variables_(options.max_variables) {
        if (options.use_known_stabilizers) {
            known_.emplace(hir.num_qubits);
        }
        output_.reserve(hir.ops.size());
        if (!hir.source_map.empty()) {
            sources_.reserve(hir.ops.size());
        }
    }

    void run() {
        for (size_t i = 0; i < hir_.ops.size();) {
            if (hir_.ops[i].op_type() == OpType::T_GATE) {
                auto block = collect(i);
                assert(block.end > i);
                if (!rewrite(i, block)) {
                    for (size_t j = i; j < block.end; ++j) {
                        emit(j);
                    }
                }
                i = block.end;
            } else {
                transform(hir_.ops[i]);
                emit(i++);
            }
        }
        hir_.ops = std::move(output_);
        hir_.source_map = std::move(sources_);
        frame_.finish(hir_);
    }

    size_t blocks_reduced = 0;
    size_t pauli_pullbacks = 0;

  private:
    struct Row {
        PauliString body;
        uint64_t coordinates;
    };

    Block collect(size_t start) {
        Block block{start, {}, {}, {}, {}, {}};
        // Entry coordinates stay fixed while barriers remove relations that
        // cannot hold after moving those operations ahead of the phase block.
        auto available = known_;
        std::map<uint32_t, Row> rows;
        for (size_t i = start; i < hir_.ops.size(); ++i) {
            const auto& op = hir_.ops[i];
            const auto type = op.op_type();
            if (type == OpType::T_GATE) {
                auto axis = frame_.read(hir_.mask_view(op), hir_.num_qubits);
                auto reduced = known_ ? known_->reduce_body(axis) : axis;
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
                    if (block.generators.size() == max_variables_) {
                        break;
                    }
                    const uint64_t bit = uint64_t{1} << block.generators.size();
                    const uint32_t pivot = pivot_of(reduced);
                    rows.emplace(pivot, Row{std::move(reduced), coordinates ^ bit});
                    block.generators.push_back(axis);
                    block.generators.back().set_sign(false);
                    coordinates = bit;
                }
                auto residual = product_axis(block.generators, coordinates, hir_.num_qubits);
                auto unsigned_axis = axis;
                unsigned_axis.set_sign(false);
                residual.right_multiply(unsigned_axis.view());
                assert(available || (residual.x().is_zero() && residual.z().is_zero()));
                const auto negative = available ? available->eigenvalue(residual)
                                                : std::optional<bool>{residual.sign()};
                if (!negative) {
                    break;
                }
                if (!residual.x().is_zero() || !residual.z().is_zero()) {
                    assert(known_);
                    residual.set_sign(residual.sign() ^ *negative);
                    block.constraints.insert(std::move(residual));
                }
                phase_detail::add_parity(block.polynomial, coordinates,
                                         *negative ? -coefficient : coefficient);
                write_axis(hir_.mask_at(op), axis);
                block.rotations.push_back(i);
            } else if (type == OpType::NOISE) {
                // Moving this site before the phase prefix must preserve the
                // relations already used. Later relations are checked against
                // the entry knowledge after applying this channel.
                const auto& channels =
                    hir_.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
                std::vector<PauliString> axes;
                for (const auto& channel : channels) {
                    auto axis =
                        frame_.read(hir_.noise_channel_masks.at(channel.mask), hir_.num_qubits);
                    if (channel.prob > 0 &&
                        (!block.constraints.commutes(axis.view()) ||
                         std::ranges::any_of(block.generators, [&](const PauliString& generator) {
                             return !axis.view().commutes(generator.view());
                         }))) {
                        return block;
                    }
                    axis.set_sign(false);
                    axes.push_back(std::move(axis));
                }
                for (size_t j = 0; j < channels.size(); ++j) {
                    write_axis(hir_.noise_channel_masks.mut_at(channels[j].mask), axes[j]);
                }
                if (available) {
                    available->advance(hir_, op);
                }
            } else if (type == OpType::MEASURE || type == OpType::CONDITIONAL_PAULI ||
                       type == OpType::EXP_VAL) {
                const auto axis = frame_.read(hir_.mask_view(op), hir_.num_qubits);
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
                    product_axis(block.generators, difference->parity, hir_.num_qubits);
                pulled.right_multiply(correction.view());
                // U^dag M U = M * omega^(-difference). The scalar also repairs
                // the imaginary phase when M anticommutes with its correction.
                pulled.add_phase(static_cast<uint8_t>((-int(difference->constant) / 2) & 3));
                assert(pulled.is_hermitian());
                block.observers.push_back({i, std::move(pulled)});
                if (available) {
                    available->intersect(axis.view());
                }
                write_axis(hir_.mask_at(op), axis);
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
        const auto synthesis = phase_detail::synthesize_parities(block.polynomial);
        const size_t output_count = std::ranges::count_if(
            synthesis, [](const auto& term) { return (term.second & 1) != 0; });
        if (output_count > block.rotations.size() ||
            (output_count == block.rotations.size() &&
             basis.core_width == block.generators.size())) {
            return false;
        }
        std::vector<PauliString> generators;
        for (uint64_t parity : basis.parities) {
            generators.push_back(product_axis(block.generators, parity, hir_.num_qubits));
        }
        for (const auto& observer : block.observers) {
            write_axis(hir_.mask_at(hir_.ops[observer.index]), observer.pulled_axis);
        }
        for (size_t i = start; i < block.end; ++i) {
            if (hir_.ops[i].op_type() != OpType::T_GATE) {
                emit(i);
            }
        }
        std::vector<uint32_t> sources;
        if (!hir_.source_map.empty()) {
            for (size_t index : block.rotations) {
                const auto& original = hir_.source_map[index];
                sources.insert(sources.end(), original.begin(), original.end());
            }
            std::ranges::sort(sources);
            sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        }
        size_t written = 0;
        for (const auto& [parity, coefficient] : synthesis) {
            const auto axis = product_axis(generators, parity, hir_.num_qubits);
            int clifford = coefficient;
            if (coefficient & 1) {
                const bool dagger = coefficient >= 5;
                auto& op = hir_.ops[block.rotations[written++]];
                hir_.demote_to_tgate(op, dagger);
                write_axis(hir_.mask_at(op), axis);
                output_.push_back(op);
                if (known_) {
                    known_->advance(hir_, op);
                }
                if (!hir_.source_map.empty()) {
                    sources_.push_back(sources);
                }
                clifford = (coefficient - (dagger ? -1 : 1)) & 7;
            }
            if (clifford) {
                frame_.absorb(axis, static_cast<uint8_t>(clifford));
            }
        }
        assert(written == output_count);
        ++blocks_reduced;
        pauli_pullbacks += block.observers.size();
        return true;
    }

    void transform(const HeisenbergOp& op) {
        const auto transform_mask = [&](MutablePauliMaskView mask) {
            write_axis(mask, frame_.read(mask, hir_.num_qubits));
        };
        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
            case OpType::MEASURE:
            case OpType::CONDITIONAL_PAULI:
            case OpType::EXP_VAL:
                transform_mask(hir_.mask_at(op));
                break;

            case OpType::INSTRUMENT: {
                transform_mask(hir_.mask_at(op));
                const auto& site =
                    hir_.instrument_sites[static_cast<uint32_t>(op.instrument_site_idx())];
                transform_mask(hir_.pauli_masks.mut_at(site.destination_flip_mask));
                break;
            }

            case OpType::NOISE:
                for (const auto& channel :
                     hir_.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels) {
                    auto mask = hir_.noise_channel_masks.mut_at(channel.mask);
                    transform_mask(mask);
                    // A channel is unchanged by replacing its Pauli with -P.
                    mask.set_sign(false);
                }
                break;

            case OpType::READOUT_NOISE:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
            case OpType::NUM_OP_TYPES:
                break;
        }
    }

    void emit(size_t index) {
        output_.push_back(hir_.ops[index]);
        if (known_) {
            known_->advance(hir_, hir_.ops[index]);
        }
        if (!hir_.source_map.empty()) {
            sources_.push_back(hir_.source_map[index]);
        }
    }

    HirModule& hir_;
    uint32_t max_variables_;
    CliffordFrame frame_;
    // Follow emitted operations: absorbed Cliffords live in frame_, so their
    // effects reach this analysis through the transformed subsequent operands.
    std::optional<KnownStabilizers> known_;
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
    input_t_count_ = output_t_count_ = hir.num_t_gates();
    if (options_.max_variables == 0 || input_t_count_ == 0 ||
        !hir.logical_noise_prefix_matches_schedule()) {
        return;
    }
    if (!hir.source_map.empty() && hir.source_map.size() != hir.ops.size()) {
        throw std::invalid_argument("HIR source map size does not match the operation count");
    }
    HirModule candidate = hir;
    candidate.logical_noise_prefix.clear();
    Rewriter rewriter(candidate, options_);
    rewriter.run();
    if (!rewriter.blocks_reduced) {
        return;
    }
    const auto after = analyze_active_width(candidate);
    // This is a local structural guard, not a prediction of later scheduling
    // or hardware throughput. The pass remains explicitly opt in.
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
    output_t_count_ = candidate.num_t_gates();
    hir = std::move(candidate);
}

}  // namespace clifft
