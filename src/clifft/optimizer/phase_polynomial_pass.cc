#include "clifft/optimizer/phase_polynomial_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/tableau/pauli_string.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {
namespace {

PauliString copy_mask(PauliMaskView mask, uint32_t num_qubits) {
    PauliString out(num_qubits);
    out.mut_x().xor_with(mask.x());
    out.mut_z().xor_with(mask.z());
    out.set_sign(mask.sign());
    return out;
}

uint32_t pivot_of(const PauliString& p) {
    const uint32_t domain = p.x().num_words() * 64;
    const uint32_t x = p.x().lowest_bit();
    return x < domain ? x : domain + p.z().lowest_bit();
}

bool body_bit(const PauliString& p, uint32_t pivot) {
    const uint32_t domain = p.x().num_words() * 64;
    return pivot < domain ? p.x().bit_get(pivot) : p.z().bit_get(pivot - domain);
}

bool identity_body(const PauliString& p) {
    return p.x().is_zero() && p.z().is_zero();
}

// Unlike dormant-width analysis, these rows have a fixed positive eigenvalue
// on every possible trajectory at this point. Unknown signs cannot justify a
// fixed Clifford decoder, even when their unsigned bodies remain stabilizers.
class KnownStabilizers {
  public:
    explicit KnownStabilizers(uint32_t num_qubits) {
        rows_.reserve(num_qubits);
        for (uint32_t q = 0; q < num_qubits; ++q) {
            PauliString z(num_qubits);
            z.set_pauli(q, false, true);
            rows_.push_back({pivot_of(z), std::move(z)});
        }
    }

    PauliString reduce_body(PauliString p) const {
        for (const auto& row : rows_) {
            if (body_bit(p, row.pivot)) {
                p.mut_x().xor_with(row.axis.x());
                p.mut_z().xor_with(row.axis.z());
            }
        }
        return p;
    }

    std::optional<bool> eigenvalue(PauliString p) const {
        for (const auto& row : rows_) {
            if (body_bit(p, row.pivot)) {
                // This query is used only after unsigned membership is proved.
                assert(p.view().commutes(row.axis.view()));
                p.right_multiply(row.axis.view());
            }
        }
        if (!identity_body(p)) {
            return std::nullopt;
        }
        assert(p.is_hermitian());
        return p.sign();
    }

    void intersect(PauliStringView p) {
        std::optional<size_t> pivot;
        for (size_t i = 0; i < rows_.size(); ++i) {
            if (!rows_[i].axis.view().commutes(p)) {
                pivot = i;
            }
        }
        if (!pivot) {
            return;
        }
        // The largest pivot keeps every other row's leading bit unchanged.
        for (size_t i = 0; i < *pivot; ++i) {
            if (!rows_[i].axis.view().commutes(p)) {
                rows_[i].axis.right_multiply(rows_[*pivot].axis.view());
            }
        }
        rows_.erase(rows_.begin() + static_cast<ptrdiff_t>(*pivot));
    }

    void conjugate_pauli(PauliStringView p) {
        for (auto& row : rows_) {
            if (!row.axis.view().commutes(p)) {
                row.axis.negate();
            }
        }
    }

    void clear() { rows_.clear(); }

  private:
    struct Row {
        uint32_t pivot;
        PauliString axis;
    };
    std::vector<Row> rows_;
};

void advance_knowledge(const HirModule& hir, const HeisenbergOp& op, KnownStabilizers& known) {
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI: {
            const auto axis = copy_mask(hir.mask_view(op), hir.num_qubits);
            // An unconditioned measurement keeps only the relations common to
            // both outcomes. No future postselection enters this proof.
            known.intersect(axis.view());
            break;
        }
        case OpType::NOISE: {
            const auto& site = hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())];
            const NoiseChannel* deterministic = nullptr;
            size_t nonzero = 0;
            for (const auto& channel : site.channels) {
                if (channel.prob > 0) {
                    ++nonzero;
                    if (channel.prob == 1) {
                        deterministic = &channel;
                    }
                }
            }
            if (nonzero == 1 && deterministic != nullptr) {
                const auto axis =
                    copy_mask(hir.noise_channel_masks.at(deterministic->mask), hir.num_qubits);
                known.conjugate_pauli(axis.view());
            } else {
                for (const auto& channel : site.channels) {
                    if (channel.prob > 0) {
                        const auto axis =
                            copy_mask(hir.noise_channel_masks.at(channel.mask), hir.num_qubits);
                        known.intersect(axis.view());
                    }
                }
            }
            break;
        }
        case OpType::INSTRUMENT:
            known.clear();
            break;
        case OpType::EXP_VAL:
        case OpType::READOUT_NOISE:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::NUM_OP_TYPES:
            break;
    }
}

using Polynomial = std::map<uint64_t, uint8_t>;

void add(Polynomial& p, uint64_t monomial, int coefficient) {
    const auto found = p.find(monomial);
    const uint8_t value =
        static_cast<uint8_t>((coefficient + (found == p.end() ? 0 : found->second)) & 7);
    if (value) {
        p[monomial] = value;
    } else if (found != p.end()) {
        p.erase(found);
    }
}

void add_parity(Polynomial& p, uint64_t parity, int coefficient) {
    // XOR has coefficients 1, -2, 4. Higher degrees vanish modulo eight.
    for (uint64_t bits = parity; bits; bits &= bits - 1) {
        const uint64_t a = bits & -bits;
        add(p, a, coefficient);
        for (uint64_t pairs = bits & (bits - 1); pairs; pairs &= pairs - 1) {
            const uint64_t b = pairs & -pairs;
            add(p, a | b, -2 * coefficient);
            for (uint64_t triples = pairs & (pairs - 1); triples; triples &= triples - 1) {
                add(p, a | b | (triples & -triples), 4 * coefficient);
            }
        }
    }
}

void insert_binary_row(std::map<uint32_t, uint64_t>& basis, uint64_t row) {
    while (row) {
        const uint32_t pivot = 63 - std::countl_zero(row);
        auto found = basis.find(pivot);
        if (found == basis.end()) {
            basis.emplace(pivot, row);
            return;
        }
        row ^= found->second;
    }
}

std::vector<uint64_t> clifford_kernel(const Polynomial& p, uint32_t width) {
    std::map<uint64_t, uint64_t> constraints;
    for (uint32_t j = 0; j < width; ++j) {
        const uint64_t bit = uint64_t{1} << j;
        Polynomial derivative;
        for (const auto& [mask, c] : p) {
            if (mask & bit) {
                add(derivative, mask ^ bit, c);
                add(derivative, mask, -2 * c);
            }
        }
        for (const auto& [mask, c] : derivative) {
            const auto degree = std::popcount(mask);
            const bool non_pauli = degree == 0   ? (c & 1) != 0
                                   : degree == 1 ? ((c / 2) & 1) != 0
                                                 : ((c / 4) & 1) != 0;
            assert(degree <= 2);
            if (non_pauli) {
                constraints[mask] |= bit;
            }
        }
    }
    std::map<uint32_t, uint64_t> pivots;
    for (const auto& [mask, row] : constraints) {
        (void)mask;
        insert_binary_row(pivots, row);
    }
    std::vector<uint64_t> kernel;
    for (uint32_t free = 0; free < width; ++free) {
        if (pivots.contains(free)) {
            continue;
        }
        uint64_t vector = uint64_t{1} << free;
        for (const auto& [pivot, row] : pivots) {
            if (std::popcount(row & vector) & 1) {
                vector |= uint64_t{1} << pivot;
            }
        }
        kernel.push_back(vector);
    }
    return kernel;
}

void substitute_cx(Polynomial& p, uint64_t control, uint64_t target) {
    Polynomial out;
    for (const auto& [mask, c] : p) {
        add(out, mask, c);
        if (mask & target) {
            const uint64_t rest = mask ^ target;
            add(out, rest | control, c);
            add(out, rest | control | target, -2 * c);
        }
    }
    p = std::move(out);
}

uint32_t put_core_first(Polynomial& p, std::vector<PauliString>& generators) {
    const uint32_t width = static_cast<uint32_t>(generators.size());
    const auto kernel = clifford_kernel(p, width);
    std::map<uint32_t, uint64_t> span;
    for (uint64_t vector : kernel) {
        insert_binary_row(span, vector);
    }
    std::vector<uint64_t> columns;
    for (uint32_t j = 0; j < width; ++j) {
        const size_t before = span.size();
        insert_binary_row(span, uint64_t{1} << j);
        if (span.size() != before) {
            columns.push_back(uint64_t{1} << j);
        }
    }
    const uint32_t core_width = static_cast<uint32_t>(columns.size());
    columns.insert(columns.end(), kernel.begin(), kernel.end());
    std::vector<uint64_t> matrix(width, 0);
    for (uint32_t j = 0; j < width; ++j) {
        for (uint32_t i = 0; i < width; ++i) {
            if (columns[j] & (uint64_t{1} << i)) {
                matrix[i] |= uint64_t{1} << j;
            }
        }
    }
    for (uint32_t j = 0; j < width; ++j) {
        uint32_t pivot = j;
        while (!(matrix[pivot] & (uint64_t{1} << j))) {
            ++pivot;
            assert(pivot < width);
        }
        if (pivot != j) {
            std::swap(matrix[j], matrix[pivot]);
            std::swap(generators[j], generators[pivot]);
            Polynomial swapped;
            const uint64_t left = uint64_t{1} << j;
            const uint64_t right = uint64_t{1} << pivot;
            for (const auto& [mask, c] : p) {
                const bool different = bool(mask & left) != bool(mask & right);
                add(swapped, different ? mask ^ (left | right) : mask, c);
            }
            p = std::move(swapped);
        }
        for (uint32_t i = 0; i < width; ++i) {
            if (i != j && (matrix[i] & (uint64_t{1} << j))) {
                matrix[i] ^= matrix[j];
                substitute_cx(p, uint64_t{1} << j, uint64_t{1} << i);
                // The dual eigenbit change is a product of commuting axes.
                generators[i].right_multiply(generators[j].view());
            }
        }
    }
    return core_width;
}

Polynomial parity_synthesis(const Polynomial& p) {
    Polynomial out;
    for (const auto& [mask, c] : p) {
        const int degree = std::popcount(mask);
        if (degree == 0) {
            continue;
        }
        if (degree == 1) {
            add(out, mask, c);
        } else if (degree == 2) {
            assert(c % 2 == 0);
            const uint64_t a = mask & -mask;
            const uint64_t b = mask ^ a;
            add(out, a, c / 2);
            add(out, b, c / 2);
            add(out, mask, -c / 2);
        } else {
            assert(degree == 3 && c == 4);
            for (uint64_t subset = mask; subset; subset = (subset - 1) & mask) {
                add(out, subset, (std::popcount(subset) & 1) ? 1 : -1);
            }
        }
    }
    return out;
}

PauliString product_axis(const std::vector<PauliString>& generators, uint64_t label,
                         uint32_t num_qubits) {
    PauliString out(num_qubits);
    while (label) {
        const uint32_t bit = std::countr_zero(label);
        out.right_multiply(generators[bit].view());
        label &= label - 1;
    }
    assert(out.is_hermitian());
    return out;
}

Polynomial derivative(Polynomial p, uint64_t flip) {
    const auto original = p;
    while (flip) {
        const uint64_t bit = flip & -flip;
        Polynomial out;
        for (const auto& [mask, c] : p) {
            if (mask & bit) {
                add(out, mask ^ bit, c);
                add(out, mask, -c);
            } else {
                add(out, mask, c);
            }
        }
        p = std::move(out);
        flip &= flip - 1;
    }
    for (const auto& [mask, c] : original) {
        add(p, mask, -c);
    }
    return p;
}

std::optional<PauliString> pullback(const PauliString& observable, const Polynomial& prefix,
                                    const std::vector<PauliString>& generators) {
    uint64_t flip = 0;
    for (size_t j = 0; j < generators.size(); ++j) {
        if (!observable.view().commutes(generators[j].view())) {
            flip |= uint64_t{1} << j;
        }
    }
    const auto difference = derivative(prefix, flip);
    int constant = 0;
    uint64_t z = 0;
    for (const auto& [mask, c] : difference) {
        if (!mask) {
            constant = c;
        } else if (std::popcount(mask) == 1 && c == 4) {
            z |= mask;
        } else {
            return std::nullopt;
        }
    }
    if (constant & 1) {
        return std::nullopt;
    }
    auto result = observable;
    const auto correction = product_axis(generators, z, observable.num_qubits());
    // U^dag M U = M * omega^(-d p). Keeping the scalar is necessary when
    // M and the diagonal Pauli anticommute, because their product is imaginary.
    result.right_multiply(correction.view());
    result.add_phase(static_cast<uint8_t>((-constant / 2) & 3));
    assert(result.is_hermitian());
    return result;
}

struct PhaseTerm {
    uint64_t parity;
    int coefficient;
};

struct QuotientRow {
    PauliString body;
    uint64_t label;
};

bool build_polynomial(const HirModule& hir, const std::vector<size_t>& rotations,
                      const KnownStabilizers& known, uint32_t max_variables, Polynomial& polynomial,
                      std::vector<PauliString>& generators, std::vector<PhaseTerm>& terms,
                      std::vector<PauliString>& constraints) {
    std::map<uint32_t, QuotientRow> quotient;
    for (size_t index : rotations) {
        const auto& op = hir.ops[index];
        auto physical = copy_mask(hir.mask_view(op), hir.num_qubits);
        const int coefficient = op.is_dagger() != physical.sign() ? -1 : 1;
        physical.set_sign(false);
        auto reduced = known.reduce_body(physical);
        uint64_t label = 0;
        for (const auto& [pivot, row] : quotient) {
            if (body_bit(reduced, pivot)) {
                reduced.mut_x().xor_with(row.body.x());
                reduced.mut_z().xor_with(row.body.z());
                label ^= row.label;
            }
        }
        if (!identity_body(reduced)) {
            if (generators.size() == max_variables) {
                return false;
            }
            const uint64_t bit = uint64_t{1} << generators.size();
            const uint32_t pivot = pivot_of(reduced);
            quotient.emplace(pivot, QuotientRow{std::move(reduced), label ^ bit});
            generators.push_back(physical);
            label = bit;
        }
        auto residual = physical;
        const auto product = product_axis(generators, label, hir.num_qubits);
        residual.right_multiply(product.view());
        const auto negative = known.eigenvalue(residual);
        assert(negative.has_value());
        if (!identity_body(residual)) {
            constraints.push_back(std::move(residual));
        }
        // A signed relation complements the parity and reverses its angle;
        // the extra constant is a physically irrelevant global phase.
        const int c = *negative ? -coefficient : coefficient;
        terms.push_back({label, c});
        add_parity(polynomial, label, c);
    }
    return true;
}

struct Block {
    size_t end;
    std::vector<size_t> rotations;
};

Block commuting_block(const HirModule& hir, size_t start) {
    Block block{start, {}};
    std::vector<PauliString> axes;
    std::vector<PauliString> noise;
    for (size_t i = start; i < hir.ops.size(); ++i) {
        const auto& op = hir.ops[i];
        if (op.op_type() == OpType::T_GATE) {
            auto axis = copy_mask(hir.mask_view(op), hir.num_qubits);
            const auto commutes = [&](const PauliString& other) {
                return axis.view().commutes(other.view());
            };
            if (!std::ranges::all_of(axes, commutes) || !std::ranges::all_of(noise, commutes)) {
                break;
            }
            axes.push_back(std::move(axis));
            block.rotations.push_back(i);
        } else if (op.op_type() == OpType::NOISE) {
            const auto& site = hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())];
            std::vector<PauliString> channels;
            bool transparent = true;
            for (const auto& channel : site.channels) {
                if (channel.prob > 0) {
                    auto axis = copy_mask(hir.noise_channel_masks.at(channel.mask), hir.num_qubits);
                    for (const auto& other : axes) {
                        transparent &= axis.view().commutes(other.view());
                    }
                    channels.push_back(std::move(axis));
                }
            }
            if (!transparent) {
                break;
            }
            for (auto& channel : channels) {
                noise.push_back(std::move(channel));
            }
        } else if (op.op_type() != OpType::DETECTOR && op.op_type() != OpType::OBSERVABLE &&
                   op.op_type() != OpType::READOUT_NOISE && op.op_type() != OpType::MEASURE &&
                   op.op_type() != OpType::CONDITIONAL_PAULI && op.op_type() != OpType::EXP_VAL) {
            break;
        }
        block.end = i + 1;
    }
    return block;
}

struct ObserverRewrite {
    size_t index;
    PauliString body;
};

std::optional<size_t> rewrite_observers(const HirModule& hir, size_t start, const Block& block,
                                        const std::vector<PauliString>& generators,
                                        const std::vector<PhaseTerm>& terms,
                                        const std::vector<PauliString>& constraints,
                                        std::vector<ObserverRewrite>& observers) {
    Polynomial prefix;
    size_t rotation = 0;
    for (size_t i = start; i < block.end; ++i) {
        const auto& op = hir.ops[i];
        if (op.op_type() == OpType::T_GATE) {
            const auto& term = terms[rotation++];
            add_parity(prefix, term.parity, term.coefficient);
        } else if (op.op_type() == OpType::MEASURE || op.op_type() == OpType::CONDITIONAL_PAULI ||
                   op.op_type() == OpType::EXP_VAL) {
            const auto body = copy_mask(hir.mask_view(op), hir.num_qubits);
            for (const auto& constraint : constraints) {
                // The polynomial is an identity only on this entry code.
                // Both measurement branches must stay inside that code.
                if (!body.view().commutes(constraint.view())) {
                    return i;
                }
            }
            auto conjugated = pullback(body, prefix, generators);
            if (!conjugated) {
                return i;
            }
            observers.push_back({i, std::move(*conjugated)});
        }
    }
    assert(rotation == block.rotations.size());
    return std::nullopt;
}

struct Replacement {
    size_t end;
    std::vector<HeisenbergOp> rotations;
    std::vector<uint32_t> sources;
};

void write_body(HirModule& hir, const HeisenbergOp& op, const PauliString& body) {
    auto mask = hir.mask_at(op);
    std::ranges::copy(body.x().words, mask.x().words.begin());
    std::ranges::copy(body.z().words, mask.z().words.begin());
    mask.set_sign(body.sign());
}

void compact(HirModule& hir, const std::vector<uint8_t>& deleted,
             const std::vector<Replacement>& replacements) {
    const bool mapped = !hir.source_map.empty();
    std::vector<HeisenbergOp> ops;
    std::vector<std::vector<uint32_t>> sources;
    ops.reserve(hir.ops.size());
    if (mapped) {
        sources.reserve(hir.ops.size());
    }
    size_t replacement = 0;
    for (size_t read = 0; read <= hir.ops.size(); ++read) {
        if (replacement < replacements.size() && replacements[replacement].end == read) {
            for (const auto& op : replacements[replacement].rotations) {
                ops.push_back(op);
                if (mapped) {
                    sources.push_back(replacements[replacement].sources);
                }
            }
            ++replacement;
        }
        if (read < hir.ops.size() && !deleted[read]) {
            ops.push_back(hir.ops[read]);
            if (mapped) {
                sources.push_back(std::move(hir.source_map[read]));
            }
        }
    }
    assert(replacement == replacements.size());
    hir.ops = std::move(ops);
    if (mapped) {
        hir.source_map = std::move(sources);
    }
}

}  // namespace

PhasePolynomialPass::PhasePolynomialPass(PhasePolynomialOptions options) : options_(options) {
    if (options.max_variables > 64) {
        throw std::invalid_argument("max_variables must be between zero and 64");
    }
}

void PhasePolynomialPass::run(HirModule& hir) {
    blocks_examined_ = blocks_reduced_ = oversized_blocks_ = expansion_rejections_ = 0;
    pauli_pullbacks_ = 0;
    input_t_count_ = output_t_count_ = hir.num_t_gates();
    applied_ = false;
    const auto incumbent = analyze_active_width(hir);
    incumbent_peak_ = result_peak_ = incumbent.peak_width;
    // A previous noise crossing needs its original axes for symbolic sign
    // correction. A frame-changing rewrite must precede those schedulers.
    if (!hir.logical_noise_prefix_matches_schedule() || options_.max_variables == 0 ||
        input_t_count_ == 0) {
        return;
    }
    if (!hir.source_map.empty() && hir.source_map.size() != hir.ops.size()) {
        throw std::invalid_argument("HIR source map size does not match the operation count");
    }
    HirModule candidate = hir;
    candidate.logical_noise_prefix.clear();
    KnownStabilizers known(candidate.num_qubits);
    std::vector<uint8_t> deleted(candidate.ops.size(), 0);
    std::vector<Replacement> replacements;
    for (size_t i = 0; i < candidate.ops.size();) {
        if (candidate.ops[i].op_type() != OpType::T_GATE) {
            advance_knowledge(candidate, candidate.ops[i++], known);
            continue;
        }
        auto block = commuting_block(candidate, i);
        assert(block.end > i && !block.rotations.empty());
        ++blocks_examined_;
        Polynomial polynomial;
        std::vector<PauliString> generators;
        std::vector<PhaseTerm> terms;
        std::vector<PauliString> constraints;
        std::vector<ObserverRewrite> observers;
        bool rewrite;
        while (true) {
            polynomial.clear();
            generators.clear();
            terms.clear();
            constraints.clear();
            observers.clear();
            rewrite = build_polynomial(candidate, block.rotations, known, options_.max_variables,
                                       polynomial, generators, terms, constraints);
            std::optional<size_t> stop;
            if (rewrite) {
                stop = rewrite_observers(candidate, i, block, generators, terms, constraints,
                                         observers);
            } else {
                // An oversized suffix need not hide a smaller valid prefix.
                for (size_t j = i + 1; j < block.end; ++j) {
                    const auto type = candidate.ops[j].op_type();
                    if (type == OpType::MEASURE || type == OpType::CONDITIONAL_PAULI ||
                        type == OpType::EXP_VAL) {
                        stop = j;
                        break;
                    }
                }
            }
            if (!stop) {
                break;
            }
            block.end = *stop;
            std::erase_if(block.rotations, [&](size_t index) { return index >= block.end; });
        }
        Polynomial synthesis;
        size_t output_count = block.rotations.size();
        if (!rewrite) {
            ++oversized_blocks_;
        } else {
            const uint32_t width = static_cast<uint32_t>(generators.size());
            const uint32_t core = put_core_first(polynomial, generators);
            synthesis = parity_synthesis(polynomial);
            output_count = std::ranges::count_if(
                synthesis, [](const auto& term) { return (term.second & 1) != 0; });
            if (output_count > block.rotations.size()) {
                ++expansion_rejections_;
                rewrite = false;
            } else {
                rewrite = output_count < block.rotations.size() || core < width;
            }
        }
        if (rewrite) {
            pauli_pullbacks_ += observers.size();
            for (const auto& observer : observers) {
                write_body(candidate, candidate.ops[observer.index], observer.body);
            }
            Replacement replacement{block.end, {}, {}};
            auto& sources = replacement.sources;
            for (size_t index : block.rotations) {
                deleted[index] = 1;
                if (!candidate.source_map.empty()) {
                    sources.insert(sources.end(), candidate.source_map[index].begin(),
                                   candidate.source_map[index].end());
                }
            }
            std::ranges::sort(sources);
            sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
            size_t written = 0;
            for (const auto& [label, c] : synthesis) {
                const auto axis = product_axis(generators, label, candidate.num_qubits);
                int clifford = c;
                if (c & 1) {
                    const bool dagger = c >= 5;
                    const size_t index = block.rotations[written++];
                    auto& op = candidate.ops[index];
                    candidate.demote_to_tgate(op, dagger);
                    write_body(candidate, op, axis);
                    replacement.rotations.push_back(op);
                    clifford = (c - (dagger ? -1 : 1)) & 7;
                }
                if (clifford == 4) {
                    internal::apply_virtual_pauli_downstream(candidate, block.end, axis.x(),
                                                             axis.z(), deleted);
                } else if (clifford) {
                    assert(clifford == 2 || clifford == 6);
                    internal::apply_virtual_s_downstream(candidate, block.end, axis.x(), axis.z(),
                                                         axis.sign(), clifford == 6, deleted);
                }
            }
            assert(written == output_count);
            replacements.push_back(std::move(replacement));
            ++blocks_reduced_;
        }
        for (; i < block.end; ++i) {
            if (!deleted[i]) {
                advance_knowledge(candidate, candidate.ops[i], known);
            }
        }
        if (rewrite) {
            for (const auto& op : replacements.back().rotations) {
                advance_knowledge(candidate, op, known);
            }
        }
    }
    if (!blocks_reduced_) {
        return;
    }
    compact(candidate, deleted, replacements);
    const auto result = analyze_active_width(candidate);
    // T count alone does not predict simulation cost on an arbitrary input.
    // Retain the original if the full structural trace would get worse.
    if (result.peak_width > incumbent.peak_width ||
        (result.peak_width == incumbent.peak_width &&
         estimate_dense_work(result) > estimate_dense_work(incumbent))) {
        blocks_reduced_ = pauli_pullbacks_ = 0;
        return;
    }
    output_t_count_ = candidate.num_t_gates();
    result_peak_ = result.peak_width;
    applied_ = true;
    hir = std::move(candidate);
}

}  // namespace clifft
