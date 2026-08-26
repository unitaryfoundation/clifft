#include "clifft/sampling/batch/presampled_program.h"

#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/plan.h"
#include "clifft/util/mask_view.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace clifft::sampling {

namespace {

struct ExpressionBlock {
    bool constant = false;
    std::vector<uint32_t> terms;
    std::vector<uint32_t> registers;
    uint32_t parent = std::numeric_limits<uint32_t>::max();
    bool invert_parent = false;
    uint32_t depth = 0;
    std::vector<uint32_t> delta_terms;
};

struct NoiseEffectBasis {
    std::vector<uint64_t> effects;
    uint32_t carrier_symbol = 0;
};

inline constexpr uint64_t kMinExpressionTerms = 1024;
inline constexpr uint64_t kCostNumerator = 3;
inline constexpr uint64_t kCostDenominator = 4;

std::vector<uint32_t> symmetric_difference(std::span<const uint32_t> left,
                                           std::span<const uint32_t> right) {
    std::vector<uint32_t> result;
    result.reserve(left.size() + right.size());
    std::ranges::set_symmetric_difference(left, right, std::back_inserter(result));
    return result;
}

uint64_t expression_hash(bool constant, std::span<const uint32_t> terms) noexcept {
    uint64_t hash = constant ? 0x9e3779b97f4a7c15ULL : 0xcbf29ce484222325ULL;
    for (uint32_t term : terms) {
        hash ^= term;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

}  // namespace

std::optional<BatchPresampledProgram> BatchPresampledProgram::build(
    const ExecutablePlan& executable, const SamplingPlan& source,
    std::span<const uint32_t> expression_terms, std::span<const uint32_t> expression_term_begins,
    std::span<const uint8_t> bound_presampled_symbols) {
#if defined(__EMSCRIPTEN__)
    (void)executable;
    (void)source;
    (void)expression_terms;
    (void)expression_term_begins;
    (void)bound_presampled_symbols;
    return std::nullopt;
#else
    if (executable.has_instruments_) {
        return std::nullopt;
    }

    const size_t num_expressions = expression_term_begins.size();
    assert(bound_presampled_symbols.size() == source.symbols.size() &&
           "presampled ownership must parallel plan symbols");
    std::vector<std::vector<uint32_t> > presampled_terms(num_expressions);
    uint64_t original_presampled_terms = 0;
    for (size_t expression = 0; expression < num_expressions; ++expression) {
        const uint32_t begin = expression_term_begins[expression];
        const uint32_t end = expression + 1 < num_expressions
                                 ? expression_term_begins[expression + 1]
                                 : static_cast<uint32_t>(expression_terms.size());
        for (uint32_t term = begin; term < end; ++term) {
            const uint32_t symbol = expression_terms[term];
            if (source.symbols[symbol] != SymbolKind::Presampled) {
                continue;
            }
            ++original_presampled_terms;
            if (bound_presampled_symbols[symbol] == 0) {
                presampled_terms[expression].push_back(symbol);
            }
        }
    }
    if (original_presampled_terms < kMinExpressionTerms) {
        return std::nullopt;
    }

    BatchPresampledProgram program;
    const size_t effect_words = num_expressions / 64 + (num_expressions % 64 != 0);
    program.outcome_assignments_.reserve(executable.noise_outcomes_.size());
    for (const PresampledNoiseSite& site : source.presampled_noise_sites) {
        std::vector<NoiseEffectBasis> basis;
        std::vector<int32_t> basis_for_effect(num_expressions, -1);
        std::vector<uint64_t> residual(effect_words, 0);
        basis.reserve(site.outcomes.size());
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            std::ranges::fill(residual, uint64_t{0});
            for (uint32_t expression :
                 executable.expression_dependencies_.dependent_registers(index(outcome.symbol))) {
                residual[expression >> 6] |= uint64_t{1} << (expression & 63);
            }

            MutableMaskView residual_view(residual);
            std::vector<uint32_t> coordinates;
            while (!residual_view.is_zero()) {
                const uint32_t pivot = residual_view.lowest_bit();
                const int32_t basis_index = basis_for_effect[pivot];
                if (basis_index >= 0) {
                    coordinates.push_back(static_cast<uint32_t>(basis_index));
                    residual_view.xor_with(
                        MaskView(basis[static_cast<size_t>(basis_index)].effects));
                    continue;
                }

                const uint32_t new_basis_index = static_cast<uint32_t>(basis.size());
                basis_for_effect[pivot] = static_cast<int32_t>(new_basis_index);
                coordinates.push_back(new_basis_index);
                basis.push_back(NoiseEffectBasis{std::move(residual), index(outcome.symbol)});
                residual.resize(effect_words, 0);
                const NoiseEffectBasis& added = basis.back();
                for (size_t word = 0; word < added.effects.size(); ++word) {
                    uint64_t pending = added.effects[word];
                    while (pending != 0) {
                        const uint32_t bit = std::countr_zero(pending);
                        const size_t expression = word * 64 + bit;
                        assert(expression < num_expressions &&
                               "batch noise effect must name a prepared expression");
                        presampled_terms[expression].push_back(added.carrier_symbol);
                        pending &= pending - 1;
                    }
                }
                break;
            }

            if (coordinates.size() >
                std::numeric_limits<uint32_t>::max() - program.assigned_carriers_.size()) {
                throw std::length_error(
                    "sampling executable batch noise assignments exceed uint32 range");
            }
            const uint32_t assignment_begin =
                static_cast<uint32_t>(program.assigned_carriers_.size());
            for (uint32_t basis_index : coordinates) {
                program.assigned_carriers_.push_back(basis[basis_index].carrier_symbol);
            }
            program.outcome_assignments_.push_back(
                {assignment_begin, static_cast<uint32_t>(coordinates.size())});
        }
    }
    if (program.outcome_assignments_.size() != executable.noise_outcomes_.size()) {
        throw std::logic_error("batch noise factorization must cover every prepared outcome");
    }
    for (std::vector<uint32_t>& terms : presampled_terms) {
        std::ranges::sort(terms);
        assert(std::ranges::adjacent_find(terms) == terms.end() &&
               "batch noise factorization must produce canonical terms");
    }

    std::unordered_multimap<uint64_t, uint32_t> interned;
    std::vector<ExpressionBlock> blocks;
    blocks.reserve(expression_term_begins.size());
    interned.reserve(expression_term_begins.size());
    for (size_t expression = 0; expression < expression_term_begins.size(); ++expression) {
        std::vector<uint32_t> terms = std::move(presampled_terms[expression]);

        const bool constant = executable.expression_register_constants_[expression] != 0;
        const uint64_t hash = expression_hash(constant, terms);
        uint32_t block_index = std::numeric_limits<uint32_t>::max();
        const auto [first, last] = interned.equal_range(hash);
        for (auto position = first; position != last; ++position) {
            const ExpressionBlock& candidate = blocks[position->second];
            if (candidate.constant == constant && candidate.terms == terms) {
                block_index = position->second;
                break;
            }
        }
        if (block_index == std::numeric_limits<uint32_t>::max()) {
            block_index = static_cast<uint32_t>(blocks.size());
            ExpressionBlock block;
            block.constant = constant;
            block.terms = std::move(terms);
            blocks.push_back(std::move(block));
            interned.emplace(hash, block_index);
        }
        blocks[block_index].registers.push_back(static_cast<uint32_t>(expression));
    }

    std::vector<std::vector<uint32_t> > blocks_by_symbol(executable.num_symbols_);
    std::vector<uint32_t> intersection_counts(blocks.size(), 0);
    std::vector<uint32_t> candidate_parents;
    uint32_t max_depth = 0;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        ExpressionBlock& block = blocks[block_index];
        block.delta_terms = block.terms;
        uint64_t best_cost = block.delta_terms.size();
        candidate_parents.clear();
        for (uint32_t symbol : block.terms) {
            for (uint32_t parent_index : blocks_by_symbol[symbol]) {
                if (intersection_counts[parent_index]++ == 0) {
                    candidate_parents.push_back(parent_index);
                }
            }
        }
        std::ranges::sort(candidate_parents);
        for (uint32_t parent_index : candidate_parents) {
            const bool invert_parent = block.constant != blocks[parent_index].constant;
            const uint64_t cost = static_cast<uint64_t>(block.terms.size()) +
                                  blocks[parent_index].terms.size() -
                                  2 * static_cast<uint64_t>(intersection_counts[parent_index]) +
                                  static_cast<uint64_t>(invert_parent);
            if (cost < best_cost) {
                best_cost = cost;
                block.parent = parent_index;
                block.invert_parent = invert_parent;
            }
        }
        for (uint32_t parent_index : candidate_parents) {
            intersection_counts[parent_index] = 0;
        }
        if (block.parent != std::numeric_limits<uint32_t>::max()) {
            block.delta_terms = symmetric_difference(block.terms, blocks[block.parent].terms);
            block.depth = blocks[block.parent].depth + 1;
        }
        max_depth = std::max(max_depth, block.depth);
        for (uint32_t symbol : block.terms) {
            blocks_by_symbol[symbol].push_back(static_cast<uint32_t>(block_index));
        }
    }

    uint64_t prepared_operations = 0;
    for (const ExpressionBlock& block : blocks) {
        prepared_operations += block.delta_terms.size();
        prepared_operations += static_cast<uint64_t>(block.invert_parent);
        prepared_operations +=
            static_cast<uint64_t>(block.parent != std::numeric_limits<uint32_t>::max());
        prepared_operations += block.registers.size() - 1;
    }
    // Parent and duplicate copies each require a full packed-column pass.
    // Keep ordinary propagation unless the shared program removes enough
    // passes to amortize its retained plan storage and reset bookkeeping.
    if (prepared_operations * kCostDenominator > original_presampled_terms * kCostNumerator) {
        return std::nullopt;
    }

    std::vector<std::vector<BatchPresampledProgram::InitializeExpression> >
        initializations_by_level(static_cast<size_t>(max_depth) + 1);
    std::vector<std::vector<BatchPresampledProgram::XorCarrierIntoExpression> >
        carrier_xors_by_level(static_cast<size_t>(max_depth) + 1);
    for (const ExpressionBlock& block : blocks) {
        assert(!block.registers.empty() && "interned expression block must have a destination");
        const uint32_t destination = block.registers.front();
        if (block.parent != std::numeric_limits<uint32_t>::max()) {
            initializations_by_level[block.depth].push_back(
                {destination, blocks[block.parent].registers.front(), block.invert_parent});
        }
        for (uint32_t symbol : block.delta_terms) {
            carrier_xors_by_level[block.depth].push_back({symbol, destination});
        }
        for (size_t register_index = 1; register_index < block.registers.size(); ++register_index) {
            program.copies_.push_back({destination, block.registers[register_index]});
        }
    }

    program.initialization_level_offsets_.push_back(0);
    program.carrier_xor_level_offsets_.push_back(0);
    for (size_t level = 0; level < initializations_by_level.size(); ++level) {
        auto& carrier_xors = carrier_xors_by_level[level];
        std::ranges::sort(carrier_xors, {},
                          &BatchPresampledProgram::XorCarrierIntoExpression::carrier);
        if (initializations_by_level[level].size() >
                std::numeric_limits<uint32_t>::max() - program.initializations_.size() ||
            carrier_xors.size() >
                std::numeric_limits<uint32_t>::max() - program.carrier_xors_.size()) {
            throw std::length_error(
                "sampling executable presampled expression tape exceeds uint32 range");
        }
        program.initializations_.insert(program.initializations_.end(),
                                        initializations_by_level[level].begin(),
                                        initializations_by_level[level].end());
        program.carrier_xors_.insert(program.carrier_xors_.end(), carrier_xors.begin(),
                                     carrier_xors.end());
        program.initialization_level_offsets_.push_back(
            static_cast<uint32_t>(program.initializations_.size()));
        program.carrier_xor_level_offsets_.push_back(
            static_cast<uint32_t>(program.carrier_xors_.size()));
    }

    constexpr uint32_t kUnassigned = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> carrier_slots(executable.num_symbols_, kUnassigned);
    const auto mark_carrier = [&](uint32_t symbol) {
        assert(symbol < source.symbols.size() &&
               source.symbols[symbol] == SymbolKind::Presampled &&
               "batch carrier must originate from a presampled symbol");
        carrier_slots[symbol] = 0;
    };
    for (uint32_t assignment : program.assigned_carriers_) {
        mark_carrier(assignment);
    }
    for (const BatchPresampledProgram::XorCarrierIntoExpression& carrier_xor :
         program.carrier_xors_) {
        mark_carrier(carrier_xor.carrier);
    }
    for (uint32_t& slot : carrier_slots) {
        if (slot != kUnassigned) {
            slot = program.num_carriers_++;
        }
    }
    for (uint32_t& assignment : program.assigned_carriers_) {
        assignment = carrier_slots[assignment];
    }
    for (BatchPresampledProgram::XorCarrierIntoExpression& carrier_xor : program.carrier_xors_) {
        carrier_xor.carrier = carrier_slots[carrier_xor.carrier];
    }

    program.validate(executable.noise_outcomes_.size(),
                     executable.expression_register_constants_.size());
    return program;
#endif
}

void BatchPresampledProgram::validate(size_t num_noise_outcomes,
                                      size_t num_expression_registers) const noexcept {
#ifndef NDEBUG
    assert(!initialization_level_offsets_.empty() &&
           initialization_level_offsets_.size() == carrier_xor_level_offsets_.size() &&
           initialization_level_offsets_.front() == 0 && carrier_xor_level_offsets_.front() == 0 &&
           initialization_level_offsets_.back() == initializations_.size() &&
           carrier_xor_level_offsets_.back() == carrier_xors_.size() &&
           "presampled expression levels must cover their operation tapes");
    for (const InitializeExpression& initialization : initializations_) {
        assert(initialization.destination < num_expression_registers &&
               initialization.parent < num_expression_registers &&
               "presampled expression initialization must name valid registers");
    }
    for (const XorCarrierIntoExpression& carrier_xor : carrier_xors_) {
        assert(carrier_xor.carrier < num_carriers_ &&
               carrier_xor.destination < num_expression_registers &&
               "presampled expression carrier XOR must name valid storage");
    }
    for (const CopyExpression& copy : copies_) {
        assert(copy.source < num_expression_registers &&
               copy.destination < num_expression_registers &&
               "presampled expression copy must name valid registers");
    }
    assert(outcome_assignments_.size() == num_noise_outcomes && num_carriers_ != 0 &&
           "batch outcome assignments must parallel scalar outcomes");
    for (const OutcomeAssignments& outcome : outcome_assignments_) {
        const size_t end = static_cast<size_t>(outcome.begin) + outcome.count;
        assert(end <= assigned_carriers_.size() &&
               "batch outcome assignment must stay in its prepared tape");
        for (size_t assignment = outcome.begin; assignment < end; ++assignment) {
            assert(assigned_carriers_[assignment] < num_carriers_ &&
                   "batch outcome assignment must name a compact carrier");
        }
    }
#else
    (void)num_noise_outcomes;
    (void)num_expression_registers;
#endif
}

}  // namespace clifft::sampling
