#include "clifft/sampling/executable_plan_builder.h"

#include "clifft/sampling/early_postselection.h"
#include "clifft/util/noise_sampling.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace clifft::sampling {

namespace {

template <typename>
[[maybe_unused]] inline constexpr bool kAlwaysFalse = false;

// Moving these small lowering helpers out of the constructor stopped non-LTO
// Release builds from inlining them and measurably regressed small-plan
// preparation, so preserve that previously implicit optimization here.
#if defined(_MSC_VER)
#define CLIFFT_BUILDER_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define CLIFFT_BUILDER_FORCE_INLINE inline __attribute__((always_inline))
#else
#define CLIFFT_BUILDER_FORCE_INLINE inline
#endif

bool activates_new_x(const ApplyInstrument& instrument, uint32_t active_after) {
    return instrument.mode == InstrumentMode::Activate && active_after > 0 &&
           instrument.source.z == 0 && instrument.source.x == (uint64_t{1} << (active_after - 1));
}

inline constexpr double kMaxSparseBatchReadoutProbability = 0.05;

// 64-bit FNV-1a seeds and multiplier for affine-expression register interning.
inline constexpr uint64_t kExpressionHashSeedFalse = 0xcbf29ce484222325ULL;
inline constexpr uint64_t kExpressionHashSeedTrue = 0x9e3779b97f4a7c15ULL;
inline constexpr uint64_t kExpressionHashPrime = 0x100000001b3ULL;

[[nodiscard]] uint64_t coefficient_size(uint32_t active_width) noexcept {
    assert(active_width < kDenseActiveWidthLimit &&
           "validated active width must fit coefficient work metadata");
    return uint64_t{1} << active_width;
}

[[nodiscard]] batch_detail::BatchLaneWork common_batch_lane_work(uint64_t work) noexcept {
    return {.common = work};
}

[[nodiscard]] batch_detail::BatchLaneWork row_output_batch_lane_work(uint64_t work) noexcept {
    return {.row_output = work};
}

[[nodiscard]] batch_detail::BatchWorkEstimate classify_batch_lane_work(
    batch_detail::BatchLaneWork work, uint32_t active_width) noexcept {
    return {.all_widths = work,
            .width_five = active_width == 5 ? work : batch_detail::BatchLaneWork{}};
}

[[nodiscard]] batch_detail::BatchLaneWork add_batch_lane_work(
    batch_detail::BatchLaneWork lhs, batch_detail::BatchLaneWork rhs) noexcept {
    return {.common = saturating_add_u64(lhs.common, rhs.common),
            .row_output = saturating_add_u64(lhs.row_output, rhs.row_output)};
}

[[nodiscard]] batch_detail::BatchWorkEstimate add_batch_work_estimate(
    batch_detail::BatchWorkEstimate lhs, batch_detail::BatchWorkEstimate rhs) noexcept {
    return {.all_widths = add_batch_lane_work(lhs.all_widths, rhs.all_widths),
            .width_five = add_batch_lane_work(lhs.width_five, rhs.width_five)};
}

[[nodiscard]] batch_detail::BatchWorkEstimate fused_batch_lane_work(
    const PreparedFusedRotation& rotation) noexcept {
    const uint64_t dimension = uint64_t{1} << rotation.orbit_rank;
    return classify_batch_lane_work(common_batch_lane_work(saturating_multiply_u64(
                                        coefficient_size(rotation.active_width), 2 * dimension)),
                                    rotation.active_width);
}

}  // namespace

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::ExpressionDependencies
ExecutablePlan::ExpressionDependencies::build(uint32_t num_symbols,
                                              std::span<const uint32_t> expression_terms,
                                              std::span<const uint32_t> expression_term_begins) {
    assert((expression_term_begins.empty() || expression_term_begins.front() == 0) &&
           "the first expression must begin at the start of the term tape");
    assert(std::ranges::is_sorted(expression_term_begins) &&
           "expression term ranges must be ordered");
    assert((expression_term_begins.empty() ||
            expression_term_begins.back() <= expression_terms.size()) &&
           "expression term ranges must stay inside the term tape");

    ExpressionDependencies result;
    result.offsets_.assign(static_cast<size_t>(num_symbols) + 1, 0);
    for (uint32_t symbol : expression_terms) {
        assert(symbol < num_symbols && "expression term must refer to a plan symbol");
        ++result.offsets_[static_cast<size_t>(symbol) + 1];
    }
    for (size_t i = 1; i < result.offsets_.size(); ++i) {
        result.offsets_[i] += result.offsets_[i - 1];
    }
    result.targets_.resize(expression_terms.size());
    std::vector<uint32_t> next_dependency = result.offsets_;
    for (size_t expression = 0; expression < expression_term_begins.size(); ++expression) {
        const uint32_t register_id = static_cast<uint32_t>(expression);
        const uint32_t begin = expression_term_begins[expression];
        const uint32_t end = expression + 1 < expression_term_begins.size()
                                 ? expression_term_begins[expression + 1]
                                 : static_cast<uint32_t>(expression_terms.size());
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t symbol = expression_terms[i];
            result.targets_[next_dependency[symbol]++] = register_id;
        }
    }
    return result;
}

ExecutablePlanBuilder::ExpressionInterner::ExpressionInterner(std::vector<uint8_t>& constants)
    : constants_(constants) {}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::ExpressionInterner::reserve(
    size_t expected_expressions, size_t expected_terms) {
    constants_.reserve(expected_expressions);
    terms_.reserve(expected_terms);
    term_begins_.reserve(expected_expressions);
    initial_table_hint_ = std::bit_ceil(std::clamp<size_t>(expected_expressions / 2, 16, 1024));
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::ExpressionInterner::ensure_term_capacity(
    size_t additional_terms) const {
    constexpr size_t kMaxExpressionTerms = std::numeric_limits<uint32_t>::max();
    // Dynamic fusion can replace source signs with denser basis expressions,
    // so the source-plan count is only a reserve estimate after lowering.
    if (additional_terms > kMaxExpressionTerms - terms_.size()) {
        throw std::length_error("sampling executable expression storage exceeds uint32 range");
    }
}

CLIFFT_BUILDER_FORCE_INLINE uint32_t
ExecutablePlanBuilder::ExpressionInterner::prepare_constant(bool constant) {
    uint32_t& slot = constant_registers_[static_cast<size_t>(constant)];
    if (slot != kNoRegister) {
        return slot;
    }
    if (constants_.size() >= kNoRegister) {
        throw std::length_error("sampling executable expression count exceeds uint32 range");
    }
    slot = static_cast<uint32_t>(constants_.size());
    term_begins_.push_back(static_cast<uint32_t>(terms_.size()));
    constants_.push_back(static_cast<uint8_t>(constant));
    return slot;
}

CLIFFT_BUILDER_FORCE_INLINE uint32_t ExecutablePlanBuilder::ExpressionInterner::intern_appended(
    bool constant, uint32_t tentative_begin, uint64_t hash) {
    const auto tentative_terms = std::span<const uint32_t>(terms_).subspan(tentative_begin);
    if ((num_interned_ + 1) * 2 > table_.size()) {
        const size_t next_capacity = table_.empty() ? initial_table_hint_ : table_.size() * 2;
        std::vector<Slot> next_table(next_capacity);
        const size_t mask = next_capacity - 1;
        for (const Slot& existing : table_) {
            if (existing.register_id == kNoRegister) {
                continue;
            }
            size_t bucket = existing.hash & mask;
            while (next_table[bucket].register_id != kNoRegister) {
                bucket = (bucket + 1) & mask;
            }
            next_table[bucket] = existing;
        }
        table_ = std::move(next_table);
    }

    const size_t mask = table_.size() - 1;
    size_t bucket = hash & mask;
    while (table_[bucket].register_id != kNoRegister) {
        if (table_[bucket].hash == hash) {
            const uint32_t candidate_id = table_[bucket].register_id;
            if ((constants_[candidate_id] != 0) == constant) {
                const uint32_t candidate_begin = term_begins_[candidate_id];
                const uint32_t candidate_end = candidate_id + 1 < term_begins_.size()
                                                   ? term_begins_[candidate_id + 1]
                                                   : tentative_begin;
                const auto candidate_terms = std::span<const uint32_t>(terms_).subspan(
                    candidate_begin, candidate_end - candidate_begin);
                if (std::ranges::equal(candidate_terms, tentative_terms)) {
                    terms_.resize(tentative_begin);
                    return candidate_id;
                }
            }
        }
        bucket = (bucket + 1) & mask;
    }

    if (constants_.size() >= kNoRegister) {
        throw std::length_error("sampling executable expression count exceeds uint32 range");
    }
    const uint32_t register_id = static_cast<uint32_t>(constants_.size());
    term_begins_.push_back(tentative_begin);
    constants_.push_back(static_cast<uint8_t>(constant));
    table_[bucket] = {hash, register_id};
    ++num_interned_;
    return register_id;
}

CLIFFT_BUILDER_FORCE_INLINE uint32_t ExecutablePlanBuilder::ExpressionInterner::prepare(
    const AffineBool& expression, uint32_t action_index, std::span<uint32_t> symbol_first_actions) {
    if (expression.terms().empty()) {
        return prepare_constant(expression.constant());
    }
    ensure_term_capacity(expression.terms().size());
    const uint32_t tentative_begin = static_cast<uint32_t>(terms_.size());
    uint64_t hash = expression.constant() ? kExpressionHashSeedTrue : kExpressionHashSeedFalse;
    for (SymbolId term : expression.terms()) {
        const uint32_t sym = index(term);
        terms_.push_back(sym);
        hash ^= sym;
        hash *= kExpressionHashPrime;
        if (!symbol_first_actions.empty()) {
            symbol_first_actions[sym] = std::min(symbol_first_actions[sym], action_index);
        }
    }
    return intern_appended(expression.constant(), tentative_begin, hash);
}

CLIFFT_BUILDER_FORCE_INLINE uint32_t
ExecutablePlanBuilder::ExpressionInterner::prepare_measurement_correction(
    const AffineBool& outcome, uint32_t branch, uint32_t action_index,
    std::span<uint32_t> symbol_first_actions) {
    if (outcome.terms().size() <= 1) {
        assert(!outcome.terms().empty() && index(outcome.terms().front()) == branch &&
               "validated measurement outcome must contain its branch");
        return prepare_constant(outcome.constant());
    }
    ensure_term_capacity(outcome.terms().size() - 1);
    const uint32_t tentative_begin = static_cast<uint32_t>(terms_.size());
    uint64_t hash = outcome.constant() ? kExpressionHashSeedTrue : kExpressionHashSeedFalse;
    for (SymbolId term : outcome.terms()) {
        const uint32_t sym = index(term);
        if (sym != branch) {
            terms_.push_back(sym);
            hash ^= sym;
            hash *= kExpressionHashPrime;
            if (!symbol_first_actions.empty()) {
                symbol_first_actions[sym] = std::min(symbol_first_actions[sym], action_index);
            }
        }
    }
    assert(terms_.size() == static_cast<size_t>(tentative_begin) + outcome.terms().size() - 1 &&
           "validated measurement outcome must contain its branch exactly once");
    return intern_appended(outcome.constant(), tentative_begin, hash);
}

batch_detail::BatchWorkEstimate ExecutablePlanBuilder::action_batch_lane_work(
    const ExecutablePlan& plan, const ExecutablePlan::Action& action, uint32_t width) {
    return std::visit(
        [&](const auto& typed) -> batch_detail::BatchWorkEstimate {
            using T = std::decay_t<decltype(typed)>;
            const auto size = coefficient_size(width);
            if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation> ||
                          std::is_same_v<T, ExecutablePlan::ExecutePromotion>) {
                return classify_batch_lane_work(
                    common_batch_lane_work(saturating_multiply_u64(size, 4)), width);
            } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement>) {
                return classify_batch_lane_work(
                    common_batch_lane_work(saturating_multiply_u64(size, 6)), width);
            } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteExpectation>) {
                return classify_batch_lane_work(
                    typed.active ? row_output_batch_lane_work(saturating_multiply_u64(size, 4))
                                 : batch_detail::BatchLaneWork{},
                    width);
            } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFusedRotation>) {
                return fused_batch_lane_work(
                    plan.fused_rotations_[typed.rotation_index].rotation());
            } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDynamicFusedRotation>) {
                return fused_batch_lane_work(plan.dynamic_fused_rotations_[typed.rotation_index]
                                                 .variants.front()
                                                 .rotation());
            } else {
                return {};
            }
        },
        action);
}

ExecutablePlan ExecutablePlanBuilder::compact_outputs(const ExecutablePlan& input,
                                                      SamplingOutputs outputs,
                                                      bool early_postselection,
                                                      bool survivor_counts) {
    if (input.has_instruments()) {
        throw std::invalid_argument("compact outputs require a fixed sampling plan");
    }
    ExecutablePlan output(input);
    const auto order = EarlyPostselection::schedule(input, early_postselection);
    output.actions_.clear();
    output.action_plan_ranges_.clear();
    std::vector<uint32_t> action_positions(input.actions_.size() + 1);
    const bool keep_observables =
        outputs.syndromes() || survivor_counts || input.has_postselection_;
    if (outputs.measurements()) {
        // Measurement rows already preserve the snapshots needed by feedback
        // and postselection. Keep their prepared expressions and noise program.
        for (size_t source_index : order) {
            action_positions[source_index] = static_cast<uint32_t>(output.actions_.size());
            const auto& action = input.actions_[source_index];
            const bool keep = std::visit(
                [&](const auto& typed) {
                    using T = std::decay_t<decltype(typed)>;
                    if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDetector>) {
                        return outputs.syndromes() || typed.postselected;
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteObservable>) {
                        return keep_observables;
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteExpectation>) {
                        return outputs.expectations();
                    } else {
                        return true;
                    }
                },
                action);
            if (keep) {
                output.actions_.push_back(action);
                if (!input.action_plan_ranges_.empty()) {
                    output.action_plan_ranges_.push_back(input.action_plan_ranges_[source_index]);
                }
            }
        }
    } else {
        const auto expressions = input.affine_registers();
        std::vector<AffineBool> symbols;
        symbols.reserve(input.num_symbols_);
        for (uint32_t symbol = 0; symbol < input.num_symbols_; ++symbol) {
            symbols.push_back(AffineBool::symbol(SymbolId{symbol}));
        }
        std::vector<AffineBool> records(size_t{input.num_visible_records_} +
                                        input.num_hidden_records_);
        const auto expand = [&](ExecutablePlan::PreparedExpression expression) {
            const auto& original = expressions[expression.register_id];
            AffineBool result(original.constant());
            for (SymbolId symbol : original.terms()) {
                result ^= symbols[index(symbol)];
            }
            return result;
        };
        const auto parity = [&](ExecutablePlan::PreparedRecordParity value) {
            AffineBool result(value.constant);
            for (uint32_t term = 0; term < value.count; ++term) {
                result ^= records[input.record_parity_terms_[value.begin + term]];
            }
            return result;
        };

        // Intern only expressions consumed by the compact program. Record snapshots
        // and derived symbols are expanded at their original write positions.
        std::vector<uint32_t> first_actions;
        if (!output.noise_action_deadlines_.empty()) {
            first_actions.assign(input.num_symbols_, std::numeric_limits<uint32_t>::max());
        }
        output.expression_register_constants_.clear();
        output.record_parity_terms_.clear();
        output.dynamic_fused_rotations_.clear();
        ExpressionInterner expressions_out(output.expression_register_constants_);
        expressions_out.reserve(input.num_expression_registers(),
                                input.expression_dependencies_.targets_.size());
        const auto prepare =
            [&](const AffineBool& expression) -> ExecutablePlan::PreparedExpression {
            return {expressions_out.prepare(
                expression, static_cast<uint32_t>(output.actions_.size()), first_actions)};
        };
        const auto zero =
            ExecutablePlan::PreparedExpression{expressions_out.prepare_constant(false)};
        for (size_t source_index : order) {
            action_positions[source_index] = static_cast<uint32_t>(output.actions_.size());
            const auto append = [&](ExecutablePlan::Action action) {
                if (output.actions_.size() >= std::numeric_limits<uint32_t>::max()) {
                    throw std::length_error("compact action storage exceeds uint32 range");
                }
                output.actions_.push_back(std::move(action));
                if (!input.action_plan_ranges_.empty()) {
                    output.action_plan_ranges_.push_back(input.action_plan_ranges_[source_index]);
                }
            };
            std::visit(
                [&](auto typed) {
                    using T = std::decay_t<decltype(typed)>;
                    if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation> ||
                                  std::is_same_v<T, ExecutablePlan::ExecutePromotion> ||
                                  std::is_same_v<T, ExecutablePlan::ExecuteFrameRotation>) {
                        typed.sign = prepare(expand(typed.sign));
                        append(typed);
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFrameClifford>) {
                        typed.condition = prepare(expand(typed.condition));
                        append(typed);
                    } else if constexpr (std::is_same_v<
                                             T, ExecutablePlan::ExecuteDynamicFusedRotation>) {
                        auto rotation = input.dynamic_fused_rotations_[typed.rotation_index];
                        for (auto& sign : rotation.sign_basis) {
                            sign = prepare(expand(sign));
                        }
                        typed.rotation_index =
                            static_cast<uint32_t>(output.dynamic_fused_rotations_.size());
                        output.dynamic_fused_rotations_.push_back(std::move(rotation));
                        append(typed);
                    } else if constexpr (std::is_same_v<T,
                                                        ExecutablePlan::ExecuteActiveMeasurement> ||
                                         std::is_same_v<
                                             T, ExecutablePlan::ExecuteDormantMeasurement> ||
                                         std::is_same_v<T,
                                                        ExecutablePlan::ExecuteFrameMeasurement>) {
                        records[typed.record] =
                            expand(typed.correction) ^ AffineBool::symbol(SymbolId{typed.branch});
                        // Measurement records are internal scratch; requested outputs
                        // retain the original correction through their expanded parity.
                        typed.correction = zero;
                        typed.record = 0;
                        append(typed);
                    } else if constexpr (std::is_same_v<T,
                                                        ExecutablePlan::ExecuteClassicalRecord>) {
                        records[typed.record] = expand(typed.outcome);
                    } else if constexpr (std::is_same_v<T,
                                                        ExecutablePlan::ExecuteSymbolDefinition>) {
                        symbols[typed.symbol] = expand(typed.value);
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteReadoutNoise>) {
                        records[typed.record] ^= AffineBool::symbol(SymbolId{typed.flip});
                        typed.source = prepare(expand(typed.source));
                        typed.record = 0;
                        append(typed);
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDetector>) {
                        if (!outputs.syndromes() && !typed.postselected) {
                            return;
                        }
                        const auto value = prepare(parity(typed.outcome));
                        append(ExecutablePlan::ExecuteClassicalRecord{value, 0});
                        typed.outcome = {static_cast<uint32_t>(output.record_parity_terms_.size()),
                                         1, false};
                        output.record_parity_terms_.push_back(0);
                        append(typed);
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteObservable>) {
                        const auto* expression =
                            std::get_if<ExecutablePlan::PreparedExpression>(&typed.outcome);
                        typed.outcome = prepare(
                            expression ? expand(*expression)
                                       : parity(std::get<ExecutablePlan::PreparedRecordParity>(
                                             typed.outcome)));
                        append(typed);
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteExpectation>) {
                        // Selective and count-only sampling omit expectation probes.
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFusedRotation>) {
                        append(typed);
                    } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteInstrument> ||
                                         std::is_same_v<T, ExecutablePlan::ExecuteBoundary>) {
                        throw std::logic_error("fixed compact program contains a continuation");
                    } else {
                        static_assert(kAlwaysFalse<T>, "Unhandled compact action alternative");
                    }
                },
                input.actions_[source_index]);
        }
        output.num_visible_records_ = 0;
        output.num_hidden_records_ = 1;
        output.expression_dependencies_ = ExecutablePlan::ExpressionDependencies::build(
            output.num_symbols_, expressions_out.terms(), expressions_out.term_begins());
        output.expression_dependencies_.validate(output.num_symbols_,
                                                 output.num_expression_registers());
        if (early_postselection) {
            prepare_noise_schedule(output, first_actions);
        }
        // Row-producing packed execution uses the factorized dense noise program.
        // Count-only sharing prepares its chosen noise strategy lazily.
        output.batch_presampled_program_ =
            outputs.rows() ? BatchPresampledProgram::build(output) : std::nullopt;
    }
    if (!outputs.syndromes() && !input.has_postselection_) {
        output.num_detectors_ = 0;
    }
    if (!keep_observables) {
        output.num_observables_ = 0;
    }
    if (!outputs.expectations()) {
        output.num_exp_vals_ = 0;
    }
    if (!early_postselection || outputs.measurements()) {
        // Keep noise draws before the same stochastic actions when only output
        // materialization changes. Early checks may move a use earlier.
        action_positions.back() = static_cast<uint32_t>(output.actions_.size());
        for (size_t i = action_positions.size() - 1; i > 0; --i) {
            action_positions[i - 1] = std::min(action_positions[i - 1], action_positions[i]);
        }
        for (auto& deadline : output.noise_action_deadlines_) {
            deadline = action_positions[deadline];
        }
    }

    // Output selection and early checks change the estimated batch work.
    output.estimated_batch_lane_work_ = {};
    uint32_t width = output.initial_active_width_;
    std::vector<batch_detail::BatchLaneWork> work;
    work.reserve(output.actions_.size());
    for (const auto& action : output.actions_) {
        const auto estimate = action_batch_lane_work(output, action, width);
        output.estimated_batch_lane_work_ =
            add_batch_work_estimate(output.estimated_batch_lane_work_, estimate);
        if (output.has_postselection_) {
            work.push_back(estimate.all_widths);
        }
        if (std::holds_alternative<ExecutablePlan::ExecutePromotion>(action)) {
            ++width;
        } else if (std::holds_alternative<ExecutablePlan::ExecuteActiveMeasurement>(action)) {
            --width;
        }
    }
    prepare_batch_compaction_costs(output, work);
    return output;
}

void ExecutablePlanBuilder::build(ExecutablePlan& output, const SamplingPlan& source) {
    ExecutablePlanBuilder builder(output, source);
    builder.compile();
}

ExecutablePlanBuilder::ExecutablePlanBuilder(ExecutablePlan& output, const SamplingPlan& source)
    : output_(output), source_(source), expressions_(output.expression_register_constants_) {}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::compile() {
    source_.validate();
    backend_ = resolve_executor_backend(clifft::internal::runtime_isa());
    output_.backend_ = backend_;
    initialize_program();
    prepare_noise_and_boundaries();
    lower_action_stream();
    output_.estimated_batch_lane_work_ = estimated_batch_lane_work_;
    prepare_batch_compaction_costs(output_, action_batch_lane_work_);
    build_expression_dependencies();
    prepare_noise_schedule(output_, symbol_first_actions_);
    output_.batch_presampled_program_ = BatchPresampledProgram::build(output_);
    validate_executable_plan();
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::initialize_program() {
    if (source_.symbols.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable symbol count exceeds uint32 range");
    }
    if (source_.actions.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable action count exceeds uint32 range");
    }

    output_.num_symbols_ = static_cast<uint32_t>(source_.symbols.size());

    const ProgramStorageEstimate storage = estimate_program_storage();
    output_.actions_.reserve(source_.actions.size());
    output_.has_postselection_ = storage.has_postselection;
    if (storage.has_postselection && source_.instrument_distributions.empty() &&
        !source_.presampled_noise_sites.empty()) {
        symbol_first_actions_.assign(source_.symbols.size(), std::numeric_limits<uint32_t>::max());
    }
    if (output_.has_postselection_) {
        action_batch_lane_work_.reserve(source_.actions.size());
    }
    if (source_.source_map.has_value()) {
        output_.action_plan_ranges_.reserve(source_.actions.size());
    }
    // The storage prepass avoids repeated growth of the temporary term tape
    // and identifies plans that need reverse compaction metadata.
    expressions_.reserve(source_.actions.size(), storage.expression_terms);
    output_.instrument_resume_offsets_.assign(source_.instrument_distributions.size(),
                                              std::numeric_limits<uint32_t>::max());
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlanBuilder::ProgramStorageEstimate
ExecutablePlanBuilder::estimate_program_storage() const {
    ProgramStorageEstimate storage;
    size_t num_terms = 0;
    for (const PlannedAction& planned : source_.actions) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, RotateActivePauli> ||
                              std::is_same_v<T, PromoteDormantRotation> ||
                              std::is_same_v<T, RotateFramePauli>) {
                    num_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, ApplyConditionalClifford>) {
                    num_terms += typed.condition.terms().size();
                } else if constexpr (std::is_same_v<T, MeasureActivePauli> ||
                                     std::is_same_v<T, MeasureDormantRandom> ||
                                     std::is_same_v<T, MeasureFramePauli>) {
                    // The current branch is stored separately so replay can
                    // solve for it from the requested record.
                    num_terms += typed.outcome.terms().size() - 1;
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    num_terms += typed.outcome.terms().size();
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    num_terms += typed.value.terms().size();
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    num_terms += typed.source.terms().size();
                } else if constexpr (std::is_same_v<T, WriteDetector>) {
                    storage.has_postselection |= typed.postselected;
                } else if constexpr (std::is_same_v<T, WriteObservable>) {
                    if (const auto* expression = std::get_if<AffineBool>(&typed.outcome)) {
                        num_terms += expression->terms().size();
                    }
                } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                    if (typed.active.has_value()) {
                        num_terms += typed.active->sign.terms().size();
                    }
                } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                    num_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                    // Boundaries have no affine payload.
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
                }
            },
            planned.action);
    }
    if (num_terms > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable expression storage exceeds uint32 range");
    }
    storage.expression_terms = num_terms;
    return storage;
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::prepare_noise_and_boundaries() {
    output_.presampled_symbols_.reserve(source_.symbols.size());
    bound_presampled_symbols_.assign(source_.symbols.size(), 0);
    output_.noise_sites_.reserve(source_.presampled_noise_sites.size());
    output_.noise_hazards_.reserve(source_.presampled_noise_sites.size());

    double cumulative_hazard = 0.0;
    std::optional<double> uniform_site_probability;
    bool uniform_site_probabilities = true;
    for (const PresampledNoiseSite& site : source_.presampled_noise_sites) {
        const uint32_t begin = static_cast<uint32_t>(output_.noise_outcomes_.size());
        double cumulative_probability = 0.0;
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            cumulative_probability += outcome.probability;
            output_.noise_outcomes_.push_back({index(outcome.symbol), cumulative_probability});
            bound_presampled_symbols_[index(outcome.symbol)] = 1;
        }
        if (output_.noise_outcomes_.size() != begin) {
            // The validated source permits roundoff-sized disagreement between
            // a channel's declared total and the sum of its outcomes. Use the
            // declared total as the final bound so scalar and grouped batch
            // sampling agree on the site's Bernoulli probability.
            cumulative_probability = site.total_probability;
            output_.noise_outcomes_.back().cumulative_probability = cumulative_probability;
        }
        output_.noise_sites_.push_back(
            {begin, static_cast<uint32_t>(output_.noise_outcomes_.size()) - begin,
             site.total_probability});
        cumulative_hazard += bernoulli_hazard(cumulative_probability);
        output_.noise_hazards_.push_back(cumulative_hazard);
        if (!uniform_site_probability.has_value()) {
            uniform_site_probability = site.total_probability;
        } else {
            uniform_site_probabilities &= site.total_probability == *uniform_site_probability;
        }
    }
    if (uniform_site_probabilities && uniform_site_probability.has_value() &&
        *uniform_site_probability > 0.0) {
        const double inverse_hazard = 1.0 / bernoulli_hazard(*uniform_site_probability);
        if (is_finite_robust(inverse_hazard)) {
            output_.uniform_noise_inverse_hazard_ = inverse_hazard;
        }
    }
    for (uint32_t symbol = 0; symbol < source_.symbols.size(); ++symbol) {
        if (source_.symbols[symbol] != SymbolKind::Presampled) {
            continue;
        }
        output_.presampled_symbols_.push_back(symbol);
        if (bound_presampled_symbols_[symbol] == 0) {
            ++output_.num_unbound_presampled_symbols_;
        }
    }

    boundary_noise_starts_.reserve(source_.instrument_distributions.size());
    for (const PlannedAction& planned : source_.actions) {
        if (const auto* boundary = std::get_if<InstrumentBoundary>(&planned.action)) {
            boundary_noise_starts_.push_back(boundary->next_noise_site);
        }
    }
    output_.initial_noise_end_ = boundary_noise_starts_.empty()
                                     ? static_cast<uint32_t>(source_.presampled_noise_sites.size())
                                     : boundary_noise_starts_.front();
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::PreparedExpression
ExecutablePlanBuilder::prepare_expression(const AffineBool& expression) {
    return {expressions_.prepare(expression, static_cast<uint32_t>(output_.actions_.size()),
                                 symbol_first_actions_)};
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::PreparedExpression
ExecutablePlanBuilder::prepare_measurement_correction(const AffineBool& outcome, uint32_t branch) {
    return {expressions_.prepare_measurement_correction(
        outcome, branch, static_cast<uint32_t>(output_.actions_.size()), symbol_first_actions_)};
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::PreparedRecordParity
ExecutablePlanBuilder::prepare_record_parity(const RecordParity& parity) {
    if (parity.records().size() >
        std::numeric_limits<uint32_t>::max() - output_.record_parity_terms_.size()) {
        throw std::length_error("sampling executable record parity exceeds uint32 range");
    }
    const uint32_t begin = static_cast<uint32_t>(output_.record_parity_terms_.size());
    for (RecordSlot record : parity.records()) {
        output_.record_parity_terms_.push_back(index(record));
    }
    return {begin, static_cast<uint32_t>(parity.records().size()), parity.constant()};
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::PreparedObservableValue
ExecutablePlanBuilder::prepare_observable_value(const ObservableValue& value) {
    if (const auto* expression = std::get_if<AffineBool>(&value)) {
        return prepare_expression(*expression);
    }
    return prepare_record_parity(std::get<RecordParity>(value));
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::lower_action(const PlannedAction& planned,
                                                                     size_t& boundary_index) {
    std::visit(
        [&](const auto& typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, RotateActivePauli>) {
                PreparedRotation rotation =
                    prepare_rotation(typed.pauli, planned.active_before, typed.half_turns);
                const DirectRotationKernel kernel =
                    resolve_direct_rotation_kernel(rotation, backend_);
                output_.actions_.emplace_back(ExecutablePlan::ExecuteRotation{
                    std::move(rotation), prepare_expression(typed.sign), kernel});
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecutePromotion{
                    prepare_promotion(typed.half_turns), prepare_expression(typed.sign)});
            } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                PreparedMeasurement measurement =
                    prepare_measurement(typed.pauli, planned.active_before, typed.active_pivot);
                const ActiveMeasurementKernel kernel =
                    resolve_active_measurement_kernel(measurement, backend_);
                output_.actions_.emplace_back(ExecutablePlan::ExecuteActiveMeasurement{
                    std::move(measurement),
                    prepare_measurement_correction(typed.outcome, index(typed.branch)),
                    index(typed.branch), index(typed.record), kernel});
            } else if constexpr (std::is_same_v<T, MeasureDormantRandom>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteDormantMeasurement{
                    prepare_measurement_correction(typed.outcome, index(typed.branch)),
                    index(typed.branch), index(typed.record)});
            } else if constexpr (std::is_same_v<T, RecordClassical>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteClassicalRecord{
                    prepare_expression(typed.outcome), index(typed.record)});
            } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteSymbolDefinition{
                    prepare_expression(typed.value), index(typed.symbol)});
            } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                output_.has_readout_noise_ = true;
                double batch_symmetric_inverse_hazard = 0.0;
                if (typed.prob_zero_to_one == typed.prob_one_to_zero &&
                    typed.prob_zero_to_one > 0.0 &&
                    typed.prob_zero_to_one <= kMaxSparseBatchReadoutProbability) {
                    const double inverse_hazard = 1.0 / bernoulli_hazard(typed.prob_zero_to_one);
                    if (is_finite_robust(inverse_hazard)) {
                        batch_symmetric_inverse_hazard = inverse_hazard;
                    }
                }
                output_.actions_.emplace_back(ExecutablePlan::ExecuteReadoutNoise{
                    prepare_expression(typed.source), index(typed.flip), index(typed.record),
                    output_.num_readout_noise_sites_++, typed.prob_zero_to_one,
                    typed.prob_one_to_zero, batch_symmetric_inverse_hazard});
            } else if constexpr (std::is_same_v<T, WriteDetector>) {
                output_.actions_.emplace_back(
                    ExecutablePlan::ExecuteDetector{prepare_record_parity(typed.outcome),
                                                    index(typed.detector),
                                                    typed.postselected,
                                                    {}});
            } else if constexpr (std::is_same_v<T, WriteObservable>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteObservable{
                    prepare_observable_value(typed.outcome), index(typed.observable)});
            } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                std::optional<ExecutablePlan::PreparedExpectation> active;
                if (typed.active.has_value()) {
                    active = ExecutablePlan::PreparedExpectation{
                        prepare_pauli(typed.active->projection, planned.active_before),
                        prepare_expression(typed.active->sign)};
                }
                output_.actions_.emplace_back(
                    ExecutablePlan::ExecuteExpectation{std::move(active), index(typed.exp_val)});
            } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                output_.has_instruments_ = true;
                const uint32_t site = index(typed.site);
                switch (typed.mode) {
                    case InstrumentMode::DormantTrap:
                        output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                            ExecutablePlan::ExecuteDormantInstrumentTrap{site}});
                        return;
                    case InstrumentMode::Classical: {
                        assert(typed.destination_flip.has_value() &&
                               "validated in-line instrument must define a destination flip");
                        output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                            ExecutablePlan::ExecuteClassicalInstrument{
                                prepare_expression(typed.sign), site,
                                index(*typed.destination_flip)}});
                        return;
                    }
                    case InstrumentMode::Active: {
                        assert(typed.destination_flip.has_value() &&
                               "validated in-line instrument must define a destination flip");
                        const uint64_t support =
                            typed.source.x != 0 ? typed.source.x : typed.source.z;
                        const uint32_t pivot = static_cast<uint32_t>(std::countr_zero(support));
                        output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                            ExecutablePlan::ExecuteActiveInstrument{
                                prepare_measurement(typed.source, planned.active_before, pivot),
                                prepare_expression(typed.sign), site,
                                index(*typed.destination_flip)}});
                        return;
                    }
                    case InstrumentMode::Activate: {
                        assert(typed.destination_flip.has_value() &&
                               "validated in-line instrument must define a destination flip");
                        const uint32_t destination_flip = index(*typed.destination_flip);
                        if (activates_new_x(typed, planned.active_after)) {
                            output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                                ExecutablePlan::ExecuteNewXInstrumentActivation{
                                    prepare_expression(typed.sign), site, destination_flip,
                                    resolve_new_x_instrument_kernel(planned.active_before,
                                                                    backend_)}});
                            return;
                        }
                        const uint64_t support =
                            typed.source.x != 0 ? typed.source.x : typed.source.z;
                        const uint32_t pivot = static_cast<uint32_t>(std::countr_zero(support));
                        output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                            ExecutablePlan::ExecuteMeasuredInstrumentActivation{
                                prepare_measurement(typed.source, planned.active_after, pivot),
                                prepare_expression(typed.sign), site, destination_flip}});
                        return;
                    }
                }
                throw std::logic_error("validated instrument mode has no executable lowering");
            } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                const uint32_t noise_end =
                    boundary_index + 1 < boundary_noise_starts_.size()
                        ? boundary_noise_starts_[boundary_index + 1]
                        : static_cast<uint32_t>(source_.presampled_noise_sites.size());
                output_.instrument_resume_offsets_[index(typed.site)] =
                    static_cast<uint32_t>(output_.actions_.size());
                output_.actions_.emplace_back(ExecutablePlan::ExecuteBoundary{
                    index(typed.site), planned.active_before,
                    boundary_noise_starts_[boundary_index], noise_end, typed.symbol_prefix_size});
                ++boundary_index;
            } else if constexpr (std::is_same_v<T, ApplyConditionalClifford>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteFrameClifford{
                    prepare_expression(typed.condition), store_frame_pauli(typed.pauli),
                    typed.rotation});
            } else if constexpr (std::is_same_v<T, RotateFramePauli>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteFrameRotation{
                    prepare_promotion(typed.half_turns), prepare_expression(typed.sign),
                    store_frame_pauli(typed.pauli)});
            } else if constexpr (std::is_same_v<T, MeasureFramePauli>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteFrameMeasurement{
                    prepare_measurement_correction(typed.outcome, index(typed.branch)),
                    index(typed.branch), index(typed.record), store_frame_pauli(typed.pauli)});
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
            }
        },
        planned.action);
    record_batch_lane_work(
        action_batch_lane_work(output_, output_.actions_.back(), planned.active_before));
}

uint32_t ExecutablePlanBuilder::store_frame_pauli(const FramePauli& pauli) {
    const size_t stride = size_t{2} * output_.frame_words();
    const size_t stored = output_.frame_paulis_.size() / stride;
    if (stored >= std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable frame Pauli count exceeds uint32 range");
    }
    output_.frame_paulis_.insert(output_.frame_paulis_.end(), pauli.x.begin(), pauli.x.end());
    output_.frame_paulis_.insert(output_.frame_paulis_.end(), pauli.z.begin(), pauli.z.end());
    return static_cast<uint32_t>(stored);
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::record_batch_lane_work(
    batch_detail::BatchWorkEstimate work) {
    estimated_batch_lane_work_ = add_batch_work_estimate(estimated_batch_lane_work_, work);
    if (output_.has_postselection_) {
        assert(action_batch_lane_work_.size() + 1 == output_.actions_.size() &&
               "each lowered action must receive one batch work estimate");
        action_batch_lane_work_.push_back(work.all_widths);
    } else {
        assert(action_batch_lane_work_.empty() &&
               "ordinary plans must not retain action-level batch work");
    }
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::lower_action_stream() {
    size_t planned_index = 0;
    size_t boundary_index = 0;
    while (planned_index < source_.actions.size()) {
        DynamicFusedRotationRun dynamic_run;
        // AVX2 dynamic fusion regressed large active states despite helping
        // narrower ones, so only the consistently profitable AVX-512 path lowers it.
        if (backend_ == ExecutorBackend::Avx512) {
            dynamic_run = prepare_dynamic_fused_rotation_run(
                std::span<const PlannedAction>(source_.actions).subspan(planned_index));
        }
        if (dynamic_run.rotation.has_value()) {
            PreparedDynamicFusedRotation prepared = std::move(*dynamic_run.rotation);
            ExecutablePlan::PreparedDynamicFusedRotationExecution execution;
            execution.sign_basis.reserve(prepared.sign_basis.size());
            for (const AffineBool& sign : prepared.sign_basis) {
                execution.sign_basis.push_back(prepare_expression(sign));
            }
            execution.variants.reserve(prepared.variants.size());
            for (PreparedFusedRotation& variant : prepared.variants) {
                execution.variants.emplace_back(std::move(variant), backend_);
            }
            const uint32_t fused_index =
                static_cast<uint32_t>(output_.dynamic_fused_rotations_.size());
            output_.dynamic_fused_rotations_.push_back(std::move(execution));
            output_.actions_.emplace_back(ExecutablePlan::ExecuteDynamicFusedRotation{fused_index});
            record_batch_lane_work(fused_batch_lane_work(
                output_.dynamic_fused_rotations_.back().variants.front().rotation()));
            record_action_origin(static_cast<uint32_t>(planned_index),
                                 static_cast<uint32_t>(planned_index + dynamic_run.action_count));
            planned_index += dynamic_run.action_count;
            continue;
        }

        FusedRotationRun run = prepare_fused_rotation_run(
            std::span<const PlannedAction>(source_.actions).subspan(planned_index));
        if (run.rotation.has_value()) {
            const uint32_t fused_index = static_cast<uint32_t>(output_.fused_rotations_.size());
            output_.fused_rotations_.emplace_back(std::move(*run.rotation), backend_);
            output_.actions_.emplace_back(ExecutablePlan::ExecuteFusedRotation{fused_index});
            record_batch_lane_work(
                fused_batch_lane_work(output_.fused_rotations_.back().rotation()));
            record_action_origin(static_cast<uint32_t>(planned_index),
                                 static_cast<uint32_t>(planned_index + run.action_count));
            planned_index += run.action_count;
            continue;
        }
        const size_t unfused_count = std::max<size_t>(run.action_count, 1);
        const size_t run_end = planned_index + unfused_count;
        for (; planned_index < run_end; ++planned_index) {
            lower_action(source_.actions[planned_index], boundary_index);
            record_action_origin(static_cast<uint32_t>(planned_index),
                                 static_cast<uint32_t>(planned_index + 1));
        }
    }
}

void ExecutablePlanBuilder::prepare_noise_schedule(ExecutablePlan& output,
                                                   std::span<const uint32_t> first_actions) {
    if (first_actions.empty() || output.noise_sites_.empty()) {
        return;
    }
    const uint32_t end = static_cast<uint32_t>(output.actions_.size());
    auto& deadlines = output.noise_action_deadlines_;
    deadlines.assign(output.noise_sites_.size(), end);
    for (size_t site = 0; site < output.noise_sites_.size(); ++site) {
        const auto& prepared = output.noise_sites_[site];
        for (uint32_t offset = 0; offset < prepared.outcome_count; ++offset) {
            const uint32_t symbol = output.noise_outcomes_[prepared.outcome_begin + offset].symbol;
            deadlines[site] = std::min(deadlines[site], first_actions[symbol]);
        }
    }
    // An earlier site cannot remain pending when a later site must be sampled.
    // Keep categorical outcomes together even when their first uses differ.
    for (size_t site = deadlines.size() - 1; site > 0; --site) {
        deadlines[site - 1] = std::min(deadlines[site - 1], deadlines[site]);
    }
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::prepare_batch_compaction_costs(
    ExecutablePlan& output, std::span<const batch_detail::BatchLaneWork> work) {
    if (!output.has_postselection_) {
        assert(work.empty() && "ordinary plans must not retain compaction metadata");
        return;
    }
    assert(work.size() == output.actions_.size() &&
           "postselected batch work must parallel executable actions");
    batch_detail::BatchLaneWork remaining_lane_work;
    for (size_t index = output.actions_.size(); index-- > 0;) {
        if (auto* detector =
                std::get_if<ExecutablePlan::ExecuteDetector>(&output.actions_[index])) {
            detector->remaining_batch_lane_work = remaining_lane_work;
            if (detector->postselected) {
                output.postselection_remaining_lane_work_ = remaining_lane_work;
            }
        }
        remaining_lane_work = add_batch_lane_work(remaining_lane_work, work[index]);
    }
    assert(remaining_lane_work.common == output.estimated_batch_lane_work_.all_widths.common &&
           remaining_lane_work.row_output ==
               output.estimated_batch_lane_work_.all_widths.row_output &&
           "forward and reverse batch work totals must agree");
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::record_action_origin(uint32_t plan_begin,
                                                                             uint32_t plan_end) {
    if (!source_.source_map.has_value()) {
        return;
    }
    assert(plan_begin < plan_end && plan_end <= source_.actions.size() &&
           "executable action must name a nonempty plan range");
    assert(output_.action_plan_ranges_.size() + 1 == output_.actions_.size() &&
           "each executable action must receive exactly one plan range");
    output_.action_plan_ranges_.push_back({plan_begin, plan_end});
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::build_expression_dependencies() {
    output_.expression_dependencies_ = ExecutablePlan::ExpressionDependencies::build(
        output_.num_symbols_, expressions_.terms(), expressions_.term_begins());
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::validate_executable_plan() const {
#ifndef NDEBUG
    if (!output_.noise_action_deadlines_.empty()) {
        assert(output_.noise_action_deadlines_.size() == output_.noise_sites_.size() &&
               "each scheduled noise site must have a deadline");
        assert(std::ranges::is_sorted(output_.noise_action_deadlines_) &&
               output_.noise_action_deadlines_.back() <= output_.actions_.size() &&
               "noise deadlines must follow executable action order");
    }
    assert(expressions_.num_registers() == output_.expression_register_constants_.size() &&
           "expression register storage is inconsistent");
    output_.expression_dependencies_.validate(output_.num_symbols_,
                                              output_.expression_register_constants_.size());
    if (output_.batch_presampled_program_.has_value()) {
        output_.batch_presampled_program_->validate(output_.noise_outcomes_.size(),
                                                    output_.expression_register_constants_.size(),
                                                    output_.presampled_symbols_.size());
    }
    const size_t num_records =
        static_cast<size_t>(output_.num_visible_records_) + output_.num_hidden_records_;
    if (source_.source_map.has_value()) {
        assert(output_.action_plan_ranges_.size() == output_.actions_.size() &&
               "executable provenance must remain parallel to the action stream");
        uint32_t expected_begin = 0;
        for (const ExecutablePlan::PlanActionRange& range : output_.action_plan_ranges_) {
            assert(range.begin == expected_begin && range.begin < range.end &&
                   range.end <= source_.actions.size() &&
                   "executable provenance must partition the plan action stream");
            expected_begin = range.end;
        }
        assert(expected_begin == source_.actions.size() &&
               "executable provenance must cover every plan action");
    } else {
        assert(output_.action_plan_ranges_.empty() &&
               "ordinary lowering must not retain debug provenance");
    }

    auto validate_expression = [&](ExecutablePlan::PreparedExpression expression) {
        assert(expression.register_id < output_.expression_register_constants_.size() &&
               "action expression is out of range");
    };
    auto validate_record_parity = [&](ExecutablePlan::PreparedRecordParity parity) {
        const size_t end = static_cast<size_t>(parity.begin) + parity.count;
        assert(end <= output_.record_parity_terms_.size() &&
               "record parity must stay in its prepared tape");
        for (size_t term = parity.begin; term < end; ++term) {
            assert(output_.record_parity_terms_[term] < num_records &&
                   "record parity must name a valid record");
        }
    };
    auto validate_observable_value = [&](const ExecutablePlan::PreparedObservableValue& value) {
        if (const auto* expression = std::get_if<ExecutablePlan::PreparedExpression>(&value)) {
            validate_expression(*expression);
        } else {
            validate_record_parity(std::get<ExecutablePlan::PreparedRecordParity>(value));
        }
    };
    auto validate_frame_pauli = [&]([[maybe_unused]] uint32_t pauli) {
        assert(output_.frame_width_ != 0 &&
               (size_t{2} * pauli + 2) * output_.frame_words() <= output_.frame_paulis_.size() &&
               "frame Pauli is out of range");
    };
    for (const ExecutablePlan::Action& action : output_.actions_) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation>) {
                    validate_expression(typed.sign);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFusedRotation>) {
                    assert(typed.rotation_index < output_.fused_rotations_.size() &&
                           "fused rotation is out of range");
                } else if constexpr (std::is_same_v<T,
                                                    ExecutablePlan::ExecuteDynamicFusedRotation>) {
                    assert(typed.rotation_index < output_.dynamic_fused_rotations_.size() &&
                           "dynamic fused rotation is out of range");
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecutePromotion>) {
                    validate_expression(typed.sign);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement>) {
                    validate_expression(typed.correction);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDormantMeasurement>) {
                    validate_expression(typed.correction);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteClassicalRecord>) {
                    validate_expression(typed.outcome);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteSymbolDefinition>) {
                    validate_expression(typed.value);
                    assert(typed.symbol < output_.num_symbols_ && "defined symbol is out of range");
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteReadoutNoise>) {
                    validate_expression(typed.source);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDetector>) {
                    validate_record_parity(typed.outcome);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteObservable>) {
                    validate_observable_value(typed.outcome);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteExpectation>) {
                    if (typed.active.has_value()) {
                        validate_expression(typed.active->sign);
                    }
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteInstrument>) {
                    std::visit(
                        [&](const auto& instrument) {
                            using Instrument = std::decay_t<decltype(instrument)>;
                            assert(instrument.site < output_.instrument_distributions_.size() &&
                                   "instrument site is out of range");
                            if constexpr (!std::is_same_v<
                                              Instrument,
                                              ExecutablePlan::ExecuteDormantInstrumentTrap>) {
                                validate_expression(instrument.sign);
                                assert(instrument.destination_flip < output_.num_symbols_ &&
                                       "instrument destination flip is out of range");
                            }
                        },
                        typed.form);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteBoundary>) {
                    assert(typed.site < output_.instrument_resume_offsets_.size() &&
                           "instrument boundary site is out of range");
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFrameClifford>) {
                    validate_expression(typed.condition);
                    validate_frame_pauli(typed.pauli);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFrameRotation>) {
                    validate_expression(typed.sign);
                    validate_frame_pauli(typed.pauli);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFrameMeasurement>) {
                    validate_expression(typed.correction);
                    validate_frame_pauli(typed.pauli);
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled executable action alternative");
                }
            },
            action);
    }
    for (const ExecutablePlan::PreparedDynamicFusedRotationExecution& rotation :
         output_.dynamic_fused_rotations_) {
        for (ExecutablePlan::PreparedExpression sign : rotation.sign_basis) {
            validate_expression(sign);
        }
    }
#endif
}

#undef CLIFFT_BUILDER_FORCE_INLINE

}  // namespace clifft::sampling
