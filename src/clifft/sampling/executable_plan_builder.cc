#include "clifft/sampling/executable_plan_builder.h"

#include "clifft/util/noise_sampling.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace clifft::sampling {

namespace {

template <typename>
inline constexpr bool kAlwaysFalse = false;

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

inline constexpr uint32_t kExperimentalCacheBlockRank = 14;
inline constexpr uint32_t kExperimentalCacheBlockDefaultMinActiveWidth = 24;
inline constexpr size_t kExperimentalCacheBlockMinExecutablePasses = 6;
inline constexpr uint32_t kAvx512LaneIndexBitsForCacheBlock = 3;
inline constexpr uint32_t kExperimentalCacheBlockMaxFusedSigns = 2;
inline constexpr uint32_t kExperimentalCacheBlockMaxFusedSelectors = 5;

[[nodiscard]] bool experimental_cache_blocking_enabled() noexcept {
    const char* const value = std::getenv("CLIFFT_EXPERIMENTAL_CACHE_BLOCKED_ROTATIONS");
    return value != nullptr && std::string_view(value) == "1";
}

[[nodiscard]] uint32_t experimental_cache_block_min_active_width() noexcept {
    const char* const text = std::getenv("CLIFFT_EXPERIMENTAL_CACHE_BLOCKED_MIN_ACTIVE_WIDTH");
    if (text == nullptr) {
        return kExperimentalCacheBlockDefaultMinActiveWidth;
    }
    uint32_t value = 0;
    const std::string_view input(text);
    const auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    if (result.ec != std::errc{} || result.ptr != input.data() + input.size() ||
        value >= kDenseActiveWidthLimit) {
        return kExperimentalCacheBlockDefaultMinActiveWidth;
    }
    return value;
}

class CacheBlockBasis {
  public:
    [[nodiscard]] bool insert(uint64_t value, uint32_t max_rank) noexcept {
        uint64_t reduced = value;
        for (int pivot = 63; pivot >= 0; --pivot) {
            const uint64_t bit = uint64_t{1} << pivot;
            if ((reduced & bit) != 0 && rows_[pivot] != 0) {
                reduced ^= rows_[pivot];
            }
        }
        if (reduced == 0) {
            return true;
        }
        if (rank_ == max_rank) {
            return false;
        }
        const uint32_t pivot = 63U - static_cast<uint32_t>(std::countl_zero(reduced));
        const uint64_t bit = uint64_t{1} << pivot;
        for (uint64_t& row : rows_) {
            if ((row & bit) != 0) {
                row ^= reduced;
            }
        }
        rows_[pivot] = reduced;
        ++rank_;
        return true;
    }

    [[nodiscard]] uint32_t rank() const noexcept { return rank_; }

    [[nodiscard]] std::vector<uint64_t> rows() const {
        std::vector<uint64_t> result;
        result.reserve(rank_);
        for (uint64_t row : rows_) {
            if (row != 0) {
                result.push_back(row);
            }
        }
        return result;
    }

    [[nodiscard]] uint64_t coordinates(uint64_t value) const noexcept {
        uint64_t result = 0;
        uint32_t coordinate = 0;
        for (uint64_t row : rows_) {
            if (row == 0) {
                continue;
            }
            const uint64_t pivot = std::bit_floor(row);
            if ((value & pivot) != 0) {
                value ^= row;
                result |= uint64_t{1} << coordinate;
            }
            ++coordinate;
        }
        assert(value == 0 && "cache-blocked Pauli X mask must belong to the block span");
        return result;
    }

  private:
    std::array<uint64_t, 64> rows_{};
    uint32_t rank_ = 0;
};

struct CacheSelector {
    uint64_t local = 0;
    uint64_t external = 0;

    CacheSelector& operator^=(const CacheSelector& other) noexcept {
        local ^= other.local;
        external ^= other.external;
        return *this;
    }
};

[[nodiscard]] bool cache_selector_bit(const CacheSelector& selector, uint32_t bit) noexcept {
    return bit < 64 ? ((selector.local >> bit) & 1U) != 0
                    : ((selector.external >> (bit - 64)) & 1U) != 0;
}

class CacheSelectorBasis {
  public:
    [[nodiscard]] bool insert(CacheSelector value, uint32_t max_rank) noexcept {
        for (int pivot = 127; pivot >= 0; --pivot) {
            if (cache_selector_bit(value, static_cast<uint32_t>(pivot)) &&
                (rows_[pivot].local != 0 || rows_[pivot].external != 0)) {
                value ^= rows_[pivot];
            }
        }
        if (value.local == 0 && value.external == 0) {
            return true;
        }
        if (rank_ == max_rank) {
            return false;
        }
        uint32_t pivot = 0;
        for (int candidate = 127; candidate >= 0; --candidate) {
            if (cache_selector_bit(value, static_cast<uint32_t>(candidate))) {
                pivot = static_cast<uint32_t>(candidate);
                break;
            }
        }
        for (CacheSelector& row : rows_) {
            if (cache_selector_bit(row, pivot)) {
                row ^= value;
            }
        }
        rows_[pivot] = value;
        ++rank_;
        return true;
    }

    [[nodiscard]] std::vector<CacheSelector> rows() const {
        std::vector<CacheSelector> result;
        result.reserve(rank_);
        for (const CacheSelector& row : rows_) {
            if (row.local != 0 || row.external != 0) {
                result.push_back(row);
            }
        }
        return result;
    }

    [[nodiscard]] uint32_t coordinates(CacheSelector value) const noexcept {
        uint32_t result = 0;
        uint32_t coordinate = 0;
        for (const CacheSelector& row : rows_) {
            if (row.local == 0 && row.external == 0) {
                continue;
            }
            uint32_t pivot = 0;
            for (uint32_t candidate = 0; candidate < 128; ++candidate) {
                if (cache_selector_bit(row, candidate)) {
                    pivot = candidate;
                }
            }
            if (cache_selector_bit(value, pivot)) {
                value ^= row;
                result |= uint32_t{1} << coordinate;
            }
            ++coordinate;
        }
        assert(value.local == 0 && value.external == 0 &&
               "cache selector must belong to its prepared basis");
        return result;
    }

  private:
    std::array<CacheSelector, 128> rows_{};
    uint32_t rank_ = 0;
};

class CacheSignBasis {
  public:
    [[nodiscard]] bool insert(const AffineBool& value, uint32_t max_rank) {
        AffineBool reduced(false, value.terms());
        while (!reduced.terms().empty()) {
            const uint32_t pivot = index(reduced.terms().back());
            if (pivot < rows_.size() && rows_[pivot].has_value()) {
                reduced ^= rows_[pivot]->expression;
                continue;
            }
            if (expressions_.size() == max_rank) {
                return false;
            }
            if (pivot >= rows_.size()) {
                rows_.resize(static_cast<size_t>(pivot) + 1);
            }
            const uint32_t coordinate = static_cast<uint32_t>(expressions_.size());
            expressions_.push_back(reduced);
            rows_[pivot] = Row{std::move(reduced), coordinate};
            return true;
        }
        return true;
    }

    [[nodiscard]] uint32_t coordinates(const AffineBool& value) const {
        AffineBool reduced(false, value.terms());
        uint32_t result = 0;
        while (!reduced.terms().empty()) {
            const uint32_t pivot = index(reduced.terms().back());
            assert(pivot < rows_.size() && rows_[pivot].has_value() &&
                   "cache sign must belong to its prepared basis");
            result ^= uint32_t{1} << rows_[pivot]->coordinate;
            reduced ^= rows_[pivot]->expression;
        }
        return result;
    }

    [[nodiscard]] const std::vector<AffineBool>& expressions() const noexcept {
        return expressions_;
    }

  private:
    struct Row {
        AffineBool expression;
        uint32_t coordinate = 0;
    };

    std::vector<std::optional<Row>> rows_;
    std::vector<AffineBool> expressions_;
};

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

[[nodiscard]] batch_detail::BatchWorkEstimate planned_batch_lane_work(
    const PlannedAction& planned) noexcept {
    const batch_detail::BatchLaneWork work = std::visit(
        [&](const auto& typed) -> batch_detail::BatchLaneWork {
            using T = std::decay_t<decltype(typed)>;
            const uint64_t size = coefficient_size(planned.active_before);
            if constexpr (std::is_same_v<T, RotateActivePauli>) {
                return common_batch_lane_work(saturating_multiply_u64(size, 4));
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                return common_batch_lane_work(saturating_multiply_u64(size, 4));
            } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                return common_batch_lane_work(saturating_multiply_u64(size, 6));
            } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                return typed.active.has_value()
                           ? row_output_batch_lane_work(saturating_multiply_u64(size, 4))
                           : batch_detail::BatchLaneWork{};
            } else {
                return {};
            }
        },
        planned.action);
    return classify_batch_lane_work(work, planned.active_before);
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

void ExecutablePlanBuilder::build(ExecutablePlan& output, const SamplingPlan& source) {
    ExecutablePlanBuilder builder(output, source);
    builder.compile();
}

ExecutablePlanBuilder::ExecutablePlanBuilder(ExecutablePlan& output, const SamplingPlan& source)
    : output_(output), source_(source) {}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::compile() {
    source_.validate();
    backend_ = resolve_executor_backend(clifft::internal::runtime_isa());
    output_.backend_ = backend_;
    initialize_program();
    prepare_noise_and_boundaries();
    lower_action_stream();
    prepare_cache_blocked_rotation_regions();
    prepare_batch_compaction_costs();
    build_expression_dependencies();
    output_.batch_presampled_program_ = BatchPresampledProgram::build(
        output_, source_, expression_terms_, expression_term_begins_, bound_presampled_symbols_);
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
    lowered_action_ranges_.reserve(source_.actions.size());
    output_.has_postselection_ = storage.has_postselection;
    if (output_.has_postselection_) {
        action_batch_lane_work_.reserve(source_.actions.size());
    }
    if (source_.source_map.has_value()) {
        output_.action_plan_ranges_.reserve(source_.actions.size());
    }
    output_.expression_register_constants_.reserve(source_.actions.size());
    // The storage prepass avoids repeated growth of the temporary term tape
    // and identifies plans that need reverse compaction metadata.
    expression_terms_.reserve(storage.expression_terms);
    expression_term_begins_.reserve(source_.actions.size());
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
                              std::is_same_v<T, PromoteDormantRotation>) {
                    num_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, MeasureActivePauli> ||
                                     std::is_same_v<T, MeasureDormantRandom>) {
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
            output_.unbound_presampled_symbols_.push_back(symbol);
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

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::ensure_expression_term_capacity(
    size_t additional_terms) const {
    constexpr size_t kMaxExpressionTerms = std::numeric_limits<uint32_t>::max();
    // Dynamic fusion can replace source signs with denser basis expressions,
    // so the source-plan count is only a reserve estimate after lowering.
    if (additional_terms > kMaxExpressionTerms - expression_terms_.size()) {
        throw std::length_error("sampling executable expression storage exceeds uint32 range");
    }
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::PreparedExpression
ExecutablePlanBuilder::prepare_expression(const AffineBool& expression) {
    if (output_.expression_register_constants_.size() >= std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable expression count exceeds uint32 range");
    }
    ensure_expression_term_capacity(expression.terms().size());
    const uint32_t register_id =
        static_cast<uint32_t>(output_.expression_register_constants_.size());
    expression_term_begins_.push_back(static_cast<uint32_t>(expression_terms_.size()));
    output_.expression_register_constants_.push_back(static_cast<uint8_t>(expression.constant()));
    for (SymbolId term : expression.terms()) {
        expression_terms_.push_back(index(term));
    }
    return {register_id};
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::PreparedExpression
ExecutablePlanBuilder::prepare_measurement_correction(const AffineBool& outcome, uint32_t branch) {
    if (output_.expression_register_constants_.size() >= std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable expression count exceeds uint32 range");
    }
    ensure_expression_term_capacity(outcome.terms().size() - 1);
    const uint32_t register_id =
        static_cast<uint32_t>(output_.expression_register_constants_.size());
    const uint32_t begin = static_cast<uint32_t>(expression_terms_.size());
    expression_term_begins_.push_back(begin);
    output_.expression_register_constants_.push_back(static_cast<uint8_t>(outcome.constant()));
    for (SymbolId term : outcome.terms()) {
        if (index(term) != branch) {
            expression_terms_.push_back(index(term));
        }
    }
    assert(expression_terms_.size() == static_cast<size_t>(begin) + outcome.terms().size() - 1 &&
           "validated measurement outcome must contain its branch exactly once");
    return {register_id};
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
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
            }
        },
        planned.action);
    record_batch_lane_work(planned_batch_lane_work(planned));
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

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::prepare_cache_blocked_rotation_regions() {
    if (!experimental_cache_blocking_enabled() || backend_ != ExecutorBackend::Avx512) {
        lowered_action_ranges_.clear();
        return;
    }

    assert(lowered_action_ranges_.size() == output_.actions_.size() &&
           "cache-block preparation requires one source range per executable action");
    constexpr uint32_t kNoRegion = std::numeric_limits<uint32_t>::max();
    output_.cache_blocked_region_by_action_.assign(output_.actions_.size(), kNoRegion);
    const uint32_t min_active_width = experimental_cache_block_min_active_width();

    size_t action_begin = 0;
    while (action_begin < output_.actions_.size()) {
        const ExecutablePlan::PlanActionRange& first_range = lowered_action_ranges_[action_begin];
        const PlannedAction& first_planned = source_.actions[first_range.begin];
        const uint32_t active_width = first_planned.active_before;
        if (active_width < min_active_width) {
            ++action_begin;
            continue;
        }

        CacheBlockBasis basis;
        for (uint32_t lane = 0; lane < kAvx512LaneIndexBitsForCacheBlock; ++lane) {
            [[maybe_unused]] const bool inserted =
                basis.insert(uint64_t{1} << lane, kExperimentalCacheBlockRank);
            assert(inserted && "AVX-512 lane coordinates must fit the cache block");
        }

        size_t action_end = action_begin;
        uint32_t expected_plan_begin = first_range.begin;
        size_t source_rotation_count = 0;
        while (action_end < output_.actions_.size()) {
            const ExecutablePlan::PlanActionRange& range = lowered_action_ranges_[action_end];
            if (range.begin != expected_plan_begin) {
                break;
            }

            CacheBlockBasis candidate = basis;
            bool eligible = true;
            for (uint32_t plan_index = range.begin; plan_index < range.end; ++plan_index) {
                const PlannedAction& planned = source_.actions[plan_index];
                const auto* rotation = std::get_if<RotateActivePauli>(&planned.action);
                if (rotation == nullptr || planned.active_before != active_width ||
                    planned.active_after != active_width ||
                    !candidate.insert(rotation->pauli.x, kExperimentalCacheBlockRank)) {
                    eligible = false;
                    break;
                }
            }
            if (!eligible) {
                break;
            }

            basis = candidate;
            source_rotation_count += range.end - range.begin;
            expected_plan_begin = range.end;
            ++action_end;
        }

        if (action_end - action_begin < kExperimentalCacheBlockMinExecutablePasses) {
            ++action_begin;
            continue;
        }

        ExecutablePlan::PreparedCacheBlockedRotationRegion region;
        region.action_begin = static_cast<uint32_t>(action_begin);
        region.action_end = static_cast<uint32_t>(action_end);
        region.active_width = active_width;
        region.block_rank = basis.rank();
        region.basis_masks = basis.rows();
        region.basis_pivots.reserve(region.basis_masks.size());
        for (uint64_t row : region.basis_masks) {
            region.basis_pivots.push_back(
                static_cast<uint32_t>(std::countr_zero(std::bit_floor(row))));
        }
        region.operations.reserve(source_rotation_count);
        uint64_t block_pivot_mask = 0;
        for (uint32_t pivot : region.basis_pivots) {
            block_pivot_mask |= uint64_t{1} << pivot;
        }

        struct LocalizedRotation {
            PreparedRotation rotation;
            AffineBool sign;
            uint64_t external_z = 0;
            bool phase_flip = false;
        };
        for (size_t action_index = action_begin; action_index < action_end; ++action_index) {
            const ExecutablePlan::PlanActionRange& range = lowered_action_ranges_[action_index];
            std::vector<LocalizedRotation> localized;
            localized.reserve(range.end - range.begin);
            for (uint32_t plan_index = range.begin; plan_index < range.end; ++plan_index) {
                const PlannedAction& planned = source_.actions[plan_index];
                const auto& source_rotation = std::get<RotateActivePauli>(planned.action);
                const PreparedRotation global_rotation = prepare_rotation(
                    source_rotation.pauli, active_width, source_rotation.half_turns);
                const uint64_t local_x = basis.coordinates(source_rotation.pauli.x);
                uint64_t local_z = 0;
                for (size_t coordinate = 0; coordinate < region.basis_masks.size(); ++coordinate) {
                    local_z |= static_cast<uint64_t>(std::popcount(region.basis_masks[coordinate] &
                                                                   source_rotation.pauli.z) &
                                                     1U)
                               << coordinate;
                }
                const PreparedPauli local_pauli =
                    prepare_pauli({local_x, local_z}, region.block_rank);
                const std::complex<double> phase_ratio =
                    global_rotation.pauli.even_phase * std::conj(local_pauli.even_phase);
                assert(std::abs(phase_ratio.imag()) < 1e-12 &&
                       std::abs(std::abs(phase_ratio.real()) - 1.0) < 1e-12 &&
                       "basis localization must preserve a Hermitian Pauli up to sign");
                localized.push_back({{local_pauli, global_rotation.cosine, global_rotation.sine},
                                     source_rotation.sign,
                                     source_rotation.pauli.z,
                                     phase_ratio.real() < 0.0});
            }

            const bool was_fused =
                std::holds_alternative<ExecutablePlan::ExecuteFusedRotation>(
                    output_.actions_[action_index]) ||
                std::holds_alternative<ExecutablePlan::ExecuteDynamicFusedRotation>(
                    output_.actions_[action_index]);
            CacheBlockBasis orbit_basis;
            CacheSignBasis sign_basis;
            bool can_fuse = was_fused;
            for (const LocalizedRotation& rotation : localized) {
                can_fuse &= orbit_basis.insert(rotation.rotation.pauli.x, 2);
                can_fuse &= sign_basis.insert(rotation.sign, kExperimentalCacheBlockMaxFusedSigns);
            }

            const std::vector<uint64_t> orbit_masks = orbit_basis.rows();
            uint64_t orbit_pivot_mask = 0;
            for (uint64_t mask : orbit_masks) {
                orbit_pivot_mask |= std::bit_floor(mask);
            }
            CacheSelectorBasis selector_basis;
            for (const LocalizedRotation& rotation : localized) {
                can_fuse &= selector_basis.insert({rotation.rotation.pauli.z & ~orbit_pivot_mask,
                                                   rotation.external_z & ~block_pivot_mask},
                                                  kExperimentalCacheBlockMaxFusedSelectors);
            }

            if (can_fuse) {
                const std::vector<CacheSelector> selector_rows = selector_basis.rows();
                std::vector<uint64_t> local_selector_masks;
                std::vector<uint64_t> external_selector_masks;
                local_selector_masks.reserve(selector_rows.size());
                external_selector_masks.reserve(selector_rows.size());
                for (const CacheSelector& row : selector_rows) {
                    local_selector_masks.push_back(row.local);
                    external_selector_masks.push_back(row.external);
                }

                std::vector<uint32_t> sign_coordinates;
                std::vector<uint32_t> selector_coordinates;
                sign_coordinates.reserve(localized.size());
                selector_coordinates.reserve(localized.size());
                for (const LocalizedRotation& rotation : localized) {
                    sign_coordinates.push_back(sign_basis.coordinates(rotation.sign));
                    selector_coordinates.push_back(
                        selector_basis.coordinates({rotation.rotation.pauli.z & ~orbit_pivot_mask,
                                                    rotation.external_z & ~block_pivot_mask}));
                }

                ExecutablePlan::PreparedCacheBlockedFusedRotation fused;
                fused.external_selector_masks = std::move(external_selector_masks);
                for (const AffineBool& expression : sign_basis.expressions()) {
                    fused.sign_basis.push_back(prepare_expression(expression));
                }
                const size_t num_sign_variants = size_t{1} << fused.sign_basis.size();
                fused.variants.reserve(num_sign_variants);
                std::vector<PreparedFusedRotationTerm> terms;
                terms.reserve(localized.size());
                for (size_t variant = 0; variant < num_sign_variants; ++variant) {
                    terms.clear();
                    for (size_t i = 0; i < localized.size(); ++i) {
                        const LocalizedRotation& rotation = localized[i];
                        const bool sign =
                            rotation.sign.constant() ^ rotation.phase_flip ^
                            ((std::popcount(static_cast<uint32_t>(variant) & sign_coordinates[i]) &
                              1U) != 0);
                        terms.push_back({rotation.rotation, selector_coordinates[i], sign});
                    }
                    fused.variants.emplace_back(
                        prepare_fused_rotation_from_terms(region.block_rank, orbit_masks,
                                                          local_selector_masks, terms),
                        backend_);
                }
                region.operations.emplace_back(std::move(fused));
                continue;
            }

            for (LocalizedRotation& rotation : localized) {
                const DirectRotationKernel kernel =
                    rotation.rotation.pauli.is_identity()
                        ? DirectRotationKernel::Scalar
                        : resolve_direct_rotation_kernel(rotation.rotation, backend_);
                region.operations.emplace_back(ExecutablePlan::PreparedCacheBlockedRotation{
                    std::move(rotation.rotation), prepare_expression(rotation.sign),
                    rotation.external_z, rotation.phase_flip, kernel});
            }
        }

        const uint32_t region_index =
            static_cast<uint32_t>(output_.cache_blocked_rotation_regions_.size());
        output_.cache_blocked_region_by_action_[action_begin] = region_index;
        output_.cache_blocked_rotation_regions_.push_back(std::move(region));
        action_begin = action_end;
    }

    if (output_.cache_blocked_rotation_regions_.empty()) {
        output_.cache_blocked_region_by_action_.clear();
    }
    lowered_action_ranges_.clear();
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::prepare_batch_compaction_costs() {
    output_.estimated_batch_lane_work_ = estimated_batch_lane_work_;
    if (!output_.has_postselection_) {
        assert(action_batch_lane_work_.empty() &&
               "ordinary plans must not retain compaction metadata");
        return;
    }
    assert(action_batch_lane_work_.size() == output_.actions_.size() &&
           "postselected batch work must parallel executable actions");
    batch_detail::BatchLaneWork remaining_lane_work;
    for (size_t index = output_.actions_.size(); index-- > 0;) {
        if (auto* detector =
                std::get_if<ExecutablePlan::ExecuteDetector>(&output_.actions_[index])) {
            detector->remaining_batch_lane_work = remaining_lane_work;
        }
        remaining_lane_work =
            add_batch_lane_work(remaining_lane_work, action_batch_lane_work_[index]);
    }
    assert(remaining_lane_work.common == estimated_batch_lane_work_.all_widths.common &&
           remaining_lane_work.row_output == estimated_batch_lane_work_.all_widths.row_output &&
           "forward and reverse batch work totals must agree");
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::record_action_origin(uint32_t plan_begin,
                                                                             uint32_t plan_end) {
    assert(plan_begin < plan_end && plan_end <= source_.actions.size() &&
           "executable action must name a nonempty plan range");
    assert(lowered_action_ranges_.size() + 1 == output_.actions_.size() &&
           "each executable action must retain one construction-time source range");
    lowered_action_ranges_.push_back({plan_begin, plan_end});
    if (source_.source_map.has_value()) {
        assert(output_.action_plan_ranges_.size() + 1 == output_.actions_.size() &&
               "each inspected executable action must receive one source range");
        output_.action_plan_ranges_.push_back({plan_begin, plan_end});
    }
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::build_expression_dependencies() {
    output_.expression_dependencies_ = ExecutablePlan::ExpressionDependencies::build(
        output_.num_symbols_, expression_terms_, expression_term_begins_);
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::validate_executable_plan() const {
#ifndef NDEBUG
    assert(expression_term_begins_.size() == output_.expression_register_constants_.size() &&
           "expression register storage is inconsistent");
    output_.expression_dependencies_.validate(output_.num_symbols_,
                                              output_.expression_register_constants_.size());
    if (output_.batch_presampled_program_.has_value()) {
        output_.batch_presampled_program_->validate(output_.noise_outcomes_.size(),
                                                    output_.expression_register_constants_.size());
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
    if (output_.cache_blocked_rotation_regions_.empty()) {
        assert(output_.cache_blocked_region_by_action_.empty() &&
               "an empty cache-block plan must not retain an action index");
    } else {
        assert(output_.cache_blocked_region_by_action_.size() == output_.actions_.size() &&
               "cache-block action indices must parallel the fallback stream");
    }
    uint32_t previous_end = 0;
    for (size_t region_index = 0; region_index < output_.cache_blocked_rotation_regions_.size();
         ++region_index) {
        const auto& region = output_.cache_blocked_rotation_regions_[region_index];
        assert(region.action_begin >= previous_end && region.action_begin < region.action_end &&
               region.action_end <= output_.actions_.size() &&
               "cache-blocked fallback ranges must be ordered and nonempty");
        assert(region.block_rank == region.basis_masks.size() &&
               region.block_rank == region.basis_pivots.size() &&
               region.block_rank <= kExperimentalCacheBlockRank &&
               "cache-block geometry must fit its prepared rank");
        assert(output_.cache_blocked_region_by_action_[region.action_begin] == region_index &&
               "cache-block action index must select its prepared region");
        for (const auto& operation : region.operations) {
            std::visit(
                [&](const auto& prepared) {
                    using Operation = std::decay_t<decltype(prepared)>;
                    if constexpr (std::is_same_v<Operation,
                                                 ExecutablePlan::PreparedCacheBlockedRotation>) {
                        assert(prepared.rotation.pauli.active_width == region.block_rank &&
                               "localized rotation width must match its cache block");
                        validate_expression(prepared.sign);
                    } else {
                        assert(!prepared.variants.empty() &&
                               prepared.external_selector_masks.size() <=
                                   kExperimentalCacheBlockMaxFusedSelectors &&
                               "cache-block fused variants must fit their controls");
                        for (ExecutablePlan::PreparedExpression sign : prepared.sign_basis) {
                            validate_expression(sign);
                        }
                        for (const PreparedFusedRotationExecution& variant : prepared.variants) {
                            assert(variant.rotation().active_width == region.block_rank &&
                                   variant.rotation().selector_masks.size() ==
                                       prepared.external_selector_masks.size() &&
                                   "cache-block fused geometry must match its region");
                        }
                    }
                },
                operation);
        }
        previous_end = region.action_end;
    }
#endif
}

#undef CLIFFT_BUILDER_FORCE_INLINE

}  // namespace clifft::sampling
