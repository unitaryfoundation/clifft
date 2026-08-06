#include "clifft/sampling/executor.h"

#include "clifft/util/numeric.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace clifft::sampling {

namespace {

template <typename>
inline constexpr bool kAlwaysFalse = false;

constexpr double kLogHalf = -std::numbers::ln2;

uint32_t index(SymbolId id) {
    return static_cast<uint32_t>(id);
}

uint32_t index(RecordSlot slot) {
    return static_cast<uint32_t>(slot);
}

uint32_t index(InstrumentSiteId site) {
    return static_cast<uint32_t>(site);
}

}  // namespace

MeasurementBranchClassification classify_measurement_branch(
    MeasurementProbabilities probabilities) noexcept {
    const double total = probabilities.total();
    assert(is_finite_robust(probabilities.zero) && probabilities.zero >= 0.0 &&
           is_finite_robust(probabilities.one) && probabilities.one >= 0.0 &&
           is_finite_robust(total) && total > 0.0 &&
           "measurement probabilities must be finite, nonnegative, and nonzero");
    const double epsilon = kMeasurementDustEpsilon * total;
    if (probabilities.one <= epsilon) {
        return {.kind = MeasurementBranchKind::DeterministicZero,
                .clamped_dust = probabilities.one > 0.0};
    }
    if (probabilities.zero <= epsilon) {
        return {.kind = MeasurementBranchKind::DeterministicOne,
                .clamped_dust = probabilities.zero > 0.0};
    }
    return {.kind = MeasurementBranchKind::Random};
}

ExecutablePlan::ExecutablePlan(const SamplingPlan& plan)
    : num_qubits_(plan.num_qubits),
      initial_active_width_(plan.initial_active_width),
      max_active_width_(plan.max_active_width),
      num_visible_records_(plan.num_visible_records),
      num_hidden_records_(plan.num_hidden_records),
      global_weight_(plan.global_weight) {
    plan.validate();
    if (plan.symbols.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable symbol count exceeds uint32 range");
    }
    num_symbols_ = static_cast<uint32_t>(plan.symbols.size());

    size_t num_expression_terms = 0;
    for (const PlannedAction& planned : plan.actions) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, RotateActivePauli> ||
                              std::is_same_v<T, PromoteDormantRotation>) {
                    num_expression_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, MeasureActivePauli> ||
                                     std::is_same_v<T, MeasureDormantRandom>) {
                    // The current measurement branch is stored separately so
                    // replay can solve for it from the requested record.
                    num_expression_terms += typed.outcome.terms().size() - 1;
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    num_expression_terms += typed.outcome.terms().size();
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    num_expression_terms += typed.value.terms().size();
                } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                    // Rejected below after all structural validation has run.
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
                }
            },
            planned.action);
    }
    if (num_expression_terms > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable expression storage exceeds uint32 range");
    }
    expression_terms_.reserve(num_expression_terms);
    actions_.reserve(plan.actions.size());
    presampled_symbols_.reserve(plan.symbols.size());
    for (uint32_t symbol = 0; symbol < plan.symbols.size(); ++symbol) {
        if (plan.symbols[symbol].kind == SymbolKind::Presampled) {
            presampled_symbols_.push_back(symbol);
        }
    }

    for (const PlannedAction& planned : plan.actions) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, RotateActivePauli>) {
                    actions_.emplace_back(ExecuteRotation{
                        prepare_rotation(typed.pauli, planned.active_before, typed.half_turns),
                        prepare_expression(typed.sign)});
                } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                    actions_.emplace_back(ExecutePromotion{prepare_promotion(typed.half_turns),
                                                           prepare_expression(typed.sign)});
                } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                    actions_.emplace_back(ExecuteActiveMeasurement{
                        prepare_measurement(typed.pauli, planned.active_before, typed.active_pivot),
                        prepare_measurement_correction(typed.outcome, index(typed.branch)),
                        index(typed.branch), index(typed.record)});
                } else if constexpr (std::is_same_v<T, MeasureDormantRandom>) {
                    actions_.emplace_back(ExecuteDormantMeasurement{
                        prepare_measurement_correction(typed.outcome, index(typed.branch)),
                        index(typed.branch), index(typed.record)});
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    actions_.emplace_back(ExecuteClassicalRecord{prepare_expression(typed.outcome),
                                                                 index(typed.record)});
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    actions_.emplace_back(ExecuteSymbolDefinition{prepare_expression(typed.value),
                                                                  index(typed.symbol)});
                } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                    throw std::invalid_argument(
                        "sampling executable does not yet support instrument boundary site " +
                        std::to_string(index(typed.site)));
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
                }
            },
            planned.action);
    }
}

ExecutablePlan::PreparedExpression ExecutablePlan::prepare_expression(
    const AffineBool& expression) {
    const uint32_t begin = static_cast<uint32_t>(expression_terms_.size());
    for (SymbolId term : expression.terms()) {
        expression_terms_.push_back(index(term));
    }
    return {begin, static_cast<uint32_t>(expression.terms().size()), expression.constant()};
}

ExecutablePlan::PreparedExpression ExecutablePlan::prepare_measurement_correction(
    const AffineBool& outcome, uint32_t branch) {
    const uint32_t begin = static_cast<uint32_t>(expression_terms_.size());
    for (SymbolId term : outcome.terms()) {
        if (index(term) != branch) {
            expression_terms_.push_back(index(term));
        }
    }
    assert(expression_terms_.size() == static_cast<size_t>(begin) + outcome.terms().size() - 1 &&
           "validated measurement outcome must contain its branch exactly once");
    return {begin, static_cast<uint32_t>(outcome.terms().size() - 1), outcome.constant()};
}

Executor::Executor(const ExecutablePlan& plan, uint64_t seed)
    : plan_(plan),
      state_(plan.max_active_width_, plan.initial_active_width_, plan.global_weight_),
      symbols_(plan.num_symbols_, 0),
      records_(static_cast<size_t>(plan.num_visible_records_) + plan.num_hidden_records_, 0),
      rng_(seed) {}

Executor::Executor(const ExecutablePlan& plan, std::nullopt_t) : Executor(plan, uint64_t{0}) {
    rng_.seed_from_entropy();
}

void Executor::run_shot(std::span<const uint8_t> presampled_values) noexcept {
    initialize_shot(presampled_values);
    (void)execute_actions<false>({});
}

ReplayResult Executor::replay_shot(std::span<const uint8_t> forced_records,
                                   std::span<const uint8_t> presampled_values) noexcept {
    assert(forced_records.size() == records_.size() &&
           "one forced value is required for every plan record");
    assert(std::ranges::all_of(forced_records, [](uint8_t value) { return value <= 1; }) &&
           "forced records must be Boolean");
    initialize_shot(presampled_values);
    return execute_actions<true>(forced_records);
}

void Executor::initialize_shot(std::span<const uint8_t> presampled_values) noexcept {
    assert(presampled_values.size() == plan_.presampled_symbols_.size() &&
           "one value is required for every presampled symbol");
    state_.reset();
    std::fill(symbols_.begin(), symbols_.end(), uint8_t{0});
    std::fill(records_.begin(), records_.end(), uint8_t{0});
    for (size_t i = 0; i < presampled_values.size(); ++i) {
        assert(presampled_values[i] <= 1 && "presampled symbols must be Boolean");
        symbols_[plan_.presampled_symbols_[i]] = presampled_values[i];
    }
}

template <bool ForceRecords>
ReplayResult Executor::execute_actions(std::span<const uint8_t> forced_records) noexcept {
    ReplayResult result;
    for (const ExecutablePlan::Action& action : plan_.actions_) {
        std::visit(
            [&](const auto& typed) noexcept {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation>) {
                    apply_rotation(state_, typed.rotation, evaluate(typed.sign));
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecutePromotion>) {
                    apply_promotion(state_, typed.promotion, evaluate(typed.sign));
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement>) {
                    const MeasurementProbabilities probabilities =
                        measurement_probabilities(state_, typed.measurement);
                    const bool correction = evaluate(typed.correction);
                    bool branch = false;
                    if constexpr (ForceRecords) {
                        branch = (forced_records[typed.record] != 0) ^ correction;
                        const std::optional<double> log_increment =
                            force_active_branch(probabilities, branch);
                        if (!log_increment.has_value()) {
                            result.reachable = false;
                            return;
                        }
                        result.log_probability += *log_increment;
                    } else {
                        branch = sample_active_branch(probabilities);
                    }
                    symbols_[typed.branch] = static_cast<uint8_t>(branch);
                    collapse_measurement(state_, typed.measurement, branch,
                                         probabilities.for_branch(branch));
                    records_[typed.record] = static_cast<uint8_t>(branch ^ correction);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDormantMeasurement>) {
                    const bool correction = evaluate(typed.correction);
                    bool branch = false;
                    if constexpr (ForceRecords) {
                        branch = (forced_records[typed.record] != 0) ^ correction;
                        result.log_probability += kLogHalf;
                    } else {
                        branch = sample_dormant_branch();
                    }
                    symbols_[typed.branch] = static_cast<uint8_t>(branch);
                    records_[typed.record] = static_cast<uint8_t>(branch ^ correction);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteClassicalRecord>) {
                    records_[typed.record] = static_cast<uint8_t>(evaluate(typed.outcome));
                    if constexpr (ForceRecords) {
                        if (records_[typed.record] != forced_records[typed.record]) {
                            result.reachable = false;
                        }
                    }
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteSymbolDefinition>) {
                    symbols_[typed.symbol] = static_cast<uint8_t>(evaluate(typed.value));
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled executable action alternative");
                }
            },
            action);
        if constexpr (ForceRecords) {
            if (!result.reachable) {
                return result;
            }
        }
    }
    return result;
}

bool Executor::evaluate(ExecutablePlan::PreparedExpression expression) const noexcept {
    assert(static_cast<uint64_t>(expression.term_begin) + expression.term_count <=
               plan_.expression_terms_.size() &&
           "prepared affine expression must stay inside term storage");
    bool value = expression.constant;
    for (uint32_t i = 0; i < expression.term_count; ++i) {
        const uint32_t symbol = plan_.expression_terms_[expression.term_begin + i];
        assert(symbol < symbols_.size() && symbols_[symbol] <= 1 &&
               "prepared affine term must refer to a Boolean symbol");
        value ^= symbols_[symbol] != 0;
    }
    return value;
}

bool Executor::sample_active_branch(MeasurementProbabilities probabilities) noexcept {
    const MeasurementBranchClassification classification =
        classify_measurement_branch(probabilities);
    switch (classification.kind) {
        case MeasurementBranchKind::Random:
            return rng_.next_double() * probabilities.total() >= probabilities.zero;
        case MeasurementBranchKind::DeterministicZero:
            dust_clamps_ += static_cast<uint64_t>(classification.clamped_dust);
            return false;
        case MeasurementBranchKind::DeterministicOne:
            dust_clamps_ += static_cast<uint64_t>(classification.clamped_dust);
            return true;
    }
    assert(false && "unhandled measurement branch classification");
    return false;
}

std::optional<double> Executor::force_active_branch(MeasurementProbabilities probabilities,
                                                    bool branch) noexcept {
    const MeasurementBranchClassification classification =
        classify_measurement_branch(probabilities);
    dust_clamps_ += static_cast<uint64_t>(classification.clamped_dust);
    switch (classification.kind) {
        case MeasurementBranchKind::Random:
            return std::log(probabilities.for_branch(branch) / probabilities.total());
        case MeasurementBranchKind::DeterministicZero:
            return branch ? std::nullopt : std::optional<double>{0.0};
        case MeasurementBranchKind::DeterministicOne:
            return branch ? std::optional<double>{0.0} : std::nullopt;
    }
    assert(false && "unhandled measurement branch classification");
    return std::nullopt;
}

bool Executor::sample_dormant_branch() noexcept {
    return rng_.next_double() >= 0.5;
}

std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed) {
    if (plan.num_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "batch sampling does not yet support plans with presampled symbols");
    }
    const size_t stride = plan.num_visible_records();
    if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error("sampling output size exceeds size_t range");
    }
    std::vector<uint8_t> records(static_cast<size_t>(shots) * stride);
    if (shots == 0) {
        return records;
    }

    auto run = [&](Executor& executor) {
        for (uint32_t shot = 0; shot < shots; ++shot) {
            executor.run_shot();
            std::ranges::copy(executor.visible_records(), records.begin() + shot * stride);
        }
    };
    if (seed.has_value()) {
        Executor executor(plan, *seed);
        run(executor);
    } else {
        Executor executor(plan, std::nullopt);
        run(executor);
    }
    return records;
}

std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                             std::span<const uint8_t> forced_records,
                                             size_t num_records) {
    if (plan.num_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "record probabilities do not yet support plans with presampled symbols");
    }
    if (plan.num_hidden_records() != 0) {
        throw std::invalid_argument(
            "record probabilities do not yet support plans with hidden records");
    }
    const size_t stride = plan.num_visible_records();
    if (stride == 0) {
        throw std::invalid_argument(
            "record probabilities require a plan with at least one visible record");
    }
    if (num_records > std::numeric_limits<size_t>::max() / stride ||
        forced_records.size() != num_records * stride) {
        throw std::invalid_argument(
            "record buffer length must equal num_records times visible records");
    }
    if (!std::ranges::all_of(forced_records, [](uint8_t value) { return value <= 1; })) {
        throw std::invalid_argument("record bytes must be Boolean");
    }

    std::vector<double> log_probabilities(num_records);
    Executor executor(plan);
    for (size_t record = 0; record < num_records; ++record) {
        const ReplayResult replay =
            executor.replay_shot(forced_records.subspan(record * stride, stride));
        log_probabilities[record] =
            replay.reachable ? replay.log_probability : std::numeric_limits<double>::lowest();
    }
    return log_probabilities;
}

}  // namespace clifft::sampling
