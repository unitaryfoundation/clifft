#include "clifft/sampling/executor.h"

#include "clifft/util/noise_sampling.h"
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

uint32_t index(DetectorSlot slot) {
    return static_cast<uint32_t>(slot);
}

uint32_t index(ObservableSlot slot) {
    return static_cast<uint32_t>(slot);
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
      num_detectors_(plan.num_detectors),
      num_observables_(plan.num_observables),
      has_postselection_(plan.has_postselection),
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
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    num_expression_terms += typed.source.terms().size();
                } else if constexpr (std::is_same_v<T, WriteDetector> ||
                                     std::is_same_v<T, WriteObservable>) {
                    num_expression_terms += typed.outcome.terms().size();
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
    std::vector<bool> bound_presampled(plan.symbols.size(), false);
    noise_sites_.reserve(plan.presampled_noise_sites.size());
    noise_hazards_.reserve(plan.presampled_noise_sites.size());
    double cumulative_hazard = 0.0;
    for (const PresampledNoiseSite& site : plan.presampled_noise_sites) {
        const uint32_t begin = static_cast<uint32_t>(noise_outcomes_.size());
        double cumulative_probability = 0.0;
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            cumulative_probability += outcome.probability;
            noise_outcomes_.push_back({index(outcome.symbol), cumulative_probability});
            bound_presampled[index(outcome.symbol)] = true;
        }
        noise_sites_.push_back(
            {begin, static_cast<uint32_t>(noise_outcomes_.size()) - begin, cumulative_probability});
        cumulative_hazard += bernoulli_hazard(cumulative_probability);
        noise_hazards_.push_back(cumulative_hazard);
    }
    for (uint32_t symbol = 0; symbol < plan.symbols.size(); ++symbol) {
        if (plan.symbols[symbol].kind == SymbolKind::Presampled) {
            presampled_symbols_.push_back(symbol);
            if (!bound_presampled[symbol]) {
                unbound_presampled_symbols_.push_back(symbol);
            }
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
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    has_readout_noise_ = true;
                    actions_.emplace_back(ExecuteReadoutNoise{
                        prepare_expression(typed.source), index(typed.flip), index(typed.record),
                        typed.prob_zero_to_one, typed.prob_one_to_zero});
                } else if constexpr (std::is_same_v<T, WriteDetector>) {
                    actions_.emplace_back(ExecuteDetector{prepare_expression(typed.outcome),
                                                          index(typed.detector),
                                                          typed.postselected});
                } else if constexpr (std::is_same_v<T, WriteObservable>) {
                    actions_.emplace_back(ExecuteObservable{prepare_expression(typed.outcome),
                                                            index(typed.observable)});
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
      detectors_(plan.num_detectors_, 0),
      observables_(plan.num_observables_, 0),
      rng_(seed) {
    previous_presampled_ones_.reserve(plan.presampled_symbols_.size());
}

void Executor::run_shot() noexcept {
    assert(plan_.unbound_presampled_symbols_.empty() &&
           "automatic execution requires every presampled symbol to have a distribution");
    reset_shot();
    sample_presampled_noise();
    (void)execute_actions<false>({});
}

void Executor::run_shot(std::span<const uint8_t> presampled_values) noexcept {
    reset_shot();
    assign_presampled_values(presampled_values);
    (void)execute_actions<false>({});
}

ReplayResult Executor::replay_shot(std::span<const uint8_t> forced_records,
                                   std::span<const uint8_t> presampled_values) noexcept {
    assert(forced_records.size() == records_.size() &&
           "one forced value is required for every plan record");
    assert(std::ranges::all_of(forced_records, [](uint8_t value) { return value <= 1; }) &&
           "forced records must be Boolean");
    reset_shot();
    assign_presampled_values(presampled_values);
    return execute_actions<true>(forced_records);
}

void Executor::reset_shot() noexcept {
    state_.reset();
    // Validation guarantees that every mutable symbol and output is overwritten
    // before use on a completed shot. Only nonfiring noise symbols need restoring.
    for (uint32_t symbol : previous_presampled_ones_) {
        assert(symbol < symbols_.size() && symbols_[symbol] == 1 &&
               "tracked presampled symbols must be set");
        symbols_[symbol] = 0;
    }
    previous_presampled_ones_.clear();
    discarded_ = false;
}

void Executor::assign_presampled_values(std::span<const uint8_t> presampled_values) noexcept {
    assert(presampled_values.size() == plan_.presampled_symbols_.size() &&
           "one value is required for every presampled symbol");
    for (size_t i = 0; i < presampled_values.size(); ++i) {
        assert(presampled_values[i] <= 1 && "presampled symbols must be Boolean");
        const uint32_t symbol = plan_.presampled_symbols_[i];
        symbols_[symbol] = presampled_values[i];
        if (presampled_values[i] != 0) {
            previous_presampled_ones_.push_back(symbol);
        }
    }
}

void Executor::sample_presampled_noise() noexcept {
    uint32_t first_candidate = 0;
    while (first_candidate < plan_.noise_sites_.size()) {
        const double current_hazard =
            first_candidate == 0 ? 0.0 : plan_.noise_hazards_[first_candidate - 1];
        if (current_hazard >= plan_.noise_hazards_.back()) {
            return;
        }
        const uint32_t site_index =
            sample_next_noise_site(plan_.noise_hazards_, first_candidate, rng_.next_double());
        if (site_index == kNoNoiseSite) {
            return;
        }
        const ExecutablePlan::PreparedNoiseSite& site = plan_.noise_sites_[site_index];
        assert(site.outcome_count > 0 && site.total_probability > 0.0 &&
               "a sampled hazard must identify a nonempty noise site");
        uint32_t outcome_index = site.outcome_begin;
        if (site.outcome_count > 1) {
            const double channel_draw = rng_.next_double() * site.total_probability;
            while (channel_draw >= plan_.noise_outcomes_[outcome_index].cumulative_probability) {
                ++outcome_index;
                assert(outcome_index < site.outcome_begin + site.outcome_count &&
                       "channel draw must select one prepared outcome");
            }
        }
        const uint32_t symbol = plan_.noise_outcomes_[outcome_index].symbol;
        assert(symbol < symbols_.size() && symbols_[symbol] == 0 &&
               "a noise site may define only one fresh symbol per shot");
        symbols_[symbol] = 1;
        previous_presampled_ones_.push_back(symbol);
        first_candidate = site_index + 1;
    }
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteRotation& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    apply_rotation(state_, action.rotation, evaluate(action.sign));
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecutePromotion& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    apply_promotion(state_, action.promotion, evaluate(action.sign));
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    const MeasurementProbabilities probabilities =
        measurement_probabilities(state_, action.measurement);
    const bool correction = evaluate(action.correction);
    bool branch = false;
    if constexpr (ForceRecords) {
        branch = (forced_records[action.record] != 0) ^ correction;
        const std::optional<double> log_increment = force_active_branch(probabilities, branch);
        if (!log_increment.has_value()) {
            result.reachable = false;
            return;
        }
        result.log_probability += *log_increment;
    } else {
        branch = sample_active_branch(probabilities);
    }
    symbols_[action.branch] = static_cast<uint8_t>(branch);
    collapse_measurement(state_, action.measurement, branch, probabilities.for_branch(branch));
    records_[action.record] = static_cast<uint8_t>(branch ^ correction);
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    const bool correction = evaluate(action.correction);
    bool branch = false;
    if constexpr (ForceRecords) {
        branch = (forced_records[action.record] != 0) ^ correction;
        result.log_probability += kLogHalf;
    } else {
        branch = sample_dormant_branch();
    }
    symbols_[action.branch] = static_cast<uint8_t>(branch);
    records_[action.record] = static_cast<uint8_t>(branch ^ correction);
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteClassicalRecord& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    records_[action.record] = static_cast<uint8_t>(evaluate(action.outcome));
    if constexpr (ForceRecords) {
        if (records_[action.record] != forced_records[action.record]) {
            result.reachable = false;
        }
    }
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    symbols_[action.symbol] = static_cast<uint8_t>(evaluate(action.value));
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteReadoutNoise& action,
                              std::span<const uint8_t>, ReplayResult& result) noexcept {
    if constexpr (ForceRecords) {
        result.reachable = false;
    } else {
        const bool source = evaluate(action.source);
        assert(records_[action.record] == static_cast<uint8_t>(source) &&
               "readout source must match the current record value");
        const double probability = source ? action.prob_one_to_zero : action.prob_zero_to_one;
        const bool flip =
            probability >= 1.0 || (probability > 0.0 && rng_.next_double() < probability);
        symbols_[action.flip] = static_cast<uint8_t>(flip);
        records_[action.record] ^= static_cast<uint8_t>(flip);
    }
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteDetector& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    const bool outcome = evaluate(action.outcome);
    detectors_[action.detector] = static_cast<uint8_t>(outcome);
    discarded_ |= action.postselected && outcome;
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteObservable& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    observables_[action.observable] = static_cast<uint8_t>(evaluate(action.outcome));
}

template <bool ForceRecords>
ReplayResult Executor::execute_actions(std::span<const uint8_t> forced_records) noexcept {
    ReplayResult result;
    for (const ExecutablePlan::Action& action : plan_.actions_) {
        std::visit(
            [&](const auto& typed) noexcept {
                execute_action<ForceRecords>(typed, forced_records, result);
            },
            action);
        if constexpr (ForceRecords) {
            if (!result.reachable) {
                return result;
            }
        }
        if (discarded_) {
            return result;
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

SamplingResult sample(const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed) {
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "batch sampling requires a distribution for every presampled symbol");
    }
    if (plan.has_postselection()) {
        throw std::invalid_argument(
            "fixed-row sampling does not support postselection; use sample_survivors");
    }

    auto checked_size = [shots](size_t stride) {
        if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
            throw std::length_error("sampling output size exceeds size_t range");
        }
        return static_cast<size_t>(shots) * stride;
    };

    SamplingResult result;
    result.measurements.resize(checked_size(plan.num_visible_records()));
    result.detectors.resize(checked_size(plan.num_detectors()));
    result.observables.resize(checked_size(plan.num_observables()));
    if (shots == 0) {
        return result;
    }

    auto run = [&](Executor& executor) {
        for (uint32_t shot = 0; shot < shots; ++shot) {
            executor.run_shot();
            std::ranges::copy(executor.visible_records(),
                              result.measurements.begin() +
                                  static_cast<size_t>(shot) * plan.num_visible_records());
            std::ranges::copy(
                executor.detectors(),
                result.detectors.begin() + static_cast<size_t>(shot) * plan.num_detectors());
            std::ranges::copy(
                executor.observables(),
                result.observables.begin() + static_cast<size_t>(shot) * plan.num_observables());
        }
    };
    if (seed.has_value()) {
        Executor executor(plan, *seed);
        run(executor);
    } else {
        Executor executor(plan);
        executor.reseed_from_entropy();
        run(executor);
    }
    return result;
}

std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed) {
    return sample(plan, shots, seed).measurements;
}

SamplingSurvivorResult sample_survivors(const ExecutablePlan& plan, uint32_t shots,
                                        std::optional<uint64_t> seed, bool keep_records) {
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "survivor sampling requires a distribution for every presampled symbol");
    }

    SamplingSurvivorResult result;
    result.total_shots = shots;
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(plan.num_observables(), 0);
    if (keep_records) {
        auto checked_reserve = [shots](size_t stride) {
            if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
                throw std::length_error("survivor output size exceeds size_t range");
            }
            return static_cast<size_t>(shots) * stride;
        };
        result.measurements.reserve(checked_reserve(plan.num_visible_records()));
        result.detectors.reserve(checked_reserve(plan.num_detectors()));
        result.observables.reserve(checked_reserve(plan.num_observables()));
    }
    auto run = [&](Executor& executor) {
        for (uint32_t shot = 0; shot < shots; ++shot) {
            executor.run_shot();
            if (executor.discarded()) {
                continue;
            }
            ++result.passed_shots;
            bool logical_error = false;
            for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
                const bool value = executor.observables()[observable] != 0;
                result.observable_ones[observable] += static_cast<uint64_t>(value);
                logical_error |= value;
            }
            result.logical_errors += static_cast<uint32_t>(logical_error);
            if (keep_records) {
                result.measurements.insert(result.measurements.end(),
                                           executor.visible_records().begin(),
                                           executor.visible_records().end());
                result.detectors.insert(result.detectors.end(), executor.detectors().begin(),
                                        executor.detectors().end());
                result.observables.insert(result.observables.end(), executor.observables().begin(),
                                          executor.observables().end());
            }
        }
    };
    if (seed.has_value()) {
        Executor executor(plan, *seed);
        run(executor);
    } else {
        Executor executor(plan);
        executor.reseed_from_entropy();
        run(executor);
    }
    return result;
}

std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                             std::span<const uint8_t> forced_records,
                                             size_t num_records) {
    if (plan.num_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "record probabilities do not yet support plans with presampled symbols");
    }
    if (plan.has_readout_noise()) {
        throw std::invalid_argument("record probabilities do not yet support readout noise");
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
