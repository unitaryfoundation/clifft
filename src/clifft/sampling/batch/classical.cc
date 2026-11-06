#include "clifft/sampling/batch/classical.h"

#include "clifft/util/fault_sampling.h"
#include "clifft/util/noise_sampling.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <stdexcept>

namespace clifft::sampling {
namespace {

[[nodiscard]] const ExecutablePlan* validate_batch_plan(const ExecutablePlan& plan,
                                                        BatchSamplingMode sampling_mode) {
    if (plan.has_instruments()) {
        throw std::invalid_argument("packed sampling does not support instrument continuations");
    }
    if (sampling_mode != BatchSamplingMode::ExternalInputs &&
        plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "packed sampling requires distributions or external inputs "
            "for every presampled symbol");
    }
    return &plan;
}

}  // namespace

PreparedBatchNoise::PreparedBatchNoise(const ExecutablePlan& plan, BatchNoiseMode mode)
    : plan_(&plan), mode_(mode) {
    if (mode == BatchNoiseMode::SiteSparse || mode == BatchNoiseMode::SiteRows) {
        sites_.reserve(plan.noise_sites_.size());
        for (const auto& site : plan.noise_sites_) {
            const double probability =
                plan.noise_outcomes_[site.outcome_begin + site.outcome_count - 1]
                    .cumulative_probability;
            sites_.emplace_back(probability);
        }
    }
    if (mode == BatchNoiseMode::SiteRows) {
        row_stride_ = (plan.num_expression_registers() + 63) / 64;
        if (row_stride_ != 0 &&
            plan.noise_outcomes_.size() > outcome_rows_.max_size() / row_stride_) {
            throw std::length_error("packed noise outcome rows exceed addressable storage");
        }
        outcome_rows_.resize(plan.noise_outcomes_.size() * row_stride_);
        for (size_t outcome = 0; outcome < plan.noise_outcomes_.size(); ++outcome) {
            for (uint32_t reg : plan.expression_dependencies_.dependent_registers(
                     plan.noise_outcomes_[outcome].symbol)) {
                outcome_rows_[outcome * row_stride_ + reg / 64] |= uint64_t{1} << (reg & 63);
            }
        }
    }
}

BatchClassicalState::BatchClassicalState(const ExecutablePlan& plan, BatchOutputMode output_mode,
                                         BatchSamplingMode sampling_mode,
                                         const batch_detail::BatchWorkerStorageLayout& storage,
                                         const PreparedBatchNoise* noise)
    : noise_(noise),
      plan_(validate_batch_plan(plan, sampling_mode)),
      output_mode_(output_mode),
      sampling_mode_(sampling_mode),
      lane_capacity_(storage.lane_capacity),
      word_capacity_(storage.word_capacity),
      shot_indices_(storage.shot_index_entries),
      symbols_(storage.symbol_columns, storage.lane_capacity),
      batch_noise_carriers_(storage.noise_carrier_columns, storage.lane_capacity),
      expression_registers_(storage.expression_register_columns, storage.lane_capacity),
      records_(storage.record_columns, storage.lane_capacity),
      detectors_(storage.detector_columns, storage.lane_capacity),
      observables_(storage.observable_columns, storage.lane_capacity),
      forced_readout_(storage.forced_readout_columns, storage.lane_capacity),
      live_words_(storage.live_word_entries, 0),
      scratch_words_(storage.scratch_word_entries, 0) {
    if (noise && noise->plan_ != &plan) {
        throw std::invalid_argument("packed noise preparation must belong to the executor plan");
    }
    if (noise && noise->mode_ != BatchNoiseMode::Dense &&
        sampling_mode != BatchSamplingMode::Ordinary) {
        throw std::invalid_argument("sparse noise preparation requires ordinary sampling");
    }
    if (noise && noise->row_stride_ != 0) {
        if (lane_capacity_ > noise_rows_.max_size() / noise->row_stride_) {
            throw std::length_error("packed noise rows exceed addressable storage");
        }
        noise_rows_.resize(size_t{lane_capacity_} * noise->row_stride_);
    }
}

void BatchClassicalState::assign_external_inputs(const BatchExternalInputs& inputs) noexcept {
    for (uint32_t input = 0; input < plan_->presampled_symbols_.size(); ++input) {
        if (!plan_->batch_presampled_program_.has_value()) {
            symbols_.assign(plan_->presampled_symbols_[input], inputs.bits_.column(input),
                            live_words_);
            continue;
        }
        const auto& program = *plan_->batch_presampled_program_;
        const auto assignment = program.input_assignments_[input];
        if (assignment.count == 0) {
            continue;
        }
        const auto values = inputs.bits_.column(input);
        for (size_t word = 0; word < word_capacity_; ++word) {
            scratch_words_[word] = values[word] & live_words_[word];
        }
        for (uint32_t i = 0; i < assignment.count; ++i) {
            // Factorization is linear. Multiple supplied outcomes can share
            // a carrier, so their effects combine by XOR rather than OR.
            batch_noise_carriers_.xor_into(program.assigned_carriers_[assignment.begin + i],
                                           scratch_words_);
        }
    }
    finalize_presampled_symbols();
    for (size_t column = 0; column < forced_readout_.num_columns(); ++column) {
        forced_readout_.assign(column, inputs.bits_.column(inputs.noise_columns_ + column),
                               live_words_);
    }
}

void BatchClassicalState::reset_classical(const SeedRoot& root, uint32_t first_shot,
                                          uint32_t shots) noexcept {
    assert(shots <= lane_capacity_ && "packed batch must fit retained capacity");
    live_count_ = shots;
    fill_low_lane_mask(live_words_, shots);
    if (plan_->batch_presampled_program_.has_value()) {
        batch_noise_carriers_.clear();
    } else {
        symbols_.clear();
    }
    expression_registers_.clear();
    records_.clear();
    detectors_.clear();
    observables_.clear();
    if (sampling_mode_ == BatchSamplingMode::FixedFaults) {
        forced_readout_.clear();
    }
    active_lanes_ = shots;
    const std::array<uint64_t, 4> rng_words =
        derive_state(root, first_shot, kBatchSamplingExecutorDomain);
    rng_.seed_full(rng_words[0], rng_words[1], rng_words[2], rng_words[3]);
    for (uint32_t lane = 0; lane < shots; ++lane) {
        shot_indices_[lane] = first_shot + lane;
    }
    initialize_expression_registers();
}

void BatchClassicalState::apply_early_postselection(const EarlyPostselection& checks) noexcept {
    assert(checks.plan_ == plan_ && "entrance checks must match the executor plan");
    for (const auto& check : checks.checks_) {
        for (size_t word = 0; word < word_capacity_; ++word) {
            uint64_t rejected = check.constant ? live_words_[word] : 0;
            for (uint32_t reg : check.registers) {
                rejected ^= expression_registers_.column(reg)[word];
            }
            rejected &= live_words_[word];
            live_words_[word] &= ~rejected;
            live_count_ -= static_cast<uint32_t>(std::popcount(rejected));
        }
        if (live_count_ == 0) {
            break;
        }
    }
}

void BatchClassicalState::fill_random_half_bits() noexcept {
    for (size_t word = 0; word < word_capacity_; ++word) {
        scratch_words_[word] = rng_() & live_words_[word];
    }
}

void BatchClassicalState::initialize_expression_registers() noexcept {
    for (size_t expression = 0; expression < plan_->expression_register_constants_.size();
         ++expression) {
        if (plan_->expression_register_constants_[expression] != 0) {
            expression_registers_.assign(expression, live_words_, live_words_);
        }
    }
}

void BatchClassicalState::initialize_presampled_expressions() noexcept {
    assert(plan_->batch_presampled_program_.has_value() &&
           "presampled expression program must contain matching levels");
    const BatchPresampledProgram& program = *plan_->batch_presampled_program_;
    const size_t levels = program.initialization_level_offsets_.size() - 1;
    for (size_t level = 0; level < levels; ++level) {
        const uint32_t initialization_begin = program.initialization_level_offsets_[level];
        const uint32_t initialization_end = program.initialization_level_offsets_[level + 1];
        for (uint32_t index = initialization_begin; index < initialization_end; ++index) {
            const BatchPresampledProgram::InitializeExpression& initialization =
                program.initializations_[index];
            expression_registers_.copy(initialization.destination, initialization.parent);
            if (initialization.invert_parent) {
                expression_registers_.xor_into(initialization.destination, live_words_);
            }
        }
        const uint32_t carrier_xor_begin = program.carrier_xor_level_offsets_[level];
        const uint32_t carrier_xor_end = program.carrier_xor_level_offsets_[level + 1];
        for (uint32_t index = carrier_xor_begin; index < carrier_xor_end; ++index) {
            const BatchPresampledProgram::XorCarrierIntoExpression& carrier_xor =
                program.carrier_xors_[index];
            expression_registers_.xor_into(carrier_xor.destination,
                                           batch_noise_carriers_.column(carrier_xor.carrier));
        }
    }
    for (const BatchPresampledProgram::CopyExpression& copy : program.copies_) {
        expression_registers_.copy(copy.destination, copy.source);
    }
}

void BatchClassicalState::finalize_presampled_symbols() noexcept {
    if (noise_ && noise_->mode_ != BatchNoiseMode::Dense) {
        if (noise_->mode_ == BatchNoiseMode::SiteRows) {
            xor_bit_rows_into_columns(noise_rows_, noise_->row_stride_, active_lanes(),
                                      expression_registers_);
        }
        return;
    }
    if (!plan_->batch_presampled_program_.has_value()) {
        for (uint32_t symbol : plan_->presampled_symbols_) {
            propagate_symbol(symbol, symbols_.column(symbol));
        }
        return;
    }
    initialize_presampled_expressions();
}

// Plan tables and the noise destination, read once per batch. The noise loops
// call std::log, which the compiler must assume may write any memory reachable
// through the executor or plan, so reading these through either object would
// repeat the loads, and the RNG stores, on every draw. Callers pass a local RNG
// for the same reason.
struct BatchClassicalState::NoiseSiteActivator {
    explicit NoiseSiteActivator(BatchClassicalState& executor)
        : sites(executor.plan_->noise_sites_),
          outcomes(executor.plan_->noise_outcomes_),
          has_program(executor.plan_->batch_presampled_program_.has_value()),
          mode(executor.noise_ ? executor.noise_->mode_ : BatchNoiseMode::Dense),
          dependencies(&executor.plan_->expression_dependencies_),
          row_stride(executor.noise_ ? executor.noise_->row_stride_ : 0),
          rows(executor.noise_rows_),
          outcome_rows(executor.noise_ ? std::span<const uint64_t>(executor.noise_->outcome_rows_)
                                       : std::span<const uint64_t>{}) {
        if (mode != BatchNoiseMode::Dense) {
            destination = executor.expression_registers_.mutable_view();
            return;
        }
        if (has_program) {
            const BatchPresampledProgram& program = *executor.plan_->batch_presampled_program_;
            outcome_assignments = program.outcome_assignments_;
            assigned_carriers = program.assigned_carriers_;
            destination = executor.batch_noise_carriers_.mutable_view();
        } else {
            destination = executor.symbols_.mutable_view();
        }
    }

    // Selects the firing site's outcome, drawing from rng only when the site
    // has several, and sets that outcome's bits for the lane.
    void activate(Xoshiro256PlusPlus& rng, uint32_t lane, uint32_t site_index) const noexcept {
        assert(site_index < sites.size() && "noise site must be prepared");
        const ExecutablePlan::PreparedNoiseSite& site = sites[site_index];
        const uint32_t outcome_end = site.outcome_begin + site.outcome_count;
        assert(site.outcome_count > 0 && outcome_end <= outcomes.size() &&
               "firing noise site must contain prepared outcomes");
        uint32_t outcome_index = site.outcome_begin;
        if (site.outcome_count > 1) {
            const double execution_probability = outcomes[outcome_end - 1].cumulative_probability;
            const double draw = rng.next_double() * execution_probability;
            while (draw >= outcomes[outcome_index].cumulative_probability) {
                ++outcome_index;
                assert(outcome_index < outcome_end &&
                       "channel draw must select a prepared outcome");
            }
        }
        if (mode == BatchNoiseMode::SiteRows) {
            for (size_t word = 0; word < row_stride; ++word) {
                rows[size_t{lane} * row_stride + word] ^=
                    outcome_rows[size_t{outcome_index} * row_stride + word];
            }
            return;
        }
        if (mode != BatchNoiseMode::Dense) {
            for (uint32_t reg : dependencies->dependent_registers(outcomes[outcome_index].symbol)) {
                destination.column(reg)[lane >> 6] ^= uint64_t{1} << (lane & 63);
            }
            return;
        }
        if (!has_program) {
            const uint32_t symbol = outcomes[outcome_index].symbol;
            assert(!destination.bit(symbol, lane) && "noise site must define a fresh lane symbol");
            destination.set_bit(symbol, lane);
            return;
        }

        const BatchPresampledProgram::OutcomeAssignments& batch_outcome =
            outcome_assignments[outcome_index];
        const uint32_t assignment_end = batch_outcome.begin + batch_outcome.count;
        assert(assignment_end <= assigned_carriers.size() &&
               "batch noise assignment must stay in its prepared tape");
        for (uint32_t assignment = batch_outcome.begin; assignment < assignment_end; ++assignment) {
            const uint32_t carrier = assigned_carriers[assignment];
            assert(!destination.bit(carrier, lane) &&
                   "batch noise carrier must be assigned once per site");
            destination.set_bit(carrier, lane);
        }
    }

    std::span<const ExecutablePlan::PreparedNoiseSite> sites;
    std::span<const ExecutablePlan::PreparedNoiseOutcome> outcomes;
    std::span<const BatchPresampledProgram::OutcomeAssignments> outcome_assignments;
    std::span<const uint32_t> assigned_carriers;
    bool has_program = false;
    BatchNoiseMode mode;
    const ExecutablePlan::ExpressionDependencies* dependencies;
    size_t row_stride;
    std::span<uint64_t> rows;
    std::span<const uint64_t> outcome_rows;
    MutableBitColumnsView destination;
};

void BatchClassicalState::sample_presampled_noise() noexcept {
    if (!noise_rows_.empty()) {
        std::ranges::fill(
            std::span(noise_rows_).first(static_cast<size_t>(active_lanes()) * noise_->row_stride_),
            uint64_t{0});
    }
    if (plan_->noise_sites_.empty()) {
        finalize_presampled_symbols();
        return;
    }
    const NoiseSiteActivator activator(*this);
    Xoshiro256PlusPlus rng = rng_;
    const uint32_t lanes = active_lanes();
    const uint32_t end = static_cast<uint32_t>(activator.sites.size());
    if (noise_ && (noise_->mode_ == BatchNoiseMode::SiteSparse ||
                   noise_->mode_ == BatchNoiseMode::SiteRows)) {
        for (uint32_t site = 0; site < end; ++site) {
            const auto& descriptor = activator.sites[site];
            const double probability =
                activator.outcomes[descriptor.outcome_begin + descriptor.outcome_count - 1]
                    .cumulative_probability;
            if (probability >= 1.0) {
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    activator.activate(rng, lane, site);
                }
            } else if (probability > 0.0 && noise_->sites_[site].inverse_hazard() > 0.0) {
                const double inverse_hazard = noise_->sites_[site].inverse_hazard();
                uint32_t lane = 0;
                while (lane < lanes) {
                    const double gap = sample_geometric_gap(inverse_hazard, rng.next_double());
                    if (gap >= lanes - lane) {
                        break;
                    }
                    lane += static_cast<uint32_t>(gap);
                    activator.activate(rng, lane, site);
                    ++lane;
                }
            } else if (probability > 0.0) {
                sample_bernoulli_bits(noise_->sites_[site], live_words_, scratch_words_, lanes,
                                      rng);
                for (size_t word = 0; word < scratch_words_.size(); ++word) {
                    uint64_t events = scratch_words_[word];
                    while (events != 0) {
                        const auto lane =
                            static_cast<uint32_t>(word * 64 + std::countr_zero(events));
                        activator.activate(rng, lane, site);
                        events &= events - 1;
                    }
                }
            }
        }
    } else if (plan_->uniform_noise_inverse_hazard_.has_value()) {
        const ExecutablePlan::PreparedNoiseSite& first_site = activator.sites.front();
        const uint32_t first_outcome_end = first_site.outcome_begin + first_site.outcome_count;
        if (activator.outcomes[first_outcome_end - 1].cumulative_probability >= 1.0) {
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                for (uint32_t site = 0; site < end; ++site) {
                    activator.activate(rng, lane, site);
                }
            }
        } else {
            const double inverse_hazard = *plan_->uniform_noise_inverse_hazard_;
            const uint64_t total_draws = static_cast<uint64_t>(lanes) * end;
            uint64_t draw_index = 0;
            while (draw_index < total_draws) {
                const double gap = sample_geometric_gap(inverse_hazard, rng.next_double());
                if (gap >= static_cast<double>(total_draws - draw_index)) {
                    break;
                }
                draw_index += static_cast<uint64_t>(gap);
                const uint32_t lane = static_cast<uint32_t>(draw_index / end);
                const uint32_t site = static_cast<uint32_t>(draw_index % end);
                activator.activate(rng, lane, site);
                ++draw_index;
            }
        }
    } else {
        const std::span<const double> hazards(plan_->noise_hazards_);
        const double last_hazard = hazards[end - 1];
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            uint32_t first_candidate = 0;
            while (first_candidate < end) {
                const double current_hazard =
                    first_candidate == 0 ? 0.0 : hazards[first_candidate - 1];
                if (current_hazard >= last_hazard) {
                    break;
                }
                const uint32_t site =
                    sample_next_noise_site(hazards, first_candidate, rng.next_double());
                if (site == kNoNoiseSite || site >= end) {
                    break;
                }
                activator.activate(rng, lane, site);
                first_candidate = site + 1;
            }
        }
    }
    rng_ = rng;
    finalize_presampled_symbols();
}

void BatchClassicalState::assign_forced_faults(KFaultSampler& fault_sampler) noexcept {
    assert(fault_sampler.num_sites() ==
               plan_->noise_sites_.size() + plan_->num_readout_noise_sites_ &&
           "conditioned batch sampler must cover every fault site");
    const NoiseSiteActivator activator(*this);
    Xoshiro256PlusPlus rng = rng_;
    const uint32_t lanes = active_lanes();
    const uint32_t quantum_sites = static_cast<uint32_t>(activator.sites.size());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const std::span<const uint32_t> selected =
            fault_sampler.sample([&]() noexcept { return rng.next_double(); });
        for (uint32_t site : selected) {
            if (site < quantum_sites) {
                activator.activate(rng, lane, site);
            } else {
                forced_readout_.set_bit(site - quantum_sites, lane);
            }
        }
    }
    rng_ = rng;
    finalize_presampled_symbols();
}

void BatchClassicalState::propagate_symbol(uint32_t symbol,
                                           std::span<const uint64_t> values) noexcept {
    for (uint32_t register_id : plan_->expression_dependencies_.dependent_registers(symbol)) {
        expression_registers_.xor_into(register_id, values);
    }
}

void BatchClassicalState::execute_action(
    const ExecutablePlan::ExecuteDormantMeasurement& action) noexcept {
    const std::span<const uint64_t> corrections = evaluate(action.correction);
    fill_random_half_bits();
    propagate_symbol(action.branch, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, scratch_words_, corrections, live_words_);
    }
}

void BatchClassicalState::execute_action(
    const ExecutablePlan::ExecuteClassicalRecord& action) noexcept {
    if (records_.num_columns() != 0) {
        records_.assign(action.record, evaluate(action.outcome), live_words_);
    }
}

void BatchClassicalState::execute_action(
    const ExecutablePlan::ExecuteSymbolDefinition& action) noexcept {
    propagate_symbol(action.symbol, evaluate(action.value));
}

void BatchClassicalState::execute_action(
    const ExecutablePlan::ExecuteReadoutNoise& action) noexcept {
    const std::span<const uint64_t> sources = evaluate(action.source);
    std::ranges::fill(scratch_words_, uint64_t{0});
    bool sampled_packed = false;
    if (sampling_mode_ == BatchSamplingMode::ExternalInputs) {
        const auto zero = forced_readout_.column(size_t{2} * action.site);
        const auto one = forced_readout_.column(size_t{2} * action.site + 1);
        for (size_t word = 0; word < word_capacity_; ++word) {
            scratch_words_[word] =
                ((~sources[word] & zero[word]) | (sources[word] & one[word])) & live_words_[word];
        }
        sampled_packed = true;
    } else if (sampling_mode_ == BatchSamplingMode::FixedFaults) {
        const std::span<const uint64_t> forced = forced_readout_.column(action.site);
        for (size_t word = 0; word < word_capacity_; ++word) {
            scratch_words_[word] = forced[word] & live_words_[word];
        }
        sampled_packed = true;
    } else if (action.prob_zero_to_one == action.prob_one_to_zero) {
        const double probability = action.prob_zero_to_one;
        if (probability <= 0.0) {
            sampled_packed = true;
        } else if (probability >= 1.0) {
            std::ranges::copy(live_words_, scratch_words_.begin());
            sampled_packed = true;
        } else if (probability == 0.5) {
            fill_random_half_bits();
            sampled_packed = true;
        } else if (action.batch_symmetric_inverse_hazard > 0.0) {
            uint64_t lane = 0;
            while (lane < active_lanes()) {
                const double gap =
                    sample_geometric_gap(action.batch_symmetric_inverse_hazard, rng_.next_double());
                if (gap >= static_cast<double>(active_lanes() - lane)) {
                    break;
                }
                lane += static_cast<uint64_t>(gap);
                scratch_words_[lane >> 6] |= uint64_t{1} << (lane & 63);
                ++lane;
            }
            sampled_packed = true;
        }
    }
    if (!sampled_packed) {
        for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
            if (!is_live(lane)) {
                continue;
            }
            const bool source = lane_bit(sources, lane);
            const double probability = source ? action.prob_one_to_zero : action.prob_zero_to_one;
            if (probability >= 1.0 || (probability > 0.0 && rng_.next_double() < probability)) {
                scratch_words_[lane >> 6] |= uint64_t{1} << (lane & 63);
            }
        }
    }
    propagate_symbol(action.flip, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, sources, scratch_words_, live_words_);
    }
}

void BatchClassicalState::execute_action(const ExecutablePlan::ExecuteObservable& action) noexcept {
    const std::span<const uint64_t> outcomes = evaluate_observable(action.outcome);
    observables_.assign(action.observable, outcomes, live_words_);
}

std::span<const uint64_t> BatchClassicalState::evaluate(
    ExecutablePlan::PreparedExpression expression) const noexcept {
    assert(expression.register_id < expression_registers_.num_columns() &&
           "prepared expression must reference a packed register");
    return expression_registers_.column(expression.register_id);
}

std::span<const uint64_t> BatchClassicalState::evaluate_observable(
    const ExecutablePlan::PreparedObservableValue& value) noexcept {
    if (const auto* expression = std::get_if<ExecutablePlan::PreparedExpression>(&value)) {
        return evaluate(*expression);
    }
    return evaluate_record_parity(std::get<ExecutablePlan::PreparedRecordParity>(value));
}

std::span<const uint64_t> BatchClassicalState::evaluate_record_parity(
    ExecutablePlan::PreparedRecordParity parity) noexcept {
    const uint32_t end = parity.begin + parity.count;
    assert(end <= plan_->record_parity_terms_.size() &&
           "prepared record parity must stay in its term tape");
    const size_t words = packed_word_count(active_lanes());
    if (parity.constant) {
        std::ranges::copy(std::span<const uint64_t>(live_words_).first(words),
                          scratch_words_.begin());
    } else {
        std::ranges::fill(std::span<uint64_t>(scratch_words_).first(words), uint64_t{0});
    }
    for (uint32_t term = parity.begin; term < end; ++term) {
        const std::span<const uint64_t> record = records_.column(plan_->record_parity_terms_[term]);
        for (size_t word = 0; word < words; ++word) {
            scratch_words_[word] ^= record[word];
        }
    }
    return scratch_words_;
}

bool BatchClassicalState::lane_bit(std::span<const uint64_t> bits, uint32_t lane) const noexcept {
    assert(bits.size() >= word_capacity_ && lane < active_lanes() &&
           "packed expression lookup must reference an active lane");
    return ((bits[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

bool BatchClassicalState::is_live(uint32_t lane) const noexcept {
    assert(lane < active_lanes() && "live lookup must reference the current lane span");
    return ((live_words_[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

uint32_t BatchClassicalState::accumulate_survivor_counts(
    std::span<uint64_t> observable_ones) const noexcept {
    assert(output_mode_ == BatchOutputMode::AggregateSurvivors &&
           observable_ones.size() == plan_->num_observables_ &&
           "aggregate survivor outputs must match the prepared observable count");
    uint32_t logical_errors = 0;
    const size_t words = packed_word_count(active_lanes());
    for (size_t word = 0; word < words; ++word) {
        const uint64_t live = live_words_[word];
        uint64_t any_observable = 0;
        for (uint32_t observable = 0; observable < plan_->num_observables_; ++observable) {
            const uint64_t ones = observables_.column(observable)[word] & live;
            observable_ones[observable] += static_cast<uint64_t>(std::popcount(ones));
            any_observable |= ones;
        }
        logical_errors += static_cast<uint32_t>(std::popcount(any_observable));
    }
    return logical_errors;
}

uint32_t BatchClassicalState::shot_index(uint32_t lane) const noexcept {
    assert(lane < live_count_ && active_lanes() == live_count_ &&
           "batch outputs require finalized live lanes");
    return shot_indices_[lane];
}

std::span<const uint32_t> BatchClassicalState::shot_indices() const noexcept {
    assert(active_lanes() == live_count_ && "batch outputs require finalized live lanes");
    return std::span<const uint32_t>(shot_indices_.data(), live_count_);
}

bool BatchClassicalState::measurement(uint32_t lane, uint32_t record) const noexcept {
    assert(lane < live_count_ && record < plan_->num_visible_records_ &&
           "measurement output must be visible and live");
    return records_.bit(record, lane);
}

bool BatchClassicalState::detector(uint32_t lane, uint32_t detector_index) const noexcept {
    assert(lane < live_count_ && detector_index < plan_->num_detectors_ &&
           "detector output must be live and in range");
    return detectors_.bit(detector_index, lane);
}

bool BatchClassicalState::observable(uint32_t lane, uint32_t observable_index) const noexcept {
    assert(lane < live_count_ && observable_index < plan_->num_observables_ &&
           "observable output must be live and in range");
    return observables_.bit(observable_index, lane);
}

}  // namespace clifft::sampling
