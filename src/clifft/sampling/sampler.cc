#include "clifft/sampling/sampler.h"

#include "clifft/sampling/executor.h"
#include "clifft/util/fault_sampling.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace clifft::sampling {

namespace {

template <typename RunShot>
SamplingResult sample_fixed_rows(const ExecutablePlan& plan, uint32_t shots,
                                 std::optional<uint64_t> seed, RunShot&& run_shot) {
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
    result.exp_vals.resize(checked_size(plan.num_exp_vals()));
    if (shots == 0) {
        return result;
    }

    auto run = [&](Executor& executor) {
        for (uint32_t shot = 0; shot < shots; ++shot) {
            run_shot(executor);
            std::ranges::copy(executor.visible_records(),
                              result.measurements.begin() +
                                  static_cast<size_t>(shot) * plan.num_visible_records());
            std::ranges::copy(
                executor.detectors(),
                result.detectors.begin() + static_cast<size_t>(shot) * plan.num_detectors());
            std::ranges::copy(
                executor.observables(),
                result.observables.begin() + static_cast<size_t>(shot) * plan.num_observables());
            std::ranges::copy(
                executor.exp_vals(),
                result.exp_vals.begin() + static_cast<size_t>(shot) * plan.num_exp_vals());
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

template <typename RunShot>
SamplingSurvivorResult sample_surviving_rows(const ExecutablePlan& plan, uint32_t shots,
                                             std::optional<uint64_t> seed, bool keep_records,
                                             RunShot&& run_shot) {
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
        result.exp_vals.reserve(checked_reserve(plan.num_exp_vals()));
    }
    auto run = [&](Executor& executor) {
        for (uint32_t shot = 0; shot < shots; ++shot) {
            run_shot(executor);
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
                result.exp_vals.insert(result.exp_vals.end(), executor.exp_vals().begin(),
                                       executor.exp_vals().end());
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

}  // namespace

SamplingResult sample(const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "fixed-plan sampling does not support instrument traps; use the trajectory driver");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "batch sampling requires a distribution for every presampled symbol");
    }
    if (plan.has_postselection()) {
        throw std::invalid_argument(
            "fixed-row sampling does not support postselection; use sample_survivors");
    }

    return sample_fixed_rows(plan, shots, seed,
                             [](Executor& executor) noexcept { executor.run_shot(); });
}

std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed) {
    return sample(plan, shots, seed).measurements;
}

SamplingSurvivorResult sample_survivors(const ExecutablePlan& plan, uint32_t shots,
                                        std::optional<uint64_t> seed, bool keep_records) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "survivor sampling does not support instrument traps; use the trajectory driver");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "survivor sampling requires a distribution for every presampled symbol");
    }

    return sample_surviving_rows(plan, shots, seed, keep_records,
                                 [](Executor& executor) noexcept { executor.run_shot(); });
}

SamplingResult sample_k(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                        std::optional<uint64_t> seed) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "forced-fault sampling does not support instrument traps or trajectory drivers");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "forced-fault sampling requires a distribution for every presampled symbol");
    }
    if (plan.has_postselection()) {
        throw std::invalid_argument(
            "fixed-row forced-fault sampling does not support postselection; use "
            "sample_k_survivors");
    }
    if (shots == 0) {
        return sample_fixed_rows(plan, shots, seed,
                                 [](Executor& executor) noexcept { executor.run_shot(); });
    }
    KFaultSampler fault_sampler(plan.noise_site_probabilities(), k);
    return sample_fixed_rows(
        plan, shots, seed, [&](Executor& executor) noexcept { executor.run_shot(fault_sampler); });
}

SamplingSurvivorResult sample_k_survivors(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                                          std::optional<uint64_t> seed, bool keep_records) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "forced-fault survivor sampling does not support instrument traps or trajectory "
            "drivers");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "forced-fault survivor sampling requires a distribution for every presampled symbol");
    }
    if (shots == 0) {
        return sample_surviving_rows(plan, shots, seed, keep_records,
                                     [](Executor& executor) noexcept { executor.run_shot(); });
    }
    KFaultSampler fault_sampler(plan.noise_site_probabilities(), k);
    return sample_surviving_rows(plan, shots, seed, keep_records, [&](Executor& executor) noexcept {
        executor.run_shot(fault_sampler);
    });
}

std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                             std::span<const uint8_t> forced_records,
                                             size_t num_records) {
    if (plan.has_instruments()) {
        throw std::invalid_argument("record probabilities do not yet support instruments");
    }
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
