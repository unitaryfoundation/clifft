#include "clifft/api/sampling_preparation.h"

#include <stdexcept>

namespace clifft {

SamplingPreparation::SamplingPreparation(const sampling::ExecutablePlan& compiled,
                                         sampling::SamplingOutputs outputs,
                                         bool early_postselection, bool survivor_counts)
    : early_postselection_(early_postselection),
      compact_(outputs.records == sampling::RecordOutputs::All
                   ? std::nullopt
                   : std::optional(
                         compiled.compact_outputs(outputs, early_postselection, survivor_counts))),
      plan_(compact_ ? &*compact_ : &compiled) {}

const sampling::EarlyPostselection& SamplingPreparation::checks() const {
    return checks_.get(*plan_, early_postselection_);
}

const sampling::PreparedBatchNoise& SamplingPreparation::noise(
    sampling::BatchNoiseMode mode) const {
    switch (mode) {
        case sampling::BatchNoiseMode::Dense:
            return dense_noise_.get(*plan_, mode);
        case sampling::BatchNoiseMode::Sparse:
            return sparse_noise_.get(*plan_, mode);
        case sampling::BatchNoiseMode::SiteSparse:
            return site_sparse_noise_.get(*plan_, mode);
        case sampling::BatchNoiseMode::SiteRows:
            return site_rows_noise_.get(*plan_, mode);
    }
    throw std::invalid_argument("unknown batch noise mode");
}

const sampling::SharingKey& SamplingPreparation::sharing_key() const {
    return sharing_key_.get(*plan_);
}

uint32_t SamplingPreparation::probe_capacity() const {
    constexpr uint32_t kProbeLanes = 256;
    return std::min(kProbeLanes, sampling::max_safe_shared_capacity(
                                     *plan_, noise(sampling::BatchNoiseMode::SiteSparse), nullptr,
                                     kProbeStateBytes));
}

sampling::TrajectoryWork SamplingPreparation::measure_work(uint32_t lanes) const {
    sampling::TrajectoryWork work;
    sampling::SharingExecutor worker(*plan_, noise(sampling::BatchNoiseMode::SiteSparse), checks(),
                                     lanes, 1, true, nullptr, kProbeStateBytes);
    worker.run_batch(make_seed_root(lanes, 0), 0, lanes, &work);
    return work;
}

const sampling::TrajectoryWork& SamplingPreparation::trajectory_work() const {
    std::call_once(profile_once_, [&] { profile_ = measure_work(probe_capacity()); });
    return profile_;
}

}  // namespace clifft
