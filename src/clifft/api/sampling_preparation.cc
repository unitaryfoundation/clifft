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

}  // namespace clifft
