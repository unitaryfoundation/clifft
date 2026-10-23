#include "clifft/sampling/batch/external_inputs.h"

#include "clifft/sampling/executable_plan.h"

#include <cassert>
#include <limits>
#include <stdexcept>

namespace clifft::sampling {

namespace {

size_t input_columns(const ExecutablePlan& plan) {
    const uint64_t columns = static_cast<uint64_t>(plan.num_presampled_symbols()) +
                             uint64_t{2} * plan.num_readout_noise_sites();
    if (columns > std::numeric_limits<size_t>::max()) {
        throw std::length_error("packed external input column count exceeds size_t range");
    }
    return static_cast<size_t>(columns);
}

}  // namespace

BatchExternalInputs::BatchExternalInputs(const ExecutablePlan& plan, uint32_t lane_capacity)
    : plan_(&plan),
      lane_capacity_(lane_capacity),
      noise_columns_(plan.num_presampled_symbols()),
      readout_sites_(plan.num_readout_noise_sites()),
      bits_(input_columns(plan), lane_capacity) {}

size_t BatchExternalInputs::storage_bytes(const ExecutablePlan& plan, uint32_t lane_capacity) {
    return packed_bit_columns_storage_bytes(input_columns(plan), lane_capacity);
}

std::span<uint64_t> BatchExternalInputs::noise(uint32_t input) noexcept {
    assert(input < noise_columns_ && "external noise input must belong to the plan");
    return bits_.column(input);
}

std::span<uint64_t> BatchExternalInputs::readout_zero(uint32_t site) noexcept {
    assert(site < readout_sites_ && "external readout input must belong to the plan");
    return bits_.column(noise_columns_ + size_t{2} * site);
}

std::span<uint64_t> BatchExternalInputs::readout_one(uint32_t site) noexcept {
    assert(site < readout_sites_ && "external readout input must belong to the plan");
    return bits_.column(noise_columns_ + size_t{2} * site + 1);
}

}  // namespace clifft::sampling
