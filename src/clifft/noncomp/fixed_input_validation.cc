#include "clifft/noncomp/fixed_input_validation.h"

#include <stdexcept>
#include <variant>

namespace clifft::detail {

void validate_fixed_input_bindings(std::span<const size_t> noise_nodes,
                                   std::span<const FixedReadoutSite> readout_sites,
                                   const TraceSiteOrigins& origins, const HirModule& hir,
                                   const sampling::SamplingPlan& plan) {
    if (!plan.instrument_distributions.empty() || hir.noise_sites.size() != noise_nodes.size() ||
        origins.noise_nodes.size() != noise_nodes.size() ||
        plan.presampled_noise_sites.size() != noise_nodes.size()) {
        throw std::logic_error("fixed noncomputational lowering lost a noise-site identity");
    }
    for (size_t site = 0; site < noise_nodes.size(); ++site) {
        if (origins.noise_nodes[site] != noise_nodes[site] ||
            plan.presampled_noise_sites[site].source_site != site) {
            throw std::logic_error("fixed noncomputational noise-site identity changed");
        }
    }
    if (hir.readout_noise.size() != readout_sites.size() ||
        origins.readout_nodes.size() != readout_sites.size()) {
        throw std::logic_error("fixed noncomputational lowering lost a readout-site identity");
    }
    for (size_t site = 0; site < readout_sites.size(); ++site) {
        if (origins.readout_nodes[site] != readout_sites[site].node ||
            hir.readout_noise[site].meas_idx != readout_sites[site].record) {
            throw std::logic_error("fixed noncomputational readout-site identity changed");
        }
    }
    size_t site = 0;
    for (const auto& action : plan.actions) {
        if (const auto* readout = std::get_if<sampling::ApplyReadoutNoise>(&action.action)) {
            if (site >= readout_sites.size() || readout->source_site != site ||
                sampling::index(readout->record) != readout_sites[site].record) {
                throw std::logic_error(
                    "fixed noncomputational planned readout-site identity changed");
            }
            ++site;
        }
    }
    if (site != readout_sites.size()) {
        throw std::logic_error("fixed noncomputational lowering lost a readout-site identity");
    }
}

}  // namespace clifft::detail
