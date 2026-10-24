#include "clifft/optimizer/reconstruction_lowering.h"

#include "clifft/optimizer/reconstruction_internal.h"
#include "clifft/optimizer/reconstruction_noise.h"

#include <algorithm>
#include <bit>
#include <utility>

namespace clifft {
namespace {

using namespace detail_reconstruction;

HirModule lower_reconstruction(const GlobalReconstruction& r, const ReconstructionNoise& noise,
                               const PostselectionMetadata& postselection,
                               const ReconstructionLimits& limits) {
    const uint32_t visible = r.outputs.num_records;
    const uint32_t width = r.num_qubits + noise.num_bits + 1;
    const uint32_t carrier = width - 1;
    size_t masks = noise.num_bits + 1;
    for (const auto& sign : noise.signs) {
        size_t terms = 0;
        for (auto w : sign.words)
            terms += std::popcount(w);
        masks += 2 * (terms + 2) + 2;
    }
    size_t noise_masks = 0;
    for (const auto& channel : noise.channels)
        noise_masks += channel.size();
    const size_t bytes_per_mask = 2 * ((size_t{width} + 63) / 64) * sizeof(uint64_t) + 1;
    require_supported(masks + noise_masks <= limits.max_lowering_bytes / bytes_per_mask,
                      "lowering mask memory limit exceeded");
    HirModule hir(width, masks, noise_masks);
    hir.num_measurements = visible;
    hir.num_detectors = r.outputs.num_detectors;
    hir.num_observables = r.outputs.num_observables;
    hir.postselection = postselection;
    for (const auto& channel : noise.channels) {
        NoiseSite site;
        for (const auto& outcome : channel) {
            if (outcome.bits.empty() || outcome.probability == 0)
                continue;
            const auto handle = hir.claim_empty_noise_channel_mask();
            auto slot = hir.noise_channel_masks.mut_at(handle);
            for (uint32_t i : outcome.bits)
                slot.x().bit_set(r.num_qubits + i, true);
            site.channels.push_back({handle, outcome.probability});
            site.total_probability += outcome.probability;
        }
        // Roundoff in convolution must not turn a normalized distribution into
        // an invalid noise site. Scale only an excess over unit total mass.
        if (site.total_probability > 1) {
            for (auto& c : site.channels)
                c.prob /= site.total_probability;
            site.total_probability = 1;
        }
        const auto index = static_cast<uint32_t>(hir.noise_sites.size());
        hir.noise_sites.push_back(std::move(site));
        hir.append_noise(NoiseSiteIdx{index});
    }
    auto hidden_measure = [&](auto&& fill) {
        const auto record = visible + hir.num_hidden_measurements++;
        hir.append_measure(MeasRecordIdx{record}, fill).set_hidden(true);
        return record;
    };
    std::vector<uint32_t> fault_records;
    for (uint32_t i = 0; i < noise.num_bits; ++i) {
        fault_records.push_back(
            hidden_measure([&](MutablePauliMaskView p) { p.z().bit_set(r.num_qubits + i, true); }));
    }
    const auto one = hidden_measure([](MutablePauliMaskView p) { p.set_sign(true); });
    std::vector<std::vector<uint32_t>> output_parities(r.outputs.num_outputs());
    for (size_t j = 0; j < r.operations.size(); ++j) {
        const auto& op = r.operations[j];
        std::vector<uint32_t> parity;
        each_bit(noise.signs[j].words, [&](uint32_t i) { parity.push_back(fault_records[i]); });
        auto fill_axis = [&](MutablePauliMaskView p) {
            for (uint32_t w = 0; w < op.axis.x().num_words(); ++w) {
                p.x().words[w] = op.axis.x().words[w];
                p.z().words[w] = op.axis.z().words[w];
            }
        };
        if (op.half_turns != 0) {
            const uint32_t pivot = std::min(op.axis.x().lowest_bit(), op.axis.z().lowest_bit());
            if (pivot >= r.num_qubits)
                continue;
            auto feedback = [&] {
                for (auto record : parity)
                    hir.append_conditional(ControllingMeasIdx{record}, [&](MutablePauliMaskView p) {
                        if (op.axis.x().bit_get(pivot))
                            p.z().bit_set(pivot, true);
                        else
                            p.x().bit_set(pivot, true);
                    });
            };
            feedback();
            hir.append_phase_rotation(op.half_turns, [&](MutablePauliMaskView p) {
                fill_axis(p);
                p.set_sign(noise.signs[j].constant);
            });
            feedback();
        } else {
            parity.push_back(hidden_measure(fill_axis));
            if (noise.signs[j].constant)
                parity.push_back(one);
            if (op.output < 0)
                continue;
            auto& output = output_parities.at(static_cast<size_t>(op.output));
            output = parity;
            if (static_cast<uint32_t>(op.output) < visible) {
                // Materialize the affine record in a zero classical carrier.
                // Undo the same controls afterwards so the carrier is reusable.
                auto copy_parity = [&] {
                    for (auto record : parity)
                        hir.append_conditional(
                            ControllingMeasIdx{record},
                            [&](MutablePauliMaskView p) { p.x().bit_set(carrier, true); });
                };
                copy_parity();
                hir.append_measure(MeasRecordIdx{static_cast<uint32_t>(op.output)},
                                   [&](MutablePauliMaskView p) { p.z().bit_set(carrier, true); });
                copy_parity();
            }
        }
    }
    // Source-order declarations preserve the original normalizer and masks.
    // Earlier checks can be added with an explicit public-index permutation.
    for (uint32_t i = 0; i < r.outputs.num_detectors; ++i) {
        hir.detector_targets.push_back(std::move(output_parities[visible + i]));
        hir.append_detector(DetectorIdx{i});
    }
    for (uint32_t i = 0; i < r.outputs.num_observables; ++i) {
        hir.observable_targets.push_back(
            std::move(output_parities[visible + r.outputs.num_detectors + i]));
        hir.append_observable(ObservableIdx{i}, i);
    }
    return hir;
}

}  // namespace

ReconstructionLoweringResult lower_global_reconstruction(const GlobalReconstruction& reconstruction,
                                                         const PostselectionMetadata& postselection,
                                                         const ReconstructionLimits& limits) {
    auto noise = compress_reconstruction_noise(reconstruction, limits);
    if (!noise.noise)
        return {std::nullopt, std::move(noise.fallback_reason)};
    try {
        return {lower_reconstruction(reconstruction, *noise.noise, postselection, limits), {}};
    } catch (const Ineligible& e) {
        return {std::nullopt, e.reason};
    }
}

}  // namespace clifft
