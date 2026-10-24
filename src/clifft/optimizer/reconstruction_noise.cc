#include "clifft/optimizer/reconstruction_noise.h"

#include "clifft/optimizer/reconstruction_internal.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <utility>

namespace clifft {
namespace {

using namespace detail_reconstruction;

Bits canonical(Bits value) {
    while (!value.empty() && value.back() == 0)
        value.pop_back();
    return value;
}

ReconstructionNoise compress_noise(const GlobalReconstruction& r,
                                   const ReconstructionLimits& limits) {
    Basis basis;
    std::vector<Bits> rows;
    std::vector<size_t> indices(r.operations.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return (r.operations[a].output < 0) < (r.operations[b].output < 0);
    });
    for (size_t i : indices) {
        const auto& terms = r.operations[i].sign.words;
        if (!basis.solve(terms)) {
            basis.insert(terms, unit(static_cast<uint32_t>(rows.size())));
            rows.push_back(terms);
        }
    }
    std::vector<Bits> columns(r.num_noise_bits);
    for (uint32_t j = 0; j < rows.size(); ++j) {
        each_bit(rows[j], [&](uint32_t i) { toggle(columns.at(i), j); });
    }
    using Distribution = std::map<Bits, double>;
    std::map<std::vector<Bits>, Distribution> grouped;
    size_t products = 0;
    for (const auto& channel : r.noise_channels) {
        Distribution image;
        for (const auto& outcome : channel) {
            if (outcome.probability == 0)
                continue;
            Bits value;
            for (auto b : outcome.bits)
                xor_into(value, columns.at(b));
            image[canonical(std::move(value))] += outcome.probability;
        }
        if (image.size() == 1 && image.begin()->first.empty())
            continue;
        std::vector<Bits> key;
        for (const auto& [value, probability] : image)
            key.push_back(value);
        auto existing = grouped.find(key);
        if (existing != grouped.end()) {
            Distribution convolved;
            for (const auto& [a, p] : existing->second) {
                for (const auto& [b, q] : image) {
                    require_supported(products++ < limits.max_channel_products,
                                      "channel convolution work limit exceeded");
                    Bits sum = a;
                    xor_into(sum, b);
                    convolved[canonical(std::move(sum))] += p * q;
                    require_supported(convolved.size() <= limits.max_channel_support,
                                      "channel support limit exceeded");
                }
            }
            existing->second = std::move(convolved);
        } else {
            require_supported(image.size() <= limits.max_channel_support,
                              "channel support limit exceeded");
            grouped.emplace(std::move(key), std::move(image));
        }
    }
    ReconstructionNoise result;
    result.num_bits = static_cast<uint32_t>(rows.size());
    for (const auto& op : r.operations)
        result.signs.push_back({*basis.solve(op.sign.words), op.sign.constant});
    for (const auto& [key, distribution] : grouped) {
        auto& channel = result.channels.emplace_back();
        for (const auto& [value, probability] : distribution) {
            ReconstructionNoiseOutcome outcome{probability, {}};
            each_bit(value, [&](uint32_t i) { outcome.bits.push_back(i); });
            channel.push_back(std::move(outcome));
        }
    }
    return result;
}

}  // namespace

ReconstructionNoiseResult compress_reconstruction_noise(const GlobalReconstruction& reconstruction,
                                                        const ReconstructionLimits& limits) {
    try {
        return {compress_noise(reconstruction, limits), {}};
    } catch (const Ineligible& e) {
        return {std::nullopt, e.reason};
    }
}

}  // namespace clifft
