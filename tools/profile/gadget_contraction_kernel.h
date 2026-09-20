#pragma once

#include "clifft/util/numeric.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gadget_study {
using Complex = std::complex<double>;

struct Reader {
    std::istream& input;
    uint64_t word() {
        uint64_t result;
        if (!(input >> result))
            throw std::invalid_argument("truncated native gadget model");
        return result;
    }
    size_t size() {
        const auto result = word();
        if (result > 2000000)
            throw std::invalid_argument("native gadget model exceeds storage limit");
        return result;
    }
    double number() {
        double result;
        if (!(input >> result) || !clifft::is_finite_robust(result))
            throw std::invalid_argument("invalid native gadget parameter");
        return result;
    }
};

struct ContractionWorkspace {
    std::vector<Complex> scratch, product;

    void prepare(size_t scratch_size, size_t product_size) {
        scratch.resize(scratch_size);
        product.resize(product_size);
    }
};

class Contraction {
    struct Gather {
        static constexpr size_t block_size = 256;
        std::vector<uint32_t> low, high;

        void compact() {
            // Small gathers do not repay the extra block-loop overhead.
            if (low.size() <= 256 || low.size() % block_size)
                return;
            std::vector<uint32_t> offsets(low.size() / block_size);
            for (size_t block = 0; block < offsets.size(); ++block) {
                if (low[block * block_size] < low[0])
                    return;
                offsets[block] = low[block * block_size] - low[0];
                for (size_t j = 0; j < block_size; ++j)
                    if (uint64_t(offsets[block]) + low[j] != low[block * block_size + j])
                        return;
            }
            // Bit projections separate into low-bit addresses and high-bit offsets.
            // Retain the expanded form if an external plan does not have this property.
            std::vector<uint32_t>(low.begin(), low.begin() + block_size).swap(low);
            high = std::move(offsets);
        }
    };
    struct Leaf {
        size_t offset;
        std::vector<uint8_t> parity;
    };
    struct Step {
        size_t offset, size;
        std::vector<Gather> gathers;
    };
    std::vector<Leaf> leaves_;
    std::vector<Step> steps_;
    std::vector<uint32_t> outputs_;
    size_t scratch_size_, product_size_;
    double normalization_;
    size_t leaf_values_;

  public:
    explicit Contraction(Reader& reader, size_t expected_rank, size_t expected_leaves,
                         size_t leaf_values = 2)
        : leaf_values_(leaf_values) {
        if (leaf_values_ != 2 && leaf_values_ != 4)
            throw std::invalid_argument("invalid local table size");
        if (reader.size() != 1 || reader.size() != expected_rank)
            throw std::invalid_argument("incorrect marginal plan rank or mode");
        const auto storage = reader.size();
        leaves_.resize(reader.size());
        if (leaves_.size() != expected_leaves || storage == 0)
            throw std::invalid_argument("incorrect marginal plan leaves");
        size_t initialized = 0;
        for (auto& leaf : leaves_) {
            leaf.offset = reader.size();
            leaf.parity.resize(reader.size());
            if (leaf.offset != initialized || leaf.parity.empty() ||
                leaf.parity.size() > storage - initialized)
                throw std::invalid_argument("invalid marginal leaf storage");
            initialized += leaf.parity.size();
            for (auto& parity : leaf.parity) {
                const auto value = reader.size();
                if (value >= leaf_values_)
                    throw std::invalid_argument("invalid marginal parity");
                parity = static_cast<uint8_t>(value);
            }
        }
        steps_.resize(reader.size());
        size_t max_product = 0;
        for (auto& step : steps_) {
            step.offset = reader.size();
            step.size = reader.size();
            if (step.offset != initialized || !step.size || step.size > storage - initialized)
                throw std::invalid_argument("invalid marginal step storage");
            step.gathers.resize(reader.size());
            for (auto& gather : step.gathers) {
                gather.low.resize(2 * step.size);
                for (auto& address : gather.low) {
                    address = reader.size();
                    if (address >= initialized)
                        throw std::invalid_argument("marginal gather reads uninitialized storage");
                }
                gather.compact();
            }
            initialized += step.size;
            max_product = std::max(max_product, 2 * step.size);
        }
        if (initialized != storage)
            throw std::invalid_argument("marginal storage size differs");
        outputs_.resize(reader.size());
        for (auto& output : outputs_) {
            output = reader.size();
            if (output >= initialized)
                throw std::invalid_argument("invalid marginal output");
        }
        if (reader.size() != 0)
            throw std::invalid_argument("native sampler expects an unbound contraction plan");
        scratch_size_ = storage;
        product_size_ = max_product;
        normalization_ = std::ldexp(1.0, -int(expected_rank));
    }

    size_t scratch_size() const noexcept { return scratch_size_; }
    size_t product_size() const noexcept { return product_size_; }

    size_t lookup_bytes() const noexcept {
        size_t result = sizeof(uint32_t) * outputs_.capacity();
        for (const auto& leaf : leaves_)
            result += sizeof(uint8_t) * leaf.parity.capacity();
        for (const auto& step : steps_)
            for (const auto& gather : step.gathers)
                result += sizeof(uint32_t) * (gather.low.capacity() + gather.high.capacity());
        return result;
    }

    Complex evaluate(const Complex* local, ContractionWorkspace& workspace) const noexcept {
        assert(workspace.scratch.size() >= scratch_size_);
        assert(workspace.product.size() >= product_size_);
        auto& scratch = workspace.scratch;
        auto& product = workspace.product;
        for (size_t i = 0; i < leaves_.size(); ++i) {
            const auto& leaf = leaves_[i];
            for (size_t j = 0; j < leaf.parity.size(); ++j)
                scratch[leaf.offset + j] = local[leaf_values_ * i + leaf.parity[j]];
        }
        for (const auto& step : steps_) {
            std::fill_n(product.begin(), 2 * step.size, Complex(1));
            for (const auto& gather : step.gathers) {
                if (gather.high.empty()) {
                    for (size_t j = 0; j < 2 * step.size; ++j)
                        product[j] *= scratch[gather.low[j]];
                } else {
                    for (size_t block = 0; block < gather.high.size(); ++block) {
                        const auto* source = scratch.data() + gather.high[block];
                        auto* target = product.data() + block * Gather::block_size;
                        for (size_t j = 0; j < Gather::block_size; ++j)
                            target[j] *= source[gather.low[j]];
                    }
                }
            }
            for (size_t j = 0; j < step.size; ++j)
                scratch[step.offset + j] = product[j] + product[j + step.size];
        }
        Complex result = normalization_;
        for (auto output : outputs_)
            result *= scratch[output];
        return result;
    }
};
}  // namespace gadget_study
