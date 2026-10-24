#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace clifft::detail_reconstruction {

using Bits = std::vector<uint64_t>;

inline void toggle(Bits& a, uint32_t i) {
    a.resize(std::max(a.size(), size_t{i / 64 + 1}), 0);
    a[i / 64] ^= uint64_t{1} << (i % 64);
}

inline void xor_into(Bits& a, const Bits& b) {
    a.resize(std::max(a.size(), b.size()), 0);
    for (size_t i = 0; i < b.size(); ++i) {
        a[i] ^= b[i];
    }
}

inline int32_t highest(const Bits& a) {
    for (size_t i = a.size(); i-- > 0;) {
        if (a[i]) {
            return static_cast<int32_t>(i * 64 + 63 - std::countl_zero(a[i]));
        }
    }
    return -1;
}

template <class F>
void each_bit(const Bits& a, F&& f) {
    for (size_t w = 0; w < a.size(); ++w) {
        for (uint64_t v = a[w]; v; v &= v - 1) {
            f(static_cast<uint32_t>(w * 64 + std::countr_zero(v)));
        }
    }
}

inline Bits unit(uint32_t i) {
    Bits a;
    toggle(a, i);
    return a;
}

class Basis {
  public:
    std::optional<Bits> insert(Bits value, Bits label) {
        reduce(value, label);
        const int32_t pivot = highest(value);
        if (pivot < 0) {
            return label;
        }
        rows_.emplace(pivot, std::make_pair(std::move(value), std::move(label)));
        return std::nullopt;
    }

    std::optional<Bits> solve(Bits value) const {
        Bits label;
        reduce(value, label);
        return highest(value) < 0 ? std::optional<Bits>{std::move(label)} : std::nullopt;
    }

    size_t size() const { return rows_.size(); }

  private:
    void reduce(Bits& value, Bits& label) const {
        for (;;) {
            const auto it = rows_.find(highest(value));
            if (it == rows_.end()) {
                return;
            }
            xor_into(value, it->second.first);
            xor_into(label, it->second.second);
        }
    }
    std::map<int32_t, std::pair<Bits, Bits>> rows_;
};

struct Ineligible {
    std::string reason;
};

inline void require_supported(bool condition, const char* reason) {
    if (!condition) {
        throw Ineligible{reason};
    }
}

}  // namespace clifft::detail_reconstruction
