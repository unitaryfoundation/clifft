#pragma once

#include "clifft/util/xoshiro.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace clifft::sampling::css {

using Complex = std::complex<double>;
inline constexpr unsigned kMaxData = 63;
inline constexpr unsigned kMaxRank = 31;

struct Check {
    bool x;
    uint64_t support;
    bool operator==(const Check&) const = default;
};

// All scopes and gathers are constructed before execution. Factors depend on
// parity variables, while bound values select eighth roots of unity.
class Factor {
  public:
    struct Leaf {
        unsigned parameter, offset;
        std::vector<uint8_t> parity;
    };
    struct Step {
        unsigned offset, size, inputs;
        std::vector<unsigned> gathers;
    };
    Factor(std::span<const std::pair<uint64_t, unsigned>> factors, std::span<const unsigned> order,
           size_t entry_limit);
    [[nodiscard]] Complex evaluate(std::span<const unsigned> parameters,
                                   std::span<Complex> scratch) const noexcept;
    [[nodiscard]] size_t storage() const { return storage_; }
    [[nodiscard]] size_t gather_entries() const { return gathers_; }

  private:
    std::vector<Leaf> leaves_;
    std::vector<Step> steps_;
    std::vector<unsigned> outputs_;
    unsigned storage_ = 0;
    size_t gathers_ = 0;
};

// The constructor certifies positive CSS checks with one encoded qubit and
// all-data logical X/Z, including generator independence and syndrome duals.
class Code {
  public:
    Code(unsigned width, std::vector<Check> checks, size_t entry_limit = 2000000);
    [[nodiscard]] unsigned width() const { return width_; }
    [[nodiscard]] unsigned rank() const { return xchecks_.size(); }
    [[nodiscard]] const std::vector<Check>& checks() const { return checks_; }
    [[nodiscard]] const std::vector<std::pair<uint64_t, uint64_t>>& duals() const { return duals_; }
    [[nodiscard]] size_t scratch_size() const { return scratch_size_; }
    [[nodiscard]] size_t gather_entries() const { return gather_entries_; }

  private:
    friend struct Engine;
    unsigned width_;
    uint64_t all_;
    std::vector<Check> checks_;
    std::vector<uint64_t> xchecks_, zchecks_, coordinates_;
    std::vector<std::pair<uint64_t, uint64_t>> duals_;
    std::vector<std::pair<bool, unsigned>> check_order_;
    std::vector<unsigned> order_;
    std::vector<Factor> prefixes_;
    std::vector<Factor> single_;
    size_t scratch_size_ = 0, gather_entries_ = 0;
    [[nodiscard]] unsigned coordinate(uint64_t bits) const noexcept;
    [[nodiscard]] uint64_t expand(unsigned bits) const noexcept;
};

struct Workspace {
    explicit Workspace(size_t slots) : coefficients(slots) {}
    // Incoming frame X/Z, then four physical fault layers X/Z, in wire order.
    std::array<uint8_t, 10 * kMaxData> inputs{};
    // Gadget record followed by the original CSS check order.
    std::array<uint8_t, kMaxData> outcomes{};
    std::vector<Complex> coefficients;
};

// Returns the joint probability conditional on physical faults. Zero in
// forced mode means unreachable. Ordinary execution never allocates or plans.
[[nodiscard]] double apply(const Code& code, std::array<Complex, 2>& logical, Workspace& workspace,
                           Xoshiro256PlusPlus& rng, bool forced) noexcept;

}  // namespace clifft::sampling::css
