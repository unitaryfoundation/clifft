#pragma once

#include "clifft/noncomp/preparation.h"

#include <memory>

namespace clifft {
namespace detail {
struct OnlineNonComputationalData;
}

// Samples without resolving the Clifford frame ahead of time, avoiding repeated
// compilation when leakage or loss changes downstream operations. Validated
// operations are shared; each sampling worker owns its mutable shot state.
class OnlineNonComputationalProgram {
  public:
    explicit OnlineNonComputationalProgram(PreparedNonComputationalRequest request);

    NonComputationalSample sample(uint32_t shots, std::optional<uint64_t> seed, uint32_t threads,
                                  bool collect_stats) const;

  private:
    std::shared_ptr<const detail::OnlineNonComputationalData> data_;
};

}  // namespace clifft
