#pragma once

#include "clifft/noncomp/preparation.h"

#include <memory>

namespace clifft {
namespace detail {
struct OnlineNonComputationalData;
}

// Immutable, validated physical operations. Each sample call owns its online
// Clifford coordinates, coefficient storage, records, and per-shot RNG streams.
class OnlineNonComputationalProgram {
  public:
    explicit OnlineNonComputationalProgram(PreparedNonComputationalRequest request);

    NonComputationalSample sample(uint32_t shots, std::optional<uint64_t> seed, uint32_t threads,
                                  bool collect_stats) const;

  private:
    std::shared_ptr<const detail::OnlineNonComputationalData> data_;
};

}  // namespace clifft
