#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
namespace clifft {
struct Circuit;
enum class QubitStatus : uint8_t;
namespace noncomp_detail {
class OnlineState;
}
}  // namespace clifft
#endif

namespace clifft::detail {

// Source outcomes are G, E, no-fire; partitions are G, E, noncomputational;
// destinations use Level order. Measurements use the local Pauli eigenvalue.
enum class BranchChoice { Source, DestinationPartition, Destination, Measurement };

#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
// Defined by the diagnostic executable, never linked into the ordinary library.
std::optional<unsigned> replay_branch(BranchChoice kind, std::span<const double> masses) noexcept;

// Inspection stays at compilation or completed-shot boundaries, where the
// diagnostic executable can allocate without changing the hot-path contract.
void replay_append_probes(Circuit& circuit);
void replay_inspect(std::span<const double> probes, std::span<const QubitStatus> status);
void replay_inspect_online(const noncomp_detail::OnlineState& quantum,
                           std::span<const QubitStatus> status);
#endif

// Both selectors consume the same supported masses. Keeping the original
// sampler at the call site preserves its RNG draw count and interval arithmetic.
template <size_t N, typename Sample>
unsigned choose_branch([[maybe_unused]] BranchChoice kind, const std::array<double, N>& masses,
                       Sample sample) noexcept {
#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
    if (auto selected = replay_branch(kind, masses))
        return *selected;
#endif
    return sample(masses);
}

}  // namespace clifft::detail
