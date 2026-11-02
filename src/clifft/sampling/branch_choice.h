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
// Classical transitions, heralds, and readout flips use binary no/yes masses.
enum class BranchChoice {
    Source,
    DestinationPartition,
    Destination,
    Measurement,
    ClassicalTransition,
    Herald,
    Readout
};

#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
// Defined by the diagnostic executable, never linked into the ordinary library.
std::optional<unsigned> replay_branch(BranchChoice kind, std::span<const double> masses) noexcept;

// Resource probes are confined to this executable as well. Ordinary reset,
// dispatch, and activation completion must remain allocation-free.
void replay_hot_path(bool enter) noexcept;
void replay_allocation() noexcept;
struct ReplayHotPathScope {
    ReplayHotPathScope() noexcept { replay_hot_path(true); }
    ~ReplayHotPathScope() { replay_hot_path(false); }
};
enum class ReplayOnlineBoundary { Reset, BeforeGrowth, AfterGrowth, Complete };
void replay_online_boundary(ReplayOnlineBoundary boundary,
                            const noncomp_detail::OnlineState& state);

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
