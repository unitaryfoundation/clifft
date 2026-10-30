#pragma once

// Included only by the diagnostic target's copies of production translation
// units. The ordinary library and Python extension contain no replay hooks.
#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/level.h"

#include <span>

namespace clifft::noncomp_detail {
class OnlineState;
}

namespace clifft::test {

// Source masses are G, E, no-fire; destination partitions are G, E, noncomp;
// full destinations use Level order. Measurement outcomes are backend-local
// bits. Only the resulting visible records enter the physical comparison, so
// adapters need not share hidden records, corrections, or decision tapes.
enum class ReplayChoice { Source, DestinationPartition, Destination, Measurement };

struct NoncompReplayHooks {
    virtual ~NoncompReplayHooks() = default;

    // Implementations preallocate their decision tape. Selection runs inside
    // noexcept dispatch and must neither allocate nor throw. Probabilities are
    // unnormalized masses; selection must retain their conditional weight.
    virtual unsigned choose(ReplayChoice kind, std::span<const double> masses) noexcept = 0;

    // These inspection boundaries run during compilation or after dispatch.
    virtual void append_probes(Circuit& circuit) = 0;
    virtual void inspect(std::span<const double> probes, std::span<const QubitStatus> status) = 0;
    virtual void inspect_online(const noncomp_detail::OnlineState& quantum,
                                std::span<const QubitStatus> status) = 0;
};

inline thread_local NoncompReplayHooks* noncomp_replay = nullptr;

}  // namespace clifft::test
