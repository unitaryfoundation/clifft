#include "clifft/circuit/parser.h"
#include "clifft/util/page_allocation.h"

#include "noncomp_replay_helpers.h"
#include "noncomp_resource_probe.h"
#include "noncomp_test_helpers.h"

#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace clifft;
using namespace clifft::test;

namespace {

struct ResourceReplay : NoncompReplayHooks {
    unsigned sources = 0;
    unsigned choose(ReplayChoice kind, std::span<const double> masses) noexcept override {
        if (kind == ReplayChoice::Source) {
            ++sources;
            return 2;
        }
        for (unsigned i = 0; i < masses.size(); ++i)
            if (masses[i] > 0)
                return i;
        return 0;
    }
    void append_probes(Circuit&) override {}
    void inspect(std::span<const double>, std::span<const QubitStatus>) override {}
    void inspect_online(const noncomp_detail::OnlineState&, std::span<const QubitStatus>) override {
    }
};

struct ResourceTrace {
    struct Snapshot {
        detail::ReplayOnlineBoundary boundary{};
        const double* data = nullptr;
        uint32_t width = 0, capacity = 0;
        unsigned sources = 0;
        double z = 0, x = 0;
    };
    std::array<Snapshot, 32> snapshots{};
    size_t count = 0;
    ResourceReplay replay;
};
thread_local ResourceTrace* resource_trace = nullptr;

void capture_boundary(detail::ReplayOnlineBoundary boundary,
                      const noncomp_detail::OnlineState& quantum) {
    auto& trace = *resource_trace;
    REQUIRE(trace.count < trace.snapshots.size());
    REQUIRE(hot_path_counters.depth == 0);
    auto& snapshot = trace.snapshots[trace.count++];
    snapshot = {boundary, quantum.state.real_data(), quantum.state.active_width(),
                quantum.state.max_active_width(), trace.replay.sources};
    PauliString axis(quantum.frame.n);
    quantum.frame.axis(0, 'Z', axis);
    auto populations = quantum.populations(axis.view());
    snapshot.z = (populations.zero - populations.one) / populations.total();
    quantum.frame.axis(0, 'X', axis);
    populations = quantum.populations(axis.view());
    snapshot.x = (populations.zero - populations.one) / populations.total();
}

struct ResourceScope {
    NoncompReplayHooks* previous_replay = noncomp_replay;
    OnlineBoundaryProbe previous_probe = online_boundary_probe;
    explicit ResourceScope(ResourceTrace& trace) {
        resource_trace = &trace;
        noncomp_replay = &trace.replay;
        online_boundary_probe = capture_boundary;
        hot_path_counters = {};
    }
    ~ResourceScope() {
        noncomp_replay = previous_replay;
        online_boundary_probe = previous_probe;
        resource_trace = nullptr;
    }
};

}  // namespace

TEST_CASE("Weighted resource probe detects ordinary and coefficient allocations",
          "[noncomp][weighted_replay]") {
    hot_path_counters = {};
    void* pointer;
    {
        detail::ReplayHotPathScope hot_path;
        // Call the replaceable function directly so allocation elision cannot
        // turn the probe's positive control into a false success.
        auto* volatile allocate = static_cast<void* (*)(size_t)>(&::operator new);
        pointer = allocate(32);
        PageAlignedAllocation coefficients(32);
    }
    ::operator delete(pointer);
    REQUIRE(hot_path_counters.depth == 0);
    REQUIRE(hot_path_counters.intervals == 1);
    REQUIRE(hot_path_counters.allocations == 2);
    hot_path_counters = {};
}

TEST_CASE("Weighted online growth retains pending decisions and capacity across shots",
          "[noncomp][weighted_replay]") {
    auto filter = zero_transition_matrix();
    filter[4][0] = 0.2;
    filter[4][1] = 0.6;
    const auto model =
        NonComputationalModel::from_spec(pure_initial_state(Level::G), {{"filter", filter}},
                                         classifier_matrix_with_column(Level::Lost, {1, 0}), {});
    const auto circuit = parse(
        "H 0 1\nSWAP 0 1\nSWAP 0 1\nLEVEL_TRANSITION[filter] 0\nT 1\nMY 0\n"
        "R_Y(0.25) 0\nM 0 1\n");
    const OnlineNonComputationalProgram program(PreparedNonComputationalRequest(circuit, model));
    ResourceTrace trace;
    ResourceScope scope(trace);
    const auto sample = program.sample(4, 17, 1, true);
    REQUIRE(hot_path_counters.allocations == 0);
    REQUIRE(hot_path_counters.depth == 0);
    REQUIRE(hot_path_counters.intervals >= 10);
    REQUIRE(trace.replay.sources == 4);
    REQUIRE(sample.stats->online_state_growths == 2);
    REQUIRE(sample.stats->continuation_compilations == 0);
    REQUIRE(trace.count == 12);
    REQUIRE(trace.snapshots[0].capacity == 0);
    for (size_t before : {size_t{1}, size_t{3}}) {
        const auto& a = trace.snapshots[before];
        const auto& b = trace.snapshots[before + 1];
        REQUIRE(a.boundary == detail::ReplayOnlineBoundary::BeforeGrowth);
        REQUIRE(b.boundary == detail::ReplayOnlineBoundary::AfterGrowth);
        REQUIRE(b.capacity == a.capacity + 1);
        REQUIRE(b.width == a.width + 1);
        REQUIRE(a.sources == b.sources);
    }
    REQUIRE(trace.snapshots[2].z == Catch::Approx(1.0 / 3));
    REQUIRE(trace.snapshots[2].x == Catch::Approx(2 * std::sqrt(0.8 * 0.4) / 1.2));
    const auto* grown = trace.snapshots[4].data;
    for (size_t i = 5; i < trace.count; ++i) {
        REQUIRE(trace.snapshots[i].data == grown);
        REQUIRE(trace.snapshots[i].capacity == 2);
        REQUIRE(trace.snapshots[i].width == 0);
    }
}

TEST_CASE("Weighted trajectory reset and dispatch allocate only outside hot intervals",
          "[noncomp][weighted_replay]") {
    const auto model = NonComputationalModel::from_spec(
        pure_initial_state(Level::G),
        {{"lose", certain_transition_from_computational(Level::Lost)}},
        classifier_matrix_with_column(Level::Lost, {1, 0}), {.reset_restores_lost = true});
    const auto circuit = parse(
        "H 0\nT 0\nCX 0 1\nLEVEL_TRANSITION[lose] 0\n"
        "HERALD_LOSS_EVENT 0\nR 0\nR_Y(0.25) 0\nCZ rec[-1] 0\nM 0 1");
    hot_path_counters = {};
    const auto replay =
        replay_physical_histories(PreparedNonComputationalRequest(circuit, model), false, false);
    REQUIRE(replay.continuations > 0);
    REQUIRE(hot_path_counters.intervals > 0);
    REQUIRE(hot_path_counters.depth == 0);
    REQUIRE(hot_path_counters.allocations == 0);
}
