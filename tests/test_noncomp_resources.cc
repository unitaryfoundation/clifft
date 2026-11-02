#include "clifft/circuit/parser.h"
#include "clifft/util/page_allocation.h"

#include "noncomp_replay_helpers.h"
#include "noncomp_resource_probe.h"
#include "noncomp_test_helpers.h"

using namespace clifft;
using namespace clifft::test;

namespace {

struct ResourceReplay : NoncompReplayHooks {
    struct Snapshot {
        const double* data = nullptr;
        uint32_t width = 0, capacity = 0;
        unsigned sources = 0;
    };
    std::array<Snapshot, 4> snapshots{};
    size_t count = 0;
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
    void inspect_online(const noncomp_detail::OnlineState& quantum,
                        std::span<const QubitStatus>) override {
        REQUIRE(count < snapshots.size());
        REQUIRE(hot_path_counters.depth == 0);
        snapshots[count++] = {quantum.state.real_data(), quantum.state.active_width(),
                              quantum.state.max_active_width(), sources};
    }
};

struct ResourceScope {
    NoncompReplayHooks* previous = noncomp_replay;
    explicit ResourceScope(ResourceReplay& replay) {
        noncomp_replay = &replay;
        hot_path_counters = {};
    }
    ~ResourceScope() { noncomp_replay = previous; }
};

}  // namespace

TEST_CASE("Weighted resource probe detects ordinary and coefficient allocations",
          "[noncomp][weighted_replay]") {
    hot_path_counters = {};
    void* pointer;
    size_t ordinary_allocations;
    {
        detail::ReplayHotPathScope hot_path;
        // Call the replaceable function directly so allocation elision cannot
        // turn the probe's positive control into a false success.
        auto* volatile allocate = static_cast<void* (*)(size_t)>(&::operator new);
        pointer = allocate(32);
        ordinary_allocations = hot_path_counters.allocations;
        PageAlignedAllocation coefficients(32);
    }
    ::operator delete(pointer);
    REQUIRE(hot_path_counters.depth == 0);
    REQUIRE(hot_path_counters.intervals == 1);
    REQUIRE(ordinary_allocations > 0);
    REQUIRE(hot_path_counters.allocations > ordinary_allocations);
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
    ResourceReplay replay;
    ResourceScope scope(replay);
    const auto sample = program.sample(4, 17, 1, true);
    REQUIRE(hot_path_counters.allocations == 0);
    REQUIRE(hot_path_counters.depth == 0);
    REQUIRE(hot_path_counters.intervals >= 10);
    REQUIRE(replay.sources == 4);
    REQUIRE(sample.stats->online_state_growths == 2);
    REQUIRE(sample.stats->continuation_compilations == 0);
    REQUIRE(replay.count == 4);
    REQUIRE(replay.snapshots[0].data != nullptr);
    for (size_t i = 0; i < replay.count; ++i) {
        REQUIRE(replay.snapshots[i].data == replay.snapshots[0].data);
        REQUIRE(replay.snapshots[i].capacity == 2);
        REQUIRE(replay.snapshots[i].width == 0);
        REQUIRE(replay.snapshots[i].sources == i + 1);
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
