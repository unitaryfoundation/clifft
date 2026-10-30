"""Generate a locally instrumented driver without modifying production sources."""

import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
preamble = r"""
#include <chrono>
#include <cstdio>
#include <array>
namespace {
struct PhaseTotals {
    std::array<double, 8> seconds{};
    std::array<unsigned long long, 8> calls{};
    ~PhaseTotals() {
        const char* names[] = {"rewrite", "trace", "optimize", "plan", "prepare",
                               "execute", "classical", "classify"};
        for (size_t i = 0; i < seconds.size(); ++i)
            std::fprintf(stderr, "PHASE %s %.9f %llu\n", names[i], seconds[i], calls[i]);
    }
};
PhaseTotals phase_totals;
struct PhaseTimer {
    size_t phase;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~PhaseTimer() {
        const auto elapsed = std::chrono::steady_clock::now() - start;
        phase_totals.seconds[phase] += std::chrono::duration<double>(elapsed).count();
        ++phase_totals.calls[phase];
    }
};
template<class F> decltype(auto) timed_phase(size_t phase, F&& f) {
    PhaseTimer timer{phase};
    return f();
}
}
"""


def wrap_calls(text, name, phase):
    start = 0
    while True:
        pos = text.find(name + "(", start)
        if pos < 0:
            break
        end = pos + len(name) + 1
        depth = 1
        while depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        replacement = f"timed_phase({phase}, [&]() {{ return {text[pos:end]}; }})"
        text = text[:pos] + replacement + text[end:]
        start = pos + len(replacement)
    return text


for name, phase in [("trace", 1), ("sampling::plan_sampling", 3), ("sampling::ExecutablePlan", 4)]:
    source = wrap_calls(source, name, phase)
source = source.replace(
    "trajectory_hir_pass_manager().run(hir);",
    "timed_phase(2, [&]() { trajectory_hir_pass_manager().run(hir); });",
)
pos = source.index("NonComputationalSample run_trajectory_driver")
head, tail = source[:pos], source[pos:]
for name, phase in [
    ("rewrite_continuation", 0),
    ("executor->run_shot", 5),
    ("executor->resume", 5),
    ("extend_classical_outcomes", 6),
    ("flags_for", 7),
]:
    tail = wrap_calls(tail, name, phase)
pathlib.Path(sys.argv[2]).write_text(preamble + head + tail)
