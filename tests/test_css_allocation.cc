#include "clifft/circuit/parser.h"
#include "clifft/sampling/css/planner.h"
#include "clifft/sampling/executor.h"

#include <cstdlib>
#include <fstream>
#include <new>
#include <sstream>

namespace {
bool forbid_new = false;
}

// Isolated executable: Catch2 and its reporters allocate while checking
// assertions, so the hot-dispatch guard must not intercept their bookkeeping.
void* operator new(std::size_t size) {
    if (forbid_new)
        std::abort();
    if (void* memory = std::malloc(size ? size : 1))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    return ::operator new(size);
}
void operator delete(void* memory) noexcept {
    std::free(memory);
}
void operator delete[](void* memory) noexcept {
    std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}
void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}

int main() {
    for (unsigned distance : {7u, 9u}) {
        std::ifstream file(std::string(CLIFFT_FIXTURES_DIR) +
                           "/../../tools/bench/fixtures/css_five_check_d" +
                           std::to_string(distance) + ".stim");
        if (!file.good())
            return 1;
        std::stringstream text;
        text << file.rdbuf();
        auto circuit = clifft::parse(text.str());
        auto candidate = clifft::sampling::try_plan_css_blocks(circuit);
        if (!candidate.plan)
            return 2;
        clifft::sampling::ExecutablePlan plan(*candidate.plan);
        clifft::sampling::Executor executor(plan, 516);
        forbid_new = true;
        for (unsigned shot = 0; shot < 32; ++shot)
            executor.run_shot();
        forbid_new = false;
    }
}
