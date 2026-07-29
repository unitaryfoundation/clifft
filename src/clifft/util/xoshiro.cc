#include "clifft/util/xoshiro.h"

#include <random>

namespace clifft {

std::array<uint64_t, 4> entropy_seed_words() {
    // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94087
    // See https://github.com/quantumlib/Stim/issues/26
#if defined(__linux__) && defined(__GLIBCXX__) && __GLIBCXX__ >= 20200128
    std::random_device rd("/dev/urandom");
#else
    std::random_device rd;
#endif
    auto rd64 = [&rd]() -> uint64_t { return (static_cast<uint64_t>(rd()) << 32) | rd(); };
    return {rd64(), rd64(), rd64(), rd64()};
}

void Xoshiro256PlusPlus::seed_from_entropy() {
    const auto w = entropy_seed_words();
    seed_full(w[0], w[1], w[2], w[3]);
}

}  // namespace clifft
