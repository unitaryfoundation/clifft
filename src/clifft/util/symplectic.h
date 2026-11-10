#pragma once

// Symplectic inner product over runtime-width mask views, needed by both
// optimizer/ (commutation analysis) and sampling/ (the planner's
// logical-noise-crossing sign correction) without either depending on the
// other.

#include "clifft/util/mask_view.h"

#include <bit>
#include <cassert>
#include <cstdint>

namespace clifft {

// Returns true if the two Pauli strings anti-commute. All four views must
// share num_words().
inline bool anti_commute(MaskView x1, MaskView z1, MaskView x2, MaskView z2) {
    assert(x1.num_words() == z1.num_words() && z1.num_words() == x2.num_words() &&
           x2.num_words() == z2.num_words());
    int parity = 0;
    for (uint32_t i = 0; i < x1.num_words(); ++i) {
        parity += std::popcount((x1.words[i] & z2.words[i]) ^ (z1.words[i] & x2.words[i]));
    }
    return (parity & 1) != 0;
}

// Reuses the nonzero word range of one Pauli across many commutation checks.
// The borrowed masks must remain unchanged for the lifetime of the checker.
// Local axes on wide registers then inspect only the words they can affect.
class PauliCommutation {
  public:
    PauliCommutation(MaskView x, MaskView z) : x_(x), z_(z), end_(x.num_words()) {
        assert(x.num_words() == z.num_words());
        while (end_ > 0 && (x.words[end_ - 1] | z.words[end_ - 1]) == 0) {
            --end_;
        }
        while (begin_ < end_ && (x.words[begin_] | z.words[begin_]) == 0) {
            ++begin_;
        }
    }

    [[nodiscard]] bool references(MaskView x, MaskView z) const {
        return x.words.data() == x_.words.data() && z.words.data() == z_.words.data() &&
               x.num_words() == x_.num_words() && z.num_words() == z_.num_words();
    }

    // Half-open range of words where the borrowed Pauli is nonzero.
    [[nodiscard]] uint32_t begin_word() const { return begin_; }
    [[nodiscard]] uint32_t end_word() const { return end_; }

    [[nodiscard]] bool anti_commutes(MaskView x, MaskView z) const {
        assert(x.num_words() == x_.num_words() && z.num_words() == z_.num_words());
        return anti_commute(MaskView{x_.words.subspan(begin_, end_ - begin_)},
                            MaskView{z_.words.subspan(begin_, end_ - begin_)},
                            MaskView{x.words.subspan(begin_, end_ - begin_)},
                            MaskView{z.words.subspan(begin_, end_ - begin_)});
    }

  private:
    MaskView x_;
    MaskView z_;
    uint32_t begin_ = 0;
    uint32_t end_;
};

}  // namespace clifft
