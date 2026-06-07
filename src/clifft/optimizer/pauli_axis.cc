#include "clifft/optimizer/pauli_axis.h"

namespace clifft {

void PauliAxis::resize(uint32_t num_words) {
    x.assign(num_words, 0);
    z.assign(num_words, 0);
}

void PauliAxis::set_from(MaskView xv, MaskView zv) {
    x.assign(xv.words.begin(), xv.words.end());
    z.assign(zv.words.begin(), zv.words.end());
}

void PauliAxis::xor_with(MaskView xv, MaskView zv) {
    for (uint32_t w = 0; w < xv.num_words(); ++w) {
        x[w] ^= xv.words[w];
        z[w] ^= zv.words[w];
    }
}

bool PauliAxis::is_zero() const {
    for (uint64_t w : x) {
        if (w != 0)
            return false;
    }
    for (uint64_t w : z) {
        if (w != 0)
            return false;
    }
    return true;
}

}  // namespace clifft
