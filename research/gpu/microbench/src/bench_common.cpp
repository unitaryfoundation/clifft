#include "bench_common.hpp"

#include <algorithm>

namespace mb {

double median(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    size_t n = xs.size();
    return (n & 1) ? xs[n / 2] : 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}

const char* op_name(Op op) {
    switch (op) {
        case Op::H: return "H";
        case Op::T: return "T";
        case Op::CZ: return "CZ";
        case Op::CNOT: return "CNOT";
        case Op::EXPAND: return "EXPAND";
        case Op::EXPAND_T: return "EXPAND_T";
        case Op::MEAS_DIAG: return "MEAS_DIAG";
        case Op::MEAS_INTERFERE: return "MEAS_INTERFERE";
    }
    return "?";
}

// One layer = H on every axis, T on every axis, a CNOT chain, then one CZ.
// Replayed `layers` times. Represents dense work on an active block of rank k.
std::vector<ScheduledOp> make_layer_schedule(unsigned k, unsigned layers) {
    std::vector<ScheduledOp> sched;
    for (unsigned l = 0; l < layers; ++l) {
        for (unsigned v = 0; v < k; ++v) sched.push_back({Op::H, v, 0});
        for (unsigned v = 0; v < k; ++v) sched.push_back({Op::T, v, 0});
        for (unsigned v = 0; v + 1 < k; ++v) sched.push_back({Op::CNOT, v, v + 1});
        if (k >= 2) sched.push_back({Op::CZ, 0, k - 1});
    }
    return sched;
}

}  // namespace mb
