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
        case Op::U2: return "U2";
        case Op::U4: return "U4";
        case Op::EXPAND: return "EXPAND";
        case Op::EXPAND_T: return "EXPAND_T";
        case Op::MEAS_DIAG: return "MEAS_DIAG";
        case Op::MEAS_INTERFERE: return "MEAS_INTERFERE";
    }
    return "?";
}

// One layer = H on every axis, T on every axis, a CNOT chain, one CZ, one
// fused U2, one fused U4. Replayed `layers` times. Represents dense work on
// an active block of rank k, including the post-optimizer fused opcodes.
std::vector<ScheduledOp> make_layer_schedule(unsigned k, unsigned layers) {
    std::vector<ScheduledOp> sched;
    for (unsigned l = 0; l < layers; ++l) {
        for (unsigned v = 0; v < k; ++v) sched.push_back({Op::H, v, 0});
        for (unsigned v = 0; v < k; ++v) sched.push_back({Op::T, v, 0});
        for (unsigned v = 0; v + 1 < k; ++v) sched.push_back({Op::CNOT, v, v + 1});
        if (k >= 2) sched.push_back({Op::CZ, 0, k - 1});
        sched.push_back({Op::U2, k / 2, 0});
        if (k >= 4) sched.push_back({Op::U4, 1, k / 2});
        else if (k >= 2) sched.push_back({Op::U4, 0, 1});
    }
    return sched;
}

// Gate layer + one completed measurement (rank k -> k-1) + one expand
// (k-1 -> k): the batch shape is identical every layer, mirroring clifft's
// static shot-invariant k trajectory.
std::vector<ScheduledOp> make_layer_schedule_meas(unsigned k, unsigned layers) {
    std::vector<ScheduledOp> sched;
    auto gates = make_layer_schedule(k, 1);
    for (unsigned l = 0; l < layers; ++l) {
        sched.insert(sched.end(), gates.begin(), gates.end());
        sched.push_back({Op::MEAS_DIAG, k - 1, 0});  // k -> k-1
        sched.push_back({Op::EXPAND, k - 1, 0});     // k-1 -> k (arg = current rank)
    }
    return sched;
}

}  // namespace mb
