#include "clifft/optimizer/zx_spacetime_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"

#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>

namespace clifft {
namespace {

// Nullopt when the module exceeds the dense limit; other planning errors
// propagate.
std::optional<uint32_t> planned_peak(const HirModule& hir) {
    try {
        return sampling::plan_sampling(hir).peak_active_width;
    } catch (const std::overflow_error&) {
        return std::nullopt;
    }
}

}  // namespace

ZxSpacetimePass::ZxSpacetimePass(ZxSpacetimeOptions options) : options_(options) {
    if (options_.max_qubits == 0) {
        throw std::invalid_argument("ZxSpacetimePass: max_qubits must be positive");
    }
    if (options_.max_reduce_work == 0) {
        throw std::invalid_argument("ZxSpacetimePass: max_reduce_work must be positive");
    }
}

void ZxSpacetimePass::run(HirModule& hir) {
    applied_ = false;
    input_peak_ = 0;
    result_peak_ = 0;
    reason_.clear();

    auto rewritten = zx::rewrite_hir(hir, options_, &reason_);
    if (!rewritten) {
        return;
    }
    HirModule baseline = hir;
    default_hir_pass_manager().run(baseline);
    HirModule candidate = *rewritten;
    default_hir_pass_manager().run(candidate);

    const std::optional<uint32_t> before = planned_peak(baseline);
    std::optional<uint32_t> after;
    try {
        after = planned_peak(candidate);
    } catch (const std::exception&) {
        // The input plans, so a rewrite that does not is a lowering bug;
        // keep the input rather than fail the compile.
        reason_ = "the rewrite could not be planned";
        return;
    }
    if (before && after) {
        input_peak_ = *before;
        result_peak_ = *after;
    } else {
        input_peak_ = analyze_active_width(baseline).peak_width;
        result_peak_ = analyze_active_width(candidate).peak_width;
    }
    if (result_peak_ >= input_peak_) {
        reason_ = "the rewrite is not narrower";
        return;
    }
    hir = std::move(*rewritten);
    applied_ = true;
}

}  // namespace clifft
