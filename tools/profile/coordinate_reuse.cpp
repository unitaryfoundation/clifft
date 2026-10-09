// Compile the unchanged planner against a research-only coordinate wrapper.
// Production builds and the ordinary planner retain their existing behavior.

#include "coordinate_reuse.h"

#include "clifft/sampling/planner_frame.h"

#include <algorithm>
#include <bit>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace coordinate_reuse {
namespace {
thread_local Result* active_result = nullptr;
}
}  // namespace coordinate_reuse

namespace clifft::sampling::internal {

class ResearchCoordinateFrame : public CoordinateFrame {
  public:
    explicit ResearchCoordinateFrame(uint32_t n) : CoordinateFrame(n) {
        if (!coordinate_reuse::active_result)
            throw std::logic_error("Missing coordinate diagnostic context");
        if (coordinate_reuse::active_result->policy == coordinate_reuse::Policy::Columns)
            columns_.resize(2 * static_cast<size_t>(n));
    }

    ~ResearchCoordinateFrame() { finish("end"); }

    PlannerPauli to_current(const PlannerPauli& initial) {
        auto& context = *coordinate_reuse::active_result;
        ++context.queries;
        auto result = convert(initial);
        if (context.verify && context.policy != coordinate_reuse::Policy::Native) {
            if (result != CoordinateFrame::to_current(initial))
                throw std::runtime_error("Research coordinate conversion differs from native");
            ++context.coordinate_checks;
        }
        if (context.audit) {
            ++interval_.queries;
            interval_.identity_queries += initial.x().is_zero() && initial.z().is_zero();
            interval_.inverse_queries += has_cached_inverse_for_testing();
            interval_.input_weight += initial.x().popcount() + initial.z().popcount();
            interval_.output_weight += result.x().popcount() + result.z().popcount();
            std::vector<uint64_t> key(initial.x().words.begin(), initial.x().words.end());
            key.insert(key.end(), initial.z().words.begin(), initial.z().words.end());
            key.push_back(initial.phase());
            unique_.insert(std::move(key));
            for (uint32_t q = 0; q < initial.num_qubits(); ++q) {
                if (initial.x().bit_get(q))
                    touched_.insert(q);
                if (initial.z().bit_get(q))
                    touched_.insert(initial.num_qubits() + q);
            }
        }
        return result;
    }

    void change_basis(const PlannerTableau& change) {
        finish("basis");
        CoordinateFrame::change_basis(change);
    }

    void promote_dormant(const PlannerPauli& p, uint32_t width, uint32_t pivot) {
        finish("promote");
        CoordinateFrame::promote_dormant(p, width, pivot);
    }

    void measure_dormant(const PlannerPauli& p, uint32_t pivot) {
        finish("measure_dormant");
        CoordinateFrame::measure_dormant(p, pivot);
    }

    void measure_active(const PlannerPauli& p, uint32_t width, uint32_t pivot) {
        finish("measure_active");
        CoordinateFrame::measure_active(p, width, pivot);
    }

  private:
    coordinate_reuse::Interval interval_;
    std::set<std::vector<uint64_t>> unique_;
    std::set<uint32_t> touched_;
    std::vector<std::optional<PlannerPauli>> columns_;
    std::vector<size_t> built_columns_;
    std::optional<PlannerTableau> inverse_;
    uint64_t lookups_ = 0;

    const PlannerPauli& column(uint32_t q, bool z) {
        auto& context = *coordinate_reuse::active_result;
        const auto& frame = current_to_initial();
        const uint32_t n = frame.num_qubits();
        const size_t index = (z ? n : 0) + q;
        if (columns_[index]) {
            ++context.column_hits;
            return *columns_[index];
        }
        PlannerPauli result(n);
        for (uint32_t k = 0; k < n; ++k) {
            const auto x = frame.x_output(k), y = frame.z_output(k);
            result.set_pauli(k, z ? y.x().bit_get(q) : y.z().bit_get(q),
                             z ? x.x().bit_get(q) : x.z().bit_get(q));
        }
        result.set_sign(false);
        // Symplectic coordinates determine the body; a forward round trip
        // recovers the inverse generator's sign, including Y contributions.
        result.set_sign(frame.apply(result.view()).sign());
        columns_[index].emplace(std::move(result));
        built_columns_.push_back(index);
        ++context.column_builds;
        context.maximum_cached_columns =
            std::max<uint64_t>(context.maximum_cached_columns, built_columns_.size());
        return *columns_[index];
    }

    PlannerPauli convert(const PlannerPauli& initial) {
        auto& context = *coordinate_reuse::active_result;
        if (context.policy == coordinate_reuse::Policy::Native)
            return CoordinateFrame::to_current(initial);
        if (initial.x().is_zero() && initial.z().is_zero()) {
            ++context.identity_fast_paths;
            return initial;
        }
        if (context.policy == coordinate_reuse::Policy::Identity)
            return CoordinateFrame::to_current(initial);
        if (context.policy == coordinate_reuse::Policy::Inverse32) {
            if (!inverse_ && lookups_++ >= 32) {
                inverse_.emplace(current_to_initial().inverse());
                ++context.inverse_builds;
            }
            return inverse_ ? inverse_->apply(initial.view())
                            : CoordinateFrame::to_current(initial);
        }
        PlannerPauli result(initial.num_qubits());
        // Preserve the raw i^p X^x Z^z phase and multiplication order, as in
        // Tableau::apply; treating Y as an unsigned product loses its sign.
        result.set_phase(initial.phase());
        for (const bool z : {false, true}) {
            const auto mask = z ? initial.z() : initial.x();
            for (uint32_t w = 0; w < mask.num_words(); ++w) {
                uint64_t pending = mask.words[w];
                while (pending) {
                    const uint32_t q = 64 * w + std::countr_zero(pending);
                    if (q < initial.num_qubits())
                        result.right_multiply(column(q, z).view());
                    pending &= pending - 1;
                }
            }
        }
        return result;
    }

    void finish(const char* reason) {
        if (coordinate_reuse::active_result->audit) {
            interval_.unique_queries = unique_.size();
            interval_.unique_generators = touched_.size();
            interval_.end = reason;
            coordinate_reuse::active_result->intervals.push_back(interval_);
            interval_ = {};
            unique_.clear();
            touched_.clear();
        }
        for (const auto index : built_columns_)
            columns_[index].reset();
        built_columns_.clear();
        inverse_.reset();
        lookups_ = 0;
    }
};

}  // namespace clifft::sampling::internal

// Header guards keep production declarations intact. Only the separately
// compiled planner body sees the alternate type and entry-point names.
#define CoordinateFrame ResearchCoordinateFrame
#define plan_sampling research_plan_sampling
#include "../../src/clifft/sampling/planner.cc"
#undef plan_sampling
#undef CoordinateFrame

namespace coordinate_reuse {

Result plan(const clifft::HirModule& hir, Policy policy, bool audit, bool verify) {
    if (active_result)
        throw std::logic_error("Coordinate diagnostics are not reentrant");
    Result result;
    result.policy = policy;
    result.audit = audit;
    result.verify = verify;
    active_result = &result;
    try {
        result.plan = clifft::sampling::research_plan_sampling(hir, {});
    } catch (...) {
        active_result = nullptr;
        throw;
    }
    active_result = nullptr;
    return result;
}

std::string Result::diagnostics() const {
    std::ostringstream out;
    out << "{\"queries\":" << queries << ",\"identity_fast_paths\":" << identity_fast_paths
        << ",\"inverse_builds\":" << inverse_builds << ",\"column_builds\":" << column_builds
        << ",\"column_hits\":" << column_hits
        << ",\"maximum_cached_columns\":" << maximum_cached_columns
        << ",\"coordinate_checks\":" << coordinate_checks << ",\"intervals\":[";
    for (size_t i = 0; i < intervals.size(); ++i) {
        const auto& row = intervals[i];
        out << (i ? ",{" : "{") << "\"queries\":" << row.queries
            << ",\"unique_queries\":" << row.unique_queries
            << ",\"identity_queries\":" << row.identity_queries
            << ",\"inverse_queries\":" << row.inverse_queries
            << ",\"input_weight\":" << row.input_weight
            << ",\"output_weight\":" << row.output_weight
            << ",\"unique_generators\":" << row.unique_generators << ",\"end\":\"" << row.end
            << "\"}";
    }
    out << "]}";
    return out.str();
}

}  // namespace coordinate_reuse
