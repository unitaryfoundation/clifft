#pragma once

#include "clifft/sampling/planner.h"

#include <cstdint>
#include <string>
#include <vector>

namespace coordinate_reuse {

enum class Policy { Native, Identity, Columns, Inverse32 };

struct Interval {
    uint64_t queries = 0;
    uint64_t unique_queries = 0;
    uint64_t identity_queries = 0;
    uint64_t inverse_queries = 0;
    uint64_t input_weight = 0;
    uint64_t output_weight = 0;
    uint64_t unique_generators = 0;
    std::string end;
};

struct Result {
    clifft::sampling::SamplingPlan plan;
    std::vector<Interval> intervals;
    Policy policy = Policy::Native;
    bool audit = false;
    bool verify = false;
    uint64_t queries = 0;
    uint64_t identity_fast_paths = 0;
    uint64_t inverse_builds = 0;
    uint64_t column_builds = 0;
    uint64_t column_hits = 0;
    uint64_t maximum_cached_columns = 0;
    uint64_t coordinate_checks = 0;
    [[nodiscard]] std::string diagnostics() const;
};

[[nodiscard]] Result plan(const clifft::HirModule& hir, Policy policy, bool audit, bool verify);

}  // namespace coordinate_reuse
