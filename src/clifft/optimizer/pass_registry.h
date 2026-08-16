#pragma once

#include "clifft/optimizer/drop_non_unitary_pass.h"
#include "clifft/optimizer/hir_pass.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"

#include <cstddef>
#include <memory>
#include <string_view>

namespace clifft {

using HirPassFactory = std::unique_ptr<HirPass> (*)();

// Whether a pass keeps every record-writing operation (visible and
// hidden measurement records alike) in program order, neither
// reordering nor removing one. Reordering commuting measurements is
// sound under sampling semantics -- the outcomes are exchangeable --
// but not under forced-outcome execution, where a forced collapse must
// happen before the operations correlated with it; pipelines that
// force outcomes run only passes declaring `preserved`. There is no
// default: a newly registered pass that does not declare its behavior
// fails to compile rather than silently joining (or silently breaking)
// those pipelines.
struct RecordOrder {
    bool preserved;
    constexpr explicit RecordOrder(bool preserved_in) : preserved(preserved_in) {}
};

inline constexpr RecordOrder kPreservesRecordOrder{true};
inline constexpr RecordOrder kBreaksRecordOrder{false};

// Whether changing the input after an INSTRUMENT can change the pass output
// through that instrument. A resumed trajectory recompiles a changed suffix
// while reusing the state produced by the already-executed prefix, so such
// passes must make an explicit stability guarantee. There is intentionally no
// default: adding a pass requires its author to opt in or keep it out of the
// trajectory pipeline.
struct InstrumentPrefixStability {
    bool preserved;
    constexpr explicit InstrumentPrefixStability(bool preserved_in) : preserved(preserved_in) {}
};

inline constexpr InstrumentPrefixStability kPreservesInstrumentPrefix{true};
inline constexpr InstrumentPrefixStability kMayChangeInstrumentPrefix{false};

struct PassInfo {
    std::string_view name;
    bool default_enabled;
    RecordOrder record_order;
    InstrumentPrefixStability instrument_prefix;
    HirPassFactory make;
};

[[nodiscard]] constexpr bool is_trajectory_compatible(const PassInfo& info) {
    return info.record_order.preserved && info.instrument_prefix.preserved;
}

template <typename T>
std::unique_ptr<HirPass> make_hir() {
    return std::make_unique<T>();
}

// Single source of truth for all available optimization passes.
// Each entry defines metadata AND the factory function used to construct it.
inline const PassInfo kRegisteredPasses[] = {
    // HIR passes
    {.name = "PeepholeFusionPass",
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .instrument_prefix = kPreservesInstrumentPrefix,
     .make = make_hir<PeepholeFusionPass>},
    {.name = "StatevectorSqueezePass",
     .default_enabled = true,
     .record_order = kBreaksRecordOrder,
     .instrument_prefix = kMayChangeInstrumentPrefix,
     .make = make_hir<StatevectorSqueezePass>},
    {.name = "RemoveNoisePass",
     .default_enabled = false,
     .record_order = kBreaksRecordOrder,
     .instrument_prefix = kMayChangeInstrumentPrefix,
     .make = make_hir<RemoveNoisePass>},
    {.name = "DropNonUnitaryPass",
     .default_enabled = false,
     .record_order = kBreaksRecordOrder,
     .instrument_prefix = kMayChangeInstrumentPrefix,
     .make = make_hir<DropNonUnitaryPass>},
};

inline constexpr size_t kNumRegisteredPasses =
    sizeof(kRegisteredPasses) / sizeof(kRegisteredPasses[0]);

}  // namespace clifft
