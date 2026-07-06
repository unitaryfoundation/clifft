#pragma once

#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/drop_non_unitary_pass.h"
#include "clifft/optimizer/expand_t_pass.h"
#include "clifft/optimizer/hir_pass.h"
#include "clifft/optimizer/multi_gate_pass.h"
#include "clifft/optimizer/noise_block_pass.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/optimizer/single_axis_fusion_pass.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/optimizer/swap_meas_pass.h"
#include "clifft/optimizer/tile_axis_fusion_pass.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace clifft {

enum class PassKind : uint8_t { HIR, Bytecode };

using HirPassFactory = std::unique_ptr<HirPass> (*)();
using BytecodePassFactory = std::unique_ptr<BytecodePass> (*)();

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

struct PassInfo {
    std::string_view name;
    PassKind kind;
    bool default_enabled;
    RecordOrder record_order;
    HirPassFactory make_hir = nullptr;
    BytecodePassFactory make_bc = nullptr;
};

template <typename T>
std::unique_ptr<HirPass> make_hir() {
    return std::make_unique<T>();
}

template <typename T>
std::unique_ptr<BytecodePass> make_bc() {
    return std::make_unique<T>();
}

// Single source of truth for all available optimization passes.
// Each entry defines metadata AND the factory function used to construct it.
inline const PassInfo kRegisteredPasses[] = {
    // HIR passes
    {.name = "PeepholeFusionPass",
     .kind = PassKind::HIR,
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .make_hir = make_hir<PeepholeFusionPass>},
    {.name = "StatevectorSqueezePass",
     .kind = PassKind::HIR,
     .default_enabled = true,
     .record_order = kBreaksRecordOrder,
     .make_hir = make_hir<StatevectorSqueezePass>},
    {.name = "RemoveNoisePass",
     .kind = PassKind::HIR,
     .default_enabled = false,
     .record_order = kPreservesRecordOrder,
     .make_hir = make_hir<RemoveNoisePass>},
    {.name = "DropNonUnitaryPass",
     .kind = PassKind::HIR,
     .default_enabled = false,
     .record_order = kBreaksRecordOrder,
     .make_hir = make_hir<DropNonUnitaryPass>},
    // Bytecode passes
    {.name = "NoiseBlockPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .make_bc = make_bc<NoiseBlockPass>},
    {.name = "MultiGatePass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .make_bc = make_bc<MultiGatePass>},
    {.name = "ExpandTPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .make_bc = make_bc<ExpandTPass>},
    {.name = "ExpandRotPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .make_bc = make_bc<ExpandRotPass>},
    {.name = "SwapMeasPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .make_bc = make_bc<SwapMeasPass>},
    {.name = "TileAxisFusionPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .make_bc = make_bc<TileAxisFusionPass>},
    {.name = "SingleAxisFusionPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .record_order = kPreservesRecordOrder,
     .make_bc = make_bc<SingleAxisFusionPass>},
};

inline constexpr size_t kNumRegisteredPasses =
    sizeof(kRegisteredPasses) / sizeof(kRegisteredPasses[0]);

}  // namespace clifft
