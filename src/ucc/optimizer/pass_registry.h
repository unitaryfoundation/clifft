#pragma once

#include "ucc/optimizer/bytecode_pass.h"
#include "ucc/optimizer/expand_t_pass.h"
#include "ucc/optimizer/hir_pass.h"
#include "ucc/optimizer/multi_gate_pass.h"
#include "ucc/optimizer/noise_block_pass.h"
#include "ucc/optimizer/peephole.h"
#include "ucc/optimizer/remove_noise_pass.h"
#include "ucc/optimizer/single_axis_fusion_pass.h"
#include "ucc/optimizer/statevector_squeeze_pass.h"
#include "ucc/optimizer/swap_meas_pass.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace ucc {

enum class PassKind : uint8_t { HIR, Bytecode };

using HirPassFactory = std::unique_ptr<HirPass> (*)();
using BytecodePassFactory = std::unique_ptr<BytecodePass> (*)();

struct PassInfo {
    std::string_view name;
    PassKind kind;
    bool default_enabled;
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
     .make_hir = make_hir<PeepholeFusionPass>},
    {.name = "StatevectorSqueezePass",
     .kind = PassKind::HIR,
     .default_enabled = true,
     .make_hir = make_hir<StatevectorSqueezePass>},
    {.name = "RemoveNoisePass",
     .kind = PassKind::HIR,
     .default_enabled = false,
     .make_hir = make_hir<RemoveNoisePass>},
    // Bytecode passes
    {.name = "NoiseBlockPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .make_bc = make_bc<NoiseBlockPass>},
    {.name = "MultiGatePass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .make_bc = make_bc<MultiGatePass>},
    {.name = "ExpandTPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .make_bc = make_bc<ExpandTPass>},
    {.name = "ExpandRotPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .make_bc = make_bc<ExpandRotPass>},
    {.name = "SwapMeasPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .make_bc = make_bc<SwapMeasPass>},
    {.name = "SingleAxisFusionPass",
     .kind = PassKind::Bytecode,
     .default_enabled = true,
     .make_bc = make_bc<SingleAxisFusionPass>},
};

inline constexpr size_t kNumRegisteredPasses =
    sizeof(kRegisteredPasses) / sizeof(kRegisteredPasses[0]);

}  // namespace ucc
