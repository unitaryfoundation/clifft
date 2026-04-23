"""MkDocs macros hook: loads opcodes.json and pass metadata for template rendering."""

import json
from pathlib import Path
from typing import Any


def define_env(env: Any) -> None:
    """Called by mkdocs-macros-plugin to inject template variables."""
    docs_dir = Path(env.conf["docs_dir"])
    data_path = docs_dir / "opcodes.json"

    with open(data_path) as f:
        data = json.load(f)

    # -- Opcodes --
    opcodes = data.get("opcodes", {})
    opcode_categories_order = [
        "Frame",
        "Array",
        "Subspace",
        "Measurement",
        "Meta",
    ]
    opcodes_by_category: dict[str, list[dict[str, str]]] = {}
    for cat in opcode_categories_order:
        opcodes_by_category[cat] = []
    for name, info in opcodes.items():
        cat = info.get("category", "Meta")
        entry = {"name": name, **info}
        if cat not in opcodes_by_category:
            opcodes_by_category[cat] = []
        opcodes_by_category[cat].append(entry)

    env.variables["opcodes"] = opcodes
    env.variables["opcode_categories"] = [
        c for c in opcode_categories_order if opcodes_by_category.get(c)
    ]
    env.variables["opcodes_by_category"] = opcodes_by_category

    # -- HIR ops --
    hir_ops = data.get("hir_ops", {})
    hir_categories_order = [
        "Non-Clifford",
        "Measurement",
        "Feedback",
        "Noise",
        "QEC",
    ]
    hir_by_category: dict[str, list[dict[str, object]]] = {}
    for cat in hir_categories_order:
        hir_by_category[cat] = []
    for name, info in hir_ops.items():
        cat = info.get("category", "Meta")
        entry = {"name": name, **info}
        if cat not in hir_by_category:
            hir_by_category[cat] = []
        hir_by_category[cat].append(entry)

    env.variables["hir_ops"] = hir_ops
    env.variables["hir_categories"] = [c for c in hir_categories_order if hir_by_category.get(c)]
    env.variables["hir_by_category"] = hir_by_category

    # -- Optimization passes --
    # Hardcoded from pass_registry.h (single source of truth in C++).
    # When a new pass is added in C++, add it here too.
    passes = [
        {
            "name": "PeepholeFusionPass",
            "kind": "HIR",
            "default_enabled": True,
            "python_name": "PeepholeFusionPass",
            "summary": "Algebraic T-gate cancellation and fusion.",
            "detail": (
                "Scans the HIR to cancel or fuse T/T_dag gates acting on the "
                "same virtual Pauli axis using the symplectic inner product as "
                "a commutation check. T+T fuses to S, T+T_dag cancels to identity."
            ),
        },
        {
            "name": "StatevectorSqueezePass",
            "kind": "HIR",
            "default_enabled": True,
            "python_name": "StatevectorSqueezePass",
            "summary": "Minimizes peak active dimension by reordering HIR operations.",
            "detail": (
                "Reduces `peak_rank` by compacting qubit lifetimes. "
                "Sweep 1 (leftward) bubbles MEASURE ops as early as possible. "
                "Sweep 2 (rightward) bubbles T_GATE and PHASE_ROTATION ops as "
                "late as possible. Measurements free active dimensions sooner, "
                "and non-Clifford expansions are deferred."
            ),
        },
        {
            "name": "RemoveNoisePass",
            "kind": "HIR",
            "default_enabled": False,
            "python_name": "RemoveNoisePass",
            "summary": "Strips all noise from the HIR.",
            "detail": (
                "Removes all stochastic noise and readout noise ops, and clears "
                "the noise_sites, readout_noise side-tables and source_map. "
                "Not included in the default pipeline. Used internally by "
                "compute_reference_syndrome() to produce a noiseless circuit copy "
                "for reference-shot extraction."
            ),
        },
        {
            "name": "NoiseBlockPass",
            "kind": "Bytecode",
            "default_enabled": True,
            "python_name": "NoiseBlockPass",
            "summary": "Coalesces contiguous noise instructions into blocks.",
            "detail": (
                "Collapses contiguous OP_NOISE instructions with consecutive site "
                "indices into single OP_NOISE_BLOCK instructions. The VM's "
                "exponential gap-sampling already skips silent noise sites in O(1), "
                "but without this pass the dispatch loop still individually fetches "
                "and decodes each OP_NOISE. For a d=5 surface code, this collapses "
                "~3400 OP_NOISE instructions into ~24 OP_NOISE_BLOCK instructions."
            ),
        },
        {
            "name": "MultiGatePass",
            "kind": "Bytecode",
            "default_enabled": True,
            "python_name": "MultiGatePass",
            "summary": "Fuses star-graph CNOT/CZ patterns into single array sweeps.",
            "detail": (
                "Fuses contiguous ARRAY_CNOT instructions sharing a target axis into "
                "OP_ARRAY_MULTI_CNOT, and contiguous ARRAY_CZ sharing a control axis "
                "into OP_ARRAY_MULTI_CZ. These star-graph patterns arise from the "
                "backend's Pauli localization pass. The fused instruction processes all "
                "controls/targets in one O(2^k) array pass using popcount-based parity."
            ),
        },
        {
            "name": "ExpandTPass",
            "kind": "Bytecode",
            "default_enabled": True,
            "python_name": "ExpandTPass",
            "summary": "Fuses EXPAND + T-phase into a single array pass.",
            "detail": (
                "Fuses contiguous OP_EXPAND + OP_PHASE_T (or T_DAG) pairs into "
                "single OP_EXPAND_T (or OP_EXPAND_T_DAG) instructions. The separate "
                "instructions cause two array passes; the fused instruction performs "
                "both in one loop: arr[i+half] = arr[i] * exp(+/-i*pi/4)."
            ),
        },
        {
            "name": "ExpandRotPass",
            "kind": "Bytecode",
            "default_enabled": True,
            "python_name": "ExpandRotPass",
            "summary": "Fuses EXPAND + continuous rotation into a single array pass.",
            "detail": (
                "Fuses contiguous OP_EXPAND + OP_PHASE_ROT pairs into single "
                "OP_EXPAND_ROT instructions, eliminating the two-pass penalty of "
                "separate expand and phase-rotate operations."
            ),
        },
        {
            "name": "SwapMeasPass",
            "kind": "Bytecode",
            "default_enabled": True,
            "python_name": "SwapMeasPass",
            "summary": "Fuses SWAP + measurement into one operation.",
            "detail": (
                "Fuses contiguous OP_ARRAY_SWAP + OP_MEAS_ACTIVE_INTERFERE pairs "
                "into single OP_SWAP_MEAS_INTERFERE instructions. The backend emits "
                "a SWAP to route the measurement axis to position k-1, followed by "
                "the interfere measurement. The fused instruction eliminates the "
                "separate O(2^k) ARRAY_SWAP memory pass."
            ),
        },
        {
            "name": "TileAxisFusionPass",
            "kind": "Bytecode",
            "default_enabled": True,
            "python_name": "TileAxisFusionPass",
            "summary": "Fuses 2-qubit tile sequences into precomputed 4x4 unitaries.",
            "detail": (
                "Fuses consecutive 2-qubit operations on a fixed axis pair {a, b} "
                "into single OP_ARRAY_U4 instructions with precomputed 4x4 unitary "
                "matrices. Pre-computes 16 matrices (one per incoming Pauli frame "
                "state on the two axes) and stores them in the ConstantPool. A run "
                "is only fused if it contains at least 3 array-touching operations. "
                "Runs before SingleAxisFusionPass so it operates on raw primitives."
            ),
        },
        {
            "name": "SingleAxisFusionPass",
            "kind": "Bytecode",
            "default_enabled": True,
            "python_name": "SingleAxisFusionPass",
            "summary": "Fuses single-axis operation chains into precomputed 2x2 unitaries.",
            "detail": (
                "Fuses consecutive single-axis operations (ARRAY_H, ARRAY_S, PHASE_T, "
                "PHASE_ROT, etc.) on the same virtual axis into a single OP_ARRAY_U2 "
                "instruction. Pre-computes 2x2 unitary matrices for all 4 possible "
                "incoming Pauli frame states (I, X, Z, Y). A run is fused if it "
                "contains at least 3 array-touching operations, or at least 2 when "
                "one is a continuous rotation."
            ),
        },
    ]

    hir_passes = [p for p in passes if p["kind"] == "HIR"]
    bytecode_passes = [p for p in passes if p["kind"] == "Bytecode"]
    default_hir = [p for p in hir_passes if p["default_enabled"]]
    default_bytecode = [p for p in bytecode_passes if p["default_enabled"]]

    env.variables["passes"] = passes
    env.variables["hir_passes"] = hir_passes
    env.variables["bytecode_passes"] = bytecode_passes
    env.variables["default_hir_passes"] = default_hir
    env.variables["default_bytecode_passes"] = default_bytecode
