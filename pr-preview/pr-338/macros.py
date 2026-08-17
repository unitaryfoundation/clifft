"""MkDocs macros hook: loads HIR operation and pass metadata."""

import json
from pathlib import Path
from typing import Any


def define_env(env: Any) -> None:
    """Called by mkdocs-macros-plugin to inject template variables."""
    docs_dir = Path(env.conf["docs_dir"])
    data_path = docs_dir / "opcodes.json"

    with open(data_path) as f:
        data = json.load(f)

    site_url = env.conf["site_url"].rstrip("/")
    env.variables["playground_url"] = f"{site_url}/playground/"

    # -- HIR ops --
    hir_ops = data.get("hir_ops", {})
    hir_categories_order = [
        "Non-Clifford",
        "Measurement",
        "Instrument",
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

    unknown_hir_categories = sorted(set(hir_by_category) - set(hir_categories_order))
    if unknown_hir_categories:
        raise ValueError(
            "docs/opcodes.json contains HIR categories missing from docs/macros.py: "
            + ", ".join(unknown_hir_categories)
        )

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
            "preserves_record_order": True,
            "preserves_instrument_prefix": True,
            "python_name": "PeepholeFusionPass",
            "summary": "Algebraic T-gate fusion and terminal-phase elimination.",
            "detail": (
                "Scans the HIR to cancel or fuse T/T_dag gates acting on the "
                "same virtual Pauli axis using the symplectic inner product as "
                "a commutation check. T+T fuses to S, T+T_dag cancels to identity. "
                "It also removes a T gate or Pauli phase rotation when a later "
                "same-axis measurement consumes the phase; intervening Pauli "
                "noise is left in place."
            ),
        },
        {
            "name": "StatevectorSqueezePass",
            "kind": "HIR",
            "default_enabled": True,
            "preserves_record_order": False,
            "preserves_instrument_prefix": False,
            "python_name": "StatevectorSqueezePass",
            "summary": "Minimizes peak active width by reordering HIR operations.",
            "detail": (
                "Attempts to reduce `peak_active_width` by compacting qubit lifetimes. "
                "Sweep 1 (leftward) bubbles MEASURE ops as early as possible. "
                "Sweep 2 (rightward) bubbles T_GATE and PHASE_ROTATION ops as "
                "late as possible. Measurements reduce active width sooner, "
                "and non-Clifford expansions are deferred."
            ),
        },
        {
            "name": "RemoveNoisePass",
            "kind": "HIR",
            "default_enabled": False,
            "preserves_record_order": False,
            "preserves_instrument_prefix": False,
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
            "name": "DropNonUnitaryPass",
            "kind": "HIR",
            "default_enabled": False,
            "preserves_record_order": False,
            "preserves_instrument_prefix": False,
            "python_name": "DropNonUnitaryPass",
            "summary": "Drops non-evolution operations from the HIR.",
            "detail": (
                "Removes MEASURE, CONDITIONAL_PAULI, NOISE, READOUT_NOISE, "
                "DETECTOR, OBSERVABLE, and EXP_VAL ops and clears the matching "
                "metadata. Not included in the default pipeline and not "
                "semantics-preserving; use only when intentionally querying a "
                "unitary-only circuit skeleton."
            ),
        },
    ]

    default_hir = [p for p in passes if p["default_enabled"]]

    env.variables["passes"] = passes
    env.variables["hir_passes"] = passes
    env.variables["default_hir_passes"] = default_hir
