"""Action census of compiled SamplingPlan programs, for GPU-benchmark calibration.

The 2026-07/08 GPU study (research/gpu on the clifft-research branch) was
calibrated by an opcode census of the LEGACY bytecode. That runtime is gone:
current main compiles circuits to an ExecutablePlan whose action vocabulary
(ROTATE / FUSED_ROTATION / PROMOTE / MEASURE_ACTIVE / frame actions) differs
from the old opcode set in both granularity and mix. Any new GPU or batch
benchmark must therefore be re-calibrated against what the new planner
actually emits.

This script compiles the same workload corpus as the old census (circuits.py,
copied verbatim from research/stabrank_profiling), walks each program's
lowered actions via Program.inspect_action(), tracks the active width through
PROMOTE (+1) and MEASURE_ACTIVE (-1) transitions, and reports:

  * dense-vs-frame action split (frame actions never touch the coefficient
    array and are effectively free on any backend);
  * per-class action counts and share of coefficient-visit-weighted work
    (visits evaluated at the actual width of each action, matching the old
    census's amps-touched semantics for comparability);
  * the width profile of dense work (visit-weighted mean and peak width,
    share of visits inside candidate on-chip residency bands);
  * dense actions and coefficient visits per active measurement;
  * promotions per active measurement.

Width-band notes: a shot's coefficient state costs 16 B/coefficient (split
real+imag f64). 227 KB of Hopper shared memory holds w <= 13; 64 KB of CDNA3
LDS holds w <= 11 (both before any reduction scratch).

Run (from the repo root, with the package installed):
  uv run python research/sampling_gpu/census.py [--md research/sampling_gpu/census.md]
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import circuits  # noqa: E402

# Dense action classes touch the 2^w coefficient array; frame classes do not.
_DENSE = {"ROTATE", "FUSED_ROTATION", "DYNAMIC_FUSED_ROTATION", "PROMOTE", "MEASURE_ACTIVE"}
_FRAME = {
    "MEASURE_DORMANT",
    "RECORD_CLASSICAL",
    "DEFINE_SYMBOL",
    "READOUT_NOISE",
    "WRITE_DETECTOR",
    "WRITE_OBSERVABLE",
    "WRITE_EXPECTATION",
}

_CLASS_ORDER = ["ROTATE", "FUSED_ROTATION", "DYNAMIC_FUSED_ROTATION", "PROMOTE", "MEASURE_ACTIVE"]

# On-chip residency bands, in bytes per coefficient = 16 (split f64 pair).
_BANDS = {"w<=11 (64KB LDS)": 11, "w<=13 (227KB smem)": 13}


def _visits(cls: str, w: int) -> int:
    """Coefficient visits at active width w, before the action's transition.

    PROMOTE reads 2^w and writes 2^(w+1); MEASURE_ACTIVE reduces over 2^w and
    collapses into 2^(w-1). Both are counted as 2*2^w to match the old
    census's EXPAND/MEAS accounting, so shares stay comparable across the two
    reports. Rotations (direct or fused) sweep the full 2^w state once.
    """
    dim = 1 << w
    if cls in ("PROMOTE", "MEASURE_ACTIVE"):
        return 2 * dim
    return dim


@dataclass
class Census:
    name: str
    peak_w: int = 0
    n_actions: int = 0
    n_frame: int = 0
    counts: Counter = field(default_factory=Counter)
    visits: Counter = field(default_factory=Counter)
    visit_w_sum: float = 0.0
    w_profile: Counter = field(default_factory=Counter)
    meas_kernels: Counter = field(default_factory=Counter)
    rot_kernels: Counter = field(default_factory=Counter)

    @property
    def n_dense(self) -> int:
        return sum(self.counts.values())

    @property
    def total_visits(self) -> int:
        return sum(self.visits.values())

    @property
    def n_meas(self) -> int:
        return self.counts["MEASURE_ACTIVE"]

    @property
    def n_promote(self) -> int:
        return self.counts["PROMOTE"]

    @property
    def n_rot(self) -> int:
        return (
            self.counts["ROTATE"]
            + self.counts["FUSED_ROTATION"]
            + self.counts["DYNAMIC_FUSED_ROTATION"]
        )

    @property
    def mean_w(self) -> float:
        return self.visit_w_sum / self.total_visits if self.total_visits else 0.0

    def band_share(self, w_max: int) -> float:
        if not self.total_visits:
            return 0.0
        inside = sum(v for w, v in self.w_profile.items() if w <= w_max)
        return inside / self.total_visits


def census_program(program: Any, name: str) -> Census:
    c = Census(name=name)
    c.peak_w = program.peak_active_width
    w = 0
    for i in range(program.num_actions):
        line = program.inspect_action(i)
        cls = line.split()[0]
        c.n_actions += 1
        if cls in _FRAME:
            c.n_frame += 1
            continue
        if cls not in _DENSE:
            raise ValueError(f"unknown action mnemonic {cls!r} in {name}: {line}")
        v = _visits(cls, w)
        c.counts[cls] += 1
        c.visits[cls] += v
        c.visit_w_sum += v * w
        c.w_profile[w] += v
        if cls == "MEASURE_ACTIVE" and "kernel=" in line:
            c.meas_kernels[line.rsplit("kernel=", 1)[1].split()[0]] += 1
        if cls == "ROTATE" and "kernel=" in line:
            c.rot_kernels[line.rsplit("kernel=", 1)[1].split()[0]] += 1
        if cls == "PROMOTE":
            w += 1
        elif cls == "MEASURE_ACTIVE":
            w = max(0, w - 1)
    if w != 0:
        print(f"warning: {name} ends at width {w}, expected 0", file=sys.stderr)
    return c


def build_workloads() -> list[tuple[str, str]]:
    return [
        ("rand_cliffT_n20_d40_t05", circuits.random_clifford_t(20, 40, 0.05, seed=2005)),
        ("rand_cliffT_n20_d40_t15", circuits.random_clifford_t(20, 40, 0.15, seed=2015)),
        ("hidden_shift_np8_tl2", circuits.hidden_shift(8, n_t_layers=2, seed=8)),
        ("qaoa_ring_n16_p3", circuits.qaoa_ring(16, p_layers=3, seed=19)),
        ("iqp_n028_t1_cz30", circuits.iqp(28, seed=28, t_per_qubit=1, cz_density=0.3)),
        ("hidden_ccz_t4", circuits.hidden_shift_ccz(4, seed=4)),
        ("conveyor_r16_w24", circuits.magic_conveyor(16, 24, t_per_round=24, seed=16)),
        ("surface_d5_r6", circuits.surface_code_memory_with_t(5, 6, n_t=4, seed=5)),
    ]


def fmt_row(c: Census) -> str:
    dense_share = c.n_dense / c.n_actions if c.n_actions else 0.0
    rpm = "inf" if c.n_meas == 0 else f"{c.n_rot / c.n_meas:.1f}"
    return (
        f"| {c.name} | {c.peak_w} | {c.n_actions} | {dense_share:.0%} "
        f"| {rpm} | {c.n_promote}/{c.n_meas} | {c.mean_w:.1f} |"
    )


def fmt_mix(c: Census) -> str:
    total = c.total_visits
    if not total:
        return f"| {c.name} | " + " | ".join("-" for _ in _CLASS_ORDER) + " |"
    cells = []
    for cls in _CLASS_ORDER:
        v = c.visits.get(cls, 0)
        cells.append(f"{v / total:.0%}" if v else "-")
    return f"| {c.name} | " + " | ".join(cells) + " |"


def fmt_bands(c: Census) -> str:
    cells = [f"{c.band_share(w):.0%}" for w in _BANDS.values()]
    return f"| {c.name} | {c.peak_w} | " + " | ".join(cells) + " |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, default=None, help="also write a markdown report")
    args = ap.parse_args()

    import clifft

    out = io.StringIO()

    def emit(s: str = "") -> None:
        print(s)
        out.write(s + "\n")

    rows: list[Census] = []
    for name, text in build_workloads():
        rows.append(census_program(clifft.compile(text), name))

    emit("# Action census: compiled SamplingPlan programs (current main)")
    emit()
    emit(f"clifft {clifft.version()}, default passes, `clifft.compile()` ->")
    emit("`Program.inspect_action()` walk. Same circuit corpus as the legacy-bytecode")
    emit("census (research/gpu/opcode_census.md on the clifft-research branch); the")
    emit("legacy numbers are quoted there and are NOT comparable action-for-action,")
    emit("only work-share-for-work-share.")
    emit()
    emit("## Shape metrics")
    emit()
    emit("dense% = share of actions touching the 2^w coefficient array (the rest are")
    emit("frame/record actions, effectively free on any backend). rot/meas = rotation")
    emit("actions (direct + fused) per active measurement. prom/meas = width-raising")
    emit("actions per width-lowering one. mean w = coefficient-visit-weighted.")
    emit()
    emit("| workload | peak w | actions | dense% | rot/meas | prom/meas | mean w |")
    emit("|---|---|---|---|---|---|---|")
    for c in rows:
        emit(fmt_row(c))
    emit()
    emit("## Share of visit-weighted dense work by action class")
    emit()
    emit("| workload | " + " | ".join(_CLASS_ORDER) + " |")
    emit("|---" * (len(_CLASS_ORDER) + 1) + "|")
    for c in rows:
        emit(fmt_mix(c))
    emit()
    emit("## On-chip residency: share of dense visits inside each width band")
    emit()
    emit("Bytes per coefficient: 16 (split f64). Bands ignore reduction scratch, so")
    emit("they are upper bounds on shared-memory/LDS eligibility.")
    emit()
    emit("| workload | peak w | " + " | ".join(_BANDS) + " |")
    emit("|---|---|" + "---|" * len(_BANDS))
    for c in rows:
        emit(fmt_bands(c))
    emit()
    emit("## Width profile of dense work (share of visits at each w, top 5)")
    emit()
    for c in rows:
        total = c.total_visits
        if not total:
            emit(f"- **{c.name}**: no dense work (frame-only program)")
            continue
        top = sorted(c.w_profile.items(), key=lambda kv: -kv[1])[:5]
        prof = ", ".join(f"w={w}: {v / total:.0%}" for w, v in top)
        emit(f"- **{c.name}**: {prof}")
    emit()
    emit("## Measurement and rotation kernel selections")
    emit()
    for c in rows:
        if not c.meas_kernels and not c.rot_kernels:
            continue
        mk = ", ".join(f"{k}: {n}" for k, n in sorted(c.meas_kernels.items()))
        rk = ", ".join(f"{k}: {n}" for k, n in sorted(c.rot_kernels.items()))
        emit(f"- **{c.name}**: meas [{mk or '-'}], rotate [{rk or '-'}]")

    if args.md:
        args.md.write_text(out.getvalue())
        print(f"\nwrote {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
