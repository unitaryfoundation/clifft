"""Opcode census: real clifft-compiled programs vs the GPU microbench's synthetic schedule.

The microbench (`microbench/`) races CPU/GPU architectures on a synthetic
layered schedule (make_layer_schedule_meas: per layer, H+T on every axis, a
CNOT chain, one CZ, one U2, one U4, then one completed measurement + expand).
Its go/no-go verdict is trustworthy only if that schedule's *shape* — which
ops carry the work, at what k, and how many gates ride between measurements —
resembles real compiled programs.

This script compiles real workloads (from ../stabrank_profiling/circuits.py),
walks the bytecode with the exact k-stack semantics (EXPAND* -> k++, active
measurements -> k--; see ../stabrank_profiling/analyzer.py), classifies every
instruction into the microbench's op classes, and reports:

  * dense-vs-frame instruction split (frame ops never touch the 2^k array);
  * per-class counts and share of amplitude-weighted work (microbench
    amps_touched() semantics, evaluated at the actual k of each instruction);
  * the k profile of the dense work (amp-weighted mean k, peak k);
  * dense gate ops per active measurement — the ratio the synthetic layer
    hard-codes at 3k+2;
  * expands per active measurement (real T-heavy code grows rank via
    EXPAND_T, the synthetic layer via one plain EXPAND per layer).

The same census runs over the synthetic schedule itself for a side-by-side.

Run:  uv run python research/gpu/opcode_census.py [--md research/gpu/opcode_census.md]
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stabrank_profiling import circuits  # noqa: E402

# ---------------------------------------------------------------------------
# Opcode -> microbench op-class mapping, and amps-touched semantics
# (mirrors microbench/include/bench_common.hpp amps_touched()).
# ---------------------------------------------------------------------------

# Dense classes, keyed by the microbench name they map onto.
_DENSE_CLASS = {
    "OP_ARRAY_H": "H",
    "OP_ARRAY_T": "T",
    "OP_ARRAY_T_DAG": "T",
    "OP_ARRAY_ROT": "T",       # same phase-waterfall shape as T
    "OP_ARRAY_S": "T",
    "OP_ARRAY_S_DAG": "T",
    "OP_ARRAY_CZ": "CZ",
    "OP_ARRAY_MULTI_CZ": "CZ",  # mask-dependent subset; CZ-class approximation
    "OP_ARRAY_CNOT": "CNOT",
    "OP_ARRAY_MULTI_CNOT": "CNOT",
    "OP_ARRAY_SWAP": "CNOT",
    "OP_ARRAY_U2": "U2",
    "OP_ARRAY_U4": "U4",
    "OP_EXPAND": "EXPAND",
    "OP_EXPAND_T": "EXPAND_T",
    "OP_EXPAND_T_DAG": "EXPAND_T",
    "OP_EXPAND_ROT": "EXPAND_T",
    "OP_MEAS_ACTIVE_DIAGONAL": "MEAS_DIAG",
    "OP_MEAS_ACTIVE_DIAGONAL_FORCED": "MEAS_DIAG",
    "OP_MEAS_ACTIVE_INTERFERE": "MEAS_INTERFERE",
    "OP_MEAS_ACTIVE_INTERFERE_FORCED": "MEAS_INTERFERE",
    "OP_SWAP_MEAS_INTERFERE": "MEAS_INTERFERE",
    "OP_SWAP_MEAS_INTERFERE_FORCED": "MEAS_INTERFERE",
}

_EXPAND_OPS = {"OP_EXPAND", "OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"}
_POP_OPS = {
    "OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED",
    "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
    "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED",
}
_MEAS_CLASSES = {"MEAS_DIAG", "MEAS_INTERFERE"}
_GROW_CLASSES = {"EXPAND", "EXPAND_T"}
_GATE_CLASSES = {"H", "T", "CZ", "CNOT", "U2", "U4"}

_CLASS_ORDER = ["H", "T", "CZ", "CNOT", "U2", "U4",
                "EXPAND", "EXPAND_T", "MEAS_DIAG", "MEAS_INTERFERE"]


def _amps_touched(cls: str, k: int) -> int:
    """Microbench amps_touched() at rank k. For EXPAND*, k is the PRE-expand
    rank (reads 2^k, writes 2^(k+1) -> counted as 2*2^k, matching the bench).
    For MEAS_*, k is the pre-pop rank."""
    dim = 1 << k
    return {
        "H": dim, "T": dim // 2, "CZ": dim // 4, "CNOT": dim // 2,
        "U2": dim, "U4": dim,
        "EXPAND": 2 * dim, "EXPAND_T": 2 * dim,
        "MEAS_DIAG": 2 * dim, "MEAS_INTERFERE": 2 * dim,
    }[cls]


@dataclass
class Census:
    name: str
    peak_k: int = 0
    n_instr: int = 0
    n_frame: int = 0                      # frame/bookkeeping ops (no array touch)
    counts: Counter = field(default_factory=Counter)      # dense class -> count
    amps: Counter = field(default_factory=Counter)        # dense class -> amps touched
    amp_k_sum: float = 0.0                # sum(amps * k) for amp-weighted mean k
    k_at_dense: Counter = field(default_factory=Counter)  # k -> amps touched at that k

    @property
    def n_dense(self) -> int:
        return sum(self.counts.values())

    @property
    def total_amps(self) -> int:
        return sum(self.amps.values())

    @property
    def n_gates(self) -> int:
        return sum(self.counts[c] for c in _GATE_CLASSES)

    @property
    def n_meas(self) -> int:
        return sum(self.counts[c] for c in _MEAS_CLASSES)

    @property
    def n_expand(self) -> int:
        return sum(self.counts[c] for c in _GROW_CLASSES)

    @property
    def gates_per_meas(self) -> float:
        return self.n_gates / self.n_meas if self.n_meas else float("inf")

    @property
    def amp_weighted_mean_k(self) -> float:
        return self.amp_k_sum / self.total_amps if self.total_amps else 0.0

    def add(self, cls: str, k: int) -> None:
        a = _amps_touched(cls, k)
        self.counts[cls] += 1
        self.amps[cls] += a
        self.amp_k_sum += a * k
        self.k_at_dense[k] += a


def census_program(program, name: str) -> Census:
    """Walk a compiled clifft.Program with exact k-stack semantics."""
    c = Census(name=name)
    k = 0
    for ins in program:
        op = ins.opcode.name
        c.n_instr += 1
        cls = _DENSE_CLASS.get(op)
        if cls is None:
            c.n_frame += 1
            continue
        c.add(cls, k)                     # k BEFORE the op (pre-expand / pre-pop)
        if op in _EXPAND_OPS:
            k += 1
        elif op in _POP_OPS:
            k = max(0, k - 1)
        c.peak_k = max(c.peak_k, k)
    return c


def census_synthetic(k: int, layers: int) -> Census:
    """The microbench's make_layer_schedule_meas, run through the same census."""
    c = Census(name=f"synthetic k={k} L={layers}")
    c.peak_k = k
    for _ in range(layers):
        for _ in range(k):
            c.add("H", k)
        for _ in range(k):
            c.add("T", k)
        for _ in range(k - 1):
            c.add("CNOT", k)
        c.add("CZ", k)
        c.add("U2", k)
        c.add("U4", k)
        c.add("MEAS_DIAG", k)             # k -> k-1
        c.add("EXPAND", k - 1)            # k-1 -> k (pre-expand rank)
        c.n_instr += 3 * k + 4
    return c


# ---------------------------------------------------------------------------
# Workload selection: the GPU-relevant band (peak k ~ 10-20) plus the
# measurement-rich extreme for context. Names match stabrank_profiling.
# ---------------------------------------------------------------------------

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
    dense_share = c.n_dense / c.n_instr if c.n_instr else 0.0
    gm = "inf" if c.n_meas == 0 else f"{c.gates_per_meas:.1f}"
    return (f"| {c.name} | {c.peak_k} | {c.n_instr} | {dense_share:.0%} "
            f"| {gm} | {c.n_expand}/{c.n_meas} | {c.amp_weighted_mean_k:.1f} |")


def fmt_mix(c: Census) -> str:
    total = c.total_amps
    if not total:
        return f"| {c.name} | " + " | ".join("–" for _ in _CLASS_ORDER) + " |"
    cells = []
    for cls in _CLASS_ORDER:
        a = c.amps.get(cls, 0)
        cells.append(f"{a / total:.0%}" if a else "–")
    return f"| {c.name} | " + " | ".join(cells) + " |"


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
    rows.append(census_synthetic(16, 4))

    emit("# Opcode census: real compiled programs vs the microbench schedule")
    emit()
    emit(f"clifft {clifft.__version__}, default optimization passes.")
    emit()
    emit("## Shape metrics")
    emit()
    emit("dense% = share of instructions touching the 2^k array (rest are frame-only,")
    emit("free on any backend). gates/meas = dense gate ops per active measurement")
    emit("(synthetic layer hard-codes 3k+2 = 50 at k=16). exp/meas = rank-raising ops")
    emit("per rank-lowering op (1/1 in the synthetic layer). mean-k = amp-weighted.")
    emit()
    emit("| workload | peak k | instrs | dense% | gates/meas | exp/meas | mean k |")
    emit("|---|---|---|---|---|---|---|")
    for c in rows:
        emit(fmt_row(c))
    emit()
    emit("## Share of amplitude-weighted dense work by op class")
    emit()
    emit("| workload | " + " | ".join(_CLASS_ORDER) + " |")
    emit("|---" * (len(_CLASS_ORDER) + 1) + "|")
    for c in rows:
        emit(fmt_mix(c))
    emit()
    emit("## k profile of dense work (share of amps touched at each k, top 5)")
    emit()
    for c in rows:
        total = c.total_amps
        if not total:
            emit(f"- **{c.name}**: no dense work (frame-only program)")
            continue
        top = sorted(c.k_at_dense.items(), key=lambda kv: -kv[1])[:5]
        prof = ", ".join(f"k={k}: {a / total:.0%}" for k, a in top)
        emit(f"- **{c.name}**: {prof}")

    if args.md:
        args.md.write_text(out.getvalue())
        print(f"\nwrote {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
