"""Curated, deterministic cases for matched sampling-backend measurements."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

PAPER_COMMIT = "db7dc9f13a2c2854690e92390c779048a1ac1400"
REGIME_SOURCE_COMMIT = "adfe94a51e0819d9a2a917f8c932addea640e8c2"

PAPER_HASHES = {
    "surface_d7_r7.stim": "30d1940101d70e05a63f0d2f877756ffaeaba7e8e17e94cd2ea40fce04b99583",
    "cultivation_d3.stim": "90a7d841e003e5ee38137cd9a3eb6529bb552e49c424bc6b0932a27d97cdb41f",
    "cultivation_d5.stim": "c2b4566917bd9bf27a5705284dac02700ef0dcc7c03c91066670db376d633a6d",
    "distillation.stim": "188bd53c48dbc21f840fb297df6f41c61f5bad6a856bba621f00ff42078921c1",
    "coherent_d3_r1.stim": "9b439238478b15977829c1015ee47dfe401976b3882092d9560ba321fb0f510a",
    "coherent_d3_r3.stim": "87d1308c83894e87c60aeb2dc31b74be89b3460a951929951f2c3ac92606827d",
    "coherent_d5_r1.stim": "2707188abe8912f693fe4f910db8c4b6bd795c71a8fba38cc153da90ee5910b8",
    "coherent_d5_r5.stim": "54088bbd5f06b441596e414f9fa99d8eeaff8a4a1a862b911e0ad40092c5e549",
}


@dataclass(frozen=True)
class Case:
    case_id: str
    regime: str
    kind: str
    source_kind: str
    source_name: str
    shots: int
    output_mode: str
    postselection: str = "none"
    forced_k: int | None = None
    nominal_k: int | None = None
    stream_blocks: int | None = None
    noncomp_probability: float | None = None
    extended: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


CASES = (
    Case(
        "surface_d7_r7_aggregate",
        "primary_qec",
        "ordinary",
        "paper",
        "surface_d7_r7.stim",
        200_000,
        "aggregate",
        "all_detectors",
    ),
    Case(
        "cultivation_d3_aggregate",
        "primary_qec",
        "ordinary",
        "paper",
        "cultivation_d3.stim",
        500_000,
        "aggregate",
        "all_detectors",
    ),
    Case(
        "cultivation_d5_aggregate",
        "primary_qec",
        "ordinary",
        "paper",
        "cultivation_d5.stim",
        50_000,
        "aggregate",
        "all_detectors",
    ),
    Case(
        "distillation_aggregate",
        "primary_qec",
        "ordinary",
        "paper",
        "distillation.stim",
        200_000,
        "aggregate",
        "all_detectors",
    ),
    Case(
        "coherent_d3_r1_aggregate",
        "primary_qec",
        "ordinary",
        "paper",
        "coherent_d3_r1.stim",
        1_000_000,
        "aggregate",
        "all_detectors",
    ),
    Case(
        "coherent_d3_r3_aggregate",
        "primary_qec",
        "ordinary",
        "paper",
        "coherent_d3_r3.stim",
        300_000,
        "aggregate",
        "all_detectors",
    ),
    Case(
        "coherent_d5_r1_aggregate",
        "primary_qec",
        "ordinary",
        "paper",
        "coherent_d5_r1.stim",
        15_000,
        "aggregate",
        "all_detectors",
    ),
    Case(
        "coherent_d5_r5_aggregate",
        "primary_qec",
        "ordinary",
        "paper",
        "coherent_d5_r5.stim",
        1,
        "aggregate",
        "all_detectors",
        extended=True,
    ),
    Case(
        "surface_d7_r7_raw",
        "raw_subset",
        "ordinary",
        "paper",
        "surface_d7_r7.stim",
        50_000,
        "raw",
    ),
    Case(
        "cultivation_d3_raw",
        "raw_subset",
        "ordinary",
        "paper",
        "cultivation_d3.stim",
        200_000,
        "raw",
    ),
    Case(
        "coherent_d3_r3_raw",
        "raw_subset",
        "ordinary",
        "paper",
        "coherent_d3_r3.stim",
        200_000,
        "raw",
    ),
    Case(
        "regime_k4_l32",
        "regime_map",
        "ordinary",
        "regime",
        "k4_l32",
        2_000_000,
        "aggregate",
        nominal_k=4,
        stream_blocks=32,
    ),
    Case(
        "regime_k4_l512",
        "regime_map",
        "ordinary",
        "regime",
        "k4_l512",
        150_000,
        "aggregate",
        nominal_k=4,
        stream_blocks=512,
    ),
    Case(
        "regime_k8_l32",
        "regime_map",
        "ordinary",
        "regime",
        "k8_l32",
        400_000,
        "aggregate",
        nominal_k=8,
        stream_blocks=32,
    ),
    Case(
        "regime_k8_l512",
        "regime_map",
        "ordinary",
        "regime",
        "k8_l512",
        120_000,
        "aggregate",
        nominal_k=8,
        stream_blocks=512,
    ),
    Case(
        "regime_k12_l32",
        "regime_map",
        "ordinary",
        "regime",
        "k12_l32",
        25_000,
        "aggregate",
        nominal_k=12,
        stream_blocks=32,
    ),
    Case(
        "regime_k12_l512",
        "regime_map",
        "ordinary",
        "regime",
        "k12_l512",
        20_000,
        "aggregate",
        nominal_k=12,
        stream_blocks=512,
    ),
    Case(
        "qv10_raw",
        "dense_control",
        "ordinary",
        "local",
        "tests/fixtures/qv10.stim",
        10_000,
        "raw",
        nominal_k=10,
    ),
    Case(
        "qv20_raw",
        "dense_control",
        "ordinary",
        "local",
        "tools/bench/fixtures/qv20_seed42.stim",
        1,
        "raw",
        nominal_k=20,
        extended=True,
    ),
    Case(
        "exp_val_k0_200_raw",
        "exp_val",
        "ordinary",
        "exp_val",
        "k0_20q_200",
        500_000,
        "raw",
        nominal_k=0,
    ),
    Case(
        "exp_val_k8_200_raw",
        "exp_val",
        "ordinary",
        "exp_val",
        "k8_8q_200",
        15_000,
        "raw",
        nominal_k=8,
    ),
    Case(
        "cultivation_d3_k0",
        "importance",
        "importance",
        "paper",
        "cultivation_d3.stim",
        400_000,
        "aggregate",
        "all_detectors",
        forced_k=0,
    ),
    Case(
        "cultivation_d3_k1",
        "importance",
        "importance",
        "paper",
        "cultivation_d3.stim",
        400_000,
        "aggregate",
        "all_detectors",
        forced_k=1,
    ),
    Case(
        "cultivation_d3_k2",
        "importance",
        "importance",
        "paper",
        "cultivation_d3.stim",
        500_000,
        "aggregate",
        "all_detectors",
        forced_k=2,
    ),
    Case(
        "cultivation_d5_k0",
        "importance",
        "importance",
        "paper",
        "cultivation_d5.stim",
        10_000,
        "aggregate",
        "all_detectors",
        forced_k=0,
    ),
    Case(
        "cultivation_d5_k1",
        "importance",
        "importance",
        "paper",
        "cultivation_d5.stim",
        20_000,
        "aggregate",
        "all_detectors",
        forced_k=1,
    ),
    Case(
        "cultivation_d5_k2",
        "importance",
        "importance",
        "paper",
        "cultivation_d5.stim",
        25_000,
        "aggregate",
        "all_detectors",
        forced_k=2,
    ),
    Case(
        "noncomp_d17_r5_lossless",
        "noncomputational",
        "noncomp",
        "noncomp",
        "d17_r5",
        500_000,
        "noncomp_raw",
        noncomp_probability=0.0,
    ),
    Case(
        "noncomp_d17_r5_low_leak",
        "noncomputational",
        "noncomp",
        "noncomp",
        "d17_r5",
        1_000,
        "noncomp_raw",
        noncomp_probability=0.01,
    ),
)

CASE_BY_ID = {case.case_id: case for case in CASES}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _paper_text(case: Case, paper_dir: Path) -> str:
    path = paper_dir / case.source_name
    if not path.is_file():
        raise FileNotFoundError(f"paper fixture not found: {path}")
    text = path.read_text()
    actual = sha256_text(text)
    expected = PAPER_HASHES[case.source_name]
    if actual != expected:
        raise ValueError(
            f"paper fixture hash mismatch for {case.source_name}: {actual} != {expected}"
        )
    return text


def regime_text(k: int, blocks: int) -> str:
    block_width = 16
    block_qubits = list(range(k, k + block_width))
    lines = [
        f"R {' '.join(map(str, range(k + block_width)))}",
        f"H {' '.join(map(str, range(k + block_width)))}",
    ]
    if k > 0:
        for _ in range(2):
            lines.append("CZ " + " ".join(f"{i} {i + 1}" for i in range(k - 1)))
            lines.append("R_Z(0.1) " + " ".join(map(str, range(k))))
            lines.append("H " + " ".join(map(str, range(k))))
    for _ in range(blocks):
        lines.append(
            "CX "
            + " ".join(f"{block_qubits[i]} {block_qubits[i + 1]}" for i in range(block_width - 1))
        )
        lines.append(f"X_ERROR(0.01) {block_qubits[7]}")
        lines.append(f"X_ERROR(0.01) {block_qubits[-1]}")
        lines.append(f"MR {block_qubits[-1]}")
        lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    if k > 0:
        lines.extend(("S 0", "H 0", "M 0", "OBSERVABLE_INCLUDE(0) rec[-1]"))
    return "\n".join(lines) + "\n"


def exp_val_text(num_qubits: int, num_probes: int, active_width: int) -> str:
    lines = ["H " + " ".join(map(str, range(num_qubits)))]
    lines.append("CX " + " ".join(f"{i} {i + 1}" for i in range(num_qubits - 1)))
    if active_width:
        lines.append("T " + " ".join(map(str, range(active_width))))
        lines.extend(("S 0", "H 0"))
    basis = ("X", "Y", "Z")
    for i in range(num_probes):
        q1 = i % num_qubits
        q2 = (i * 7 + 3) % num_qubits
        q3 = (i * 11 + 5) % num_qubits
        if q2 == q1:
            q2 = (q2 + 1) % num_qubits
        if q3 == q1 or q3 == q2:
            q3 = (q3 + 2) % num_qubits
        lines.append(
            f"EXP_VAL {basis[i % 3]}{q1}*{basis[(i // 3) % 3]}{q2}*{basis[(i // 9) % 3]}{q3}"
        )
    return "\n".join(lines) + "\n"


def noncomp_text(distance: int = 17, rounds: int = 5) -> str:
    data = list(range(distance))
    ancillas = list(range(distance, 2 * distance - 1))
    lines = ["H " + " ".join(map(str, data))]
    for _ in range(rounds):
        for i in range(distance - 1):
            lines.append(f"CX {data[i]} {ancillas[i]}")
            lines.append(f"CX {data[i + 1]} {ancillas[i]}")
        lines.append("S " + " ".join(map(str, data)))
        lines.append("MR " + " ".join(map(str, ancillas)))
    lines.append("M " + " ".join(map(str, data)))
    return "\n".join(lines) + "\n"


def load_circuit(case: Case, repo_root: Path, paper_dir: Path) -> tuple[str, str]:
    if case.source_kind == "paper":
        text = _paper_text(case, paper_dir)
        source = (
            f"unitaryfoundation/clifft-paper@{PAPER_COMMIT}:qec_bench/circuits/{case.source_name}"
        )
    elif case.source_kind == "local":
        path = repo_root / case.source_name
        text = path.read_text()
        source = f"clifft@HEAD:{case.source_name}"
    elif case.source_kind == "regime":
        assert case.nominal_k is not None and case.stream_blocks is not None
        text = regime_text(case.nominal_k, case.stream_blocks)
        source = (
            "corrected-output-relevant-regime-generator@"
            f"{REGIME_SOURCE_COMMIT}:k={case.nominal_k},L={case.stream_blocks}"
        )
    elif case.source_kind == "exp_val":
        if case.nominal_k == 0:
            text = exp_val_text(20, 200, 0)
        else:
            text = exp_val_text(8, 200, 8)
        source = case.source_name
    elif case.source_kind == "noncomp":
        text = noncomp_text()
        source = "tools/bench/test_bench_noncomp.py:d17-r5"
    else:
        raise ValueError(f"unknown source kind: {case.source_kind}")
    return text, source
