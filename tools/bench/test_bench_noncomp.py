"""pytest-benchmark cases for the noncomputational (leakage/loss) path.

A ladder on one repetition-code memory circuit (d data qubits, r rounds,
n = 2d-1 qubits; a hooked S layer on the data each round): plain sampling,
the noncomputational pipeline with a lossless model (isolating the
pipeline's per-shot overhead over plain sampling -- the per-call
rewrite-and-compile cost amortizes out at this batch size), and a
representative low-probability leakage-plus-loss model (drop on
leaked/lost operands, ternary herald classifier).
"""

from __future__ import annotations

from typing import Any

import pytest

import clifft
from clifft import noncomp

CONFIGS = [
    pytest.param(17, 5, id="d17-r5"),
]

LEAK_P = 0.01

# The plain and lossless rungs cost microseconds per shot, so a larger
# batch keeps their tracked means well above timer and runner noise. The
# leak rung traps and compiles continuations, which puts 100 shots in the
# tens of milliseconds already.
FAST_SHOTS = 2000
LEAK_SHOTS = 100


def rep_code_text(d: int, r: int) -> str:
    """Repetition-code memory circuit with a hooked S layer on the data each round."""
    data = list(range(d))
    anc = list(range(d, 2 * d - 1))
    lines = ["H " + " ".join(map(str, data))]
    for _ in range(r):
        for i in range(d - 1):
            lines.append(f"CX {data[i]} {anc[i]}")
            lines.append(f"CX {data[i + 1]} {anc[i]}")
        lines.append("S " + " ".join(map(str, data)))
        lines.append("MR " + " ".join(map(str, anc)))
    lines.append("M " + " ".join(map(str, data)))
    return "\n".join(lines) + "\n"


def leak_loss_matrix(p: float) -> list[list[float]]:
    """Source-dependent: e leaks to leak_e (0.8p) or is lost (0.2p); g is quiet."""
    m = [[0.0] * 5 for _ in range(5)]
    m[noncomp.Level.LEAK_E][noncomp.Level.E] = 0.8 * p
    m[noncomp.Level.LOST][noncomp.Level.E] = 0.2 * p
    return m


def ternary_classifier() -> noncomp.Classifier:
    """g/leak_g read 0, e/leak_e read 1, lost heralds."""
    m = [[0.0] * 5 for _ in range(3)]
    m[0][noncomp.Level.G] = 1.0
    m[0][noncomp.Level.LEAK_G] = 1.0
    m[1][noncomp.Level.E] = 1.0
    m[1][noncomp.Level.LEAK_E] = 1.0
    m[2][noncomp.Level.LOST] = 1.0
    return noncomp.Classifier(m)


def noncomp_model(p: float) -> noncomp.Model:
    return noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": leak_loss_matrix(p)} if p > 0 else {},
        classifier=ternary_classifier(),
    )


@pytest.mark.parametrize("d,r", CONFIGS)
def test_bench_noncomp_plain(benchmark: Any, d: int, r: int) -> None:
    program = clifft.compile(rep_code_text(d, r))
    clifft.sample(program, 2, 1)  # warm
    benchmark.extra_info["shots"] = FAST_SHOTS
    benchmark(clifft.sample, program, FAST_SHOTS, 1)


@pytest.mark.parametrize("d,r", CONFIGS)
def test_bench_noncomp_lossless(benchmark: Any, d: int, r: int) -> None:
    circuit = clifft.parse(rep_code_text(d, r))
    model = noncomp_model(0.0)
    noncomp.sample(circuit, model, shots=2, seed=1)  # warm
    benchmark.extra_info["shots"] = FAST_SHOTS
    benchmark(noncomp.sample, circuit, model, shots=FAST_SHOTS, seed=1)


@pytest.mark.parametrize("d,r", CONFIGS)
def test_bench_noncomp_leak(benchmark: Any, d: int, r: int) -> None:
    circuit = clifft.parse(rep_code_text(d, r))
    model = noncomp_model(LEAK_P)
    noncomp.sample(circuit, model, shots=2, seed=1)  # warm
    benchmark.extra_info["shots"] = LEAK_SHOTS
    benchmark(noncomp.sample, circuit, model, shots=LEAK_SHOTS, seed=1)
