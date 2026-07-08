"""pytest-benchmark cases for the noncomputational (leakage/loss) path.

A ladder per repetition-code memory circuit (d data qubits, r rounds,
n = 2d-1 qubits; a hooked S layer on the data each round): plain sampling,
the noncomputational pipeline with a lossless model (isolating the shared
main line's overhead over plain sampling), and a representative
low-probability leakage-plus-loss model (drop on leaked/lost operands,
ternary herald classifier). The retired ahead-of-time pipeline's baseline
-- 13-14x slower on the lossless rung -- is recorded in the design note's
step-7 results.

A final cross-simulator rung runs the same circuit and noise model on the
cirq-superstaq leakage simulator. That package is not a committed dependency
(its simulator lives on an unmerged upstream branch), so the rung skips
unless it is importable in the local environment; everything else runs
everywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

import clifft
from clifft import noncomp

CONFIGS = [
    pytest.param(3, 3, 200, id="d3-r3"),
    pytest.param(17, 5, 100, id="d17-r5"),
]

LEAK_P = 0.01


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


@pytest.mark.parametrize("d,r,shots", CONFIGS)
def test_bench_noncomp_plain(benchmark: Any, d: int, r: int, shots: int) -> None:
    program = clifft.compile(rep_code_text(d, r))
    clifft.sample(program, 2, 1)  # warm
    benchmark.extra_info["shots"] = shots
    benchmark(clifft.sample, program, shots, 1)


@pytest.mark.parametrize("d,r,shots", CONFIGS)
def test_bench_noncomp_lossless(benchmark: Any, d: int, r: int, shots: int) -> None:
    circuit = clifft.parse(rep_code_text(d, r))
    model = noncomp_model(0.0)
    noncomp.sample(circuit, model, shots=2, seed=1)  # warm
    benchmark.extra_info["shots"] = shots
    benchmark(noncomp.sample, circuit, model, shots=shots, seed=1)


@pytest.mark.parametrize("d,r,shots", CONFIGS)
def test_bench_noncomp_leak(benchmark: Any, d: int, r: int, shots: int) -> None:
    circuit = clifft.parse(rep_code_text(d, r))
    model = noncomp_model(LEAK_P)
    noncomp.sample(circuit, model, shots=2, seed=1)  # warm
    benchmark.extra_info["shots"] = shots
    benchmark(noncomp.sample, circuit, model, shots=shots, seed=1)


@pytest.mark.parametrize("d,r,shots", CONFIGS)
def test_bench_noncomp_sqale_comparison(benchmark: Any, d: int, r: int, shots: int) -> None:
    cirq = pytest.importorskip("cirq")
    leakage_sim = pytest.importorskip("cirq_superstaq.sim.leakage_sim")
    sqale_model = pytest.importorskip("cirq_superstaq.sim.sqale_leakage_model")

    qs = cirq.LineQubit.range(2 * d - 1)
    data, anc = qs[:d], qs[d:]
    ops: list = [cirq.H.on_each(*data)]
    for k in range(r):
        for i in range(d - 1):
            ops.append(cirq.CX(data[i], anc[i]))
            ops.append(cirq.CX(data[i + 1], anc[i]))
        ops.append(cirq.S.on_each(*data))
        ops.append(cirq.measure(*anc, key=f"m{k}"))
        ops.extend(cirq.reset(a) for a in anc)
    ops.append(cirq.measure(*data, key="final"))
    model = sqale_model.SqaleLeakageModel(
        rz_transition_matrix=leak_loss_matrix(LEAK_P), classifier_errors=(0, 0)
    )
    noisy = cirq.Circuit(ops).with_noise(model)

    sq_shots = min(shots, 20)  # their per-shot stabilizer runs are slow at scale
    benchmark.extra_info["shots"] = sq_shots
    benchmark.pedantic(
        leakage_sim.sample_circuit,
        args=(noisy,),
        kwargs={"repetitions": sq_shots, "max_workers": 0, "progrcssbar": False},
        rounds=2,
        iterations=1,
    )
