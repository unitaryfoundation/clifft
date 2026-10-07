"""Measure native optimizer behavior after measured CSS preparation.

The logical replacement is an independently checked control, never applied
to noisy negative cases. Use separate environments to compare compiler builds.
Run from the repository root with the development environment installed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

import clifft

N = 15
ALL = (1 << N) - 1
X_ROWS = [sum(1 << (v - 1) for v in range(1, 16) if v >> b & 1) for b in range(4)]


def support(mask: int) -> list[int]:
    return [q for q in range(mask.bit_length()) if mask >> q & 1]


def product(mask: int, axis: str = "Z", offset: int = 0) -> str:
    return "*".join(f"{axis}{q + offset}" for q in support(mask))


def css_basis() -> tuple[dict[int, int], list[tuple[int, int]]]:
    """Systematic basis of punctured RM(1,4) and its orthogonal checks."""
    basis: dict[int, int] = {}
    for row in [*X_ROWS, ALL]:
        while row:
            pivot = (row & -row).bit_length() - 1
            if pivot in basis:
                row ^= basis[pivot]
            else:
                basis[pivot] = row
                break
    for pivot in sorted(basis, reverse=True):
        for other in basis:
            if other != pivot and basis[other] >> pivot & 1:
                basis[other] ^= basis[pivot]
    checks = [
        (q, (1 << q) ^ sum(1 << p for p, row in basis.items() if row >> q & 1))
        for q in range(N)
        if q not in basis
    ]
    assert len(basis) == 5 and len(checks) == 10
    assert all((row & check).bit_count() % 2 == 0 for row in basis.values() for _, check in checks)
    assert all(
        ((check >> q) & 1) == (i == j)
        for i, (q, _) in enumerate(checks)
        for j, (_, check) in enumerate(checks)
    )
    return basis, checks


BASIS, CHECKS = css_basis()


def preparation(offset: int = 0, readout: float = 0) -> list[str]:
    lines = ["H " + " ".join(str(q + offset) for q in range(N))]
    for i, (q, check) in enumerate(CHECKS):
        # One corrupted record isolates the distinction between collapse and report.
        gate = f"MPP({readout})" if i == 0 and readout else "MPP"
        lines += [f"{gate} {product(check, offset=offset)}", f"CX rec[-1] {q + offset}"]
    return lines


def unitary_preparation() -> list[str]:
    lines = ["H " + " ".join(map(str, BASIS))]
    for pivot, row in BASIS.items():
        lines += [f"CX {pivot} {q}" for q in support(row) if q != pivot]
    return lines


def observables() -> list[tuple[int, int]]:
    # Logical X and Y plus the complete stabilizer generators expose wrong cosets.
    return [(ALL, 0), (ALL, ALL), *[(row, 0) for row in X_ROWS], *[(0, c) for _, c in CHECKS]]


def pauli_text(x: int, z: int) -> str:
    return "*".join(f"{'IXZY'[((x >> q) & 1) + 2 * ((z >> q) & 1)]}{q}" for q in support(x | z))


def rm_circuit(
    *, readout: float = 0, fault: str = "", logical: bool = False, unitary: bool = False
) -> str:
    lines = unitary_preparation() if unitary else preparation(readout=readout)
    if fault:
        lines.append(fault)
    lines += [f"R_PAULI(-0.25) {product(ALL)}" if logical else "T " + " ".join(map(str, range(N)))]
    lines += [f"EXP_VAL {pauli_text(x, z)}" for x, z in observables()]
    # Outputs are retained on both sides; no postselection or decoder is used.
    lines += [f"MPP {product(ALL, 'X')}", "OBSERVABLE_INCLUDE(0) rec[-1]"]
    return "\n".join(lines) + "\n"


def slicing_circuit(
    *, readout: float = 0, fault: str = "", unequal: bool = False, cancelled: bool = False
) -> str:
    lines = ["H 0 1", f"MPP({readout}) Z0*Z1", "CX rec[-1] 1"]
    if fault:
        lines.append(fault)
    if not cancelled:
        lines += ["R_Z(0.125) 0", f"R_Z({-0.1 if unequal else -0.125}) 1"]
    lines += ["EXP_VAL X0*X1", "EXP_VAL Z0*Z1", "MPP X0*X1", "OBSERVABLE_INCLUDE(0) rec[-1]"]
    return "\n".join(lines) + "\n"


def ccz_factory() -> str:
    lines = [line for block in range(3) for line in preparation(offset=N * block)]
    lines += [f"CCZ {q} {q + N} {q + 2 * N}" for q in range(N)]
    for block in range(3):
        lines += [f"MPP {product(ALL, 'X', N * block)}", f"OBSERVABLE_INCLUDE({block}) rec[-1]"]
    return "\n".join(lines) + "\n"


def profile(
    source: str, repeats: int, shots: int, *, phase: bool = True, small_batches: bool = False
) -> dict[str, Any]:
    timings: list[list[float]] = []
    pass_ = None
    for _ in range(repeats):
        start = perf_counter()
        circuit = clifft.parse(source)
        parsed = perf_counter()
        hir = clifft.trace(circuit)
        traced = perf_counter()
        if phase:
            manager = clifft.default_hir_pass_manager()
        else:
            manager = clifft.HirPassManager()
            manager.add(clifft.PeepholeFusionPass())
            manager.add(clifft.RotationSimplificationPass())
            manager.add(clifft.StatevectorSqueezePass())
        manager.run(hir)
        optimized = perf_counter()
        program = clifft.lower(hir)
        lowered = perf_counter()
        timings.append(
            [
                parsed - start,
                traced - parsed,
                optimized - traced,
                lowered - optimized,
                lowered - start,
            ]
        )
    # Independently check the complete public default path, including its options.
    if phase:
        public = clifft.compile(source)
        assert public.inspect() == program.inspect()
        assert public.peak_active_width == program.peak_active_width
    raw = clifft.trace(clifft.parse(source))
    if phase:
        diagnostic = clifft.HirPassManager()
        diagnostic.add(clifft.PeepholeFusionPass())
        pass_ = clifft.PhasePolynomialPass()
        diagnostic.add(pass_)
        diagnostic.run(raw)
    plan = program.inspect()
    kinds = Counter(line.split()[1] for line in plan.splitlines()[1:])
    sample_times = []
    if shots:
        clifft.sample(program, shots=min(shots, 64), seed=87, threads=1)
        for i in range(repeats):
            start = perf_counter()
            clifft.sample(program, shots=shots, seed=100 + i, threads=1)
            sample_times.append(perf_counter() - start)
    compile_s = statistics.median(t[-1] for t in timings)
    sample_s = statistics.median(sample_times) if sample_times else None
    small_batch_times = {}
    if shots and small_batches:
        for batch in (1, 16, 256, 4096):
            clifft.sample(program, shots=batch, seed=87, threads=1)
            trials = []
            for i in range(repeats):
                start = perf_counter()
                clifft.sample(program, shots=batch, seed=100 + i, threads=1)
                trials.append(perf_counter() - start)
            small_batch_times[str(batch)] = trials
    return {
        "qubits": program.num_qubits,
        "records": program.num_measurements,
        "hir_t_count": hir.num_t_gates,
        "peak_active_width": program.peak_active_width,
        "action_counts": dict(kinds),
        "phase_pass": None
        if pass_ is None
        else {
            "input_t": pass_.input_t_count,
            "output_t": pass_.output_t_count,
            "applied": pass_.applied,
        },
        "compile_columns": ["parse", "trace", "optimize", "plan_and_prepare", "total"],
        "compile_seconds": timings,
        "median_compile_seconds": compile_s,
        "sampling_shots": shots,
        "sampling_seconds": sample_times,
        "small_batch_sampling_seconds": small_batch_times,
        "median_shots_per_second": shots / sample_s if sample_s else None,
        "amortized_seconds_per_shot": {
            str(reuse): compile_s / (shots * reuse) + sample_s / shots for reuse in (1, 10, 1000)
        }
        if sample_s
        else {},
        "plan": plan,
    }


def aer_conditional_states(fault: str = "", readout_flip: bool = False) -> tuple[np.ndarray, float]:
    qc = QuantumCircuit(N, len(CHECKS))
    qc.h(range(N))
    for i, (target, check) in enumerate(CHECKS):
        others = [q for q in support(check) if q != target]
        for q in others:
            qc.cx(q, target)
        qc.measure(target, i)
        for q in reversed(others):
            qc.cx(q, target)
        with qc.if_test((qc.clbits[i], 1)):
            qc.x(target)
    if fault:
        getattr(qc, fault)(CHECKS[0][0])
    qc.t(range(N))
    qc.save_statevector(pershot=True)
    model = NoiseModel()
    if readout_flip:
        model.add_readout_error(ReadoutError([[0, 1], [1, 0]]), [CHECKS[0][0]])
    result = (
        AerSimulator(method="statevector", noise_model=model, max_parallel_threads=1)
        .run(qc, shots=8, seed_simulator=1234, memory=True)
        .result()
    )
    assert result.success
    states = [np.asarray(s) for s in result.data()["statevector"]]
    reference = states[0]
    # Feedback makes each deterministic fault/readout branch independent of the
    # random preparation records, up to trajectory-global phase.
    minimum_overlap = min(float(abs(np.vdot(reference, s)) ** 2) for s in states)
    assert minimum_overlap > 1 - 1e-10
    expectations = []
    for x, z in observables():
        label = "".join("IXZY"[((x >> q) & 1) + 2 * ((z >> q) & 1)] for q in reversed(range(N)))
        expectations.append(
            float(result.data()["statevector"][0].expectation_value(Pauli(label)).real)
        )
    return np.asarray(expectations), minimum_overlap


def validate() -> dict[str, Any]:
    # Checking all basis vectors proves the diagonal replacement on the entire
    # codespace, including arbitrary amplitudes and entangled references.
    codewords = [0]
    for row in BASIS.values():
        codewords += [word ^ row for word in codewords]
    residuals = [(word.bit_count() + (word.bit_count() % 2)) % 8 for word in codewords]
    assert residuals == [0] * 32
    # Transversal CCZ has phase equal to the product of the three logical bits.
    assert all(
        ((a & b & c).bit_count() % 2)
        == ((a.bit_count() % 2) * (b.bit_count() % 2) * (c.bit_count() % 2))
        for a in codewords
        for b in codewords
        for c in codewords
    )

    state_checks = []
    branch_expectations = {}
    for flip in (False, True):
        for fault in ("", "x", "y", "z"):
            expected, overlap = aer_conditional_states(fault, flip)
            fault_text = f"{fault.upper()} {CHECKS[0][0]}" if fault else ""
            source = rm_circuit(readout=float(flip), fault=fault_text)
            sampled = clifft.sample(clifft.compile(source), shots=32, seed=5, threads=1)
            error = float(np.max(np.abs(sampled.exp_vals - expected)))
            assert error < 1e-10
            branch_expectations[(flip, fault)] = expected
            state_checks.append(
                {
                    "readout_flip": flip,
                    "fault": fault or "I",
                    "aer_branch_overlap": overlap,
                    "max_expectation_error": error,
                }
            )

    # The full noiseless joint record distribution is small enough to enumerate.
    originals = []
    for logical in (False, True):
        source = rm_circuit(logical=logical)
        source = "\n".join(
            line
            for line in source.splitlines()
            if not line.startswith(("EXP_VAL", "OBSERVABLE_INCLUDE"))
        )
        program = clifft.compile(source)
        records = [format(k, "011b") for k in range(1 << 11)]
        originals.append(clifft.record_probabilities(program, records))
    joint_error = float(np.max(np.abs(originals[0] - originals[1])))
    assert joint_error < 1e-12 and abs(sum(originals[0]) - 1) < 1e-10

    # A categorical Pauli channel and independent report flip are checked
    # against the exact mixture of independently simulated conditional states.
    p_readout = 0.125
    probabilities = {"": 0.7, "x": 0.1, "y": 0.08, "z": 0.12}
    expected = np.sum(
        [
            prob * (p_readout if flip else 1 - p_readout) * branch_expectations[(flip, fault)]
            for fault, prob in probabilities.items()
            for flip in (False, True)
        ],
        axis=0,
    )
    source = rm_circuit(readout=p_readout, fault=f"PAULI_CHANNEL_1(0.1,0.08,0.12) {CHECKS[0][0]}")
    samples = clifft.sample(clifft.compile(source), shots=32768, seed=731, threads=1)
    mixture_error = float(np.max(np.abs(samples.exp_vals.mean(axis=0) - expected)))
    assert mixture_error < 7 / np.sqrt(32768)

    # Clifford-only preparation uses Stim as an independent stochastic oracle.
    prep = preparation(readout=p_readout) + [f"E(0.2) X{CHECKS[0][0]} X{CHECKS[1][0]}"]
    prep += [f"MPP {product(c)}" for _, c in CHECKS]
    text = "\n".join(prep)
    stim_rows = stim.Circuit(text).compile_sampler(seed=923).sample(shots=32768)
    clifft_rows = clifft.sample(
        clifft.compile(text), shots=32768, seed=1923, threads=1
    ).measurements

    # Record marginals and every pair parity retain sensitivity to the shared fault.
    def moments(rows: np.ndarray) -> np.ndarray:
        return np.array(
            [np.mean(rows[:, i]) for i in range(rows.shape[1])]
            + [np.mean(rows[:, i] ^ rows[:, j]) for i in range(rows.shape[1]) for j in range(i)]
        )

    stim_error = float(np.max(np.abs(moments(stim_rows) - moments(clifft_rows))))
    assert stim_error < 7 / np.sqrt(32768)

    # The wrong unconditional replacement must be observably distinguishable.
    wrong = clifft.sample(
        clifft.compile(rm_circuit(readout=1, logical=True)), shots=8, seed=1, threads=1
    ).exp_vals[0]
    refusal_gap = float(np.max(np.abs(wrong - branch_expectations[(True, "")])))
    assert refusal_gap > 0.1
    return {
        "codewords_checked": len(codewords),
        "ccz_basis_triples_checked": 32**3,
        "aer_conditional_checks": state_checks,
        "joint_records_checked": 2**11,
        "joint_probability_max_error": joint_error,
        "categorical_noise_mixture_max_error": mixture_error,
        "stim_record_moment_max_error": stim_error,
        "invalid_readout_replacement_expectation_gap": refusal_gap,
        "slicing": validate_slicing(),
    }


def validate_slicing() -> list[dict[str, Any]]:
    checks = []
    for flip, fault, unequal in (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ):
        qc = QuantumCircuit(2, 1)
        qc.h([0, 1])
        qc.cx(0, 1)
        qc.measure(1, 0)
        qc.cx(0, 1)
        with qc.if_test((qc.clbits[0], 1)):
            qc.x(1)
        if fault:
            qc.x(1)
        qc.rz(np.pi * 0.125, 0)
        qc.rz(np.pi * (-0.1 if unequal else -0.125), 1)
        qc.save_expectation_value(Pauli("XX"), [0, 1], label="xx")
        qc.save_expectation_value(Pauli("ZZ"), [0, 1], label="zz")
        model = NoiseModel()
        if flip:
            model.add_readout_error(ReadoutError([[0, 1], [1, 0]]), [1])
        result = (
            AerSimulator(noise_model=model, max_parallel_threads=1)
            .run(qc, shots=16, seed_simulator=131)
            .result()
        )
        assert result.success
        expected = np.array([result.data()["xx"], result.data()["zz"]])
        source = slicing_circuit(readout=float(flip), fault="X 1" if fault else "", unequal=unequal)
        actual = clifft.sample(clifft.compile(source), shots=16, seed=2, threads=1).exp_vals
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-12
        assert (
            (expected[0] < 1 - 1e-4)
            if (flip or fault or unequal)
            else (abs(expected[0] - 1) < 1e-12)
        )
        checks.append(
            {
                "readout_flip": flip,
                "x_fault": fault,
                "unequal": unequal,
                "aer_expectations": expected.tolist(),
                "max_error": error,
            }
        )
    distributions = []
    for cancelled in (False, True):
        source = slicing_circuit(cancelled=cancelled)
        source = "\n".join(
            line
            for line in source.splitlines()
            if not line.startswith(("EXP_VAL", "OBSERVABLE_INCLUDE"))
        )
        probabilities = clifft.record_probabilities(
            clifft.compile(source), ["00", "01", "10", "11"]
        )
        np.testing.assert_allclose(sorted(probabilities), [0, 0, 0.5, 0.5], atol=1e-12)
        distributions.append(probabilities)
    np.testing.assert_allclose(distributions[0], distributions[1], atol=1e-12)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--small-batches", action="store_true")
    parser.add_argument(
        "--corpus", action="store_true", help="Include compilation of repository fixtures"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.shots < 1:
        parser.error("repeats and shots must be positive")
    circuits = {
        "rm_measured": rm_circuit(),
        "rm_unitary_control": rm_circuit(unitary=True),
        "rm_logical_control": rm_circuit(logical=True),
        "rm_z_fault": rm_circuit(fault="Z_ERROR(0.02) 4"),
        "rm_x_fault": rm_circuit(fault="X_ERROR(0.02) 4"),
        "rm_probability_one_fault": rm_circuit(fault="X_ERROR(1) 4"),
        "rm_correlated_fault": rm_circuit(fault="E(0.02) X4 X5"),
        "rm_readout": rm_circuit(readout=0.02),
        "slicing_measured": slicing_circuit(),
        "slicing_cancelled_control": slicing_circuit(cancelled=True),
        "slicing_readout": slicing_circuit(readout=0.02),
        "slicing_x_fault": slicing_circuit(fault="X_ERROR(0.02) 1"),
        "slicing_unequal": slicing_circuit(unequal=True),
        "ccz_factory_compile_only": ccz_factory(),
        "planner_already_suffices": "R_X(0.37) 0\nM 0\nCX rec[-1] 0\nT 0\nM 0\n",
    }
    results = {}
    for name, source in circuits.items():
        print(f"Profiling {name}", flush=True)
        results[name] = profile(
            source,
            args.repeats,
            0 if name.endswith("compile_only") else args.shots,
            small_batches=args.small_batches,
        )
    results["rm_measured_without_phase"] = profile(
        circuits["rm_measured"],
        args.repeats,
        args.shots,
        phase=False,
        small_batches=args.small_batches,
    )
    if args.corpus:
        for name in ("qv10", "cultivation_d5", "coherent_d5_r5", "surface_d7_r7_p001"):
            print(f"Profiling corpus {name}", flush=True)
            fixture = Path(__file__).resolve().parents[2] / f"tests/fixtures/{name}.stim"
            source = fixture.read_text()
            results[f"corpus_{name}"] = profile(source, args.repeats, 0)
            results[f"corpus_{name}"].pop("plan")
    print("Validating independent references", flush=True)
    validation = None if args.skip_validation else validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "version": clifft.version(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "isa": clifft.runtime_isa(),
                "circuits": circuits,
                "profiles": results,
                "validation": validation,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
