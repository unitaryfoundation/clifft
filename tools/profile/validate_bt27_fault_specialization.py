"""Compare fixed-fault BT27 records and conditional Pauli probes with Merlin.

Requires the optional merlin-sim package, ideally from the fixture's pinned
generator revision. These finite probes are not full-state tomography.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import stim
from profile_bt27_fault_specialization import LOGICAL_QUBITS, source

import clifft


def probes() -> list[str]:
    result = [f"{p}{q}" for q in LOGICAL_QUBITS for p in "XYZ"]
    result += [f"X{a}*X{b}" for i, a in enumerate(LOGICAL_QUBITS) for b in LOGICAL_QUBITS[i + 1 :]]
    rng = random.Random(81173)
    for _ in range(32):
        axes = [(q, rng.choice("IXYZ")) for q in LOGICAL_QUBITS]
        result.append("*".join(f"{p}{q}" for q, p in axes if p != "I"))
    return result


def parity_features(records: np.ndarray, seed: int, count: int) -> np.ndarray:
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        columns = rng.sample(range(records.shape[1]), min(5, records.shape[1]))
        parity = np.logical_xor.reduce(records[:, columns], axis=1)
        result.append(1.0 - 2.0 * parity)
    return np.column_stack(result)


def moments(records: np.ndarray, detectors: np.ndarray, exp_vals: np.ndarray) -> np.ndarray:
    record_parities = parity_features(records, 412, 32)
    syndrome_parities = parity_features(detectors, 913, 32)
    return np.column_stack(
        (
            1.0 - 2.0 * records,
            1.0 - 2.0 * detectors,
            record_parities,
            syndrome_parities,
            exp_vals,
            exp_vals**2,
            *(exp_vals * syndrome_parities[:, j : j + 1] for j in range(8)),
            *(exp_vals * record_parities[:, j : j + 1] for j in range(4)),
        )
    )


def conditional_groups(detectors: np.ndarray, values: np.ndarray) -> dict[bytes, np.ndarray]:
    groups: dict[bytes, list[int]] = {}
    for i, bits in enumerate(np.packbits(detectors, axis=1)):
        groups.setdefault(bits.tobytes(), []).append(i)
    return {key: values[indices] for key, indices in groups.items()}


def compare(c: Any, m: Any, converter: Any, reference: dict[str, Any]) -> dict[str, Any]:
    c_det, c_obs = converter.convert(
        measurements=c.measurements.astype(bool), separate_observables=True
    )
    m_det, _ = converter.convert(
        measurements=m.measurements.astype(bool), separate_observables=True
    )
    # Merlin automatically references the faulty circuit. Recompute raw parities
    # from its records instead of comparing those shifted detector bits.
    np.testing.assert_array_equal(c.detectors, c_det ^ np.asarray(reference["detectors"], bool))
    np.testing.assert_array_equal(c.observables, c_obs ^ np.asarray(reference["observables"], bool))
    left = moments(c.measurements, c_det, c.exp_vals)
    right = moments(m.measurements, m_det, m.exp_vals)
    difference = np.abs(left.mean(axis=0) - right.mean(axis=0))
    variance = left.var(axis=0, ddof=1) / len(left) + right.var(axis=0, ddof=1) / len(right)
    # A small bounded-observation allowance avoids pretending zero empirical
    # variance proves a deterministic feature in a finite sample.
    allowance = 2 / len(left) + 2 / len(right)
    scores = np.maximum(0, difference - allowance) / np.maximum(np.sqrt(variance), 1e-12)
    if scores.max() > 6:
        raise AssertionError(f"Independent joint-moment mismatch: {scores.max():.3f} sigma")
    c_groups = conditional_groups(c_det, c.exp_vals)
    m_groups = conditional_groups(m_det, m.exp_vals)
    checked = 0
    max_difference = 0.0
    for key in c_groups.keys() & m_groups.keys():
        c_values, m_values = c_groups[key], m_groups[key]
        if min(len(c_values), len(m_values)) < 8 or c_values.shape[1] == 0:
            continue
        spread = max(float(np.ptp(c_values, axis=0).max()), float(np.ptp(m_values, axis=0).max()))
        if spread > 1e-10:
            continue
        error = float(np.max(np.abs(c_values.mean(axis=0) - m_values.mean(axis=0))))
        max_difference = max(max_difference, error)
        np.testing.assert_allclose(c_values.mean(axis=0), m_values.mean(axis=0), atol=1e-10, rtol=0)
        checked += 1
    return {
        "joint_features": left.shape[1],
        "maximum_moment_score": float(scores.max()),
        "common_syndrome_patterns": len(c_groups.keys() & m_groups.keys()),
        "constant_conditional_probe_groups_checked": checked,
        "maximum_conditional_probe_difference": max_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=512)
    args = parser.parse_args()
    if args.shots < 16:
        parser.error("at least 16 shots are required")
    merlin = import_module("merlin")
    merlin_core = import_module("merlin._core")
    assert merlin_core.__file__ is not None

    study = json.loads(args.study.read_text())
    text = source()
    # Removing T gates preserves record numbering and detector definitions. The
    # converter skips its own reference and serves only as a parity evaluator.
    clifford = stim.Circuit(
        "\n".join(line for line in text.splitlines() if not line.startswith(("T ", "T_DAG ")))
    )
    converter = clifford.compile_m2d_converter(skip_reference_sample=True)
    probe_text = "".join(f"EXP_VAL {p}\n" for p in probes())
    reference = study["reference"]
    results = {}
    started = perf_counter()
    for key, row in study["results"].items():
        if "faults" not in row:
            continue
        faults = tuple((q, p) for q, p in row["faults"])
        circuit = source(faults) + probe_text
        program = clifft.compile(
            circuit,
            expected_detectors=reference["detectors"],
            expected_observables=reference["observables"],
        )
        if program.peak_active_width > 16:
            raise AssertionError("Probe execution exceeded the study's width budget")
        c = clifft.sample(program, shots=args.shots, seed=10531, threads=1, batch_size=1)
        m = merlin.CircuitSampler(circuit, seed=97131).sample(args.shots)
        results[key] = compare(c, m, converter, reference)
        if len(results) % 16 == 0:
            print(f"Validated {len(results)} patterns", flush=True)
    document = {
        "study_sha256": hashlib.sha256(args.study.read_bytes()).hexdigest(),
        "shots_per_pattern_per_simulator": args.shots,
        "probes": probes(),
        "merlin_version": importlib.metadata.version("merlin-sim"),
        "merlin_extension_sha256": hashlib.sha256(
            Path(merlin_core.__file__).read_bytes()
        ).hexdigest(),
        "elapsed_seconds": perf_counter() - started,
        "limitations": (
            "Finite Pauli probes and statistical joint moments; not tomography or a proof "
            "for untested fault patterns. Conditional comparisons use observed syndrome "
            "groups with at least eight samples per simulator and constant probe values."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Validated {len(results)} patterns against Merlin", flush=True)


if __name__ == "__main__":
    main()
