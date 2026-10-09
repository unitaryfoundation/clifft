"""Validate fragment composition and direct preparation rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from compiled_prefix_reuse import ReusablePhase
from prefix_trace_reuse import PreparedPhase, TraceWorker
from qiskit.quantum_info import Pauli
from validate_automatic_specialization import all_histories, remap
from validate_compiled_prefix_reuse import check
from validate_conditional_phase_frontend import leaves
from validate_regional_phase_specialization import cq_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unitary = []
    for seed in range(48):
        rng = random.Random(seed + 9871)

        def piece() -> str:
            lines = ["I 2"]
            for _ in range(10):
                a, b = rng.sample(range(3), 2)
                lines += [
                    f"{rng.choice(['H', 'S', 'S_DAG', 'T', 'T_DAG', 'Y'])} {a}",
                    f"CX {a} {b}",
                ]
            return "\n".join(lines) + "\n"

        prefix, tail = piece(), piece()
        if seed % 2:
            tail += "R_PAULI(.137) X0*Y1\n"
        expected = cq_blocks(prefix + tail)[0]
        with TraceWorker(args.worker, prefix, max_width=6, phase=True) as worker:
            row = worker.run("traced", tail, seed)
        state = np.asarray([complex(*v) for v in row["statevector"]])
        actual = np.outer(state, state.conj())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        unitary.append(float(np.max(abs(actual - expected))))

    prefix = "I 2\nH 0 1\nT 0\nS_DAG 1\nCX 0 2\n"
    expected_state = cq_blocks(prefix)[0]
    expected_probe = float(np.trace(expected_state @ Pauli("IZX").to_matrix()).real)
    with TraceWorker(args.worker, prefix, max_width=6, phase=True) as worker:
        empty_tail = worker.run("traced", "", 1)
        probe = worker.run("traced", "EXP_VAL X0*Z1\n", 1)
    state = np.asarray([complex(*v) for v in empty_tail["statevector"]])
    np.testing.assert_allclose(np.outer(state, state.conj()), expected_state, rtol=0, atol=1e-10)
    np.testing.assert_allclose(probe["exp_vals"], [expected_probe], rtol=0, atol=1e-10)

    tails = [
        "H 0\nMY 1\nR 2\nCX rec[-1] 2\nMPP !Y0*Z2\n",
        "MPAD 1\nCZ rec[-1] 1\nMY 1\nR 0\nH 0\nM !0\n",
        "T_DAG 0\nH 1\nM 1\nCZ rec[-1] 2\nT 2\nMY 2\n",
        "RY 1\nMPP X0*Y1\nCX rec[-1] 2\nM 2\n",
    ]
    instruments = []
    for index, tail in enumerate(tails):
        prefix = "I 2\nH 0 1 2\nS_DAG 1\nT 0 1\nCX 0 2\nT_DAG 2\n"
        tail += "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
        expected = np.trace(cq_blocks(prefix + tail), axis1=1, axis2=2).real
        with TraceWorker(args.worker, prefix, max_width=6, phase=True) as worker:
            for mode in ("parsed", "traced"):
                row = worker.run(mode, tail, index)
                np.testing.assert_allclose(
                    row["record_probabilities"], expected, atol=1e-10, rtol=0
                )
                instruments.append(
                    float(np.max(abs(np.asarray(row["record_probabilities"]) - expected)))
                )

    sources = {
        "ccz": "H 0 1 2\nCCZ 0 1 2\nH 0\nMY 1\nR 2\nCX 0 2\n",
        "observed": (
            "H 0 1 2\nM(.13) 2\nCX rec[-1] 0\nT 0 1\nCX 0 1\n"
            "E(.17) X0 Y1\nELSE_CORRELATED_ERROR(.23) Z1\nH 0\nM 0\n"
            "CZ rec[-1] 1\nMY 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
        ),
        "later_magic": "H 0 1\nT 0 1\nH 0\nT_DAG 0\nM 1\nCZ rec[-1] 0\nMY 0\n",
        "carrier": "H 0 1\nT 0\nM 1\nCX rec[-1] 0\nT_DAG 0\nMX 0\n",
        "unsupported": "H 0\nR_PAULI(.137) Z0\nMY 0\n",
    }
    sources |= {name + "_renamed": remap(s) for name, s in sources.items()}
    frontend = []
    trace_checks = 0
    for name, source in sources.items():
        front = PreparedPhase(source, args.exporter)
        checks = []
        for history, _ in all_histories(front.model):
            checks.append(check(front, history))
            assert front.rewrite(history, 517) == ReusablePhase.rewrite(front, history, 517)
            if front.reuse_info["eligible"]:
                with TraceWorker(
                    args.worker, front.optimized_prefix, max_width=6, phase=front.phase_required
                ) as worker:
                    for branch, _ in leaves(front, history):
                        tail = branch.source[len(front.optimized_prefix) :]
                        expected = np.trace(cq_blocks(branch.source), axis1=1, axis2=2).real
                        row = worker.run("traced", tail, 719)
                        trace_checks += int(row["checked"])
                        np.testing.assert_allclose(
                            row["record_probabilities"], expected, atol=1e-10, rtol=0
                        )
        frontend.append({"case": name, "eligible": front.reuse_info["eligible"], "checks": checks})

    rejected = []
    for name, prefix, tail in [
        ("measured_prefix", "H 0\nM 0\n", ""),
        ("noisy_prefix", "X_ERROR(.1) 0\n", ""),
        ("live_tail_noise", "H 0\nT 0\n", "Z_ERROR(.1) 0\nM 0\n"),
        ("wider_tail", "H 0\nT 0\n", "CX 0 1\n"),
    ]:
        try:
            with TraceWorker(args.worker, prefix, max_width=6, phase=True) as worker:
                worker.run("traced", tail, 1)
        except RuntimeError as error:
            rejected.append({"case": name, "reason": str(error)})
        else:
            raise AssertionError("Unsupported fragment was accepted: " + name)
    result: dict[str, Any] = {
        "unitary_density_errors": unitary,
        "empty_continuation_checked": True,
        "expectation_error": abs(probe["exp_vals"][0] - expected_probe),
        "instrument_record_errors": instruments,
        "frontend": frontend,
        "frontend_exact_trace_checks": trace_checks,
        "rejected": rejected,
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("prefix_trace_reuse.py"),
                Path(__file__).with_name("profile_prefix_trace_reuse.cpp"),
            ]
        },
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("All fragment and rendering checks passed", flush=True)


if __name__ == "__main__":
    main()
