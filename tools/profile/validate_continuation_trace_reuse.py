"""Check continuation response reuse with full traces and independent state laws."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import stim
from continuation_trace_reuse import ContinuationPhase, ContinuationShot, ContinuationWorker
from prefix_trace_reuse import PreparedPhase
from shared_phase_specialization import bits
from validate_automatic_specialization import all_histories, remap
from validate_compiled_prefix_reuse import check
from validate_regional_phase_specialization import cq_blocks
from validate_shared_phase_specialization import exact_law, prefix_values, record_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("continuation", "diagonal", "audit"), default="continuation"
    )
    args = parser.parse_args()
    unitary = []
    for seed in range(32):
        rng = random.Random(seed + 8251)

        def clifford() -> str:
            lines = ["I 2"]
            for _ in range(7):
                a, b = rng.sample(range(3), 2)
                lines += [f"{rng.choice(['H', 'S', 'S_DAG', 'X', 'Y'])} {a}", f"CX {a} {b}"]
            return "\n".join(lines) + "\n"

        prefix = clifford() + "T 0\nT_DAG 1\n" + clifford()
        pieces = [clifford() for _ in range(4)]
        axes = ["Y0", "X0 Z2", "Y1 X2"]
        template = pieces[0] + "".join(f"E(.5) {a}\n{p}" for a, p in zip(axes, pieces[1:]))
        with ContinuationWorker(
            args.worker, prefix, max_width=6, phase=True, continuation=template
        ) as worker:
            for mask in range(8):
                correction = rng.choice(["", "S 0\nCZ 0 1\nZ 2\n", "S_DAG 2\nCZ 1 2\n"])
                tail = correction + pieces[0]
                for j, (axis, piece) in enumerate(zip(axes, pieces[1:])):
                    if (mask >> j) & 1:
                        tail += "".join(f"{p[0]} {p[1:]}\n" for p in axis.split())
                    tail += piece
                row = worker.instantiate(
                    ContinuationShot(correction, tuple(bits(mask)), ()),
                    seed,
                    reference=tail,
                    mode=args.mode,
                )
                state = np.asarray([complex(*v) for v in row["statevector"]])
                expected = cq_blocks(prefix + tail)[0]
                actual = np.outer(state, state.conj())
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
                unitary.append(float(np.max(abs(actual - expected))))

    # All local Pauli bases give an informationally complete outgoing-state
    # check conditioned on every earlier visible record, including reset sums.
    instruments = []
    stim_errors = []
    for magic in (False, True):
        prefix = "H 0 1\nS 1\nCX 0 1\n" + ("T 0\n" if magic else "")
        for bases in product("XYZ", repeat=2):
            tomography = "".join(
                f"{'M' if a == 'Z' else 'M' + a} {q}\n" for q, a in enumerate(bases)
            )
            template = (
                "E(.5) Y0\nH 0\nM !0\nCX rec[-1] 1\n"
                "E(.5) X1 Z0\nR 1\nH 1\nCZ rec[-1] 1\n"
                "E(.5) Z1\nMPP Y0*X1\n"
                "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n" + tomography
            )
            with ContinuationWorker(
                args.worker, prefix, max_width=6, phase=True, continuation=template
            ) as worker:
                for mask in range(16):
                    tail = (
                        ("Y 0\n" if mask & 1 else "")
                        + "H 0\n"
                        + ("M 0\n" if mask & 8 else "M !0\n")
                        + "CX rec[-1] 1\n"
                        + ("X 1\nZ 0\n" if mask & 2 else "")
                        + "R 1\nH 1\nCZ rec[-1] 1\n"
                        + ("Z 1\n" if mask & 4 else "")
                        + "MPP Y0*X1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
                        + tomography
                    )
                    row = worker.instantiate(
                        ContinuationShot("", tuple(bits(mask & 7)), (0,) if mask & 8 else ()),
                        715,
                        reference=tail,
                        mode=args.mode,
                    )
                    observed = np.asarray(row["record_probabilities"])
                    expected = np.trace(cq_blocks(prefix + tail), axis1=1, axis2=2).real
                    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-10)
                    instruments.append(float(np.max(abs(observed - expected))))
                    if not magic:
                        circuit = record_only(stim.Circuit(prefix + tail))
                        n = circuit.num_measurements
                        law = exact_law(circuit, list(range(n)), n)
                        support = np.asarray(
                            [
                                all(
                                    ((r | (1 << n)) & constraint).bit_count() % 2 == 0
                                    for constraint in law
                                )
                                for r in range(1 << n)
                            ],
                            dtype=float,
                        )
                        support /= sum(support)
                        np.testing.assert_allclose(observed, support, rtol=0, atol=1e-10)
                        stim_errors.append(float(np.max(abs(observed - support))))

    originals = {
        "tail_channels": (
            "H 0 1 2\nCCZ 0 1 2\nH 0\nDEPOLARIZE1(.12) 0\nM(.13) !0\n"
            "CX rec[-1] 1\nE(.17) X1 Y2\nELSE_CORRELATED_ERROR(.23) Z1\n"
            "R 2\nH 2\nCX rec[-1] 2\nCZ rec[-1] 2\nMY 1\nDETECTOR rec[-2] rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-2]\n"
        ),
        "prefix_records": (
            "H 0 1 2\nM(.13) 2\nCX rec[-1] 0\nT 0 1\nCX 0 1\n"
            "E(.17) X0 Y1\nELSE_CORRELATED_ERROR(.23) Z1\nH 0\n"
            "M 0\nCZ rec[-1] 1\nMY 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
        ),
        "later_magic": "H 0 1\nT 0 1\nH 0\nT_DAG 0\nM 1\nCZ rec[-1] 0\nMY 0\n",
        "carrier": "H 0 1\nT 0\nM 1\nCX rec[-1] 0\nT_DAG 0\nMX 0\n",
        "unsupported": "H 0\nR_PAULI(.137) Z0\nMY 0\n",
    }
    frontend = []
    trace_checks = 0
    for name, source in (
        originals | {n + "_renamed": remap(s) for n, s in originals.items()}
    ).items():
        front = ContinuationPhase(source, args.exporter)
        histories = all_histories(front.model)
        checks = [check(front, history) for history, _ in histories]
        row_info: dict[str, Any] = {
            "case": name,
            "continuation": front.continuation_info,
            "checks": checks,
        }
        if front.continuation_info["eligible"]:
            assert front.first is not None
            values = prefix_values(front.first.region.shared)
            errors = []
            with ContinuationWorker(
                args.worker,
                front.optimized_prefix,
                max_width=6,
                phase=front.phase_required,
                continuation=front.template_source,
            ) as worker:
                for history, _ in histories:
                    for value in values:

                        def choose(shared: Any, stage: int) -> int:
                            return value

                        branch = front.rewrite(history, 197, choose_prefix=choose)
                        payload = front.payload(history, 197, choose_prefix=choose)
                        row = worker.instantiate(
                            payload,
                            715,
                            reference=branch.source[len(front.optimized_prefix) :],
                            mode=args.mode,
                        )
                        trace_checks += int(row["checked"])
                        expected = np.trace(cq_blocks(branch.source), axis1=1, axis2=2).real
                        np.testing.assert_allclose(
                            row["record_probabilities"], expected, rtol=0, atol=1e-10
                        )
                        errors.append(
                            float(np.max(abs(np.asarray(row["record_probabilities"]) - expected)))
                        )
            row_info["native_record_errors"] = errors
        else:
            for history, _ in histories:
                assert front.rewrite(history, 197) == PreparedPhase.rewrite(front, history, 197)
        frontend.append(row_info)
        print(name, "passed", flush=True)

    rejected = []
    for name, template, shot in [
        ("live_channel", "X_ERROR(.1) 0\n", ContinuationShot("", (), ())),
        ("live_readout", "M(.1) 0\n", ContinuationShot("", (), ())),
        ("later_magic", "T 0\n", ContinuationShot("", (), ())),
        ("wider_tail", "CX 0 2\n", ContinuationShot("", (), ())),
        ("bad_generator", "E(.5) X0\n", ContinuationShot("", (1,), ())),
        ("bad_record", "M 0\n", ContinuationShot("", (), (1,))),
        ("bad_correction", "M 0\n", ContinuationShot("H 0", (), ())),
    ]:
        try:
            with ContinuationWorker(
                args.worker, "I 1\nH 0\nT 0\n", max_width=6, phase=True, continuation=template
            ) as worker:
                worker.instantiate(shot, 1, mode=args.mode)
        except RuntimeError as error:
            rejected.append({"case": name, "reason": str(error)})
        else:
            raise AssertionError("Invalid continuation accepted: " + name)

    # Restoring a visible bit only after feedback cannot implement readout noise.
    right = cq_blocks("I 1\nM !0\nCX rec[-1] 1\n")
    postprocessed = cq_blocks("I 1\nM 0\nCX rec[-1] 1\n")[::-1]
    witness = float(np.max(abs(right - postprocessed)))
    assert witness > 0.5
    result: dict[str, Any] = {
        "mode": args.mode,
        "unitary_density_errors": unitary,
        "instrument_tomography_record_errors": instruments,
        "stim_exact_record_errors": stim_errors,
        "frontend": frontend,
        "frontend_exact_trace_checks": trace_checks,
        "postprocessed_readout_state_error": witness,
        "rejected": rejected,
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("continuation_trace_reuse.py"),
                Path(__file__).with_name("profile_prefix_trace_reuse.cpp"),
                Path(__file__).with_name("planning_reuse_audit.h"),
            ]
        },
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("All continuation response checks passed", flush=True)


if __name__ == "__main__":
    main()
