"""Check diagonal boundary updates across complete small groups and mask words."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from continuation_trace_reuse import ContinuationShot, ContinuationWorker
from validate_regional_phase_specialization import cq_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "diagonal",
            "squeeze",
            "coordinate-identity",
            "coordinate-columns",
            "coordinate-inverse32",
        ),
        default="diagonal",
    )
    args = parser.parse_args()
    statuses: Counter[str] = Counter()
    coordinate_checks = 0

    def observe(row: dict[str, Any]) -> None:
        nonlocal coordinate_checks
        statuses[row["squeeze_status"]] += 1
        if args.mode != "diagonal":
            assert row["optimized_checked"]
        if args.mode.startswith("coordinate-"):
            assert row["plan_checked"]
            diag = row["coordinate_diagnostics"]
            assert diag["coordinate_checks"] == diag["queries"]
            coordinate_checks += diag["coordinate_checks"]

    prefix = "H 0 1 2\nS 1\nCX 0 1\nT 0\nCX 1 2\nH 1\nY 2\nT_DAG 2\n"
    if args.mode != "diagonal":
        # Independent rotations survive the earlier passes, so the complete
        # group exercises schedule reuse instead of its optimizer fallback.
        prefix = "H 0 1 2\nT 0\nT_DAG 1\nT 2\n"
    template = "H 0\nE(.5) Y0 X2\nS_DAG 2\nCX 0 2\nH 1\n"
    errors = []
    with ContinuationWorker(
        args.worker, prefix, max_width=6, phase=True, continuation=template
    ) as worker:
        for powers in product(range(4), repeat=3):
            for edges in range(8):
                lines = []
                for q, power in enumerate(powers):
                    if power:
                        lines.append(f"{['I', 'S', 'Z', 'S_DAG'][power]} {q}")
                for j, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                    if edges >> j & 1:
                        lines.append(f"CZ {a} {b}")
                # Equivalent cancellations exercise aggregation, not just one
                # canonical gate for each diagonal polynomial coefficient.
                lines += ["S 0", "S 0", "Z 0", "S_DAG 2", "S 2", "CZ 2 0", "CZ 0 2", "I 1"]
                correction = "\n".join(lines) + "\n"
                active = (0,) if (sum(powers) + edges) % 2 else ()
                tail = correction + template.replace("E(.5) Y0 X2", "Y 0\nX 2" if active else "")
                row = worker.instantiate(
                    ContinuationShot(correction, active, ()), 75, reference=tail, mode=args.mode
                )
                observe(row)
                state = np.asarray([complex(*v) for v in row["statevector"]])
                expected = cq_blocks(prefix + tail)[0]
                actual = np.outer(state, state.conj())
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
                errors.append(float(np.max(abs(actual - expected))))
    if args.mode != "diagonal":
        assert statuses == {"reused": 512}
    wide = []
    for width in (65, 129, 193):
        rng = random.Random(9115 + width)
        chosen = sorted({0, 1, 63, 64, width - 1})
        prefix = f"I {width - 1}\n"
        for q in chosen:
            prefix += f"H {q}\nS_DAG {q}\nT {q}\n"
        prefix += "".join(f"CX {a} {b}\n" for a, b in zip(chosen, chosen[1:]))
        template = (
            f"E(.5) Y0 Z{width - 1}\nH 63\nM !63\nCX rec[-1] {width - 1}\n"
            f"R 64\nE(.5) X64\nH 64\nMPP Y0*X{width - 1}\n"
            "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
        )
        with ContinuationWorker(
            args.worker, prefix, max_width=8, phase=True, continuation=template
        ) as worker:
            for index in range(32):
                lines = []
                for _ in range(20):
                    a, b = rng.sample(range(width), 2)
                    lines += [f"{rng.choice(['S', 'S_DAG', 'Z'])} {a}", f"CZ {a} {b}"]
                lines += [f"S_DAG {q}" for q in chosen]
                correction = "\n".join(lines) + "\n"
                active = tuple(i for i in range(2) if index >> i & 1)
                flips = (0,) if index & 4 else ()
                concrete = template.replace(
                    "E(.5) Y0 Z" + str(width - 1), f"Y 0\nZ {width - 1}" if 0 in active else ""
                )
                concrete = concrete.replace("E(.5) X64", "X 64" if 1 in active else "")
                if flips:
                    concrete = concrete.replace("M !63", "M 63")
                shot = ContinuationShot(correction, active, flips)
                baseline = worker.instantiate(shot, 781, reference=correction + concrete)
                composed = worker.instantiate(
                    shot, 781, reference=correction + concrete, mode=args.mode
                )
                observe(composed)
                for key in ("measurements", "detectors", "observables", "width", "t_count"):
                    assert composed[key] == baseline[key], (width, index, key)
                assert composed["width"] <= 8
                wide.append(
                    {
                        "physical_width": width,
                        "active_width": composed["width"],
                        "fault_generators": len(active),
                        "readout_flips": len(flips),
                    }
                )
    result: dict[str, Any] = {
        "mode": args.mode,
        "squeeze_statuses": dict(statuses),
        "coordinate_checks": coordinate_checks,
        "diagonal_group_density_errors": errors,
        "wide_cases": wide,
        "wide_exact_trace_checks": 2 * len(wide),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("profile_prefix_trace_reuse.cpp"),
                Path(__file__).with_name("continuation_trace_reuse.py"),
                Path(__file__).with_name("planning_reuse_audit.h"),
                Path(__file__).with_name("squeeze_schedule_reuse.h"),
                Path(__file__).with_name("coordinate_reuse.h"),
                Path(__file__).with_name("coordinate_reuse.cpp"),
            ]
        },
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Complete diagonal group and mask-word boundary checks passed", flush=True)


if __name__ == "__main__":
    main()
