"""Exercise identity, long-lived, dense, and changing planner coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from continuation_trace_reuse import ContinuationShot, ContinuationWorker
from validate_regional_phase_specialization import cq_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "identity_records": (
            "I 2\nH 0\nT 0\n",
            "E(.5) Y0\nMPAD 0\nMPAD 1\nM 1\nM !1\nMX 0\n"
            "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        ),
        "stable_frame": (
            "H 0 1 2\n" + "R_PAULI(.137) X0*Y1\nR_PAULI(.173) Z0*X2\n" * 32,
            "E(.5) Y0\nH 1\nS 2\n",
        ),
        "dense_axes": (
            "H 0 1 2 3\n" + "R_PAULI(.137) Y0*Y1*Y2*Y3\nR_PAULI(.173) X0*Z1*Y2*X3\n" * 32,
            "E(.5) Y0\nS_DAG 1\nCX 2 3\n",
        ),
        "changing_frame": (
            "H 0 1 2\nT 0 1 2\n",
            "E(.5) Y0\nM 0\nRX 0\nM 1\nR 1\nH 1\nCX rec[-1] 2\nMPP X0*Y1\nMX 2\n",
        ),
        "repeated_nonidentity": (
            "I 2\nH 0\nT 0\n",
            "E(.5) Y0\nH 1\nM 1\nM 1\nM 1\nMX 0\n",
        ),
    }
    modes = ("coordinate-identity", "coordinate-columns", "coordinate-inverse32")
    result: dict[str, Any] = {"cases": {}}
    for name, (prefix, template) in cases.items():
        rows = []
        with ContinuationWorker(
            args.worker, prefix, max_width=6, phase=False, continuation=template
        ) as worker:
            for active in ((), (0,)):
                for correction in ("", "S 0\nCZ 0 1\nS_DAG 2\n"):
                    tail = correction + template.replace("E(.5) Y0", "Y 0" if active else "")
                    expected = cq_blocks(prefix + tail)
                    shot = ContinuationShot(correction, active, ())
                    audit = worker.instantiate(shot, 41, mode="coordinate-audit", reference=tail)
                    assert audit["plan_checked"]
                    if name in {"stable_frame", "dense_axes"}:
                        assert any(
                            i["inverse_queries"] > 0
                            for i in audit["coordinate_diagnostics"]["intervals"]
                        )
                    for mode in modes:
                        row = worker.instantiate(shot, 41, mode=mode, reference=tail)
                        assert row["optimized_checked"] and row["plan_checked"]
                        diag = row["coordinate_diagnostics"]
                        assert diag["coordinate_checks"] == diag["queries"]
                        if name == "identity_records":
                            assert diag["identity_fast_paths"] >= 2
                        if name in {"stable_frame", "dense_axes"}:
                            if mode == "coordinate-columns":
                                assert diag["column_hits"] > 0
                            if mode == "coordinate-inverse32":
                                assert diag["inverse_builds"] > 0
                        if len(expected) == 1:
                            state = np.asarray([complex(*v) for v in row["statevector"]])
                            observed = np.outer(state, state.conj())
                            reference = expected[0]
                        else:
                            observed = np.asarray(row["record_probabilities"])
                            reference = np.trace(expected, axis1=1, axis2=2).real
                        np.testing.assert_allclose(observed, reference, rtol=0, atol=1e-10)
                        rows.append(
                            {
                                "mode": mode,
                                "active": active,
                                "correction": correction,
                                "coordinate_diagnostics": diag,
                                "maximum_error": float(np.max(abs(observed - reference))),
                            }
                        )
        result["cases"][name] = rows
    result["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            *(
                Path(__file__).with_name(n)
                for n in (
                    "coordinate_reuse.cpp",
                    "coordinate_reuse.h",
                    "profile_prefix_trace_reuse.cpp",
                    "continuation_trace_reuse.py",
                )
            ),
        ]
    }
    result["worker_sha256"] = hashlib.sha256(args.worker.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Coordinate policies match independent dense and instrument references", flush=True)


if __name__ == "__main__":
    main()
