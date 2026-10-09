"""Check that reuse diagnostics distinguish constant changes and structural changes."""

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
    result: dict[str, Any] = {}
    for label, prefix in (
        ("commuting_boundary", "H 0\nT 0\n"),
        ("noncommuting_boundary", "H 0\nT 0\nH 0\n"),
        ("optimizer_changes_certified_input", "H 0\nT 0\nT 0\n"),
        ("optimizer_changes_same_op_count", "H 0\nR_PAULI(.25) Z0\n"),
        ("continuous_rotation", "H 0\nR_PAULI(.125) Z0\n"),
    ):
        tail = "H 0\nM 0\n"
        rows = []
        statuses = []
        with ContinuationWorker(
            args.worker, prefix, max_width=6, phase=False, continuation=tail
        ) as worker:
            certificate = worker.setup["diagonal_boundary_fixes_prefix_axes"]
            assert certificate == (label != "noncommuting_boundary")
            for correction, flips in (("", ()), ("", (0,)), ("S 0\n", ()), ("S_DAG 0\n", ())):
                source = correction + tail.replace("M 0", "M !0" if flips else "M 0")
                shot = ContinuationShot(correction, (), flips)
                baseline = worker.instantiate(shot, 93, reference=source, mode="diagonal")
                audit = worker.instantiate(shot, 93, reference=source, mode="audit")
                squeezed = worker.instantiate(shot, 93, reference=source, mode="squeeze")
                assert squeezed["optimized_checked"]
                statuses.append(squeezed["squeeze_status"])
                for key in ("measurements", "width", "t_count", "record_probabilities"):
                    assert baseline[key] == audit[key], key
                    assert squeezed[key] == audit[key], key
                expected = np.trace(cq_blocks(prefix + source), axis1=1, axis2=2).real
                np.testing.assert_allclose(
                    audit["record_probabilities"], expected, atol=1e-10, rtol=0
                )
                rows.append(audit["audit"])
        a, b = rows[0]["plan"], rows[1]["plan"]
        assert a["full_actions"] != b["full_actions"]
        assert a["without_constants"] == b["without_constants"]
        assert a["kind_width"] == b["kind_width"]
        if not certificate:
            assert rows[0]["PeepholeFusionPass"]["changed"]
            assert not rows[2]["PeepholeFusionPass"]["changed"]
            assert (
                rows[0]["PeepholeFusionPass"]["hir"]["origins"]
                != rows[2]["PeepholeFusionPass"]["hir"]["origins"]
            )
            assert rows[0]["plan"]["kind_width"] != rows[2]["plan"]["kind_width"]
        expected_status = {
            "commuting_boundary": "reused",
            "noncommuting_boundary": "boundary_not_certified",
            "optimizer_changes_certified_input": "optimizer_changed",
            "optimizer_changes_same_op_count": "optimizer_changed",
            "continuous_rotation": "reused",
        }[label]
        assert statuses == [expected_status] * 4
        if label == "optimizer_changes_same_op_count":
            assert all(
                r["initial"]["ops"] == r["RotationSimplificationPass"]["hir"]["ops"] for r in rows
            )
        result[label] = {
            "diagonal_boundary_fixes_prefix_axes": certificate,
            "constant_normalization_checked": True,
            "peephole_changes": [r["PeepholeFusionPass"]["changed"] for r in rows],
            "plan_kinds": [r["plan"]["kind_width"] for r in rows],
            "squeeze_statuses": statuses,
        }
    result["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path(__file__).with_name("planning_reuse_audit.h"),
            Path(__file__).with_name("profile_prefix_trace_reuse.cpp"),
            Path(__file__).with_name("squeeze_schedule_reuse.h"),
        ]
    }
    result["worker_sha256"] = hashlib.sha256(args.worker.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Reuse diagnostics and boundary/optimizer fallback controls passed", flush=True)


if __name__ == "__main__":
    main()
