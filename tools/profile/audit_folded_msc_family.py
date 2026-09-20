"""Cross-check physical f5 histories with the offline coherent-Clifford oracle."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from folded_msc_family import make_circuit
from folded_msc_oracle import evaluate
from validate_folded_msc import bind_faults, native_replay, sample_faults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("build-study/replay_cultivation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=int, default=24)
    args = parser.parse_args()
    directory = Path(__file__).parent
    report = dict(
        source_files_sha256={
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "folded_msc.py",
                "folded_msc_family.py",
                "folded_msc_oracle.py",
                "clifford_gadget.py",
                "validate_folded_msc.py",
            )
        },
        native_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        f5_circuit_sha256=make_circuit(0.001).manifest()["sha256"],
        small_aer_crosschecks=[],
        f5_histories=[],
    )
    previous = json.loads((directory / "research/folded_msc_validation.json").read_text())
    source = make_circuit(0.02, distance=3)
    assert source.manifest()["sha256"] == previous["circuit"]["sha256"]
    for case in previous["trajectories"]:
        physical = bind_faults(source, {int(i): label for i, label in case["faults"].items()})
        result = evaluate(physical, case["records"])
        error = abs(result["log_probability"] - case["aer_log_probability"])
        if error > 1e-10:
            raise AssertionError("folded oracle disagrees with the dense physical Aer reference")
        report["small_aer_crosschecks"].append(
            dict(
                faults=case["faults"],
                records=case["records"],
                oracle=result,
                absolute_log_error=error,
            )
        )
    rng = np.random.default_rng(9801)
    for case in range(args.traces):
        probability = 0 if case == 0 else (0.001, 0.01, 0.03)[(case - 1) % 3]
        source = make_circuit(probability)
        faults = sample_faults(source, rng)
        physical = bind_faults(source, faults)
        native = native_replay(physical, None, args.reference, seed=800 + case)
        start = time.perf_counter()
        result = evaluate(physical, native["records"])
        elapsed = time.perf_counter() - start
        error = abs(result["log_probability"] - native["log_probability"])
        if not native["reachable"] or error > 1e-10:
            raise AssertionError(f"physical f5 history disagrees with the folded oracle: {case}")
        row = dict(
            case=case,
            probability=probability,
            faults=faults,
            native=native,
            oracle=result,
            oracle_seconds=elapsed,
            absolute_log_error=error,
        )
        report["f5_histories"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    case=case,
                    faults=len(faults),
                    peak_terms=result["peak_terms"],
                    absolute_log_error=error,
                )
            ),
            flush=True,
        )
    print(
        json.dumps(
            dict(
                f5_histories=len(report["f5_histories"]),
                max_log_error=max(row["absolute_log_error"] for row in report["f5_histories"]),
                peak_terms=max(row["oracle"]["peak_terms"] for row in report["f5_histories"]),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
