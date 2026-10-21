"""Compare scalar fixed execution with trajectories on matching output contracts.

Run against an optimized local Clifft build. This uses private validation
bindings; it is not a public prepared-program API or a timing assertion.
"""

import argparse
import json
import statistics
import time

import stim

from clifft import _clifft_core, noncomp


def circuit_text(distance, rounds, rate, variant):
    source = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.001,
    ).flattened()
    lines = []
    inserted = False
    for instruction in source:
        if instruction.name in {"QUBIT_COORDS", "SHIFT_COORDS"}:
            continue
        lines.append(str(instruction))
        if instruction.name == "H" and variant != "clifford" and not inserted:
            gate = "T" if variant == "t" else "R_Z(0.173)"
            lines.append(f"{gate} {instruction.targets_copy()[0].value}")
            inserted = True
        if instruction.name == "CX":
            targets = " ".join(str(target.value) for target in instruction.targets_copy())
            lines.append(f"LEAKAGE({rate}) {targets}")
    # A final reset erases every leaked label, so both paths can return the
    # same full final-status contract as well as records and heralds.
    lines.append("R " + " ".join(str(q) for q in range(source.num_qubits)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rates", type=float, nargs="+", default=[0, 0.001, 0.003])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["clifford", "t", "rotation"],
        default=["clifford", "t", "rotation"],
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    effect = noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25))
    model = noncomp.Model(
        classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]]),
        gate_partner_effects={"leaked": effect, "lost": effect},
    )
    for distance in args.distances:
        for variant in args.variants:
            for rate in args.rates:
                circuit = _clifft_core.parse(circuit_text(distance, args.rounds, rate, variant))
                start = time.perf_counter()
                program = _clifft_core._prepare_noncomputational_fixed(circuit, model._handle, True)
                preparation = time.perf_counter() - start
                if program is None:
                    raise RuntimeError("benchmark request unexpectedly rejected fixed execution")
                program.sample(1, 0)
                noncomp.sample(circuit, model, shots=1, seed=0)
                fixed_times: list[float] = []
                trajectory_times: list[float] = []
                for repetition in range(args.repetitions):
                    calls = [
                        (fixed_times, lambda: program.sample(args.shots, repetition, 1)),
                        (
                            trajectory_times,
                            lambda: noncomp.sample(
                                circuit, model, shots=args.shots, seed=repetition, threads=1
                            ),
                        ),
                    ]
                    for values, call in calls[:: 1 if repetition % 2 == 0 else -1]:
                        start = time.perf_counter()
                        call()
                        values.append(time.perf_counter() - start)
                fixed = statistics.median(fixed_times)
                trajectory = statistics.median(trajectory_times)
                print(
                    json.dumps(
                        {
                            "distance": distance,
                            "rounds": args.rounds,
                            "variant": variant,
                            "rate": rate,
                            "shots": args.shots,
                            "threads": 1,
                            "preparation_seconds": preparation,
                            "fixed_seconds": fixed,
                            "trajectory_seconds": trajectory,
                            "speedup": trajectory / fixed,
                            **program.metrics,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
