"""Compare scalar and packed fixed execution with trajectories on matching outputs.

Run against an optimized local Clifft build. Measurements use the public
prepared-program API; this is not a timing assertion.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any

import stim

from clifft import _clifft_core, noncomp


def circuit_text(distance, rounds, rate, variant, event="LEAKAGE"):
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
            gate = {"s": "S", "t": "T", "rotation": "R_Z(0.173)"}[variant]
            lines.append(f"{gate} {instruction.targets_copy()[0].value}")
            inserted = True
        if instruction.name == "CX":
            targets = " ".join(str(target.value) for target in instruction.targets_copy())
            lines.append(f"{event}({rate}) {targets}")
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
        choices=["clifford", "s", "t", "rotation"],
        default=["clifford", "t", "rotation"],
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-sizes", nargs="+", default=["1", "65", "auto"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--event", choices=["LEAKAGE", "LOSS"], default="LEAKAGE")
    parser.add_argument("--skip-trajectories", action="store_true")
    args = parser.parse_args()
    batch_sizes = [int(value) if value != "auto" else value for value in args.batch_sizes]
    if (
        args.shots <= 0
        or args.repetitions <= 0
        or any(isinstance(value, int) and value <= 0 for value in batch_sizes)
    ):
        parser.error("shots, repetitions, and numeric batch sizes must be positive")
    effect = noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25))
    model = noncomp.Model(
        classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]]),
        gate_partner_effects={"leaked": effect, "lost": effect},
    )
    for distance in args.distances:
        for variant in args.variants:
            for rate in args.rates:
                circuit = _clifft_core.parse(
                    circuit_text(distance, args.rounds, rate, variant, args.event)
                )
                start = time.perf_counter()
                program = noncomp.compile(circuit, model, final_status=True, execution="fixed")
                preparation = time.perf_counter() - start
                times: dict[str, list[float]] = {}
                stats = {}
                calls: list[tuple[str, Callable[[int], Any]]] = []
                for batch_size in batch_sizes:
                    name = f"fixed_{batch_size}"
                    times[name] = []
                    calls.append(
                        (
                            name,
                            partial(
                                program.sample,
                                args.shots,
                                threads=args.threads,
                                batch_size=batch_size,
                                collect_stats=True,
                            ),
                        )
                    )
                if not args.skip_trajectories:
                    times["trajectory"] = []
                    calls.append(
                        (
                            "trajectory",
                            lambda seed: noncomp.sample(
                                circuit,
                                model,
                                shots=args.shots,
                                seed=seed,
                                threads=args.threads,
                                execution="trajectory",
                                final_status=True,
                            ),
                        )
                    )
                # Shot count and threads must match to warm the measured execution policy.
                for _, call in calls:
                    call(0)
                for repetition in range(args.repetitions):
                    for name, call in calls[:: 1 if repetition % 2 == 0 else -1]:
                        start = time.perf_counter()
                        result = call(repetition)
                        times[name].append(time.perf_counter() - start)
                        if name != "trajectory":
                            stats[name] = asdict(result.stats)
                medians = {name: statistics.median(values) for name, values in times.items()}
                print(
                    json.dumps(
                        {
                            "distance": distance,
                            "rounds": args.rounds,
                            "variant": variant,
                            "event": args.event,
                            "rate": rate,
                            "shots": args.shots,
                            "threads": args.threads,
                            "preparation_seconds": preparation,
                            "seconds": medians,
                            "execution": stats,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
