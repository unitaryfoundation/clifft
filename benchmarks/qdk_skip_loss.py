"""Reproduce the surface-code skip-gate loss comparison through public APIs.

Run each backend in an isolated environment and pin all runs to the same CPU.
The matched-loss variant preserves the original loss locations but disables
Pauli/readout noise, restores lost sites on reset, and reports loss heralds.
It avoids the tools' different policies for noise touching a lost operand.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path


def make_circuit(distance, variant, loss_rate):
    import stim

    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0.001,
        before_round_data_depolarization=0.001,
        before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001,
    ).flattened()
    lines = []
    loss_sites = 0
    for instruction in base:
        name = instruction.name
        if name in {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        if variant.startswith("matched") and name in {
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "X_ERROR",
            "Y_ERROR",
            "Z_ERROR",
            "M",
            "MX",
            "MY",
            "MR",
            "MRX",
            "MRY",
        }:
            instruction = stim.CircuitInstruction(name, instruction.targets_copy(), [0])
        lines.append(str(instruction))
        targets = [str(t.value) for t in instruction.targets_copy() if t.is_qubit_target]
        if targets:
            loss_sites += len(targets)
            if not variant.endswith("no-loss") and loss_rate:
                lines.append(f"LOSS({loss_rate}) " + " ".join(targets))
    return "\n".join(lines) + "\n", {
        "loss_sites": loss_sites,
        "physical_qubits": len(base.get_final_qubit_coordinates()),
        "qubit_index_span": base.num_qubits,
        "measurements": base.num_measurements,
    }


def loss_model(noncomp, restore):
    return noncomp.Model(
        classifier=noncomp.Classifier([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1]]),
        reset_restores_lost=restore,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["clifft", "qdk", "stim"], required=True)
    parser.add_argument(
        "--variant",
        choices=["reported", "matched-loss", "no-loss", "matched-no-loss"],
        required=True,
    )
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--loss-rate", type=float, default=0.001)
    parser.add_argument("--shots", type=int, nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--execution", choices=["auto", "trajectory"], default="auto")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--circuits", type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    text, metadata = make_circuit(args.distance, args.variant, args.loss_rate)
    if args.circuits:
        args.circuits.mkdir(parents=True, exist_ok=True)
        (args.circuits / f"d{args.distance}-{args.variant}-{args.loss_rate}.stim").write_text(text)
    metadata.update(
        backend=args.backend,
        variant=args.variant,
        distance=args.distance,
        loss_rate=args.loss_rate,
        cpu=args.cpu,
        python=platform.python_version(),
        stim=importlib.metadata.version("stim"),
        clifft_reset_restores_lost=args.variant != "reported",
        output_contract="raw measurements and loss heralds; no detectors or final status",
        execution=args.execution,
    )
    start = time.perf_counter()
    if args.backend == "qdk":
        from qdk import stim as qdk_stim
        from qdk.simulation import run_qir

        start = time.perf_counter()
        qir, noise = qdk_stim.compile(text.replace("LOSS(", "LOSS_ERROR("))
        prepare = time.perf_counter() - start
        metadata.update(version=importlib.metadata.version("qdk"), strategy="stabilizer")

        def sample(shots, seed):
            return run_qir(qir, shots=shots, seed=seed, noise=noise, type="stabilizer")

    elif args.backend == "clifft":
        import clifft
        from clifft import noncomp

        model = loss_model(noncomp, args.variant != "reported")
        start = time.perf_counter()
        if hasattr(noncomp, "compile"):
            program = noncomp.compile(text, model, final_status=False, execution=args.execution)
            prepare = time.perf_counter() - start
            metadata.update(strategy=program.strategy, explanation=program.explain())

            def sample(shots, seed):
                return program.sample(shots, seed=seed, threads=1, collect_stats=True)

        else:
            prepare = None
            metadata.update(
                strategy="trajectory",
                api="convenience; preparation included per call",
                output_contract="raw measurements, loss heralds, and mandatory final status",
            )

            def sample(shots, seed):
                return noncomp.sample(text, model, shots=shots, seed=seed, threads=1)

        metadata.update(version=clifft.version(), isa=clifft.runtime_isa())
    else:
        import stim

        if "LOSS(" in text:
            raise ValueError("Stim cannot execute loss")
        start = time.perf_counter()
        program = stim.Circuit(text).compile_sampler(seed=42)
        prepare = time.perf_counter() - start
        metadata.update(version=stim.__version__, strategy="frame")

        def sample(shots, seed):
            return program.sample(shots)

    metadata["preparation_seconds"] = prepare
    print(json.dumps({"event": "prepared", **metadata}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output:
        for repetition in range(args.repetitions):
            counts = args.shots if repetition % 2 == 0 else list(reversed(args.shots))
            for shots in counts:
                start = time.perf_counter()
                result = sample(shots, 17001 + repetition * 101 + shots)
                elapsed = time.perf_counter() - start
                stats = getattr(result, "stats", None)
                row = {
                    **metadata,
                    "repetition": repetition,
                    "shots": shots,
                    "seconds": elapsed,
                    "stats": dataclasses.asdict(stats) if stats is not None else None,
                }
                if args.backend == "qdk":
                    assert len(result) == shots
                    assert len(result[0]) == metadata["measurements"]
                else:
                    records = result.measurements if hasattr(result, "measurements") else result
                    assert records.shape == (shots, metadata["measurements"])
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {
                            "event": "timed",
                            "seconds": elapsed,
                            "shots": shots,
                            "repetition": repetition,
                            "stats": row["stats"],
                        }
                    ),
                    flush=True,
                )
                del result


if __name__ == "__main__":
    main()
