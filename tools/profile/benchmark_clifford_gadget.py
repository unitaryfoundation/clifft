"""Reproducible physical-history checks and timings of the offline gadget."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import cirq  # type: ignore[import-not-found]
import numpy as np
import stim
from clifford_gadget import (
    CoherentState,
    Gadget,
    atoms_from_text,
    bind_faults,
    compile_gadgets,
    evaluate,
    native_replay_text,
    sample,
)
from soft_cultivation_study import source_text

import clifft


def prepare_terminal(program, width, records):
    begin = max(i for i, operation in enumerate(program) if isinstance(operation, Gadget))
    state = CoherentState(width)
    for operation in program[:begin]:
        if isinstance(operation, Gadget):
            state.compress_qubit()
            reset = state.gadget(
                operation, records[operation.measurement.record] ^ operation.measurement.flip
            )
            if reset != records[operation.reset.record]:
                raise AssertionError("prefix reset history is inconsistent")
        elif operation.record >= 0:
            state.project(operation, records[operation.record] ^ operation.flip)
        else:
            state.gate(operation)
    state.compress_qubit()
    probability = state.norm()
    for term in state.terms:
        term.coefficient /= math.sqrt(probability)
    return state, program[begin:], math.log(probability)


def write_slots(program):
    slots = []
    for operation in program:
        if isinstance(operation, Gadget):
            slots.extend([operation.measurement.record, operation.reset.record])
        elif operation.record >= 0:
            slots.append(operation.record)
    return slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = args.native.resolve()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    source, manifest = source_text()
    report = dict(
        provenance=manifest,
        python=platform.python_version(),
        clifft=clifft.__version__,
        cirq=cirq.__version__,
        stim=stim.__version__,
        numpy=np.__version__,
        cpu=next(
            x.split(":", 1)[1].strip()
            for x in Path("/proc/cpuinfo").read_text().splitlines()
            if x.startswith("model name")
        ),
        affinity=cpu,
        threads=1,
        histories=[],
        terminal_samples=[],
        full_samples=[],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory() as directory:
        path, record_path = Path(directory) / "physical.stim", Path(directory) / "records.txt"

        def reference(text, seed, records=None):
            path.write_text(native_replay_text(text))
            command = [str(native), str(path), str(seed)]
            if records is not None:
                record_path.write_text("".join(map(str, records)))
                command.append(str(record_path))
            result = json.loads(subprocess.check_output(command, text=True))
            if not result["reachable"]:
                raise AssertionError("native replay declined a reachable history")
            return result

        for noise, seed in [
            (0, 71),
            (0, 72),
            (None, 71),
            (None, 72),
            (None, 73),
            (None, 74),
            (0.005, 75),
            (0.005, 76),
        ]:
            start = time.perf_counter()
            text, faults = bind_faults(source, seed, noise)
            width, visible, hidden, atoms = atoms_from_text(text)
            program = compile_gadgets(width, atoms)
            preparation = time.perf_counter() - start
            truth = reference(text, seed)
            records = list(map(int, truth["records"]))
            start = time.perf_counter()
            actual = evaluate(program, width, records)
            elapsed = time.perf_counter() - start
            error = abs(actual["log_probability"] - truth["log_probability"])
            if error > 1e-9:
                raise AssertionError("coherent physical-history probability differs")
            start = time.perf_counter()
            initial, suffix, prefix_log = prepare_terminal(program, width, records)
            boundary_seconds = time.perf_counter() - start
            row = dict(
                noise="source" if noise is None else noise,
                seed=seed,
                faults=faults,
                bound_sha256=hashlib.sha256(text.encode()).hexdigest(),
                records=truth["records"],
                native_log_probability=truth["log_probability"],
                coherent=actual,
                absolute_log_error=error,
                fault_binding_and_plan_seconds=preparation,
                coherent_replay_seconds=elapsed,
                prefix_preparation_seconds=boundary_seconds,
                terminal_entry_terms=len(initial.terms),
            )
            report["histories"].append(row)
            print(
                "replay",
                row["noise"],
                seed,
                "faults",
                len(faults),
                "error",
                error,
                "entry terms",
                len(initial.terms),
                flush=True,
            )
            # Four independent conditional samples per selected boundary, with
            # original prefix records restored before native full-history replay.
            if seed == 71:
                for sample_seed in (181, 182, 183, 184):
                    start = time.perf_counter()
                    observed = sample(suffix, width, visible + hidden, sample_seed, initial)
                    seconds = time.perf_counter() - start
                    combined = records.copy()
                    for slot in write_slots(suffix):
                        combined[slot] = observed["records"][slot]
                    checked = reference(text, sample_seed, combined)
                    expected_log = prefix_log + observed["log_probability"]
                    error = abs(checked["log_probability"] - expected_log)
                    if error > 1e-9:
                        raise AssertionError(
                            "terminal-generated history differs from physical replay"
                        )
                    report["terminal_samples"].append(
                        dict(
                            noise=row["noise"],
                            fault_seed=seed,
                            sample_seed=sample_seed,
                            seconds=seconds,
                            records="".join(map(str, combined)),
                            peak_terms=observed["peak_terms"],
                            absolute_log_error=error,
                            conditional_log_probability=observed["log_probability"],
                        )
                    )
            save()

        for seed in (281, 282, 283):
            start = time.perf_counter()
            text, faults = bind_faults(source, seed)
            width, visible, hidden, atoms = atoms_from_text(text)
            program = compile_gadgets(width, atoms)
            prepare_seconds = time.perf_counter() - start
            start = time.perf_counter()
            observed = sample(program, width, visible + hidden, seed)
            seconds = time.perf_counter() - start
            checked = reference(text, seed, observed["records"])
            error = abs(checked["log_probability"] - observed["log_probability"])
            if error > 1e-9:
                raise AssertionError("full generated history differs from physical replay")
            report["full_samples"].append(
                dict(
                    seed=seed,
                    faults=faults,
                    prepare_seconds=prepare_seconds,
                    seconds=seconds,
                    records="".join(map(str, observed["records"])),
                    peak_terms=observed["peak_terms"],
                    absolute_log_error=error,
                    log_probability=observed["log_probability"],
                )
            )
            print("full sample", seed, seconds, "error", error, flush=True)
            save()

    start = time.perf_counter()
    ordinary = clifft.compile(source)
    compile_seconds = time.perf_counter() - start
    clifft.sample(ordinary, 16, seed=71, threads=1, batch_size=1)
    timings = []
    for seed in (71, 72, 73):
        start = time.perf_counter()
        clifft.sample(ordinary, 256, seed=seed, threads=1, batch_size=1)
        timings.append(time.perf_counter() - start)
    report["ordinary"] = dict(
        compile_seconds=compile_seconds,
        seconds=timings,
        shots_per_repeat=256,
        attempts_per_second=256 / statistics.median(timings),
        peak_active_width=ordinary.peak_active_width,
    )
    save()


if __name__ == "__main__":
    main()
