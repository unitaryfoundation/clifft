"""Validate and time a fixed contraction on complete physical terminal records."""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from benchmark_clifford_gadget import prepare_terminal, write_slots
from clifford_gadget import atoms_from_text, bind_faults, compile_gadgets, native_replay_text
from compiled_gadget_contraction import TerminalCode, TerminalMarginals, write_native_plan
from soft_cultivation_study import source_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=Path("build-study/replay_cultivation"))
    parser.add_argument(
        "--kernel", type=Path, default=Path("build-study/profile_gadget_contraction")
    )
    parser.add_argument(
        "--prior", type=Path, default=Path("tools/profile/research/clifford_gadget_data.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-histories", type=int, default=24)
    args = parser.parse_args()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    source, manifest = source_text()
    prior = json.loads(args.prior.read_text())
    histories = [
        (r["noise"], r["seed"], r["records"], r["native_log_probability"])
        for r in prior["histories"]
    ]
    histories += [
        ("source" if i % 2 == 0 else 0.005, 900 + i, None, None)
        for i in range(args.fresh_histories)
    ]
    report = dict(
        provenance=manifest,
        python=platform.python_version(),
        numpy=np.__version__,
        affinity=cpu,
        cpu=prior["cpu"],
        histories=[],
        samples=[],
        ordinary_prior=prior["ordinary"],
    )
    code = None
    marginals = None
    cases = []
    marginal_cases: list[list] = []
    with tempfile.TemporaryDirectory() as directory:
        physical = Path(directory) / "physical.stim"
        forced = Path(directory) / "records.txt"

        def native(records, seed):
            command = [str(args.native.resolve()), str(physical), str(seed)]
            if records is not None:
                forced.write_text("".join(map(str, records)))
                command.append(str(forced))
            return json.loads(subprocess.check_output(command, text=True))

        for noise, seed, previous, native_log in histories:
            text, faults = bind_faults(source, seed, None if noise == "source" else noise)
            width, _, _, atoms = atoms_from_text(text)
            program = compile_gadgets(width, atoms)
            physical.write_text(native_replay_text(text))
            if previous is None:
                truth = native(None, seed)
                previous, native_log = truth["records"], truth["log_probability"]
            records = list(map(int, previous))
            start = time.perf_counter()
            state, suffix, prefix_log = prepare_terminal(program, width, records)
            prefix_seconds = time.perf_counter() - start
            if code is None:
                start = time.perf_counter()
                code = TerminalCode(suffix[0], suffix)
                report["plan"] = dict(
                    compile_seconds=time.perf_counter() - start,
                    rank=code.rank,
                    data_qubits=len(code.data),
                    largest_scope=code.plan.peak_scope,
                    sum_step_entries=code.plan.work,
                    scratch_complex_entries=code.plan.storage,
                    gather_entries=code.plan.gather_entries,
                    order=code.plan.order,
                )
                start = time.perf_counter()
                marginals = TerminalMarginals(code)
                report["marginal_plan_seconds"] = time.perf_counter() - start
                report["marginal_plans"] = [
                    dict(
                        measured=i,
                        largest_scope=p.peak_scope,
                        sum_step_entries=p.work,
                        scratch_complex_entries=p.storage,
                        gather_entries=p.gather_entries,
                    )
                    for i, p in enumerate(marginals.plans)
                ]
                marginal_cases = [[] for _ in marginals.plans]
            start = time.perf_counter()
            boundary = code.certify_boundary(state)
            adapter_seconds = time.perf_counter() - start
            start = time.perf_counter()
            inputs = code.bind(boundary, suffix, records)
            bind_seconds = time.perf_counter() - start
            start = time.perf_counter()
            probability = code.probability(inputs)
            python_seconds = time.perf_counter() - start
            expected = math.exp(native_log - prefix_log)
            error = abs(probability - expected)
            if error > 1e-11:
                raise AssertionError("terminal probability differs from native full replay")
            cases.append(inputs)
            assert marginals is not None
            choices = code.sampling_inputs(boundary, suffix, len(records))
            m = records[suffix[0].measurement.record] ^ suffix[0].measurement.flip
            y = records[code.logical_measurement.record] ^ code.logical_measurement.flip
            x_bits = sum((records[a.record] ^ a.flip) << i for i, a in enumerate(code.x_checks))
            for measured, queries in enumerate(marginal_cases):
                queries.append(marginals.bind_query(choices[m][1], measured, x_bits, y))
            if (noise, seed) in [(0, 71), ("source", 71), (0.005, 75), (0.005, 901)]:
                for sample_seed in (1011, 1012):
                    start = time.perf_counter()
                    observed = marginals.sample(choices, sample_seed)
                    sample_seconds = time.perf_counter() - start
                    combined = records.copy()
                    for slot in write_slots(suffix):
                        combined[slot] = observed["records"][slot]
                    truth = native(combined, sample_seed)
                    log_error = abs(
                        truth["log_probability"] - prefix_log - observed["log_probability"]
                    )
                    if not truth["reachable"] or log_error > 1e-9:
                        raise AssertionError(
                            "contraction-generated records differ from physical replay"
                        )
                    report["samples"].append(
                        dict(
                            noise=noise,
                            fault_seed=seed,
                            sample_seed=sample_seed,
                            seconds=sample_seconds,
                            absolute_log_error=log_error,
                            records="".join(map(str, combined)),
                            conditional_log_probability=observed["log_probability"],
                        )
                    )
            variants = []
            for slot in [
                code.logical_measurement.record,
                code.x_checks[0].record,
                code.z_checks[0].record,
                suffix[0].reset.record,
            ]:
                changed = records.copy()
                changed[slot] ^= 1
                actual = code.probability(code.bind(boundary, suffix, changed))
                truth = native(changed, seed)
                expected_variant = (
                    math.exp(truth["log_probability"] - prefix_log) if truth["reachable"] else 0.0
                )
                if abs(actual - expected_variant) > 1e-11:
                    raise AssertionError("changed terminal record probability differs")
                variants.append(dict(slot=slot, probability=actual, expected=expected_variant))
            report["histories"].append(
                dict(
                    noise=noise,
                    seed=seed,
                    faults=len(faults),
                    records=previous,
                    conditional_probability=probability,
                    expected=expected,
                    absolute_error=error,
                    prefix_seconds=prefix_seconds,
                    adapter_seconds=adapter_seconds,
                    bind_seconds=bind_seconds,
                    python_contraction_seconds=python_seconds,
                    changed_records=variants,
                )
            )
            print("history", noise, seed, "faults", len(faults), "error", error, flush=True)
        assert code is not None
        plan_path = Path(directory) / "plan.txt"
        write_native_plan(plan_path, code.plan, cases)
        results = []
        for _ in range(3):
            result = json.loads(
                subprocess.check_output(
                    [str(args.kernel.resolve()), str(plan_path), "1000"], text=True
                )
            )
            np.testing.assert_allclose(
                result["probabilities"],
                [r["conditional_probability"] for r in report["histories"]],
                atol=1e-13,
                rtol=1e-12,
            )
            results.append(result)
        report["native_kernel"] = dict(
            runs=results,
            median_seconds_per_probability=statistics.median(
                r["seconds"] / r["evaluations"] for r in results
            ),
        )
        assert marginals is not None
        marginal_timings = []
        for measured, (plan, queries) in enumerate(
            zip(marginals.plans, marginal_cases, strict=True)
        ):
            write_native_plan(plan_path, plan, queries, marginal=True)
            result = json.loads(
                subprocess.check_output(
                    [str(args.kernel.resolve()), str(plan_path), "1000"], text=True
                )
            )
            expected_marginals = [
                sum((a * plan.evaluate(local)).real for a, local in query) for query in queries
            ]
            np.testing.assert_allclose(
                result["probabilities"], expected_marginals, atol=1e-13, rtol=1e-11
            )
            marginal_timings.append(
                dict(
                    measured=measured,
                    seconds_per_query=result["seconds"] / result["evaluations"],
                    **result,
                )
            )
        report["native_marginal_kernels"] = marginal_timings
        report["sampling_query_budget_seconds"] = 5 * marginal_timings[0][
            "seconds_per_query"
        ] + sum(r["seconds_per_query"] for r in marginal_timings[1:])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "native seconds per complete terminal probability",
        report["native_kernel"]["median_seconds_per_probability"],
    )


if __name__ == "__main__":
    main()
