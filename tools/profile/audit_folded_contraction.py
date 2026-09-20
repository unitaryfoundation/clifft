"""Check fixed contractions on complete physical histories and native continuations."""

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import stim
from compiled_gadget_contraction import write_native_plan
from folded_check_contraction import CodeBoundary, FoldedChecks
from folded_msc_family import make_circuit
from validate_folded_msc import bind_faults


def pauli(axis, row, width):
    p = stim.PauliString(width)
    for q in range(width):
        if row >> q & 1:
            p[q] = axis
    return p


def pauli_text(p):
    if p.sign != 1:
        raise ValueError("probe requires a positive physical Pauli")
    return "*".join(f"{'IXYZ'[axis]}{q}" for q, axis in enumerate(p) if axis)


def replay(text, records, binary, directory):
    source, forced = directory / "physical.stim", directory / "records.txt"
    source.write_text(text)
    forced.write_text(records)
    return json.loads(
        subprocess.check_output([str(binary.resolve()), str(source), "0", str(forced)], text=True)
    )


def boundary_at_prefix(kernel, physical, saved_records, reference, directory):
    start_stage = "d5_logical_check_1" if kernel.surface.distance == 5 else "logical_check_1"
    stop = next(i for i, op in enumerate(physical.operations) if op.stage == start_stage)
    prefix_ops = physical.operations[:stop]
    visible = sum(op.name == "M" for op in prefix_ops)
    hidden = sum(op.name == "R" for op in prefix_ops)
    full_visible = len(physical.measurements)
    prefix = replace(physical, operations=prefix_ops, measurements=physical.measurements[:visible])
    width = physical.manifest()["num_qubits"]
    logicals = [pauli("X", kernel.logical_x, width), pauli("Z", kernel.logical_z, width)]
    logicals.insert(1, 1j * logicals[0] * logicals[1])
    probes = [
        pauli(axis, row, width)
        for axis, rows in (("X", kernel.x_rows), ("Z", kernel.z_rows))
        for row in rows
    ]
    probes += logicals + [pauli("Z", 1 << q, width) for q in range(kernel.n, width)]
    text = prefix.text() + "".join(f"EXP_VAL {pauli_text(p)}\n" for p in probes)
    records = saved_records[:visible] + saved_records[full_visible : full_visible + hidden]
    native = replay(text, records, reference, directory)
    if not native["reachable"]:
        raise AssertionError("saved prefix is unreachable")
    values = native["expectation_values"]
    rank = kernel.rank
    if any(abs(abs(v) - 1) > 1e-9 for v in values[: 2 * rank] + values[2 * rank + 3 :]):
        raise AssertionError("prefix is not in a signed CSS code with product ancillas")
    offset = 0
    for value, dual in zip(values[rank : 2 * rank], kernel.z_duals[:-1], strict=True):
        if value < 0:
            offset ^= dual
    x, y, z = values[2 * rank : 2 * rank + 3]
    if abs(x * x + y * y + z * z - 1) > 1e-9:
        raise AssertionError("prefix logical state is not pure")
    if z >= 0:
        a = math.sqrt((1 + z) / 2)
        logical = np.array([a, complex(x, y) / (2 * a)])
    else:
        b = math.sqrt((1 - z) / 2)
        logical = np.array([complex(x, -y) / (2 * b), b])
    boundary = CodeBoundary(
        offset,
        sum((v < 0) << i for i, v in enumerate(values[:rank])),
        logical,
        sum((v < 0) << i for i, v in enumerate(values[2 * rank + 3 :])),
    )
    return boundary, native, prefix, logicals


def css_continuation(kernel, operations, bound, syndrome, amplitudes):
    """Offline compile/bind of Pauli faults through sequential CSS extraction."""
    width = kernel.n + (6 if kernel.surface.distance == 5 else 3)
    frame = stim.PauliString(width)
    ancillas = bound["ancillas"]
    records = []
    check = 0
    for op in operations:
        q = op.qubits[0]
        if op.name in {"X", "Y", "Z"}:
            frame = pauli(op.name, 1 << q, width) * frame
        elif op.name in {"H", "CX"}:
            frame = frame.after(stim.CircuitInstruction(op.name, op.qubits))
        elif op.name in {"M", "R"}:
            a = q - kernel.n
            if a < 0:
                raise ValueError("CSS extraction must measure or reset an ancilla")
            if op.name == "M":
                index = check % (2 * kernel.rank)
                ideal = (
                    ((syndrome >> index) & 1)
                    if index < kernel.rank
                    else ((bound["offset"] & kernel.z_rows[index - kernel.rank]).bit_count() % 2)
                )
                check += 1
                ancillas = (ancillas & ~(1 << a)) | (ideal << a)
            else:
                ideal = (ancillas >> a) & 1
                ancillas &= ~(1 << a)
            records.append(ideal ^ int(frame[q] in (1, 2)))
            if op.name == "R":
                frame[q] = "I"
        else:
            raise ValueError("unsupported CSS continuation operation")
    if check != 4 * kernel.rank:
        raise ValueError("expected noisy checks followed by noiseless final checks")
    x = sum((frame[q] in (1, 2)) << q for q in range(kernel.n))
    z = sum((frame[q] in (2, 3)) << q for q in range(kernel.n))
    logical = amplitudes / np.linalg.norm(amplitudes)
    logical[1] *= (-1) ** ((z & kernel.logical_x).bit_count() % 2)
    boundary = CodeBoundary(
        bound["offset"] ^ x,
        syndrome ^ sum(((z & row).bit_count() % 2) << i for i, row in enumerate(kernel.x_rows)),
        logical,
        ancillas ^ sum((frame[q] in (1, 2)) << (q - kernel.n) for q in range(kernel.n, width)),
    )
    return tuple(records), boundary


def chronological_records(ops, records, full_visible, visible_start, hidden_start):
    output = []
    m, r = visible_start, full_visible + hidden_start
    for op in ops:
        if op.name in {"M", "R"}:
            i = m if op.name == "M" else r
            output.append(int(records[i]))
            m += op.name == "M"
            r += op.name == "R"
    return tuple(output)


def physical_records(groups):
    visible: list[str] = []
    hidden: list[str] = []
    for operations, records in groups:
        index = 0
        for op in operations:
            if op.name in {"M", "R"}:
                (visible if op.name == "M" else hidden).append(str(records[index]))
                index += 1
        if index != len(records):
            raise ValueError("incorrect chronological record count")
    return "".join(visible + hidden)


def export_bundle(directory, kernel, cases):
    directory.mkdir()
    (directory / "dimensions.txt").write_text(
        f"{kernel.rank} {len(kernel.masks)} {kernel.characters}\n"
    )
    write_native_plan(directory / "amplitude.txt", kernel.amplitude_plan, [], marginal=True)
    for i, plan in enumerate(kernel.marginals):
        write_native_plan(directory / f"marginal_{i}.txt", plan, [], marginal=True)
    with (directory / "inputs.txt").open("w") as f:
        f.write(f"{len(cases)}\n")
        for choices in cases:
            for bound in choices:
                for terms in bound["inputs"]:
                    for scalar, local in terms:
                        for value in (scalar, *local.flatten()):
                            f.write(f"{value.real:.17g} {value.imag:.17g}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("build-study/replay_cultivation"))
    parser.add_argument("--sampler", type=Path, default=Path("build-study/sample_folded_checks"))
    parser.add_argument("--distance", type=int, choices=(3, 5), default=5)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    root = Path(__file__).parent / "research"
    old = json.loads(
        (
            root
            / (
                "folded_msc_validation.json"
                if args.distance == 3
                else "folded_msc_f5_validation.json"
            )
        ).read_text()
    )
    histories = old["trajectories"] if args.distance == 3 else old["f5_histories"]
    kernel = FoldedChecks(args.distance)
    plans = [kernel.amplitude_plan, *kernel.marginals]
    report: dict = dict(
        distance=args.distance,
        scope="native contraction sampling with offline-bound faults and incoming states; "
        "not full-attempt throughput",
        sampler_sha256=hashlib.sha256(args.sampler.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "folded_check_contraction.py",
                "audit_folded_contraction.py",
                "sample_folded_checks.cpp",
                "gadget_contraction_kernel.h",
            )
        },
        rank=kernel.rank,
        peak_scope=max(p.peak_scope for p in plans),
        marginal_sum_work=sum(p.work for p in kernel.marginals),
        plan_payload_bytes=sum(
            16 * (p.storage + (1 << p.peak_scope))
            + 8 * (p.gather_entries + sum(len(labels) for _, labels in p.leaves) + len(p.outputs))
            for p in plans
        ),
        histories=[],
        native_traces=[],
    )
    cases, contexts = [], []
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for index, case in enumerate(histories[: args.cases]):
            circuit = make_circuit(
                0.02 if args.distance == 3 else case["probability"], distance=args.distance
            )
            physical = bind_faults(circuit, {int(i): v for i, v in case["faults"].items()})
            records = case["records"] if args.distance == 3 else case["native"]["records"]
            expected = (
                case["aer_log_probability"]
                if args.distance == 3
                else case["oracle"]["log_probability"]
            )
            boundary, prefix_result, prefix, logicals = boundary_at_prefix(
                kernel, physical, records, args.reference, directory
            )
            stages = (
                ("d5_logical_check_1", "d5_logical_check_2")
                if args.distance == 5
                else ("logical_check_1", "logical_check_2")
            )
            block = [op for op in physical.operations if op.stage in stages]
            after_stages = (
                ("d5_post_checks", "final_syndrome")
                if args.distance == 5
                else ("post_checks", "final_syndrome")
            )
            after = [op for op in physical.operations if op.stage in after_stages]
            final = [op for op in physical.operations if op.stage == "final_logical"]
            prefix_records = chronological_records(
                prefix.operations, records, len(physical.measurements), 0, 0
            )
            body_records = chronological_records(
                block,
                records,
                len(physical.measurements),
                prefix_result["visible"],
                prefix_result["hidden"],
            )
            started = time.perf_counter()
            choices = [
                kernel.bind(block, boundary, outcome)
                for outcome in itertools.product((0, 1), repeat=2)
            ]
            binding_seconds = time.perf_counter() - started
            selected = next(b for b in choices if b["records"] == body_records)
            # Infer the ideal X syndrome from the physical noisy extraction by
            # compiling its Pauli record flips with the all-zero ideal syndrome.
            observed = chronological_records(
                after,
                records,
                len(physical.measurements),
                prefix_result["visible"] + sum(op.name == "M" for op in block),
                prefix_result["hidden"] + sum(op.name == "R" for op in block),
            )
            zero_records, _ = css_continuation(
                kernel, after, selected, 0, np.array([1, 0], complex)
            )
            measurement_slots = [
                i for i, op in enumerate(o for o in after if o.name in {"M", "R"}) if op.name == "M"
            ]
            syndrome = sum(
                (observed[i] ^ zero_records[i]) << j
                for j, i in enumerate(measurement_slots[: kernel.rank])
            )
            amplitudes = kernel.amplitudes(selected, syndrome)
            probability = float(np.vdot(amplitudes, amplitudes).real)
            post_records, continuation = css_continuation(
                kernel, after, selected, syndrome, amplitudes
            )
            if post_records != observed:
                raise AssertionError("CSS continuation record mismatch")
            final_records = chronological_records(
                final,
                records,
                len(physical.measurements),
                len(physical.measurements) - sum(op.name == "M" for op in final),
                sum(op.name == "R" for op in physical.operations)
                - sum(op.name == "R" for op in final),
            )
            endings = [kernel.bind(final, continuation, (bit,)) for bit in (0, 1)]
            ending = next(b for b in endings if b["records"] == final_records)
            final_probability = kernel.marginal(ending, 0, 0)
            actual = (
                prefix_result["log_probability"]
                + math.log(probability)
                + math.log(final_probability)
            )
            error = abs(actual - expected)
            if error > 1e-9:
                raise AssertionError(
                    f"complete physical history mismatch {index}: {actual} vs {expected}"
                )
            report["histories"].append(
                dict(
                    case=index,
                    absolute_log_error=error,
                    binding_seconds=binding_seconds,
                    block_probability=probability,
                    final_probability=final_probability,
                )
            )
            cases.append(choices)
            contexts.append(
                (physical, prefix, block, after, prefix_records, prefix_result, logicals)
            )
            print(f"f{args.distance} complete history {index}: error {error:.3g}", flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        bundle = directory / "bundle"
        export_bundle(bundle, kernel, cases)
        output = subprocess.check_output(
            [str(args.sampler.resolve()), str(bundle), str(args.shots), str(2 * len(cases)), "812"],
            text=True,
        )
        rows = [json.loads(line) for line in output.splitlines()]
        report["benchmark"] = rows[-1]
        report["kernel_seconds_per_sample"] = statistics.median(rows[-1]["seconds"]) / args.shots
        for shot in rows[:-1]:
            case = shot["shot"] % len(cases)
            physical, prefix, block, after, prefix_records, prefix_result, logicals = contexts[case]
            bound = cases[case][shot["root_outcomes"]]
            amplitudes = np.array([complex(*a) for a in shot["logical"]])
            after_records, continuation = css_continuation(
                kernel, after, bound, shot["syndrome"], amplitudes
            )
            records = physical_records(
                [
                    (prefix.operations, prefix_records),
                    (block, bound["records"]),
                    (after, after_records),
                ]
            )
            ops = prefix.operations + block + after
            visible = sum(op.name == "M" for op in ops)
            source = replace(physical, operations=ops, measurements=physical.measurements[:visible])
            text = source.text() + "".join(f"EXP_VAL {pauli_text(p)}\n" for p in logicals)
            result = replay(text, records, args.reference, directory)
            log_error = abs(
                result["log_probability"]
                - prefix_result["log_probability"]
                - shot["log_probability"]
            )
            a, b = continuation.logical
            sign = (-1) ** ((continuation.offset & kernel.logical_z).bit_count() % 2)
            bloch = [
                2 * (a.conjugate() * b).real,
                sign * 2 * (a.conjugate() * b).imag,
                sign * (abs(a) ** 2 - abs(b) ** 2),
            ]
            state_error = max(
                abs(a - b) for a, b in zip(bloch, result["expectation_values"], strict=True)
            )
            if not result["reachable"] or max(log_error, state_error) > 1e-9:
                raise AssertionError(f"native continuation mismatch: {log_error} {state_error}")
            report["native_traces"].append(
                dict(
                    **shot, case=case, absolute_log_error=log_error, logical_bloch_error=state_error
                )
            )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["benchmark"]), flush=True)


if __name__ == "__main__":
    main()
