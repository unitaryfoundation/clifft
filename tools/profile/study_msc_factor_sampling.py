"""Bounded coherent CSS factor-sampling study, including synthetic large blocks."""

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import stim
from clifford_branches import BranchState, root, value
from export_msc_sampler import Writer
from fold_check import PhaseMonomial
from msc_factor_sampling import FactorCode, generated_color_checks
from msc_growth_contraction import FactoredTerm
from msc_sampling import SparseCode


def synthetic(code, seed):
    rng = np.random.default_rng(seed)
    logical = rng.normal(size=2) + 1j * rng.normal(size=2)
    logical /= np.linalg.norm(logical)
    frame = tuple(map(int, rng.integers(1 << code.width, size=2)))
    terms: list[FactoredTerm] = []
    for k in range(4):
        flip = int(rng.integers(1 << code.width))
        if k and (seed & 1 or k & 1):
            anchor = 0 if seed & 1 else k - 1
            flip = (
                terms[anchor].data.flips
                ^ (code.all_x if k & 1 else 0)
                ^ code.expand(int(rng.integers(1 << code.rank)))
            )
        terms.append(
            FactoredTerm(
                PhaseMonomial(
                    code.width,
                    flip,
                    int(rng.integers(8)),
                    (2 * rng.integers(4, size=code.width)).tolist(),
                ),
                np.empty((0, 2), dtype=complex),
                complex(*rng.normal(size=2)) / 4,
            )
        )
    scale = np.sqrt(code.norm(logical, frame, terms))
    for term in terms:
        term.weight /= scale
    return logical, frame, terms


def clifford_reference(checks, logical, frame, terms, syndrome, count, order):
    """Independent CH/Stim projection without enumerating computational support."""
    width = len(checks[0])
    reference = BranchState(width)
    zero = reference.terms[0]
    preparation = stim.Tableau.from_stabilizers(
        [*checks, stim.PauliString("Z" * width)]
    ).to_circuit()
    for op in preparation:
        targets = [t.value for t in op.targets_copy()]
        arity = 2 if op.name in {"CX", "CZ"} else 1
        for k in range(0, len(targets), arity):
            zero.gate(op.name, tuple(targets[k : k + arity]))
    anchor = zero.ch.inner_product_of_state_and_x(0)
    zero.ch.apply_global_phase(abs(anchor) / anchor)
    states = []
    amplitudes = []
    source_terms = []
    xchecks = [p for p in checks if any(axis == 1 for axis in p)]
    zchecks = [p for p in checks if any(axis == 3 for axis in p)]
    xs, zs = syndrome
    selected = [(p, (zs >> k) & 1) for k, p in enumerate(zchecks)]
    selected += [(xchecks[k], (xs >> k) & 1) for k in order[:count]]
    for t, payload in enumerate(terms):
        op = payload.data
        if any(c & 1 for c in op.linear):
            raise ValueError("large-block CH oracle requires Clifford monomials")
        for source in (0, 1):
            state = zero.copy()
            for q in range(width):
                if source:
                    state.gate("X", (q,))
            for q in range(width):
                if frame[1] >> q & 1:
                    state.gate("Z", (q,))
            for q in range(width):
                if frame[0] >> q & 1:
                    state.gate("X", (q,))
                for _ in range(op.linear[q] // 2):
                    state.gate("S", (q,))
                if op.flips >> q & 1:
                    state.gate("X", (q,))
            state.coefficient = root(op.global_phase)
            reachable = True
            for p, bit in selected:
                support = [q for q in range(width) if p[q]]
                x = p[support[0]] == 1
                if x:
                    for q in support:
                        state.gate("H", (q,))
                for q in support[:-1]:
                    state.gate("CX", (q, support[-1]))
                if not state.project(support[-1], bit):
                    reachable = False
                    break
                for q in reversed(support[:-1]):
                    state.gate("CX", (q, support[-1]))
                if x:
                    for q in support:
                        state.gate("H", (q,))
            if reachable:
                state.canonicalize()
                states.append(state)
                amplitudes.append(logical[source] * payload.weight * value(state.coefficient))
                source_terms.append(t)
    result = 0j
    for a, left in enumerate(states):
        for b, right in enumerate(states):
            spectator = np.prod(
                np.sum(
                    terms[source_terms[a]].ancillas.conj() * terms[source_terms[b]].ancillas, axis=1
                )
            )
            result += (
                amplitudes[a].conjugate()
                * amplitudes[b]
                * reference.overlap(left, right)
                * spectator
            )
    if abs(result.imag) > 1e-10:
        raise AssertionError("CH norm is not real")
    return float(result.real)


def write_factor_plan(w, plan):
    w.put(plan.storage, len(plan.leaves))
    for leaf in plan.leaves:
        w.put(leaf.parameter, leaf.offset)
        w.vector(leaf.parity.tolist())
    w.put(len(plan.steps))
    for step in plan.steps:
        w.put(step.offset, step.size, len(step.gathers))
        for row in step.gathers:
            for index in row:
                w.put(int(index))
    w.vector(plan.outputs)


def export_native(code, cases, path):
    if code.rank != len(code.zchecks) or code.rank >= 32 or code.width > 128:
        raise ValueError("native worker requires a bounded balanced CSS code")
    w = Writer()
    w.put(code.width, code.rank, len(cases))
    w.vector(code.sampling_order)
    w.vector([sum(1 << k for k in scope) for scope in code.supports])
    for plan in code.prefixes:
        write_factor_plan(w, plan)
    for logical, frame, terms in cases:
        prepared = code.prepare(logical, frame, terms)
        w.put(len(terms))
        w.vector(prepared.labels)
        for t in range(len(terms)):
            for output in (0, 1):
                w.complex(prepared.amplitudes[t, output])
                for phase in prepared.phases[t, output]:
                    w.put(int(phase))
        for coefficient in prepared.gram.reshape(-1):
            w.complex(coefficient)
    path.write_text("\n".join(w.tokens) + "\n")


def validate_native(native, path, code, checks, cases):
    results = []
    output = subprocess.check_output([str(native), str(path), "sample", "12", "913"], text=True)
    for k, line in enumerate(output.splitlines()):
        sample = json.loads(line)
        logical, frame, terms = cases[sample["case"]]
        prepared = code.prepare(logical, frame, terms)
        expected = prepared.mass(sample["zs"], code.rank, sample["xs"])
        expected /= sum(prepared.mass(label, 0, 0) for label in set(prepared.labels))
        error = abs(sample["probability"] / expected - 1)
        if error > 1e-10:
            raise AssertionError("native factor sample weight differs")
        sample["probability_relative_error"] = error
        if k < 2:
            reference = clifford_reference(
                checks,
                logical,
                frame,
                terms,
                (sample["xs"], sample["zs"]),
                code.rank,
                code.sampling_order,
            )
            reference_error = abs(sample["probability"] / reference - 1)
            if reference_error > 1e-9:
                raise AssertionError("native sample differs from CH reference")
            sample["ch_probability_relative_error"] = reference_error
        results.append(sample)
    return results


def block_circuit(checks):
    """Logical injection plus one ideal conjugated-Y check and CSS readout.

    This is a synthetic original-gate planning control, not full cultivation.
    """
    n = len(checks[0])
    preparation = stim.Tableau.from_stabilizers([*checks, stim.PauliString("X" * n)]).to_circuit()
    lines = [str(preparation).rstrip()]
    # Compute logical Z parity to apply the sole logical injection T.
    ladder = [f"CX {q} {n - 1}" for q in range(n - 1)]
    lines += [*ladder, f"T {n - 1}", *reversed(ladder)]
    data = " ".join(map(str, range(n)))
    lines += [f"T {data}", "MPP " + "*".join(f"Y{q}" for q in range(n)), f"T_DAG {data}"]
    lines.append(
        "MPP " + " ".join("*".join(f"{'_XYZ'[p[q]]}{q}" for q in range(n) if p[q]) for p in checks)
    )
    return "\n".join(lines) + "\n"


def study(native, baseline, output, shots):
    results = []
    cpu = min(os.sched_getaffinity(0))
    for distance in (3, 5, 7, 9):
        checks, width = generated_color_checks(distance)
        code = FactorCode(checks, width)
        cases = [synthetic(code, 762 + k) for k in range(4)]
        result = {
            "distance_label": distance,
            "scope": "synthetic CSS blocks",
            **code.statistics(),
            "validation": [],
            "timings": [],
            "cpu": cpu,
        }
        sparse = SparseCode(checks, width) if distance <= 5 else None
        for k, (logical, frame, terms) in enumerate(cases):
            prepared = code.prepare(logical, frame, terms)
            syndrome, probability = prepared.sample(random.Random(721 + k))
            xs = sum(bit << i for bit, (x, i) in zip(syndrome, code.order, strict=True) if x)
            zs = sum(bit << i for bit, (x, i) in zip(syndrome, code.order, strict=True) if not x)
            validations = []
            for count in (0, code.rank // 2, code.rank):
                actual = prepared.mass(zs, count, xs)
                expected = clifford_reference(
                    checks, logical, frame, terms, (xs, zs), count, code.sampling_order
                )
                error = abs(actual - expected) / max(expected, 1e-15)
                if error > 1e-9:
                    raise AssertionError((distance, k, count, actual, expected))
                validations.append(
                    {"prefix": count, "probability": actual, "relative_error": error}
                )
            if sparse:
                labels, weights = sparse.syndrome_weights(logical, frame, terms)
                expected = (
                    sum(weights[t, xs] for t, label in enumerate(labels) if label == zs)
                    / weights.sum()
                )
                if abs(expected - probability) > 1e-10:
                    raise AssertionError("sparse sample weight differs")
            result["validation"].append(
                {
                    "case": k,
                    "syndrome": syndrome,
                    "probability": probability,
                    "prefixes": validations,
                }
            )
            print(distance, "CH case", k, "passed", flush=True)
        with tempfile.TemporaryDirectory(prefix="msc-factor-") as directory:
            path = Path(directory) / "plan.txt"
            export_native(code, cases, path)
            result["serialized_plan_bytes"] = path.stat().st_size
            result["native_samples"] = validate_native(native, path, code, checks, cases)
            for mode in ("factor", "fourier") if distance <= 7 else ("factor",):
                for repeat in range(3):
                    count = shots if mode == "factor" or distance <= 5 else max(100, shots // 10)
                    timing = json.loads(
                        subprocess.check_output(
                            [
                                "taskset",
                                "-c",
                                str(cpu),
                                str(native),
                                str(path),
                                mode,
                                str(count),
                                str(31 + repeat),
                            ],
                            text=True,
                        )
                    )
                    result["timings"].append({"mode": mode, **timing})
                    print(distance, mode, timing, flush=True)
            circuit = Path(directory) / "block.stim"
            circuit.write_text(block_circuit(checks))
            result["original_gate_plan"] = json.loads(
                subprocess.check_output([str(baseline), str(circuit), "0", "plan"], text=True)
            )
        results.append(result)
        output.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=1000)
    args = parser.parse_args()
    study(args.native, args.baseline, args.output, args.shots)
