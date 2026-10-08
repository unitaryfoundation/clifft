"""Bounded offline study of reusable preparation and conditional classical sampling maps."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
from copy import copy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import stim
from bt27_circuit_corrections import Correction, bits
from bt27_scored_clifford import ScoredClifford
from profile_bt27_fault_specialization import ROOT
from profile_external_feedback import MERLIN_REVISION
from scored_clifford_core import (
    BooleanSampler,
    DiagonalCore,
    affine_forms,
    pack_forms,
    rank,
    unpack_forms,
    verify_forms,
)
from study_bt27_direct_clifford import extension_hash
from study_bt27_scored_sampling import sources
from study_bt81_scored_clifford import fixed_histories
from study_circuit_noise import History, Model, synthetic_bt_noise
from validate_bt27_scored_equivalence import exact_law, histories

ZERO = Correction(0, 0, 0, 0, 0, 0, 0)


def digest(value: Any) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def map_metrics(
    maps: list[int], laws: list[tuple[int, ...]], width: int, controls: int
) -> dict[str, Any]:
    started = perf_counter()
    sampler = BooleanSampler(maps[: 1 << controls], controls, width)
    compile_seconds = perf_counter() - started
    for assignment, law in enumerate(laws[: 1 << controls]):
        actual = sampler.matrix(assignment)
        if actual != maps[assignment]:
            raise AssertionError("Boolean coefficient compilation changed an affine map")
        verify_forms(unpack_forms(actual, width), law)
    rng = random.Random(171819)
    queries = [(rng.randrange(1 << controls), rng.getrandbits(width)) for _ in range(2048)]
    started = perf_counter()
    checksum = 0
    for assignment, random_bits in queries:
        checksum ^= sampler.sample(assignment, random_bits)
    elapsed = perf_counter() - started
    return {
        "controls": controls,
        "assignments_checked": 1 << controls,
        "distinct_output_laws": len(set(laws[: 1 << controls])),
        "distinct_unsigned_spaces": len(
            {tuple(row & ((1 << width) - 1) for row in law) for law in laws[: 1 << controls]}
        ),
        "nonzero_vector_monomials": len(sampler.terms),
        "scalar_coefficient_monomials": sum(value.bit_count() for _, value in sampler.terms),
        "maximum_control_degree": max((mask.bit_count() for mask, _ in sampler.terms), default=0),
        "coefficient_payload_bytes": sum(
            (value.bit_length() + 7) // 8 + (mask.bit_length() + 7) // 8
            for mask, value in sampler.terms
        ),
        "enumerated_map_payload_bytes": sum(
            (value.bit_length() + 7) // 8 for value in maps[: 1 << controls]
        ),
        "expression_compile_seconds_after_enumeration": compile_seconds,
        "evaluation_seconds_per_sample": elapsed / len(queries),
        "evaluation_checksum": checksum,
        "coefficient_sha256": digest(sampler.terms),
    }


def small_exhaustive() -> dict[str, Any]:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    gates: list[tuple[str, tuple[int, ...]]] = [("CZ", (0, 1)), ("CZ", (1, 2)), ("CZ", (0, 2))]
    gates += [(gate, (q,)) for gate in ("S", "Z") for q in range(3)]
    circuits = []
    laws = []
    maps = []
    for assignment in range(1 << len(gates)):
        circuit = QuantumCircuit(3)
        circuit.h(range(3))
        lines = ["RX 0 1 2"]
        for index, (gate, targets) in enumerate(gates):
            if assignment >> index & 1:
                getattr(circuit, gate.lower())(*targets)
                lines.append(gate + " " + " ".join(map(str, targets)))
        circuit.h(range(3))
        circuit.save_statevector()
        circuits.append(circuit)
        law = exact_law(stim.Circuit("\n".join(lines + ["MX 0 1 2"])), [0, 1, 2], 3)
        laws.append(law)
        maps.append(pack_forms(affine_forms(law, 3)))
    result = AerSimulator(method="statevector", max_parallel_threads=1).run(circuits).result()
    sampler = BooleanSampler(maps, len(gates), 3)
    maximum_error = 0.0
    for assignment in range(len(circuits)):
        probabilities = (
            np.bincount([sampler.sample(assignment, r) for r in range(8)], minlength=8) / 8
        )
        expected = np.abs(np.asarray(result.get_statevector(assignment))) ** 2
        maximum_error = max(maximum_error, float(np.max(np.abs(probabilities - expected))))
        np.testing.assert_allclose(probabilities, expected, atol=1e-12, rtol=0)
    # The triangle has odd output parity. Sampling only the image of its
    # adjacency matrix would incorrectly force even parity.
    if laws[7] != (15,):
        raise AssertionError("Expected the triangle's nonzero affine offset")
    single_laws = [
        exact_law(stim.Circuit("RX 0\n" + ("S 0\n" if f else "") + "MX 0"), [0], 1)
        for f in range(2)
    ]
    single = BooleanSampler([pack_forms(affine_forms(law, 1)) for law in single_laws], 1, 1)
    assert all(single.sample(f, r) == f & r for f in range(2) for r in range(2))
    return {
        "circuits": len(circuits),
        "random_assignments_per_circuit": 8,
        "aer_maximum_probability_error": maximum_error,
        "conditional_s_is_fault_and_random": True,
        "triangle_zero_offset_negative_detected": True,
        "growth": [map_metrics(maps, laws, 3, k) for k in (2, 4, 6, 8, 9)],
    }


def geometry(core: DiagonalCore, correction: Correction) -> int:
    s, _, cz = core.restrict(correction)
    return s | (cz << core.width)


def control_rank(core: DiagonalCore) -> dict[str, Any]:
    reducer = core.reducer
    controls = reducer.controls
    transform = []
    for field, width in zip(
        ("x", "z", "rotations", "records", "final_x", "final_z"), controls.fields
    ):
        for q in range(width):
            if field in ("z", "records", "final_z"):
                transform.append(0)
                continue
            if field == "rotations":
                s, _, cz = controls.rotation_actions[q]
                correction = replace(ZERO, s=s, cz=cz)
            else:
                correction = replace(ZERO, **{field: 1 << q})
            reduced, _ = reducer.reduce(correction)
            transform.append(geometry(core, reduced))
    active = sum(1 << i for i, value in enumerate(transform) if value)

    def project(packed: int) -> int:
        value = 0
        for i in bits(packed & active):
            value ^= transform[i]
        return value

    columns = [project(value) for value in controls.responses]
    rng = random.Random(199183)
    for _ in range(128):
        history = reducer.model.draw(rng)
        predicted = 0
        for site, outcome in history:
            for generator in bits(controls.outcomes[site][outcome]):
                predicted ^= columns[generator]
        actual = geometry(core, reducer.evaluate(history)[0])
        if predicted != actual:
            raise AssertionError("Unsigned geometry is not the precomputed linear fault map")
    unique = set(columns)
    return {
        "primitive_fault_bits": len(columns),
        "unique_generator_responses": len(unique - {0}),
        "unsigned_core_control_rank": rank(list(unique)),
        "linear_map_fresh_history_checks": 128,
        "response_sha256": digest(columns),
        "scope": (
            "Distinct S-parity and CZ coefficients; this is not a lower bound on distinct "
            "measurement laws or minimum sampling-program size"
        ),
    }


def select_switches(core: DiagonalCore, count: int) -> History:
    rng = random.Random(778013)
    selected: list[tuple[int, int]] = []
    columns: list[int] = []
    model = core.reducer.model
    positions = ("before_first_t", "between_t_gates", "after_last_t")
    for position in (positions * count)[:count]:
        candidates = [
            i
            for i, site in enumerate(model.sites)
            if site.position == position and i not in {j for j, _ in selected}
        ]
        rng.shuffle(candidates)
        for site in candidates:
            for outcome, pauli in enumerate(model.sites[site].outcomes):
                if "X" not in pauli and "Y" not in pauli:
                    continue
                column = geometry(core, core.reducer.evaluate(((site, outcome),))[0])
                if rank(columns + [column]) > len(columns):
                    selected.append((site, outcome))
                    columns.append(column)
                    break
            else:
                continue
            break
        else:
            raise AssertionError("Could not select an independent control in this region")
    return tuple(selected)


def model_study(name: str, checkout: Path, switches: int) -> dict[str, Any]:
    started = perf_counter()
    if name == "bt27":
        model, tail, source = sources(checkout, "gate_noise")
        baseline = json.loads(
            (ROOT / "research/conditional_clifford/bt27-scored-sampling.json").read_text()
        )
        cases = {key: history for key, (_, history) in histories(model, baseline).items()}
        reference = json.loads(
            (ROOT / "research/conditional_clifford/bt27-scored-exact-equivalence.json").read_text()
        )["cases"]
    else:
        build = importlib.import_module(
            "benchmarks.protocols.code_switching"
        ).build_code_switching_case
        ideal = build(name, p_phys=0, target_scoring=False).circuit
        scored = build(name, p_phys=0, target_scoring=True).circuit
        assert scored.startswith(ideal)
        tail = scored[len(ideal) :]
        model = Model(synthetic_bt_noise(ideal, {"prep", "phase", "decoder"}, 0.001))
        source = model.render(None) + tail
        cases = fixed_histories(model)
        reference = json.loads(
            (ROOT / "research/conditional_clifford/bt81-scored-clifford.json").read_text()
        )["fixed_validation"]["cases"]
    reducer = ScoredClifford(model, tail)
    core = DiagonalCore(reducer)
    wrong = copy(reducer)
    flipped = next(q for q, row in enumerate(core.rows) if row)
    wrong.prefix = reducer.prefix + [f"Z {flipped}"]
    try:
        DiagonalCore(wrong)
    except ValueError as error:
        if "positive product state" not in str(error):
            raise
    else:
        raise AssertionError("Signed product-state negative control was not detected")
    setup = perf_counter() - started
    rows = {}
    for label, history in cases.items():
        correction, _ = reducer.evaluate(history)
        law = core.law(core.restrict(correction))
        verify_forms(affine_forms(law, core.width), law)
        full_law = core.joined_law(law, correction)
        if digest(full_law) != reference[label]["joint_law_sha256"]:
            raise AssertionError(f"Factorized law differs from reference: {name} {label}")
        rows[label] = {
            "core_random_dimension": core.width - len(law),
            "joint_law_sha256": digest(full_law),
        }
    print(f"{name}: {len(rows)} factorized joint laws match", flush=True)
    rank_info = control_rank(core)
    selected = select_switches(core, switches)
    laws = []
    maps = []
    validation_start = perf_counter()
    for assignment in range(1 << switches):
        history = tuple(selected[i] for i in bits(assignment))
        correction, _ = reducer.evaluate(history)
        law = core.law(core.restrict(correction))
        # All combinations in this bounded slice are also checked against the
        # physical circuit, independently of the core restriction formula.
        physical = exact_law(
            stim.Circuit(reducer.render(correction)),
            list(range(core.visible)),
            core.visible,
            correction.records,
        )
        if core.joined_law(law, correction) != physical:
            raise AssertionError("The exhaustive fault slice broke factorization")
        laws.append(law)
        maps.append(pack_forms(affine_forms(law, core.width)))
        if (assignment + 1) % 64 == 0:
            print(f"{name}: fault-switch assignments {assignment + 1}/{1 << switches}", flush=True)
    return {
        "source_sha256": digest(source),
        "physical_qubits": reducer.width,
        "core_qubits": core.width,
        "prefix_visible_records": core.prefix_records,
        "prefix_random_dimension": core.prefix_records - len(core.prefix_law),
        "deterministic_product_state_constraints": core.product_constraints,
        "signed_product_state_negative_detected": True,
        "fixed_z_readouts": len(core.z_positions),
        "factorized_reference_cases": rows,
        "setup_seconds": setup,
        "geometry": rank_info,
        "selected_fault_switches": [
            {
                "site": site,
                "outcome": outcome,
                "pauli": model.sites[site].outcomes[outcome],
                "position": model.sites[site].position,
            }
            for site, outcome in selected
        ],
        "exhaustive_enumeration_seconds": perf_counter() - validation_start,
        "growth": [map_metrics(maps, laws, core.width, k) for k in (2, 4, 6, 8) if k <= switches],
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--switches", type=int, choices=(2, 4, 6, 8), default=8)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = perf_counter()
    sources(args.merlin_checkout, "ideal")
    document: dict[str, Any] = {
        "merlin_revision": MERLIN_REVISION,
        "initial_state": "All zero at circuit entry with live preparation and feedback",
        "clifft_extension_sha256": extension_hash("clifft._clifft_core"),
        "stim_version": stim.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "small_exhaustive": small_exhaustive(),
        "models": {},
        "limits": (
            "Offline finite-slice Boolean compilation, not a full-noise reusable sampler. "
            "No executor change. Canonical-map complexity depends on the chosen random-bit "
            "basis; these sizes are not complexity lower bounds."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in ("bt27", "bt81"):
        document["models"][name] = model_study(name, args.merlin_checkout, args.switches)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
    document["elapsed_seconds"] = perf_counter() - started
    args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
