"""Check complete quantum-state and record interfaces across a reduced region."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from automatic_specialization import (
    ANNOTATIONS,
    History,
    exact_record_probabilities,
    instruction_text,
    specialize_prefix,
)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from regional_phase_specialization import RegionalPhase
from shared_phase_specialization import SharedPhase
from study_automatic_specialization import candidates, sample_parities, stress_histories
from study_regional_phase_specialization import parity_echo, small_cases
from validate_automatic_specialization import (
    all_histories,
    record_pauli,
    remap,
    stim_to_qiskit_noiseless,
    terminal_unitary,
)
from validate_shared_phase_specialization import exact_law, prefix_law, prefix_values

import clifft


def materialized_lines(region: RegionalPhase, history: History) -> list[str]:
    lines = region.model.lines.copy()
    for site, outcome in history:
        lines[region.model.sites[site].line] = region.model.sites[site].replacements[outcome]
    return lines


def entry_source(region: RegionalPhase, history: History) -> str:
    return (
        "\n".join(materialized_lines(region, history)[: region.boundary_line])
        + f"\nI {region.model.num_qubits - 1}\n"
    )


def unitary_part(source: str) -> str:
    return "\n".join(
        line
        for line in source.splitlines()
        if line and line.split(" ")[0].split("(")[0] not in ANNOTATIONS | {"MPAD"}
    )


def cq_blocks(source: str) -> np.ndarray:
    """Unnormalized density matrices conditioned on every visible record string."""
    parsed = clifft.parse(source)
    n, m = parsed.num_qubits, parsed.num_measurements
    if n + m > 10:
        raise ValueError("The complete instrument oracle exceeds its density-matrix budget")
    circuit = QuantumCircuit(n + m)
    records = 0
    for node in parsed.nodes:
        gate, targets = node.gate.name, list(node.targets)
        if gate in ANNOTATIONS:
            continue
        if gate in {"M", "MX", "MY", "MPP"}:
            terms = (
                [(repr(t).lstrip("!")[0], t.value) for t in targets]
                if gate == "MPP"
                else [({"M": "Z", "MX": "X", "MY": "Y"}[gate], targets[0].value)]
            )
            record_pauli(
                circuit,
                terms,
                n + records,
                inverted=bool(sum(repr(t).startswith("!") for t in targets) % 2),
            )
            records += 1
        elif gate == "MPAD":
            if targets[0].value:
                circuit.x(n + records)
            records += 1
        elif gate in {"R", "RX", "RY"}:
            for t in targets:
                circuit.reset(t.value)
                if gate != "R":
                    circuit.h(t.value)
                if gate == "RY":
                    circuit.s(t.value)
        elif any(t.is_rec for t in targets):
            if gate not in {"CX", "CY", "CZ"} or not targets[0].is_rec:
                raise ValueError("Unsupported instrument-reference feedback")
            getattr(circuit, gate.lower())(n + targets[0].value, targets[1].value)
        else:
            piece = stim_to_qiskit_noiseless(instruction_text(node, records))
            circuit.compose(piece, qubits=list(range(piece.num_qubits)), inplace=True)
    circuit.save_density_matrix()
    result = AerSimulator(method="density_matrix", max_parallel_threads=1).run(circuit).result()
    if not result.success:
        raise AssertionError("Aer instrument reference failed")
    density = np.asarray(result.data()["density_matrix"]).reshape(1 << m, 1 << n, 1 << m, 1 << n)
    # Record wires only control later gates. Reading diagonal record blocks
    # implements classical dephasing while retaining physical coherences.
    blocks: np.ndarray = np.stack([density[record, :, record, :] for record in range(1 << m)])
    if abs(np.trace(blocks, axis1=1, axis2=2).sum() - 1) > 1e-10:
        raise AssertionError("Instrument reference lost probability mass")
    return blocks


def compare_instrument(region: RegionalPhase, history: History) -> dict[str, Any]:
    values = prefix_values(region.shared)
    prefix, _ = region.split_history(history)
    prefix_law(region.shared, prefix)
    expected_entry = cq_blocks(entry_source(region, history))
    observed_entry = np.zeros_like(expected_entry)
    expected_final = cq_blocks(region.model.render(history))
    observed_final = np.zeros_like(expected_final)
    observed_records = np.zeros(len(expected_final))
    for value in values:
        controls = region.shared.controls(prefix, value)
        state_source = region.shared.render_state(controls)
        state_program = clifft.lower(clifft.trace(clifft.parse(unitary_part(state_source))))
        state = np.asarray(clifft.get_statevector(state_program))
        record = (controls >> 1) & ((1 << region.shared.prefix_records) - 1)
        observed_entry[record] += np.outer(state, state.conj()) / len(values)
        composed = region.compose(history, value)
        observed_final += cq_blocks(composed) / len(values)
        program = clifft.lower(clifft.trace(clifft.parse(composed)))
        observed_records += np.asarray(
            exact_record_probabilities(program, region.model.num_records)
        ) / len(values)
        sample_parities(region.model.source, clifft.sample(program, shots=3, seed=78, threads=1))
    expected_records = np.trace(expected_final, axis1=1, axis2=2).real
    for observed, expected in (
        (observed_entry, expected_entry),
        (observed_final, expected_final),
        (observed_records, expected_records),
    ):
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-10)
    return {
        "branches": len(values),
        "entry_error": float(np.max(abs(observed_entry - expected_entry))),
        "final_instrument_error": float(np.max(abs(observed_final - expected_final))),
        "record_probability_error": float(np.max(abs(observed_records - expected_records))),
    }


def validate_small() -> dict[str, Any]:
    originals = small_cases()
    sources = originals | {name + "_renamed": remap(source) for name, source in originals.items()}
    rows = []
    for index, (name, source) in enumerate(sources.items()):
        region = RegionalPhase(source)
        rng = random.Random(32851 + index)
        histories: list[History] = [()]
        if region.model.sites:
            histories.append(
                tuple(
                    (site, rng.randrange(1, len(region.model.sites[site].replacements)))
                    for site in sorted(
                        rng.sample(range(len(region.model.sites)), min(5, len(region.model.sites)))
                    )
                )
            )
            histories.append(region.model.draw(rng))
        histories = list(dict.fromkeys(histories))
        checks = [compare_instrument(region, history) for history in histories]
        rows.append({"case": name, "histories": histories, "checks": checks, **region.metadata()})
        print(name, "instrument passed", flush=True)
    return {"cases": rows}


def validate_mixture() -> dict[str, Any]:
    region = RegionalPhase(small_cases()["noise_mixture"])
    histories = all_histories(region.model)
    checks = [compare_instrument(region, history) for history, _ in histories]
    # Each history preserves the full instrument, and weights are the original
    # categorical law. Summing these bounds also bounds the noisy mixture.
    return {
        "histories": len(histories),
        "probability_mass": sum(p for _, p in histories),
        "weighted_final_instrument_error_bound": sum(
            p * row["final_instrument_error"] for (_, p), row in zip(histories, checks)
        ),
        "maximum_record_probability_error": max(row["record_probability_error"] for row in checks),
    }


def validate_unitary_exits(panel: dict[str, tuple[str, str]]) -> dict[str, Any]:
    rows = []
    sources = {name: panel[name][1] for name in ("15to1_direct_x", "bh_direct_x")}
    sources["parity_echo_6"] = parity_echo(6, 2)
    simulator = AerSimulator(method="statevector", max_parallel_threads=1)
    for name, source in sources.items():
        region = RegionalPhase(source)
        if region.shared.prefix_records:
            raise AssertionError("Statevector check requires an unmeasured preparation")
        values = prefix_values(region.shared)
        if len(values) != 1:
            raise AssertionError("Statevector check requires a pure unbranched preparation")
        worst = 0.0
        histories = stress_histories(region.model)
        for history in histories:
            prefix, _ = region.split_history(history)
            prefix_law(region.shared, prefix)
            unitary, readouts = terminal_unitary(entry_source(region, history))
            if readouts:
                raise AssertionError("The quantum-state boundary must precede readout")
            circuit = stim_to_qiskit_noiseless(unitary)
            circuit.save_statevector()
            expected = np.asarray(simulator.run(circuit).result().data()["statevector"])
            text = region.shared.render_state(region.shared.controls(prefix, values[0]))
            if region.model.num_qubits <= 10:
                observed = np.asarray(
                    clifft.get_statevector(
                        clifft.lower(clifft.trace(clifft.parse(unitary_part(text))))
                    )
                )
            else:
                # Clifft deliberately limits dense state export to ten wires.
                # Aer can check the larger physical exit without changing that API.
                normalized, _ = terminal_unitary(unitary_part(text))
                candidate = QuantumCircuit(region.model.num_qubits)
                piece = stim_to_qiskit_noiseless(normalized)
                candidate.compose(piece, qubits=list(range(piece.num_qubits)), inplace=True)
                candidate.save_statevector()
                observed = np.asarray(simulator.run(candidate).result().data()["statevector"])
            overlap = np.vdot(expected, observed)
            observed = observed * (np.conj(overlap) / abs(overlap))
            error = float(np.max(abs(observed - expected)))
            worst = max(worst, error)
            if error > 1e-10:
                raise AssertionError("Regional statevector differs from original Aer state")
        rows.append(
            {
                "case": name,
                "histories": histories,
                "maximum_amplitude_error": worst,
                "candidate_backend": "clifft" if region.model.num_qubits <= 10 else "aer",
            }
        )
        print(name, "statevector passed", flush=True)
    return {"cases": rows}


def validate_guards() -> dict[str, Any]:
    failures = 0
    for action in (
        lambda: SharedPhase("H 0\nT 0\nM 0", preserve_state=True),
        lambda: SharedPhase("H 0\nT 0\nM 0").render_state(1),
        lambda: SharedPhase("H 0\nT 0", preserve_state=True).render(1, None),
        lambda: RegionalPhase("H 0\nM 0"),
    ):
        try:
            action()
        except ValueError:
            failures += 1
        else:
            raise AssertionError("An unsupported interface did not reject")
    coherence = RegionalPhase(small_cases()["coherence"])
    value = prefix_values(coherence.shared)[0]
    text = coherence.shared.render_state(coherence.shared.controls((), value))
    good = cq_blocks(text)
    bad = cq_blocks("\n".join(line for line in text.splitlines() if not line.startswith("S ")))
    np.testing.assert_allclose(
        np.diagonal(good, axis1=1, axis2=2), np.diagonal(bad, axis1=1, axis2=2)
    )
    coherence_error = float(np.max(abs(good - bad)))
    if coherence_error < 0.1:
        raise AssertionError("The state oracle missed a phase invisible to Z-readout probabilities")
    correlation = RegionalPhase(small_cases()["record_correlation"])
    good = cq_blocks(entry_source(correlation, ()))
    bad = good[::-1].copy()
    np.testing.assert_allclose(good.sum(axis=0), bad.sum(axis=0))
    correlation_error = float(np.max(abs(good - bad)))
    if correlation_error < 0.1:
        raise AssertionError("The instrument oracle missed an incorrect record-state correlation")
    return {
        "unsupported_interfaces_rejected": failures,
        "detected_coherence_error": coherence_error,
        "detected_record_state_correlation_error": correlation_error,
    }


def validate_bt(panel: dict[str, tuple[str, str]], binary: Path) -> dict[str, Any]:
    import stim

    region = RegionalPhase(panel["bt27_scored"][1])
    histories = stress_histories(region.model)
    hashes = []
    with tempfile.TemporaryDirectory(prefix="regional-phase-check-") as directory:
        input_path, output_path = Path(directory) / "input.stim", Path(directory) / "output.stim"

        def law(source: str) -> tuple[int, ...]:
            input_path.write_text(source)
            subprocess.run(
                [str(binary.resolve()), "--export-source", str(input_path), str(output_path)],
                check=True,
                capture_output=True,
            )
            exported = output_path.read_text()
            order = [
                int(line.split()[2])
                for line in exported.splitlines()
                if line.startswith("# RECORD ")
            ]
            return exact_law(stim.Circuit(exported), order, region.model.num_records)

        for history in histories:
            prefix, tail = region.split_history(history)
            prefix_law(region.shared, prefix)
            fixed = materialized_lines(region, history)
            preparation = stim.Circuit("\n".join(fixed[: region.shared.first]))
            for seed in (17, 419):
                simulator = stim.TableauSimulator(seed=seed)
                simulator.set_num_qubits(region.model.num_qubits)
                simulator.do(preparation)
                values = list(map(int, simulator.current_measurement_record()))
                for row in region.shared.probe_rows:
                    expectation = simulator.peek_observable_expectation(row)
                    if expectation not in (-1, 1):
                        raise AssertionError("Preparation lacks certified stabilizers")
                    values.append(int(expectation == -1))
                fault_controls = 0
                for site, outcome in prefix:
                    fault_controls ^= region.shared.outcome_controls[site][outcome]
                controls = (
                    1
                    | (sum(bit << q for q, bit in enumerate(values)) << 1)
                    | (fault_controls << region.shared.fault_base)
                )
                composed = region.shared.render_state(controls) + "\n".join(tail) + "\n"
                reference, info = specialize_prefix(region.model.render(history), seed)
                if info["outcomes"] != values[: region.shared.prefix_records]:
                    raise AssertionError("Conditional references selected different records")
                observed, expected = law(composed), law(reference)
                if observed != expected:
                    raise AssertionError(f"Regional BT joint law differs: {history=} {seed=}")
                hashes.append(hashlib.sha256(str(observed).encode()).hexdigest())
    return {
        "histories": histories,
        "conditional_joint_laws": len(hashes),
        "law_hashes": hashes,
        "scope": "Conditional whole-output laws relative to the existing optimizer on both paths",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=Path("build-profile/profile_bt27_scored_sampling")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = candidates(args.merlin_checkout)
    result = {
        "guards": validate_guards(),
        "small": validate_small(),
        "noise_mixture": validate_mixture(),
        "unitary_exits": validate_unitary_exits(panel),
        "bt27": validate_bt(panel, args.binary),
    }
    result["source_hashes"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            Path(__file__).name,
            "shared_phase_specialization.py",
            "regional_phase_specialization.py",
            "study_regional_phase_specialization.py",
        )
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("all regional validations passed", flush=True)


if __name__ == "__main__":
    main()
