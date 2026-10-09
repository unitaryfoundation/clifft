"""Carry one Pauli T rotation through a Clifford instrument into another region.

The research host represents a trajectory as one rotation of a stabilizer
state. Only measurements commuting with that rotation and resets disjoint
from it can be sampled on the reference stabilizer state. Unsupported work
remains in the ordinary continuation, after the exact conditional state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import stim
from automatic_specialization import ANNOTATIONS, FaultModel, History
from regional_phase_specialization import RegionalPhase
from shared_phase_specialization import bits, pauli_masks

import clifft


class NeedDecision(Exception):
    """Request both branches from the exhaustive validation driver."""


def representative(
    pauli: stim.PauliString,
    simulator: stim.TableauSimulator,
    *,
    zero_x: int,
    zero_z: int = 0,
) -> stim.PauliString | None:
    """Use a commuting stabilizer to clear selected Pauli components, if possible."""
    x, z = pauli_masks(pauli)
    n = len(pauli)
    target = (x & zero_x) | ((z & zero_z) << n)
    if not target:
        return pauli.copy()
    rows = simulator.canonical_stabilizers()
    basis: dict[int, tuple[int, int]] = {}
    for i, row in enumerate(rows):
        sx, sz = pauli_masks(row)
        # Commutation is needed: an anticommuting stabilizer would introduce
        # an imaginary factor and turn a phase rotation into a different state.
        mask = (
            (sx & zero_x)
            | ((sz & zero_z) << n)
            | ((((x & sz).bit_count() + (z & sx).bit_count()) & 1) << (2 * n))
        )
        choice = 1 << i
        while mask:
            pivot = mask.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = mask, choice
                break
            other, combination = basis[pivot]
            mask ^= other
            choice ^= combination
    remaining, choice = target, 0
    while remaining:
        pivot = remaining.bit_length() - 1
        if pivot not in basis:
            return None
        other, combination = basis[pivot]
        remaining ^= other
        choice ^= combination
    result = pauli.copy()
    for i in bits(choice):
        result *= rows[i]
    rx, rz = pauli_masks(result)
    if (rx & zero_x) or (rz & zero_z) or result.sign not in (1, -1):
        raise AssertionError("Invalid equivalent Pauli representative")
    return result


def rotation_lines(pauli: stim.PauliString | None, coefficient: int) -> list[str]:
    if pauli is None:
        return []
    support = [q for q in range(len(pauli)) if pauli[q]]
    if not support:
        return []
    if pauli.sign not in (1, -1) or coefficient not in (-1, 1):
        raise ValueError("Expected a Hermitian Pauli and a signed T angle")
    basis: list[str] = []
    inverse: list[str] = []
    for q in support:
        if pauli[q] == 1:
            basis.append(f"H {q}")
            inverse.insert(0, f"H {q}")
        elif pauli[q] == 2:
            basis += [f"S_DAG {q}", f"H {q}"]
            inverse[0:0] = [f"H {q}", f"S {q}"]
    target = support[-1]
    compute = [f"CX {q} {target}" for q in support[:-1]]
    gate = "T" if coefficient * pauli.sign.real > 0 else "T_DAG"
    return basis + compute + [f"{gate} {target}"] + compute[::-1] + inverse


@dataclass
class TransportResult:
    source: str
    metadata: dict[str, Any]


def transport(
    source: str, seed: int, *, decisions: tuple[int, ...] | None = None
) -> TransportResult:
    model = FaultModel(source)
    if model.sites:
        raise ValueError("Transport requires an already materialized fault history")
    n = model.num_qubits
    simulator = stim.TableauSimulator(seed=seed)
    simulator.set_num_qubits(n)
    pauli: stim.PauliString | None = None
    coefficient = 1
    records: list[int] = []
    classical: list[str] = []
    choices: list[int] = []
    counts: dict[str, int] = {}
    seen_t = False
    erased = False
    representative_changes = []
    stop = len(model.lines)
    reason = "end"

    def measure(observable: stim.PauliString) -> int:
        expectation = simulator.peek_observable_expectation(observable)
        if decisions is None:
            value = int(simulator.measure_observable(observable))
        else:
            if expectation == 0:
                if len(choices) >= len(decisions):
                    raise NeedDecision()
                value = decisions[len(choices)]
                if value not in (0, 1):
                    raise ValueError("A forced validation decision must be a bit")
            else:
                value = int(expectation == -1)
            # Only exhaustive validation forces branches, with their complete
            # probabilities. Actual sampling uses Stim's measurement above.
            simulator.postselect_observable(observable, desired_value=bool(value))
        if expectation == 0:
            choices.append(value)
        return value

    for index, line in enumerate(model.lines):
        if not line:
            continue
        gate, *targets = line.split()
        bare = gate.split("(")[0]
        if gate in {"T", "T_DAG"}:
            if seen_t:
                stop, reason = index, "next_phase"
                break
            seen_t = True
            coefficient = 1 if gate == "T" else -1
            pauli = stim.PauliString(n)
            pauli[int(targets[0])] = "Z"
        elif bare in ANNOTATIONS:
            classical.append(line)
        elif gate == "MPAD":
            value = int(targets[0])
            records.append(value)
            classical.append(f"MPAD {value}")
        elif gate in {"M", "MX", "MY", "MPP"}:
            observable = stim.PauliString(n)
            if gate == "MPP":
                for term in targets[0].split("*"):
                    if term.startswith("!"):
                        observable.sign *= -1
                        term = term[1:]
                    observable[int(term[1:])] = term[0]
            else:
                observable[int(targets[0].lstrip("!"))] = {"M": "Z", "MX": "X", "MY": "Y"}[gate]
                if targets[0].startswith("!"):
                    observable.sign = -1
            if pauli is not None and not pauli.commutes(observable):
                stop, reason = index, "noncommuting_measurement"
                break
            value = measure(observable)
            records.append(value)
            classical.append(f"MPAD {value}")
        elif gate in {"R", "RX", "RY"}:
            q = int(targets[0])
            if pauli is not None and pauli[q]:
                cleared = representative(pauli, simulator, zero_x=1 << q, zero_z=1 << q)
                if cleared is None:
                    stop, reason = index, "overlapping_reset"
                    break
                representative_changes.append(
                    {"line": index, "before": str(pauli), "after": str(cleared)}
                )
                pauli = cleared
            observable = stim.PauliString(n)
            observable[q] = "Z"
            if measure(observable):
                simulator.x(q)
            if gate != "R":
                simulator.h(q)
            if gate == "RY":
                simulator.s(q)
        else:
            try:
                data = stim.gate_data(gate)
            except IndexError:
                stop, reason = index, "unsupported_bridge"
                break
            if not data.is_unitary:
                stop, reason = index, "unsupported_bridge"
                break
            resolved = line
            if any(t.startswith("rec[") for t in targets):
                if gate not in {"CX", "CY", "CZ"} or not targets[0].startswith("rec["):
                    stop, reason = index, "unsupported_feedback"
                    break
                value = records[int(targets[0][4:-1])]
                resolved = f"{gate[1] if value else 'I'} {targets[1]}"
            instruction = stim.Circuit(resolved)
            simulator.do(instruction)
            if pauli is not None:
                pauli = pauli.after(instruction)
        if seen_t:
            counts[bare] = counts.get(bare, 0) + 1
        if pauli is not None and simulator.peek_observable_expectation(pauli) != 0:
            pauli = None
            erased = True

    if not seen_t:
        raise ValueError("No initial T rotation to carry")
    if decisions is not None and len(choices) != len(decisions):
        raise ValueError("Unused forced validation decisions")
    carried = str(pauli) if pauli is not None else None
    if reason == "next_phase" and pauli is not None:
        diagonal = representative(pauli, simulator, zero_x=(1 << n) - 1)
        if diagonal is None:
            reason = "non_diagonal_entry"
        else:
            pauli = diagonal
    preparation = str(simulator.current_inverse_tableau().inverse().to_circuit())
    emitted = (
        f"I {n - 1}\n"
        + preparation
        + "\n"
        + "\n".join(classical + rotation_lines(pauli, coefficient) + model.lines[stop:])
        + "\n"
    )
    if FaultModel(emitted).num_records != model.num_records:
        raise AssertionError("Transport changed the complete visible record count")
    return TransportResult(
        emitted,
        {
            "stop_reason": reason,
            "stop_line": stop,
            "carried_pauli": carried,
            "entry_pauli": str(pauli) if pauli is not None else None,
            "coefficient": coefficient,
            "rotation_erased": erased,
            "representative_changes": representative_changes,
            "sampled_records": records,
            "random_decisions": choices,
            "decision_probability": 2.0 ** -len(choices),
            "transported_gate_counts": counts,
            "tableau_qubits": n,
            "carried_rotations": int(pauli is not None),
        },
    )


@dataclass
class CoreResult:
    source: str
    first_source: str
    transported_source: str
    candidate_source: str
    metadata: dict[str, Any]


class OneCorePhase:
    def __init__(self, source: str):
        self.first = RegionalPhase(source)
        if self.first.shared.metadata()["residual_t"] != 1:
            raise ValueError("The first reduction must leave exactly one T rotation")
        self.model = self.first.model

    def rewrite(
        self,
        history: History,
        seed: int,
        *,
        prefix_bits: int | None = None,
        decisions: tuple[int, ...] | None = None,
        attempt_second: bool = True,
    ) -> CoreResult:
        rng = random.Random(seed)
        if prefix_bits is None:
            sample = self.first.shared.prefix.compile_sampler(seed=rng.getrandbits(64)).sample(
                1, bit_packed=True
            )[0]
            prefix_bits = int.from_bytes(sample.tobytes(), "little")
        first = self.first.compose(history, prefix_bits)
        carried = transport(first, rng.getrandbits(64), decisions=decisions)
        metadata = carried.metadata.copy()
        metadata["second_analysis_skipped"] = not attempt_second
        if metadata["stop_reason"] != "next_phase" or not attempt_second:
            return CoreResult(carried.source, first, carried.source, carried.source, metadata)
        try:
            second = RegionalPhase(carried.source)
        except ValueError as error:
            metadata.update(stop_reason="second_analysis_rejected", rejection=str(error))
            return CoreResult(carried.source, first, carried.source, carried.source, metadata)
        metadata["second_region"] = second.metadata()
        emitted = second.compose(())

        def cost(text: str) -> tuple[int, int]:
            hir = clifft.trace(clifft.parse(text))
            return clifft.active_width_trace(hir)["peak"], hir.num_t_gates

        before, after = cost(carried.source), cost(emitted)
        accepted = after < before
        metadata.update(
            transported_cost={"width": before[0], "t_count": before[1]},
            candidate_cost={"width": after[0], "t_count": after[1]},
            accepted_second_region=accepted,
        )
        # Both choices encode the same already-sampled record/state branch.
        # Returning the original unsampled prefix here could bias that law.
        selected = emitted if accepted else carried.source
        return CoreResult(selected, first, carried.source, emitted, metadata)
