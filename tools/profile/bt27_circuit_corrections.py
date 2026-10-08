"""Offline fixed-fault controls for BT27's preparation, phase region, and decoder.

This diagnostic precomputes binary dependencies in the physical circuit. It
does not add runtime topology discovery or a conditional executor backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from analyze_bt27_phase_corrections import region
from study_circuit_noise import History, Model


def bits(mask: int) -> list[int]:
    result = []
    while mask:
        low = mask & -mask
        result.append(low.bit_length() - 1)
        mask ^= low
    return result


@dataclass(frozen=True)
class Correction:
    x: int
    s: int
    z: int
    cz: int
    records: int
    final_x: int
    final_z: int


class Controls:
    def __init__(self, model: Model, phase_bounds: tuple[int, int] | None = None):
        self.model = model
        self.width = width = model.num_qubits
        ideal = model.render(())
        if phase_bounds is None:
            prefix, body, _, _ = region(ideal)
            phase_bounds = len(prefix), len(prefix) + len(body)
        first, end = phase_bounds
        self.phase_bounds = phase_bounds
        self.outcomes: list[tuple[int, ...]] = []
        generators: list[tuple[tuple[int, int, int], ...]] = []
        count = 0
        for site in model.sites:
            entries = []
            if site.gate == "READOUT_NOISE":
                self.outcomes.append((1 << count,))
                entries.append((-1, 0, 1 << count))
                count += 1
            elif site.gate.startswith("DEPOLARIZE"):
                for target_text in site.targets:
                    entries.append((int(target_text), 1 << count, 1 << (count + 1)))
                    count += 2
                self.outcomes.append(
                    tuple(self.pauli_mask(outcome, entries) for outcome in site.outcomes)
                )
            else:
                bit = 1 << count
                axis = site.gate[0]
                entries.append(
                    (int(site.targets[0]), bit if axis in "XY" else 0, bit if axis in "YZ" else 0)
                )
                self.outcomes.append((bit,))
                count += 1
            generators.append(tuple(entries))
        self.generator_count = count

        x, z = [0] * width, [0] * width
        coordinates = [1 << q for q in range(width)]
        records: list[int] = []
        rotations: list[tuple[int, int, int]] = []
        phase_x: list[int] | None = None
        phase_z: list[int] | None = None
        ideal_index = 0
        for entry in model.entries:
            # Noise belongs after the preceding ideal instruction, including
            # the final CNOT that closes the phase region's coordinate map.
            position = ideal_index - 1 if entry.sites else ideal_index
            in_phase = first <= position < end
            if not entry.sites and ideal_index == end:
                phase_x, phase_z = x.copy(), z.copy()
                x, z = [0] * width, [0] * width
            if entry.sites:
                for site_index in entry.sites:
                    for q, x_bit, z_bit in generators[site_index]:
                        x[q] ^= x_bit
                        if in_phase:
                            for variable in bits(coordinates[q]):
                                z[variable] ^= z_bit
                        else:
                            z[q] ^= z_bit
                continue
            gate, targets = entry.gate, entry.targets
            if gate == "CX":
                if len(targets) != 2:
                    raise ValueError("Expected one CNOT per parsed instruction")
                target = int(targets[1])
                if targets[0].startswith("rec["):
                    if in_phase:
                        raise ValueError("Feedback inside the phase region is unsupported")
                    record = len(records) + int(targets[0][4:-1])
                    if not 0 <= record < len(records):
                        raise ValueError("Invalid feedback reference")
                    x[target] ^= records[record]
                else:
                    control = int(targets[0])
                    x[target] ^= x[control]
                    if in_phase:
                        coordinates[target] ^= coordinates[control]
                    else:
                        z[control] ^= z[target]
            elif gate in ("T", "T_DAG"):
                if not in_phase or len(targets) != 1:
                    raise ValueError("T gates must belong to the selected phase region")
                q = int(targets[0])
                rotations.append((coordinates[q], 1 if gate == "T" else -1, x[q]))
            elif gate in ("M", "MX", "RX"):
                if in_phase:
                    raise ValueError("Nonunitary instruction inside phase region")
                for target_text in targets:
                    q = int(target_text.removeprefix("!"))
                    if gate == "RX":
                        x[q] = z[q] = 0
                    else:
                        flip = x[q] if gate == "M" else z[q]
                        if entry.readout_site is not None:
                            if len(targets) != 1:
                                raise ValueError("Expected single-target noisy measurements")
                            flip ^= self.outcomes[entry.readout_site][0]
                        records.append(flip)
            elif gate not in ("DETECTOR", "I"):
                raise ValueError(f"Unsupported instruction in this physical-control study: {gate}")
            ideal_index += 1
        if ideal_index == end:
            phase_x, phase_z = x, z
            x, z = [0] * width, [0] * width
        if phase_x is None or phase_z is None or coordinates != [1 << q for q in range(width)]:
            raise ValueError("Expected a completed identity-coordinate phase region")
        if len(records) != model.num_measurements:
            raise AssertionError("Record dependencies are incomplete")

        self.pairs = sorted(
            {pair for mask, _, _ in rotations for pair in combinations(bits(mask), 2)}
        )
        pair_indices = {pair: i for i, pair in enumerate(self.pairs)}
        self.rotation_actions = [
            (
                mask,
                mask if sign == 1 else 0,
                sum(1 << pair_indices[pair] for pair in combinations(bits(mask), 2)),
            )
            for mask, sign, _ in rotations
        ]
        self.rotation_count = len(rotations)
        self.fields = (
            width,
            width,
            len(rotations),
            len(records),
            width,
            width,
        )
        dependencies = phase_x + phase_z + [dep for _, _, dep in rotations] + records + x + z
        # Transpose once so a sparse sampled history needs only a few integer
        # XORs, rather than a new traversal of the circuit's topology.
        responses = [0] * count
        for output, dependency in enumerate(dependencies):
            for generator in bits(dependency):
                responses[generator] ^= 1 << output
        self.responses = responses
        self.dependency_nonzeros = sum(dep.bit_count() for dep in dependencies)

    @staticmethod
    def pauli_mask(outcome: str, entries: list[tuple[int, int, int]]) -> int:
        mask = 0
        for axis, (_, x, z) in zip(outcome, entries):
            if axis in "XY":
                mask ^= x
            if axis in "YZ":
                mask ^= z
        return mask

    def evaluate(self, history: History) -> Correction:
        packed = 0
        for site, outcome in history:
            for generator in bits(self.outcomes[site][outcome]):
                packed ^= self.responses[generator]
        fields = []
        for width in self.fields:
            fields.append(packed & ((1 << width) - 1))
            packed >>= width
        x, z, rotations, records, final_x, final_z = fields
        s = cz = 0
        # A flipped T contributes -2 * sign * parity(input coordinates).
        # Modulo eight this contains only S/Z and CZ terms. Carries when
        # adding two S controls are essential for multiple simultaneous faults.
        for index in bits(rotations):
            add_s, add_z, add_cz = self.rotation_actions[index]
            z ^= add_z ^ (s & add_s)
            s ^= add_s
            cz ^= add_cz
        return Correction(x, s, z, cz, records, final_x, final_z)

    def gates(self, correction: Correction) -> list[str]:
        gates = []
        for q in bits(correction.s | correction.z):
            coefficient = 2 * (correction.s >> q & 1) + 4 * (correction.z >> q & 1)
            gate = {2: "S", 4: "Z", 6: "S_DAG"}[coefficient]
            gates.append(f"{gate} {q}")
        gates += [f"CZ {self.pairs[i][0]} {self.pairs[i][1]}" for i in bits(correction.cz)]
        gates += [f"X {q}" for q in bits(correction.x)]
        return gates

    def metadata(self) -> dict[str, Any]:
        return {
            "primitive_fault_controls": self.generator_count,
            "linear_output_controls": sum(self.fields),
            "linear_dependency_nonzeros": self.dependency_nonzeros,
            "response_payload_bytes": sum(
                (value.bit_length() + 7) // 8 for value in self.responses
            ),
            "t_sign_controls": self.rotation_count,
            "candidate_cz_pairs": len(self.pairs),
            "response_payload_excludes": (
                "Python object/container overhead and other control tables"
            ),
        }
