"""Derive a complete scored Clifford circuit from fixed BT27 fault controls.

This offline construction uses Boolean phase identities and the preparation's
computational support. It does not invoke a HIR optimizer or change execution.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Any

from analyze_bt27_phase_corrections import phase_polynomial, region
from bt27_circuit_corrections import Controls, Correction, bits
from study_circuit_noise import History, Model


def product(forms: list[int]) -> set[int]:
    """Boolean ANF product of linear forms, with x squared equal to x."""
    terms = {0}
    for form in forms:
        updated: set[int] = set()
        for term in terms:
            for q in bits(form):
                monomial = term | (1 << q)
                updated.symmetric_difference_update((monomial,))
        terms = updated
    return terms


def substitute(terms: set[int], forms: list[int]) -> set[int]:
    result: set[int] = set()
    for term in terms:
        result.symmetric_difference_update(product([forms[q] for q in bits(term)]))
    return result


def xor_forms(form: int, inputs: list[int]) -> int:
    result = 0
    for q in bits(form):
        result ^= inputs[q]
    return result


def derivative_tables(
    rows: list[int],
    triples: list[tuple[int, ...]],
    width: int,
    existing_pairs: list[tuple[int, ...]],
) -> tuple[list[tuple[int, ...]], list[int], dict[tuple[int, int], int]]:
    first: list[set[int]] = [set() for _ in rows]
    mixed: dict[tuple[int, int], int] = {}
    for triple in triples:
        for i in triple:
            first[i].symmetric_difference_update(product([rows[j] for j in triple if j != i]))
        for i, j in combinations(triple, 2):
            k = next(q for q in triple if q not in (i, j))
            mixed[i, j] = mixed.get((i, j), 0) ^ rows[k]
    pairs = set(existing_pairs)
    for terms in first:
        pairs.update(tuple(bits(term)) for term in terms if term.bit_count() == 2)
    ordered = sorted(pairs)
    indices = {pair: i for i, pair in enumerate(ordered)}
    responses = []
    for terms in first:
        packed = 0
        for term in terms:
            if term.bit_count() == 1:
                packed ^= term
            elif term.bit_count() == 2:
                packed ^= 1 << (width + indices[tuple(bits(term))])
            else:
                raise AssertionError("A cubic derivative was not quadratic")
        responses.append(packed)
    return ordered, responses, mixed


def phase_response(shift: int, first: list[int], mixed: dict[tuple[int, int], int]) -> int:
    packed = 0
    for i in bits(shift):
        packed ^= first[i]
    # Two simultaneous shift bits create Z terms. XORing single-fault CZ
    # responses alone would omit them; the cubic term is only global phase.
    for (i, j), response in mixed.items():
        if shift >> i & shift >> j & 1:
            packed ^= response
    return packed


class ScoredClifford:
    def __init__(self, model: Model, tail: str):
        self.model = model
        self.controls = Controls(model)
        self.prefix, _, self.suffix, physical = region(model.render(()))
        lines = tail.splitlines()
        readout_start = next(i for i, line in enumerate(lines) if line.startswith("MX "))
        score, coordinates = phase_polynomial(lines[:readout_start], model.num_qubits)
        if coordinates != [1 << q for q in range(model.num_qubits)]:
            raise ValueError("Scoring must end in its input coordinates")
        if any(
            mask.bit_count() != 3 or coefficient != 4
            for mask, coefficient in list(physical.items()) + list(score.items())
        ):
            raise ValueError("Expected homogeneous CCZ phases")
        self.readout = lines[readout_start:]
        self.logical = sorted({q for term in score for q in bits(term)})
        if self.logical != list(map(int, self.readout[0].split()[1:])):
            raise ValueError("Expected all scoring qubits measured in the same X readout")
        indices = {q: i for i, q in enumerate(self.logical)}
        self.triples = [tuple(indices[q] for q in bits(term)) for term in sorted(score)]
        self.width = model.num_qubits
        decoder = [1 << q for q in range(self.width)]
        measured: set[int] = set()
        for line in self.suffix:
            gate, *targets = line.split()
            if gate == "CX" and not measured:
                a, b = map(int, targets)
                decoder[b] ^= decoder[a]
            elif gate in ("M", "MX"):
                measured.update(map(int, targets))
            elif gate != "DETECTOR":
                raise ValueError("Expected a CNOT decoder followed by syndrome measurements")
        if measured.intersection(self.logical):
            raise ValueError("Scoring must commute with the syndrome measurements")
        self.rows = [decoder[q] for q in self.logical]
        self.certificate = self.preparation_identity(set(physical), set(score), decoder)

        self.pairs, self.first, self.mixed = derivative_tables(
            self.rows, self.triples, self.width, list(self.controls.pairs)
        )
        pair_indices = {pair: i for i, pair in enumerate(self.pairs)}
        self.phase_pair_masks = [1 << pair_indices[pair] for pair in self.controls.pairs]

    def preparation_identity(
        self, physical: set[int], score: set[int], decoder: list[int]
    ) -> dict[str, Any]:
        # Every RX introduces a free computational bit. Z measurements expose
        # the current form, so feedback uses that same form on every branch.
        # Vanishing on this support is sufficient even if a reset traced out
        # earlier coherence; no measurement value is fixed or postselected.
        forms = [0] * self.width
        records = []
        variables = 0
        for line in self.prefix:
            gate, *targets = line.split()
            if gate == "RX":
                forms[int(targets[0])] = 1 << variables
                variables += 1
            elif gate == "M":
                records.append(forms[int(targets[0])])
            elif gate == "CX":
                control, target = targets
                if control.startswith("rec["):
                    index = len(records) + int(control[4:-1])
                    if not 0 <= index < len(records):
                        raise ValueError("Invalid preparation feedback")
                    value = records[index]
                else:
                    value = forms[int(control)]
                forms[int(target)] ^= value
            else:
                raise ValueError("Preparation is outside the supported CSS gate subset")
        decoded = [xor_forms(row, forms) for row in decoder]
        left = substitute(physical, forms)
        right = substitute(score, decoded)
        if left != right:
            raise ValueError("The physical and target phases do not cancel on prepared support")
        unrestricted = physical ^ substitute(score, decoder)
        if not unrestricted:
            raise AssertionError("Expected a support-restricted identity in BT27")
        return {
            "free_preparation_bits": variables,
            "live_preparation_records": len(records),
            "physical_cubic_terms": len(physical),
            "logical_cubic_terms": len(score),
            "restricted_physical_terms": [hex(term) for term in sorted(left)],
            "restricted_scoring_terms": [hex(term) for term in sorted(right)],
            "restricted_difference_terms": 0,
            "unrestricted_difference_terms": len(unrestricted),
        }

    def shift(self, correction: Correction) -> int:
        return sum(
            (((row & correction.x).bit_count() & 1) ^ (correction.final_x >> q & 1)) << i
            for i, (q, row) in enumerate(zip(self.logical, self.rows))
        )

    def derivative(self, shift: int) -> int:
        return phase_response(shift, self.first, self.mixed)

    def reduce(self, correction: Correction) -> tuple[Correction, int]:
        shift = self.shift(correction)
        packed = self.derivative(shift)
        cz = packed >> self.width
        for index in bits(correction.cz):
            cz ^= self.phase_pair_masks[index]
        return replace(
            correction, z=correction.z ^ (packed & ((1 << self.width) - 1)), cz=cz
        ), shift

    def evaluate(self, history: History) -> tuple[Correction, int]:
        return self.reduce(self.controls.evaluate(history))

    def gates(self, correction: Correction) -> list[str]:
        gates = []
        for q in bits(correction.s | correction.z):
            coefficient = 2 * (correction.s >> q & 1) + 4 * (correction.z >> q & 1)
            gate = {2: "S", 4: "Z", 6: "S_DAG"}[coefficient]
            gates.append(f"{gate} {q}")
        gates += [f"CZ {self.pairs[i][0]} {self.pairs[i][1]}" for i in bits(correction.cz)]
        gates += [f"X {q}" for q in bits(correction.x)]
        return gates

    def render(self, correction: Correction) -> str:
        final = [f"Z {q}" for q in bits(correction.final_z)]
        final += [f"X {q}" for q in bits(correction.final_x)]
        return (
            "\n".join(self.prefix + self.gates(correction) + self.suffix + final + self.readout)
            + "\n"
        )
