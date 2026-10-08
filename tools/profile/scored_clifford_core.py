"""Offline factorization into preparation records and a diagonal Clifford core."""

from __future__ import annotations

from itertools import combinations

import stim
from bt27_circuit_corrections import Correction, bits
from bt27_scored_clifford import ScoredClifford, product
from validate_bt27_scored_equivalence import canonical, exact_law, output_maps


def rank(vectors: list[int]) -> int:
    basis: dict[int, int] = {}
    for vector in vectors:
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = vector
                break
            vector ^= basis[pivot]
    return len(basis)


def affine_forms(law: tuple[int, ...], width: int) -> list[int]:
    """A fixed random bit per column; constrained columns use their solved forms."""
    mask = (1 << width) - 1
    forms = [1 << q for q in range(width)]
    pivots = 0
    for row in law:
        pivot = (row & mask).bit_length() - 1
        if pivot < 0:
            raise ValueError("Invalid affine constraint")
        pivots |= 1 << pivot
        forms[pivot] = row ^ (1 << pivot)
    if any(form & pivots for form in forms):
        raise ValueError("Expected reduced constraints with free columns only")
    return forms


def pack_forms(forms: list[int]) -> int:
    return sum(form << ((len(forms) + 1) * q) for q, form in enumerate(forms))


def unpack_forms(packed: int, width: int) -> list[int]:
    return [(packed >> ((width + 1) * q)) & ((1 << (width + 1)) - 1) for q in range(width)]


def verify_forms(forms: list[int], law: tuple[int, ...]) -> None:
    width = len(forms)
    mask = (1 << width) - 1
    for row in law:
        expression = (row >> width) << width
        for q in bits(row & mask):
            expression ^= forms[q]
        if expression:
            raise AssertionError("Compiled affine map violates a constraint")
    if rank([form & mask for form in forms]) != width - len(law):
        raise AssertionError("Compiled map is not uniform on the complete output space")


class BooleanSampler:
    """Finite exhaustive study: compile affine maps to Boolean coefficient expressions."""

    def __init__(self, maps: list[int], controls: int, width: int):
        if len(maps) != 1 << controls:
            raise ValueError("Expected every control assignment")
        self.width = width
        coefficients = maps.copy()
        for q in range(controls):
            for index in range(len(coefficients)):
                if index >> q & 1:
                    coefficients[index] ^= coefficients[index ^ (1 << q)]
        self.terms = [(mask, value) for mask, value in enumerate(coefficients) if value]

    def matrix(self, controls: int) -> int:
        packed = 0
        for mask, coefficient in self.terms:
            if controls & mask == mask:
                packed ^= coefficient
        return packed

    def sample(self, controls: int, random_bits: int) -> int:
        packed = self.matrix(controls)
        random_bits |= 1 << self.width
        mask = (1 << (self.width + 1)) - 1
        output = 0
        for q in range(self.width):
            output |= ((packed & mask & random_bits).bit_count() & 1) << q
            packed >>= self.width + 1
        return output


class DiagonalCore:
    def __init__(self, reducer: ScoredClifford):
        self.reducer = reducer
        self.decoder = [
            tuple(map(int, line.split()[1:])) for line in reducer.suffix if line.startswith("CX ")
        ]
        measurements = [
            (line.split()[0], int(q))
            for line in reducer.suffix + reducer.readout
            if line.startswith(("M ", "MX "))
            for q in line.split()[1:]
        ]
        self.prefix_records = reducer.certificate["live_preparation_records"]
        self.visible = self.prefix_records + len(measurements)
        self.x_qubits = [q for gate, q in measurements if gate == "MX"]
        self.width = len(self.x_qubits)
        self.positions = [
            self.prefix_records + i for i, (gate, _) in enumerate(measurements) if gate == "MX"
        ]
        self.z_positions = [
            (self.prefix_records + i, q) for i, (gate, q) in enumerate(measurements) if gate == "M"
        ]
        if len(set(q for _, q in measurements)) != len(measurements):
            raise ValueError("Expected each data qubit measured once")
        data = {q for _, q in measurements}
        if any(a not in data or b not in data for a, b in self.decoder):
            raise ValueError("Decoder couples outside the certified data product state")
        if any(a not in data or b not in data for a, b in reducer.pairs):
            raise ValueError("A conditional CZ couples outside the certified data product state")
        decoded = stim.Circuit(
            "R "
            + " ".join(map(str, range(reducer.width)))
            + "\n"
            + "\n".join(reducer.prefix + [f"CX {a} {b}" for a, b in self.decoder])
        )
        for gate, q in measurements:
            decoded.append(gate, [q])
        certificate = exact_law(decoded, list(range(self.visible)), self.visible)
        # Jointly certain outcomes for one Pauli on every data qubit certify
        # the product state independently of every earlier record. Use exact
        # flow generators; Stim's signed has_all_flows uses randomized tests.
        if any(
            1 << position not in certificate
            for position in range(self.prefix_records, self.visible)
        ):
            raise ValueError("Decoded data do not have a branch-independent positive product state")
        self.product_constraints = len(measurements)
        prefix = stim.Circuit("\n".join(reducer.prefix))
        self.prefix_law = exact_law(prefix, list(range(self.prefix_records)), self.prefix_records)
        self.prefix_forms = affine_forms(self.prefix_law, self.prefix_records)
        verify_forms(self.prefix_forms, self.prefix_law)
        self.rows = [0] * reducer.width
        for i, q in enumerate(self.x_qubits):
            self.rows[q] = 1 << i
        for a, b in reversed(self.decoder):
            self.rows[b] ^= self.rows[a]
        self.pairs = list(combinations(range(self.width), 2))
        indices = {pair: i for i, pair in enumerate(self.pairs)}
        self.s_pairs = [
            sum(1 << indices[a, b] for a, b in combinations(bits(form), 2)) for form in self.rows
        ]
        self.cz_responses = []
        for a, b in reducer.pairs:
            z = cz = 0
            for term in product([self.rows[a], self.rows[b]]):
                if term.bit_count() == 1:
                    z ^= term
                else:
                    first, second = bits(term)
                    cz ^= 1 << indices[first, second]
            self.cz_responses.append((z, cz))
        ideal = stim.Circuit(reducer.render(reducer.evaluate(())[0]))
        self.output_maps = output_maps(ideal, list(range(self.visible)))

    def restrict(self, correction: Correction) -> tuple[int, int, int]:
        s = z = cz = 0
        for q in bits(correction.s):
            z ^= s & self.rows[q]
            s ^= self.rows[q]
            cz ^= self.s_pairs[q]
        for q in bits(correction.z):
            z ^= self.rows[q]
        for index in bits(correction.cz):
            add_z, add_cz = self.cz_responses[index]
            z ^= add_z
            cz ^= add_cz
        return s, z, cz

    def circuit(self, phase: tuple[int, int, int]) -> stim.Circuit:
        s, z, cz = phase
        lines = ["RX " + " ".join(map(str, range(self.width)))]
        for q in bits(s | z):
            gate = {2: "S", 4: "Z", 6: "S_DAG"}[2 * (s >> q & 1) + 4 * (z >> q & 1)]
            lines.append(f"{gate} {q}")
        lines += [f"CZ {self.pairs[i][0]} {self.pairs[i][1]}" for i in bits(cz)]
        lines.append("MX " + " ".join(map(str, range(self.width))))
        return stim.Circuit("\n".join(lines))

    def law(self, phase: tuple[int, int, int]) -> tuple[int, ...]:
        return exact_law(self.circuit(phase), list(range(self.width)), self.width)

    def offset(self, correction: Correction) -> int:
        x = correction.x
        for a, b in self.decoder:
            x ^= ((x >> a) & 1) << b
        offset = correction.records
        for position, q in self.z_positions:
            offset ^= ((x >> q) & 1) << position
        for i, q in enumerate(self.reducer.logical):
            offset ^= ((correction.final_z >> q) & 1) << (
                self.visible - len(self.reducer.logical) + i
            )
        return offset

    def joined_law(self, law: tuple[int, ...], correction: Correction) -> tuple[int, ...]:
        columns = self.visible + len(self.output_maps)
        offset = self.offset(correction)
        rows = []
        for constraints, positions in (
            (self.prefix_law, list(range(self.prefix_records))),
            (law, self.positions),
        ):
            for row in constraints:
                mask = sum(1 << positions[q] for q in bits(row & ((1 << len(positions)) - 1)))
                sign = (row >> len(positions)) ^ ((mask & offset).bit_count() & 1)
                rows.append(mask | (sign << columns))
        rows += [
            (1 << position) | (((offset >> position) & 1) << columns)
            for position, _ in self.z_positions
        ]
        rows += [mask | (1 << (self.visible + i)) for i, mask in enumerate(self.output_maps)]
        return canonical(rows, columns)
