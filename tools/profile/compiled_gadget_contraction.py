"""Fixed-schedule CSS contractions for the physical terminal gadget.

Planning and boundary certification use Stim/Cirq offline. Evaluation uses
only precomputed gathers and arithmetic. This is a terminal probability
experiment, not a production sampler or a fault-to-boundary compiler.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import stim
from clifford_gadget import Atom, Gadget


class ParityContraction:
    """Compile sum_x product_q local[q, parity(mask[q] & x)]."""

    def __init__(self, masks: list[int], rank: int, entry_limit: int = 2_000_000):
        if rank < 0 or any(mask < 0 or mask >> rank for mask in masks):
            raise ValueError("invalid variable mask or rank")
        scopes = [tuple(i for i in range(rank) if mask >> i & 1) for mask in masks]
        self.rank = rank
        self.leaves = []
        self.steps = []
        self.storage = 0
        active = []
        for scope in scopes:
            size = 1 << len(scope)
            if self.storage + size > entry_limit:
                raise ValueError("contraction storage limit exceeded")
            self.leaves.append((self.storage, np.array([i.bit_count() % 2 for i in range(size)])))
            active.append((scope, self.storage))
            self.storage += size
        self.peak_scope = 0
        self.work = 0
        self.gather_entries = 0
        self.order = []
        for _ in range(rank):
            variables = set().union(*(set(s) for s, _ in active))
            if not variables:
                raise ValueError("every sum variable must occur in a factor")
            unions = {v: set().union(*(set(s) for s, _ in active if v in s)) for v in variables}
            pivot = min(variables, key=lambda v: (len(unions[v]), v))
            union = tuple(sorted(unions[pivot] - {pivot})) + (pivot,)
            selected = [(s, offset) for s, offset in active if pivot in s]
            size = 1 << len(union)
            if (
                self.storage + size // 2 > entry_limit
                or self.gather_entries + size * len(selected) > entry_limit
            ):
                raise ValueError("contraction storage limit exceeded")
            gathers = []
            for scope, offset in selected:
                positions = [union.index(v) for v in scope]
                gathers.append(
                    np.array(
                        [
                            offset + sum(((i >> p) & 1) << j for j, p in enumerate(positions))
                            for i in range(size)
                        ],
                        dtype=np.int64,
                    )
                )
            self.steps.append((self.storage, size // 2, gathers))
            active = [(s, offset) for s, offset in active if pivot not in s]
            active.append((union[:-1], self.storage))
            self.storage += size // 2
            self.order.append(pivot)
            self.peak_scope = max(self.peak_scope, len(union))
            self.work += size
            self.gather_entries += size * len(gathers)
        self.outputs = [offset for scope, offset in active]

    def evaluate(self, local):
        scratch = np.empty(self.storage, dtype=np.complex128)
        for (offset, parity), values in zip(self.leaves, local, strict=True):
            scratch[offset : offset + len(parity)] = values[parity]
        for offset, size, gathers in self.steps:
            product = np.ones(2 * size, dtype=np.complex128)
            for gather in gathers:
                product *= scratch[gather]
            scratch[offset : offset + size] = product[:size] + product[size:]
        return np.prod(scratch[self.outputs]) / (1 << self.rank)


def mask_of(p, axis):
    return sum((value == axis) << q for q, value in enumerate(p))


def binary_duals(rows, width):
    """Plan a fixed right inverse of independent binary check rows."""
    if any(row < 0 or row >> width for row in rows):
        raise ValueError("check exceeds physical width")
    work = [(row, 1 << i) for i, row in enumerate(rows)]
    pivots = []
    for i in range(len(rows)):
        candidate = next((j for j in range(i, len(rows)) if work[j][0]), None)
        if candidate is None:
            raise ValueError("dependent code checks")
        work[i], work[candidate] = work[candidate], work[i]
        row, labels = work[i]
        pivot = (row & -row).bit_length() - 1
        pivots.append(pivot)
        for j, (other, label) in enumerate(work):
            if j != i and other >> pivot & 1:
                work[j] = (other ^ row, label ^ labels)
    return [
        sum(((labels >> i) & 1) << p for p, (_, labels) in zip(pivots, work, strict=True))
        for i in range(len(rows))
    ]


@dataclass
class Boundary:
    offset: int
    x_signs: int
    logical: np.ndarray
    ancillas: dict[int, tuple[str, int]]


PAULIS = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def single_state(axis, bit):
    return (
        np.array([1, (-1) ** bit], dtype=complex) / math.sqrt(2)
        if axis == "X"
        else np.array([1 - bit, bit], dtype=complex)
    )


class TerminalCode:
    def __init__(self, gadget: Gadget, suffix: list[Atom | Gadget]) -> None:
        self.width = gadget.width
        self.data = tuple(sorted(gadget.signs))
        self.data_mask = sum(1 << q for q in self.data)
        measurements = [a for a in suffix if isinstance(a, Atom) and a.name == "MPP"]
        self.logical_measurement = measurements[0]
        self.checks = measurements[1:]
        if mask_of(self.logical_measurement.pauli, 2) != self.data_mask:
            raise ValueError("terminal logical measurement must be all Y on data")
        self.x_checks = [a for a in self.checks if mask_of(a.pauli, 1)]
        self.z_checks = [a for a in self.checks if mask_of(a.pauli, 3)]
        self.x_rows = [mask_of(a.pauli, 1) for a in self.x_checks]
        self.z_rows = [mask_of(a.pauli, 3) for a in self.z_checks]
        self.rank = len(self.x_rows)
        if len(self.data) != 2 * self.rank + 1 or len(self.z_rows) != self.rank:
            raise ValueError("requires a balanced one-logical-qubit CSS code")
        for a in self.checks:
            if a.pauli is None:
                raise ValueError("terminal check has no Pauli")
            if (
                a.pauli.sign != 1
                or mask_of(a.pauli, 2)
                or bool(mask_of(a.pauli, 1)) == bool(mask_of(a.pauli, 3))
            ):
                raise ValueError("requires positive pure CSS checks")
        if any((x & z).bit_count() % 2 for x in self.x_rows for z in self.z_rows):
            raise ValueError("CSS checks do not commute")
        if any(row & ~self.data_mask or row.bit_count() % 2 for row in self.x_rows + self.z_rows):
            raise ValueError("all-data X and Z must be logical operators")
        binary_duals(self.x_rows, self.width)
        self.z_duals = binary_duals(self.z_rows + [self.data_mask], self.width)[:-1]
        # Separate unary characters bind arbitrary incoming/outgoing X syndromes.
        masks = [sum(((row >> q) & 1) << i for i, row in enumerate(self.x_rows)) for q in self.data]
        self.masks = masks + [1 << i for i in range(self.rank)]
        self.plan = ParityContraction(self.masks, self.rank)
        self.signs = dict(gadget.signs)
        self.parity = gadget.parity.copy()
        if any(self.parity[q] != 1 for q in self.data):
            raise ValueError("gadget parity must restrict to all-data X")

    def certify_boundary(self, state):
        """Reference-only adapter; no claim of a compiled prefix handoff."""
        signs = []
        for check in self.x_checks + self.z_checks:
            values = {t.tableau.peek_observable_expectation(check.pauli) for t in state.terms}
            if len(values) != 1 or 0 in values:
                raise ValueError("entry is not in one signed CSS code space")
            signs.append(int(values == {-1}))
        offset = 0
        for bit, dual in zip(signs[self.rank :], self.z_duals, strict=True):
            if bit:
                offset ^= dual
        ancillas = {}
        anchor = offset
        x_count = 0
        for q in range(self.width):
            if q in self.signs:
                continue
            for axis in "XZ":
                p = stim.PauliString(self.width)
                p[q] = axis
                values = {t.tableau.peek_observable_expectation(p) for t in state.terms}
                if len(values) == 1 and 0 not in values:
                    bit = int(values == {-1})
                    ancillas[q] = (axis, bit)
                    anchor |= (bit if axis == "Z" else 0) << q
                    x_count += axis == "X"
                    break
            else:
                raise ValueError("ancilla is not a common product X or Z state")
            vector = single_state(*ancillas[q])
            if abs(np.vdot(vector, PAULIS[self.parity[q]] @ vector)) < 1 - 1e-9:
                raise ValueError("gadget branches act differently on a spectator")
        logical = []
        for logical_bit in (0, 1):
            bits = anchor ^ (self.data_mask if logical_bit else 0)
            big_endian = sum(((bits >> q) & 1) << (self.width - 1 - q) for q in range(self.width))
            logical.append(
                sum(
                    t.coefficient * t.ch.inner_product_of_state_and_x(big_endian)
                    for t in state.terms
                )
                * 2 ** ((self.rank + x_count) / 2)
            )
        if abs(sum(abs(a) ** 2 for a in logical) - 1) > 1e-8:
            raise ValueError("boundary amplitudes do not span the complete input state")
        return Boundary(
            offset,
            sum(bit << i for i, bit in enumerate(signs[: self.rank])),
            np.array(logical),
            ancillas,
        )

    def bind(
        self, boundary: Boundary, suffix: list[Atom | Gadget], records: list[int]
    ) -> list[tuple[complex, np.ndarray]]:
        """Offline/reference binding; returns only scalar and local-table inputs.

        Fault propagation in Gadget and boundary adaptation remain outside the
        timed contraction. Record-dependent work here is not a compiler bridge.
        """
        gadget = suffix[0]
        if not isinstance(gadget, Gadget):
            raise ValueError("terminal block must start with a gadget")
        if gadget.signs != self.signs or gadget.parity != self.parity:
            raise ValueError("gadget topology differs from the compiled code")
        bit = records[gadget.measurement.record] ^ gadget.measurement.flip
        operators, reset = gadget.terms(bit)
        if reset != records[gadget.reset.record]:
            return []
        q_base = operators[0][1]
        vectors = {
            q: PAULIS[q_base[q]] @ single_state(*spec) for q, spec in boundary.ancillas.items()
        }
        ratios = []
        for _, p in operators:
            ratios.append(
                np.prod(
                    [
                        np.vdot(vectors[q], PAULIS[p[q]] @ single_state(*spec))
                        for q, spec in boundary.ancillas.items()
                    ]
                )
            )
        post = stim.PauliString(self.width)
        ancilla_probability = 1.0
        seen_data_measurement = False
        for atom in suffix[1:]:
            if not isinstance(atom, Atom):
                raise ValueError("another gadget follows the terminal block")
            if atom.name == "MPP":
                seen_data_measurement = True
            elif atom.name in {"X", "Y", "Z"}:
                q = atom.targets[0]
                if q in vectors:
                    vectors[q] = PAULIS["IXYZ".index(atom.name)] @ vectors[q]
                else:
                    if seen_data_measurement:
                        raise ValueError("data fault after terminal measurement")
                    p = stim.PauliString(self.width)
                    p[q] = atom.name
                    post = p * post
            elif atom.name in {"MX", "M"} and atom.targets[0] in vectors:
                q = atom.targets[0]
                desired = records[atom.record] ^ atom.flip
                output = single_state("X" if atom.name == "MX" else "Z", desired)
                probability = abs(np.vdot(output, vectors[q])) ** 2
                ancilla_probability *= probability
                vectors[q] = output
            else:
                raise ValueError(f"unsupported terminal operation {atom.name}")
        if ancilla_probability < 1e-20:
            return []
        q_x = mask_of(q_base, 1) | mask_of(q_base, 2)
        post_x = mask_of(post, 1) | mask_of(post, 2)
        output_offset = boundary.offset ^ ((q_x ^ post_x) & self.data_mask)
        for check, row in zip(self.z_checks, self.z_rows, strict=True):
            if (output_offset & row).bit_count() % 2 != (records[check.record] ^ check.flip):
                return []
        output_x = sum((records[a.record] ^ a.flip) << i for i, a in enumerate(self.x_checks))
        y_bit = records[self.logical_measurement.record] ^ self.logical_measurement.flip
        # Y^n = i^n X^n Z^n. The output code uses output_offset as its zero anchor.
        y_phase = (1j ** len(self.data)) * (-1) ** (output_offset.bit_count() % 2)
        inputs = []
        for branch, ((coefficient, p), ratio) in enumerate(zip(operators, ratios, strict=True)):
            for logical in (0, 1):
                scalar = (
                    boundary.logical[logical]
                    * coefficient
                    * ratio
                    * p.sign
                    * post.sign
                    * math.sqrt(ancilla_probability / 2)
                )
                if logical ^ branch:
                    scalar *= (-1) ** y_bit * y_phase.conjugate()
                local = np.ones((len(self.data) + self.rank, 2), dtype=np.complex128)
                for i, q in enumerate(self.data):
                    u = np.diag([1, np.exp(1j * self.signs[q] * np.pi / 4)])
                    op = PAULIS[post[q]] @ u.conj().T @ PAULIS[p[q]] @ u
                    for value in (0, 1):
                        input_bit = value ^ ((boundary.offset >> q) & 1) ^ logical
                        output_bit = input_bit ^ int(p[q] in (1, 2)) ^ int(post[q] in (1, 2))
                        local[i, value] = op[output_bit, input_bit]
                for i in range(self.rank):
                    local[len(self.data) + i, 1] = (-1) ** (
                        ((boundary.x_signs ^ output_x) >> i) & 1
                    )
                inputs.append((scalar, local))
        return inputs

    def probability(self, inputs):
        return abs(sum(scalar * self.plan.evaluate(local) for scalar, local in inputs)) ** 2

    def sampling_inputs(self, boundary, suffix, num_records):
        """Prepare both gadget outcomes offline, including their record maps.

        The selected fixture's spectator measurements are deterministic given
        the gadget bit and faults. Reject a boundary for which that is false.
        """
        choices = []
        gadget = suffix[0]
        for bit in (0, 1):
            records = [0] * num_records
            records[gadget.measurement.record] = bit ^ gadget.measurement.flip
            operators, reset = gadget.terms(bit)
            records[gadget.reset.record] = reset
            q_base = operators[0][1]
            vectors = {
                q: PAULIS[q_base[q]] @ single_state(*spec) for q, spec in boundary.ancillas.items()
            }
            data_flip = (mask_of(q_base, 1) | mask_of(q_base, 2)) & self.data_mask
            for atom in suffix[1:]:
                if atom.name in {"X", "Y", "Z"}:
                    q = atom.targets[0]
                    if q in vectors:
                        vectors[q] = PAULIS["IXYZ".index(atom.name)] @ vectors[q]
                    elif atom.name in {"X", "Y"}:
                        data_flip ^= 1 << q
                elif atom.name in {"MX", "M"}:
                    q = atom.targets[0]
                    axis = "X" if atom.name == "MX" else "Z"
                    p0 = abs(np.vdot(single_state(axis, 0), vectors[q])) ** 2
                    if min(abs(p0), abs(1 - p0)) > 1e-9:
                        raise ValueError(
                            "terminal sampler requires deterministic spectator measurements"
                        )
                    desired = int(p0 < 0.5)
                    records[atom.record] = desired ^ atom.flip
                    vectors[q] = single_state(axis, desired)
            for check, row in zip(self.z_checks, self.z_rows, strict=True):
                records[check.record] = (
                    (boundary.offset ^ data_flip) & row
                ).bit_count() % 2 ^ check.flip
            for check in self.x_checks + [self.logical_measurement]:
                records[check.record] = check.flip
            choices.append((records, self.bind(boundary, suffix, records)))
        return choices


class TerminalMarginals:
    """Precompile paired sums for every prefix of the X-check outcomes."""

    def __init__(self, code):
        self.code = code
        self.plans = []
        for measured in range(code.rank + 1):
            # Summing an unobserved output character identifies the two copies
            # of its input variable. This depends only on measurement order.
            right = [
                sum(
                    ((mask >> i) & 1) << (code.rank + i if i < measured else i)
                    for i in range(code.rank)
                )
                for mask in code.masks
            ]
            self.plans.append(ParityContraction(code.masks + right, code.rank + measured))

    def bind_query(self, inputs, measured, x_bits, y_bit):
        terms = []
        for index, (scalar, source) in enumerate(inputs):
            local = source.copy()
            for i in range(measured):
                local[len(self.code.data) + i, 1] *= (-1) ** ((x_bits >> i) & 1)
            if index in (1, 2) and y_bit:
                scalar = -scalar
            terms.append((scalar, local))
        pairs = []
        for i, (a, left) in enumerate(terms):
            for j, (b, right) in enumerate(terms[: i + 1]):
                pairs.append(
                    (
                        (1 if i == j else 2) * a * b.conjugate(),
                        np.concatenate([left, right.conjugate()]),
                    )
                )
        return pairs

    def probability(self, inputs, measured, x_bits, y_bit):
        total = sum(
            (scalar * self.plans[measured].evaluate(local)).real
            for scalar, local in self.bind_query(inputs, measured, x_bits, y_bit)
        )
        if total < -1e-9:
            raise AssertionError("invalid marginal probability")
        return max(0.0, float(total))

    def sample(self, choices, seed):
        rng = random.Random(seed)
        y0 = self.probability(choices[0][1], 0, 0, 0)
        m0 = y0 + self.probability(choices[0][1], 0, 0, 1)
        m1 = self.probability(choices[1][1], 0, 0, 0) + self.probability(choices[1][1], 0, 0, 1)
        if abs(m0 + m1 - 1) > 1e-8:
            raise AssertionError("terminal outcomes do not normalize")

        def draw(zero, total):
            if not -1e-9 <= zero <= total + 1e-9:
                raise AssertionError("conditional branch exceeds parent probability")
            return int(rng.random() * total >= min(total, max(0.0, zero)))

        m = draw(m0, 1)
        records, inputs = choices[m]
        records = records.copy()
        parent = m1 if m else m0
        if m:
            y0 = self.probability(inputs, 0, 0, 0)
        y = draw(y0, parent)
        parent = parent - y0 if y else y0
        records[self.code.logical_measurement.record] ^= y
        x_bits = 0
        for measured in range(1, self.code.rank + 1):
            zero = self.probability(inputs, measured, x_bits, y)
            bit = draw(zero, parent)
            parent = parent - zero if bit else zero
            x_bits |= bit << (measured - 1)
            records[self.code.x_checks[measured - 1].record] ^= bit
        return dict(records=records, log_probability=math.log(parent))


def write_native_plan(path, plan, cases, marginal=False):
    tokens = [str(int(marginal)), str(plan.rank), str(plan.storage), str(len(plan.leaves))]
    for offset, parity in plan.leaves:
        tokens.extend(map(str, [offset, len(parity), *parity]))
    tokens.append(str(len(plan.steps)))
    for offset, size, gathers in plan.steps:
        tokens.extend(map(str, [offset, size, len(gathers)]))
        for gather in gathers:
            tokens.extend(map(str, gather))
    tokens.extend(map(str, [len(plan.outputs), *plan.outputs, len(cases)]))
    for case in cases:
        tokens.append(str(len(case)))
        for scalar, local in case:
            for value in [scalar, *local.flatten()]:
                tokens.extend([repr(float(value.real)), repr(float(value.imag))])
    path.write_text("\n".join(tokens) + "\n")
