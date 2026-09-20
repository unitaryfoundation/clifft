"""Offline coherent-Clifford reference for T-conjugated parity gadgets.

This research tool uses dynamic Stim/Cirq tableaux outside Clifft execution.
It does not add a production backend or change the ordinary compiler.
Requires cirq-core==1.6.1 in an isolated research environment.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass

import cirq  # type: ignore[import-not-found]
import numpy as np
import stim
from soft_cultivation_study import clifford_proxy


@dataclass
class Atom:
    name: str
    targets: tuple[int, ...] = ()
    record: int = -1
    flip: int = 0
    pauli: stim.PauliString | None = None


def bind_faults(source, seed, probability=None):
    """Draw original physical sites and emit a deterministic-fault circuit."""
    if probability is not None and not 0 <= probability <= 1:
        raise ValueError("fault probability must be between zero and one")
    rng = random.Random(seed)
    require_t_dialect(source)
    proxy = stim.Circuit(clifford_proxy(source))
    out = stim.Circuit()
    selected = []
    site = 0
    for op in proxy.flattened():
        name, targets, args = op.name, op.targets_copy(), op.gate_args_copy()
        if name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
            size = 2 if name == "DEPOLARIZE2" else 1
            p = args[0] if probability is None else probability
            for start in range(0, len(targets), size):
                if rng.random() < p:
                    choices = (
                        ["".join(x) for x in itertools.product("IXYZ", repeat=size)][1:]
                        if name.startswith("DEPOLARIZE")
                        else [name[0]]
                    )
                    choice = rng.choice(choices)
                    for target, axis in zip(targets[start : start + size], choice, strict=True):
                        if axis != "I":
                            out.append(axis, [target])
                    selected.append(
                        dict(
                            site=site,
                            kind=name,
                            targets=[t.value for t in targets[start : start + size]],
                            pauli=choice,
                        )
                    )
                site += 1
        elif name in {"M", "MX", "MPP"} and args:
            groups = op.target_groups() if name == "MPP" else [[t] for t in targets]
            for group in groups:
                flip = int(rng.random() < (args[0] if probability is None else probability))
                if name == "MPP":
                    combined: list[stim.GateTarget] = []
                    for target in group:
                        if combined:
                            combined.append(stim.target_combiner())
                        combined.append(target)
                    group = combined
                out.append(name, group, flip)
                if flip:
                    selected.append(dict(site=site, kind="readout", pauli="flip"))
                site += 1
        else:
            out.append(op)
    # This source contains T/T_DAG but no genuine S/S_DAG instructions.
    text = str(out).replace("S_DAG ", "T_DAG ")
    text = "\n".join(
        "T " + line[2:] if line.startswith("S ") else line for line in text.splitlines()
    )
    return text, selected


def atoms_from_text(text):
    require_t_dialect(text)
    circuit = stim.Circuit(clifford_proxy(text)).flattened()
    width = circuit.num_qubits
    visible, hidden = 0, circuit.num_measurements
    atoms = []
    for op in circuit:
        name, targets, args = op.name, op.targets_copy(), op.gate_args_copy()
        if name in {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        if name in {"M", "MX", "MPP", "R", "RX"}:
            reset = name in {"R", "RX"}
            if args and args[0] not in (0, 1):
                raise ValueError("materialize readout faults before evaluation")
            groups = op.target_groups() if name == "MPP" else [[t] for t in targets]
            for group in groups:
                p = stim.PauliString(width)
                for target in group:
                    if target.is_inverted_result_target:
                        raise ValueError("inverted targets are unsupported")
                    p[target.value] = (
                        "XYZ"[int(target.is_y_target) + 2 * int(target.is_z_target)]
                        if name == "MPP"
                        else "X"
                        if name.endswith("X")
                        else "Z"
                    )
                atoms.append(
                    Atom(
                        name,
                        tuple(t.value for t in group),
                        hidden if reset else visible,
                        int(args[0]) if args else 0,
                        p,
                    )
                )
                if reset:
                    hidden += 1
                else:
                    visible += 1
        elif name in {"S", "S_DAG", "X", "Y", "Z", "H", "CX", "CZ"}:
            size = 2 if name in {"CX", "CZ"} else 1
            if any(t.is_measurement_record_target for t in targets):
                raise ValueError("feedback is outside this reference")
            for start in range(0, len(targets), size):
                atoms.append(
                    Atom(
                        {"S": "T", "S_DAG": "T_DAG"}.get(name, name),
                        tuple(t.value for t in targets[start : start + size]),
                    )
                )
        else:
            raise ValueError(f"unsupported bound instruction {name}")
    return width, visible, hidden - visible, atoms


def native_replay_text(text):
    """Express fixed readout flips as inverted observables for native replay.

    Ordinary Clifft replay declines readout-channel actions. Inverting the
    measured Pauli changes only its bit convention, retaining the physical
    projection and the records consumed by subsequent instructions.
    """
    require_t_dialect(text)
    out = stim.Circuit()
    for op in stim.Circuit(clifford_proxy(text)).flattened():
        if op.name in {"M", "MX", "MPP"} and op.gate_args_copy():
            flip = op.gate_args_copy()[0]
            if flip not in (0, 1):
                raise ValueError("replay conversion requires fixed readout faults")
            groups = op.target_groups() if op.name == "MPP" else [[t] for t in op.targets_copy()]
            for group in groups:
                targets = []
                for i, target in enumerate(group):
                    if i:
                        targets.append(stim.target_combiner())
                    if flip and i == 0:
                        if target.is_x_target:
                            target = stim.target_x(target.value, invert=True)
                        elif target.is_y_target:
                            target = stim.target_y(target.value, invert=True)
                        elif target.is_z_target:
                            target = stim.target_z(target.value, invert=True)
                        else:
                            target = stim.target_inv(target.value)
                    targets.append(target)
                out.append(op.name, targets)
        else:
            out.append(op)
    return "\n".join(
        "T" + line[1:] if line.startswith(("S ", "S_DAG ")) else line
        for line in str(out).splitlines()
    )


class Gadget:
    def __init__(self, width, forward, middle, reverse):
        self.width = width
        self.signs = {a.targets[0]: 1 if a.name == "T" else -1 for a in forward}
        opposite = {a.targets[0]: -1 if a.name == "T" else 1 for a in reverse}
        if (
            len(self.signs) != len(forward)
            or len(opposite) != len(reverse)
            or self.signs != opposite
        ):
            raise ValueError("T layers are not distinct inverses")
        self.before: list[Atom] = []
        self.between: list[Atom] = []
        self.after: list[Atom] = []
        segment = self.before
        measurement: Atom | None = None
        reset: Atom | None = None
        for atom in middle:
            if atom.name == "MX" and measurement is None:
                measurement = atom
                segment = self.between
            elif atom.name == "RX" and reset is None and measurement is not None:
                reset = atom
                segment = self.after
            elif atom.name in {"X", "Y", "Z", "CX"}:
                if segment is self.between and atom.name == "CX":
                    raise ValueError("entangling operation between measurement and reset")
                segment.append(atom)
            else:
                raise ValueError("unsupported parity gadget middle")
        if measurement is None or reset is None or measurement.targets != reset.targets:
            raise ValueError("expected one measured and reset X wire")
        self.measurement = measurement
        self.reset = reset
        self.qubit = self.measurement.targets[0]
        a, b = stim.Circuit(), stim.Circuit()
        a.append("I", [width - 1])
        b.append("I", [width - 1])
        for segment, circuit in [(self.before, a), (self.after, b)]:
            for atom in segment:
                if atom.name == "CX":
                    circuit.append("CX", atom.targets)
        self.inverse = a.to_tableau().inverse()
        if self.inverse != b.to_tableau():
            raise ValueError("Clifford networks are not inverses")
        x = stim.PauliString(width)
        x[self.qubit] = "X"
        self.parity = self.inverse(x)
        self.pre, self.mid, self.post = [
            self.frame(s) for s in (self.before, self.between, self.after)
        ]

    def frame(self, atoms):
        frame = stim.PauliString(self.width)
        for atom in atoms:
            if atom.name == "CX":
                frame = frame.after(stim.CircuitInstruction("CX", atom.targets))
            else:
                p = stim.PauliString(self.width)
                p[atom.targets[0]] = atom.name
                frame = p * frame
        return frame

    def terms(self, true_outcome):
        reset_outcome = true_outcome ^ int(self.mid[self.qubit] in (2, 3))
        correction = stim.PauliString(self.width)
        correction[self.qubit] = "Z" if reset_outcome else "I"
        q = self.post * self.inverse(correction * self.mid * self.pre)
        sign = (-1) ** (true_outcome ^ int(self.pre[self.qubit] in (2, 3)))
        return [(0.5, q), (0.5 * sign, q * self.parity)], reset_outcome


def compile_gadgets(width, atoms):
    result = []
    index = 0
    while index < len(atoms):
        if atoms[index].name not in {"T", "T_DAG"}:
            result.append(atoms[index])
            index += 1
            continue
        stop = index
        while stop < len(atoms) and atoms[stop].name in {"T", "T_DAG"}:
            stop += 1
        if stop - index == 1:
            result.append(atoms[index])
            index = stop
            continue
        inverse = stop
        while inverse < len(atoms) and atoms[inverse].name not in {"T", "T_DAG"}:
            inverse += 1
        end = inverse
        while end < len(atoms) and atoms[end].name in {"T", "T_DAG"}:
            end += 1
        result.append(Gadget(width, atoms[index:stop], atoms[stop:inverse], atoms[inverse:end]))
        index = end
    return result


def ch_gate(state, name, targets):
    if name in {"S", "S_DAG"}:
        state.apply_z(targets[0], exponent=0.5 if name == "S" else -0.5)
    elif name != "I":
        getattr(state, "apply_" + name.lower())(*targets)


class Term:
    def __init__(self, width):
        self.coefficient = 1.0 + 0j
        self.ch = cirq.StabilizerStateChForm(width)
        self.tableau = stim.TableauSimulator()
        self.tableau.set_num_qubits(width)

    def copy(self):
        other = object.__new__(Term)
        other.coefficient = self.coefficient
        other.ch = self.ch.copy()
        other.tableau = self.tableau.copy()
        return other

    def gate(self, name, targets):
        ch_gate(self.ch, name, targets)
        self.tableau.do(stim.CircuitInstruction(name, targets))

    def conjugated_pauli(self, p, signs):
        self.coefficient *= p.sign
        for q, axis in enumerate(p):
            if not axis:
                continue
            self.gate("_XYZ"[axis], (q,))
            if axis in (1, 2) and q in signs:
                sign = signs[q]
                self.gate("S_DAG" if sign == 1 else "S", (q,))
                self.coefficient *= np.exp(1j * sign * np.pi / 4)

    def project(self, p, bit):
        if p.sign != 1 or not p.weight:
            raise ValueError("this reference projects nonidentity positive Paulis")
        expectation = self.tableau.peek_observable_expectation(p)
        if expectation == -(1 - 2 * bit):
            return False
        if expectation:
            return True
        support = [q for q, axis in enumerate(p) if axis]
        pivot = support[0]
        gates: list[tuple[str, tuple[int, ...]]] = []
        for q in support:
            if p[q] == 2:
                gates.append(("S_DAG", (q,)))
            if p[q] in (1, 2):
                gates.append(("H", (q,)))
        gates += [("CX", (q, pivot)) for q in support[1:]]
        for name, targets in gates:
            self.gate(name, targets)
        self.ch.project_Z(pivot, bit)
        self.tableau.postselect_z(pivot, desired_value=bool(bit))
        self.coefficient *= 2**-0.5
        for name, targets in reversed(gates):
            self.gate("S" if name == "S_DAG" else name, targets)
        return True


def overlap(left, right):
    for generator in left.tableau.canonical_stabilizers():
        if right.tableau.peek_observable_expectation(generator) == -1:
            return 0j
    # Synthesizing a Clifford inverse here is deliberately offline reference
    # work. Production dispatch cannot perform this topology discovery.
    inverse = left.tableau.current_inverse_tableau().to_circuit()
    a, b = left.ch.copy(), right.ch.copy()
    for op in inverse:
        targets = op.targets_copy()
        size = 2 if op.name in {"CX", "CZ", "SWAP"} else 1
        for start in range(0, len(targets), size):
            group = tuple(t.value for t in targets[start : start + size])
            ch_gate(a, op.name, group)
            ch_gate(b, op.name, group)
    anchor = a.inner_product_of_state_and_x(0)
    if abs(abs(anchor) - 1) > 1e-7:
        raise AssertionError("CH state disagrees with its Clifford tableau")
    return anchor.conjugate() * b.inner_product_of_state_and_x(0)


class CoherentState:
    def __init__(self, width):
        self.terms = [Term(width)]
        self.peak_terms = 1

    def copy(self):
        other = object.__new__(CoherentState)
        other.terms = [t.copy() for t in self.terms]
        other.peak_terms = self.peak_terms
        return other

    def norm(self):
        diagonal = sum(abs(t.coefficient) ** 2 for t in self.terms)
        result = diagonal
        for i, a in enumerate(self.terms):
            for b in self.terms[i + 1 :]:
                result += 2 * (a.coefficient.conjugate() * b.coefficient * overlap(a, b)).real
        if result < -1e-10 * diagonal:
            raise AssertionError("negative coherent probability")
        return float(result) if result > 1e-12 * diagonal else 0.0

    def merge(self):
        groups = {}
        anchors = {}
        for term in self.terms:
            key = tuple(map(str, term.tableau.canonical_stabilizers()))
            if key not in groups:
                groups[key] = term
                continue
            other = groups[key]
            if key not in anchors:
                tab = other.tableau.copy()
                bits = 0
                for q in range(other.ch.n):
                    bit = int(tab.peek_z(q) == -1)
                    tab.postselect_z(q, desired_value=bool(bit))
                    bits = (bits << 1) | bit
                anchors[key] = bits, other.ch.inner_product_of_state_and_x(bits)
            bits, anchor = anchors[key]
            phase = term.ch.inner_product_of_state_and_x(bits) / anchor
            if abs(abs(phase) - 1) > 1e-7:
                raise AssertionError("phase anchor disagrees with stabilizer identity")
            other.coefficient += term.coefficient * phase
        scale = max((abs(t.coefficient) for t in groups.values()), default=0)
        self.terms = [t for t in groups.values() if abs(t.coefficient) > 1e-12 * scale]

    def compress_qubit(self):
        """Offline certificate: all terms lie in one common stabilizer code.

        Finding these generators per history is reference work, not an allowed
        production boundary implementation. A failed certificate retains terms.
        """
        self.merge()
        if len(self.terms) <= 2:
            return
        width = self.terms[0].ch.n
        generators = [t.tableau.canonical_stabilizers() for t in self.terms]
        constraints = []
        for group in generators[1:]:
            for p in group:
                constraints.append(
                    sum((not g.commutes(p)) << i for i, g in enumerate(generators[0]))
                )
        basis = nullspace(constraints, width)

        def product(rows, mask):
            p = stim.PauliString(width)
            for i, generator in enumerate(rows):
                if mask >> i & 1:
                    p *= generator
            return p

        common = [product(generators[0], row) for row in basis]
        signs = []
        for term in self.terms[1:]:
            expectations = [term.tableau.peek_observable_expectation(p) for p in common]
            if any(e == 0 for e in expectations):
                raise AssertionError("intersection contains a non-stabilizer")
            signs.append(sum((e == -1) << i for i, e in enumerate(expectations)))
        common = [product(common, row) for row in nullspace(signs, len(common))]
        if len(common) < width - 1:
            return
        encoder = stim.Tableau.from_stabilizers(common, allow_underconstrained=True)
        zero = Term(width)
        for op in encoder.to_circuit():
            targets = op.targets_copy()
            size = 2 if op.name in {"CX", "CZ", "SWAP"} else 1
            for start in range(0, len(targets), size):
                zero.gate(op.name, tuple(t.value for t in targets[start : start + size]))
        outputs = [zero]
        if len(common) == width - 1:
            one = zero.copy()
            p = encoder.x_output(width - 1)
            one.coefficient *= p.sign
            for q, axis in enumerate(p):
                if axis:
                    one.gate("_XYZ"[axis], (q,))
            # Move the basis vector's global Pauli phase into its CH state so
            # inner products recover coordinates in this exact orthogonal pair.
            one.ch.apply_global_phase(one.coefficient)
            one.coefficient = 1.0 + 0j
            outputs.append(one)
        for output in outputs:
            output.coefficient = sum(t.coefficient * overlap(output, t) for t in self.terms)
        self.terms = outputs

    def gate(self, atom):
        if atom.name in {"T", "T_DAG"}:
            phase = np.exp((1 if atom.name == "T" else -1) * 1j * np.pi / 4)
            expanded = []
            for term in self.terms:
                other = term.copy()
                other.gate("Z", atom.targets)
                other.coefficient *= (1 - phase) / 2
                term.coefficient *= (1 + phase) / 2
                expanded.extend([term, other])
            self.terms = expanded
            self.peak_terms = max(self.peak_terms, len(expanded))
        else:
            for term in self.terms:
                term.gate(atom.name, atom.targets)

    def project(self, atom, true_outcome):
        self.terms = [t for t in self.terms if t.project(atom.pauli, true_outcome)]
        if atom.name in {"R", "RX"} and true_outcome:
            self.gate(Atom("Z" if atom.name == "RX" else "X", atom.targets))

    def gadget(self, gadget, true_outcome):
        operators, reset = gadget.terms(true_outcome)
        expanded = []
        for term in self.terms:
            for coefficient, p in operators:
                other = term.copy()
                other.coefficient *= coefficient
                other.conjugated_pauli(p, gadget.signs)
                expanded.append(other)
        self.terms = expanded
        self.peak_terms = max(self.peak_terms, len(expanded))
        return reset


def evaluate(program, width, records):
    state = CoherentState(width)
    for operation in program:
        if isinstance(operation, Gadget):
            state.compress_qubit()
            bit = records[operation.measurement.record] ^ operation.measurement.flip
            reset = state.gadget(operation, bit)
            if reset != records[operation.reset.record]:
                return dict(probability=0.0, log_probability=None, peak_terms=state.peak_terms)
        elif operation.record >= 0:
            state.project(operation, records[operation.record] ^ operation.flip)
        else:
            state.gate(operation)
    state.merge()
    probability = state.norm()
    return dict(
        probability=probability,
        log_probability=math.log(probability) if probability else None,
        peak_terms=state.peak_terms,
    )


def sample(program, width, num_records, seed, initial=None):
    rng = random.Random(seed)
    state = CoherentState(width) if initial is None else initial.copy()
    records = [0] * num_records
    log_probability = 0.0
    for operation in program:
        if isinstance(operation, Gadget):
            state.compress_qubit()
            zero = state.copy()
            zero.gadget(operation, 0)
            p0 = zero.norm()
        elif operation.record >= 0:
            values = {t.tableau.peek_observable_expectation(operation.pauli) for t in state.terms}
            if values == {1} or values == {-1}:
                bit = int(values == {-1})
                state.project(operation, bit)
                records[operation.record] = bit ^ operation.flip
                continue
            zero = state.copy()
            zero.project(operation, 0)
            p0 = zero.norm()
        else:
            state.gate(operation)
            continue
        if not -1e-9 <= p0 <= 1 + 1e-9:
            raise AssertionError(f"invalid coherent branch probability {p0}")
        p0 = min(1.0, max(0.0, p0))
        bit = int(rng.random() >= p0)
        probability = 1 - p0 if bit else p0
        if bit:
            if isinstance(operation, Gadget):
                state.gadget(operation, 1)
            else:
                state.project(operation, 1)
        else:
            state = zero
        for term in state.terms:
            term.coefficient /= math.sqrt(probability)
        log_probability += math.log(probability)
        if isinstance(operation, Gadget):
            _, reset = operation.terms(bit)
            records[operation.measurement.record] = bit ^ operation.measurement.flip
            records[operation.reset.record] = reset
            state.compress_qubit()
        else:
            records[operation.record] = bit ^ operation.flip
    state.merge()
    if abs(state.norm() - 1) > 1e-7:
        raise AssertionError("sampled coherent state lost normalization")
    return dict(records=records, log_probability=log_probability, peak_terms=state.peak_terms)


def nullspace(rows, width):
    pivots: dict[int, int] = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    basis = []
    for free in range(width):
        if free in pivots:
            continue
        vector = 1 << free
        for pivot in sorted(pivots):
            if (pivots[pivot] & vector).bit_count() % 2:
                vector ^= 1 << pivot
        basis.append(vector)
    return basis


def require_t_dialect(text):
    if re.search(r"(?m)^[ \t]*S(?:_DAG)?(?:[ \t]|$)", text):
        raise ValueError("this reference reserves the S proxy spelling for physical T gates")
