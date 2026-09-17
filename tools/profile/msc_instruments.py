"""Offline Clifford-instrument certificates for weighted MSC boundaries.

Compilation exposes reset outcomes and eliminates stabilizer flows. Binding a
record uses fixed parity rows only. Contracting those rows with a coherent input
state is deliberately left to a separate reference implementation.
"""

from dataclasses import dataclass

import stim
from fold_blocks import masks
from msc_boundaries import FaultBits, Row, pauli


def pauli_key(p, width):
    x, z = masks(p)
    return x | (z << width)


def eliminate(flow, basis, output, width):
    p = flow.output_copy() if output else flow.input_copy()
    key = pauli_key(p, width)
    while key:
        pivot = key.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = (key, flow)
            return None
        other_key, other_flow = basis[pivot]
        key ^= other_key
        flow *= other_flow
    return flow


@dataclass(frozen=True)
class BoundInstrument:
    reachable: bool
    input_signs: tuple[int, ...]
    logical_signs: tuple[int, ...]
    probability_scale: float


class InstrumentPlan:
    """K dagger K = scale times a signed commuting input projector.

    Logical output probes pull back to fixed input Paulis times that same
    projector. Scalar Kraus phases disappear, but interference between the
    incoming coherent terms must be retained during projector contraction.
    """

    def __init__(self, boundary, *, logical_outputs=None):
        program = self.program = boundary.program
        self.bits = FaultBits(program)
        self.events: list[int] = []
        self.delta = {}
        self.x = [0] * program.width
        self.z = [0] * program.width
        self.ideal = stim.Circuit(f"I {program.width - 1}")
        local = {}
        for op in boundary.instructions:
            if op.name in {"M", "MX", "MPP", "R", "RX"}:
                for offset, word in enumerate(op.targets):
                    event = program.measurement_at[op.line, offset]
                    local[event] = len(self.events)
                    self.events.append(event)
                    measurement = program.measurements[event]
                    delta = 0
                    for axis, q in measurement.pauli:
                        if axis in "XY":
                            delta ^= self.z[q]
                        if axis in "YZ":
                            delta ^= self.x[q]
                    self.delta[event] = delta
                    name = {"R": "M", "RX": "MX"}.get(op.name, op.name)
                    self.ideal += stim.Circuit(f"{name} {word}")
                    if measurement.reset:
                        self.ideal += stim.Circuit(
                            f"{'CX' if op.name == 'R' else 'CZ'} rec[-1] {word}"
                        )
                        q = int(word)
                        self.x[q] = self.z[q] = 0
            elif op.name in {"CX", "CZ"}:
                for offset in range(0, len(op.targets), 2):
                    a, b = op.targets[offset : offset + 2]
                    q = int(b)
                    if (op.line, offset) in program.controls:
                        event = program.records[program.controls[op.line, offset]]
                        if event not in local:
                            raise ValueError("instrument feedback depends on an external record")
                        a = f"rec[{local[event] - len(self.events)}]"
                        delta = self.delta[event]
                        source = program.measurements[event]
                        site = program.site_at.get((source.line, source.offset))
                        if site is not None:
                            delta ^= self.bits.bit(site, -1, "readout")
                        (self.x if op.name == "CX" else self.z)[q] ^= delta
                    else:
                        p = int(a)
                        if op.name == "CX":
                            self.x[q] ^= self.x[p]
                            self.z[p] ^= self.z[q]
                        else:
                            self.z[p] ^= self.x[q]
                            self.z[q] ^= self.x[p]
                    self.ideal += stim.Circuit(f"{op.name} {a} {b}")
            elif op.name in {"H", "S", "S_DAG", "X", "Y", "Z"}:
                self.ideal += stim.Circuit(op.name + " " + " ".join(op.targets))
                for word in op.targets:
                    q = int(word)
                    if op.name == "H":
                        self.x[q], self.z[q] = self.z[q], self.x[q]
                    elif op.name in {"S", "S_DAG"}:
                        self.z[q] ^= self.x[q]
            elif op.name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
                arity = 2 if op.name == "DEPOLARIZE2" else 1
                for offset in range(0, len(op.targets), arity):
                    site = program.site_at[op.line, offset]
                    for q in program.sites[site].wires:
                        self.x[q] ^= self.bits.bit(site, q, "X")
                        self.z[q] ^= self.bits.bit(site, q, "Z")
            elif op.name not in {
                "QUBIT_COORDS",
                "TICK",
                "SHIFT_COORDS",
                "DETECTOR",
                "OBSERVABLE_INCLUDE",
            }:
                raise ValueError("instrument is not Clifford")
        output_basis: dict[int, tuple[int, stim.Flow]] = {}
        input_basis: dict[int, tuple[int, stim.Flow]] = {}
        constraints = []
        for flow in self.ideal.flow_generators():
            flow = eliminate(flow, output_basis, True, program.width)
            if flow is None:
                continue
            flow = eliminate(flow, input_basis, False, program.width)
            if flow is not None:
                constraints.append(flow)
        self.input_flows = [flow for _, flow in input_basis.values()]
        self.constraint_flows = constraints
        self.input_paulis = [self.positive(flow.input_copy()) for flow in self.input_flows]
        self.input_rows = [self.row(flow) for flow in self.input_flows]
        self.constraints = [self.row(flow) for flow in constraints]
        self.random_power = len(self.events) - len(self.input_rows) - len(self.constraints)
        if self.random_power < 0:
            raise ValueError("invalid instrument normalization rank")
        lx = pauli(program.width, "X", boundary.data)
        lz = pauli(program.width, "Z", boundary.data)
        self.output_paulis = (
            [lx, 1j * lx * lz, lz] if logical_outputs is None else list(logical_outputs)
        )
        self.logical_paulis = []
        self.logical_rows = []
        self.logical_flows = []
        for target in self.output_paulis:
            key = pauli_key(target, program.width)
            flow = stim.Flow(input=stim.PauliString(program.width))
            while key:
                pivot = key.bit_length() - 1
                if pivot not in output_basis:
                    raise ValueError("logical probe has no input pullback")
                other_key, other_flow = output_basis[pivot]
                key ^= other_key
                flow *= other_flow
            self.logical_flows.append(flow)
            self.logical_paulis.append(self.positive(flow.input_copy()))
            self.logical_rows.append(self.row(flow, int(target.sign == -1)))
        if any(not a.commutes(b) for a in self.input_paulis for b in self.input_paulis):
            raise ValueError("input projector generators do not commute")
        if any(not a.commutes(b) for a in self.input_paulis for b in self.logical_paulis):
            raise ValueError("logical probe does not preserve the input projector")

    @staticmethod
    def positive(p):
        if p.sign not in (1, -1):
            raise ValueError("non-Hermitian flow")
        p.sign = 1
        return p

    def row(self, flow, target_sign=0):
        incoming, outgoing = flow.input_copy(), flow.output_copy()
        records = faults = 0
        for k in flow.measurements_copy():
            event = self.events[k]
            records ^= 1 << event
            faults ^= self.delta[event]
        for q in range(len(outgoing)):
            if outgoing[q] in (1, 2):
                faults ^= self.z[q]
            if outgoing[q] in (2, 3):
                faults ^= self.x[q]
        return Row(records, faults, int(incoming.sign * outgoing.sign == -1) ^ target_sign)

    def bind(self, outcomes, history):
        if len(outcomes) != len(self.program.measurements) or any(
            b not in (0, 1) for b in outcomes
        ):
            raise ValueError("one true bit is required for every measurement and reset")
        records = sum(int(b) << k for k, b in enumerate(outcomes))
        faults = self.bits.encode(self.program, history)
        return BoundInstrument(
            not any(row.evaluate(records, faults) for row in self.constraints),
            tuple(row.evaluate(records, faults) for row in self.input_rows),
            tuple(row.evaluate(records, faults) for row in self.logical_rows),
            2.0**-self.random_power,
        )
