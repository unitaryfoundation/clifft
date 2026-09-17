"""Offline signed boundary certificates and fixed parity maps for actual MSC.

Construction uses Stim flow solving. Evaluation only encodes physical fault
bits and applies precomputed parity rows; it performs no tableau work.
"""

from __future__ import annotations

from dataclasses import dataclass

import stim
from fold_blocks import masks


@dataclass(frozen=True)
class Row:
    records: int = 0
    faults: int = 0
    constant: int = 0

    def evaluate(self, records, faults):
        return (
            (self.records & records).bit_count()
            ^ (self.faults & faults).bit_count()
            ^ self.constant
        ) & 1


class FaultBits:
    def __init__(self, program):
        self.slots: dict[tuple[int, int, str], int] = {}
        for k, site in enumerate(program.sites):
            if site.readout:
                self.slots[k, -1, "readout"] = len(self.slots)
            else:
                for q in site.wires:
                    for axis in "XZ":
                        self.slots[k, q, axis] = len(self.slots)

    def bit(self, site, q, axis):
        return 1 << self.slots[site, q, axis]

    def encode(self, program, history):
        program.validate_history(history)
        result = 0
        for k, choice in history.items():
            site = program.sites[k]
            if site.readout:
                result ^= self.bit(k, -1, "readout")
            else:
                for q, axis in zip(site.wires, choice, strict=True):
                    if axis in "XY":
                        result ^= self.bit(k, q, "X")
                    if axis in "YZ":
                        result ^= self.bit(k, q, "Z")
        return result


def pauli(width, axis, wires):
    result = stim.PauliString(width)
    for q in wires:
        result[q] = axis
    return result


def code_checks(program, region, d3_program):
    source = d3_program if len(region.data) == 7 else program
    source_data = list(source.regions.values())[-1].data
    mapping = dict(zip(source_data, region.data, strict=True))
    check_line = next(op for op in reversed(source.instructions) if op.name == "MPP")
    result: list[stim.PauliString] = []
    for word in check_line.targets:
        p = stim.PauliString(program.width)
        for factor in word.split("*"):
            p[mapping[int(factor[1:])]] = factor[0]
        result.append(p)
    return result


class BoundaryPlan:
    def __init__(self, program, start, end, data, checks, input_data=(), input_checks=()):
        self.program = program
        self.start, self.end = start, end
        self.data = tuple(data)
        self.bits = FaultBits(program)
        self.checks = list(checks)
        self.code_count = len(checks)
        if self.code_count != len(data) - 1:
            raise ValueError("boundary is not a single-logical-qubit code")
        self.instructions = [op for op in program.instructions if start <= op.line <= end]
        self.local_records = []
        record_index = {event: k for k, event in enumerate(program.records)}
        self.ideal = stim.Circuit()
        self.x = [0] * program.width
        self.z = [0] * program.width
        self.record_delta = {}
        quantum = {"R", "RX", "M", "MX", "MPP", "CX", "CZ", "H", "S", "S_DAG", "X", "Y", "Z"}
        for op in self.instructions:
            if op.name in {"T", "T_DAG"}:
                raise ValueError("boundary contains a non-Clifford gate")
            if op.name in quantum:
                self.ideal += stim.Circuit(op.name + " " + " ".join(op.targets))
            if op.name in {"M", "MX", "MPP"}:
                for offset in range(len(op.targets)):
                    event = program.measurement_at[op.line, offset]
                    record = record_index[event]
                    self.local_records.append(record)
                    delta = 0
                    for axis, q in program.measurements[event].pauli:
                        if axis in "XY":
                            delta ^= self.z[q]
                        if axis in "YZ":
                            delta ^= self.x[q]
                    site = program.site_at.get((op.line, offset))
                    if site is not None:
                        delta ^= self.bits.bit(site, -1, "readout")
                    self.record_delta[record] = delta
            elif op.name in {"R", "RX"}:
                for word in op.targets:
                    self.x[int(word)] = self.z[int(word)] = 0
            elif op.name in {"CX", "CZ"}:
                for offset in range(0, len(op.targets), 2):
                    a, b = op.targets[offset : offset + 2]
                    q = int(b)
                    if (op.line, offset) in program.controls:
                        source = program.controls[op.line, offset]
                        if source not in self.record_delta:
                            raise ValueError("boundary feedback depends on an external record")
                        (self.x if op.name == "CX" else self.z)[q] ^= self.record_delta[source]
                    else:
                        p = int(a)
                        if op.name == "CX":
                            self.x[q] ^= self.x[p]
                            self.z[p] ^= self.z[q]
                        else:
                            self.z[q] ^= self.x[p]
                            self.z[p] ^= self.x[q]
            elif op.name in {"H", "S", "S_DAG"}:
                for word in op.targets:
                    q = int(word)
                    if op.name == "H":
                        self.x[q], self.z[q] = self.z[q], self.x[q]
                    else:
                        self.z[q] ^= self.x[q]
            elif op.name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
                arity = 2 if op.name == "DEPOLARIZE2" else 1
                for offset in range(0, len(op.targets), arity):
                    site = program.site_at[op.line, offset]
                    for q in program.sites[site].wires:
                        self.x[q] ^= self.bits.bit(site, q, "X")
                        self.z[q] ^= self.bits.bit(site, q, "Z")
        self.rows = [self.flow_row(p)[0] for p in self.checks]
        self.isolated = []
        for q in range(program.width):
            if q in data:
                continue
            for axis in "XZ":
                p = pauli(program.width, axis, [q])
                try:
                    row, _ = self.flow_row(p)
                    break
                except ValueError:
                    pass
            else:
                p, row = self.isolated_spectator(q)
                self.isolated.append(q)
            self.checks.append(p)
            self.rows.append(row)
        # n-1 independent output stabilizers leave precisely one logical qubit;
        # the single-wire complement checks also prove data/ancilla separation.
        stim.Tableau.from_stabilizers(self.checks, allow_underconstrained=True)
        if len(self.checks) != program.width - 1:
            raise ValueError("boundary leaves unaccounted spectator degrees of freedom")
        self.logical = {}
        for axis in "XZ" if input_data else "":
            incoming = pauli(program.width, axis, input_data)
            outgoing = pauli(program.width, axis, data)
            self.logical[axis] = self.flow_row(outgoing, incoming)
        self.input_checks = list(input_checks)
        self.input_rows = [self.flow_row(stim.PauliString(program.width), p) for p in input_checks]
        local = [stim.PauliString("".join(str(p)[q + 1] for q in data)) for p in checks]
        logical_z = pauli(len(data), "Z", range(len(data)))
        logical_x = pauli(len(data), "X", range(len(data)))
        tableau = stim.Tableau.from_stabilizers(local + [logical_z])
        self.duals = []
        for k in range(self.code_count):
            correction = tableau.x_output(k)
            if not correction.commutes(logical_x):
                correction *= logical_z
            self.duals.append(masks(correction))

    def flow_row(self, output, incoming=None):
        flow = (
            stim.Flow(output=output)
            if incoming is None
            else stim.Flow(input=incoming, output=output)
        )
        measurements = self.ideal.solve_flow_measurements([flow])[0]
        if measurements is None:
            raise ValueError("missing boundary stabilizer flow")
        positive = stim.Flow(input=flow.input_copy(), output=output, measurements=measurements)
        sign = int(not self.ideal.has_flow(positive))
        if sign and not self.ideal.has_flow(
            stim.Flow(input=flow.input_copy(), output=-output, measurements=measurements)
        ):
            raise ValueError("unsigned flow has no consistent sign")
        fault_mask = 0
        for q in range(self.program.width):
            if output[q] in (1, 2):
                fault_mask ^= self.z[q]
            if output[q] in (2, 3):
                fault_mask ^= self.x[q]
        record_mask = 0
        for k in measurements:
            record = self.local_records[k]
            record_mask ^= 1 << record
            fault_mask ^= self.record_delta[record]
        return Row(record_mask, fault_mask, sign), measurements

    def isolated_spectator(self, q):
        """Certify a last single-wire preparation/measurement followed by idling."""
        anchor = None
        for op in self.program.instructions:
            if op.line > self.end:
                break
            if op.name in {"R", "RX", "M", "MX"} and str(q) in op.targets:
                anchor = op
        axis = "X" if anchor is not None and anchor.name.endswith("X") else "Z"
        line = 0 if anchor is None else anchor.line
        quantum = {
            "R",
            "RX",
            "M",
            "MX",
            "MPP",
            "CX",
            "CZ",
            "H",
            "S",
            "S_DAG",
            "X",
            "Y",
            "Z",
            "T",
            "T_DAG",
        }
        for op in self.program.instructions:
            if line < op.line <= self.end and op.name in quantum:
                wires = {
                    int(f[1:] if f[0] in "XYZ" else f)
                    for word in op.targets
                    if not word.startswith("rec[")
                    for f in word.split("*")
                }
                if q in wires:
                    raise ValueError("unfixed spectator interacts after its proposed anchor")
        record_mask = fault_mask = 0
        if anchor is not None and anchor.name in {"M", "MX"}:
            offset = anchor.targets.index(str(q))
            event = self.program.measurement_at[anchor.line, offset]
            record_mask = 1 << self.program.records.index(event)
            site = self.program.site_at.get((anchor.line, offset))
            if site is not None:
                fault_mask ^= self.bits.bit(site, -1, "readout")
        for k, site in enumerate(self.program.sites):
            if line < site.line <= self.end and q in site.wires:
                fault_mask ^= self.bits.bit(k, q, "Z" if axis == "X" else "X")
        return pauli(self.program.width, axis, [q]), Row(record_mask, fault_mask)

    def evaluate(self, records, history):
        record_bits = sum(int(b) << k for k, b in enumerate(records))
        fault_bits = self.bits.encode(self.program, history)
        signs = [row.evaluate(record_bits, fault_bits) for row in self.rows]
        x = z = 0
        for sign, (dx, dz) in zip(signs[: self.code_count], self.duals, strict=True):
            if sign:
                x ^= dx
                z ^= dz
        return signs, (x, z)


def compile_boundaries(program, d3_program):
    found = list(program.regions.values())[:-1]
    previous = max(
        op.line
        for op in program.instructions
        if op.name in {"T", "T_DAG"} and op.line < found[0].start
    )
    result: list[BoundaryPlan] = []
    for region in found:
        result.append(
            BoundaryPlan(
                program,
                previous + 1,
                region.start - 1,
                region.data,
                code_checks(program, region, d3_program),
                found[0].data if len(region.data) > 7 else (),
                result[0].checks[: result[0].code_count] if result else (),
            )
        )
        previous = region.end
    return result
