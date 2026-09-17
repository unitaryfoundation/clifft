"""Static one-magic-input Clifford instrument for the original MSC injection.

A postselected Clifford replacement of the single T gate gives a two-qubit
Choi certificate. Construction reduces it to four two-by-two matrices and
fixed outcome/fault parity rows. Evaluation performs no stabilizer evolution.
"""

from types import SimpleNamespace

import numpy as np
import stim
from fold_blocks import masks
from msc_boundaries import pauli
from msc_instruments import InstrumentPlan, eliminate
from msc_protocol import Program


class InjectionPlan:
    def __init__(self, program, boundary):
        self.program = program
        self.end = boundary.end
        prefix = [op for op in program.instructions if op.line <= self.end]
        rotations = [op for op in prefix if op.name in {"T", "T_DAG"}]
        if len(rotations) != 1 or len(rotations[0].targets) != 1:
            raise ValueError("injection prefix must contain exactly one T gate")
        if any(line <= self.end for line, _ in program.controls):
            raise ValueError("injection prefix feedback is outside this compiler")
        resource, reference = program.width, program.width + 1
        lines = [f"H {resource}", f"CX {resource} {reference}"]
        source_lines = {}
        for op in prefix:
            if op.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
                continue
            source_lines[op.line] = len(lines) + 1
            if op.name in {"T", "T_DAG"}:
                lines.extend([f"CX {op.targets[0]} {resource}", f"M {resource}"])
                virtual_line = len(lines)
            else:
                args = "(" + ",".join(map(str, op.args)) + ")" if op.args else ""
                lines.append(op.name + args + " " + " ".join(op.targets))
        virtual = self.virtual = Program("\n".join(lines))
        self.event_map = [
            (k, virtual.measurement_at[source_lines[event.line], event.offset])
            for k, event in enumerate(program.measurements)
            if event.line <= self.end
        ]
        self.postselection_event = virtual.measurement_at[virtual_line, 0]
        self.site_map = {
            k: virtual.site_at[source_lines[site.line], site.offset]
            for k, site in enumerate(program.sites)
            if site.line <= self.end
        }
        shell = SimpleNamespace(
            program=virtual, instructions=virtual.instructions, data=boundary.data
        )
        self.instrument = InstrumentPlan(shell, logical_outputs=[])
        self.ideal = (
            stim.Circuit("R " + " ".join(map(str, range(virtual.width)))) + self.instrument.ideal
        )
        output_basis: dict[int, tuple[int, stim.Flow]] = {}
        constraints = []
        self.certificates = []
        for flow in self.ideal.flow_generators():
            if any(masks(flow.input_copy())):
                raise ValueError("initial preparation leaves an unspecified quantum input")
            remainder = eliminate(flow, output_basis, True, virtual.width)
            if remainder is not None:
                constraints.append(self.instrument.row(remainder))
                self.certificates.append(remainder)
        self.constraints = constraints
        self.random_power = len(self.instrument.events) - len(constraints)
        if self.random_power < 0:
            raise ValueError("invalid Choi measurement normalization")
        logical_x = pauli(virtual.width, "X", boundary.data)
        logical_z = pauli(virtual.width, "Z", boundary.data)
        logical_axes = [
            stim.PauliString(virtual.width),
            logical_x,
            1j * logical_x * logical_z,
            logical_z,
        ]
        joint_basis = {}
        self.joint_paulis = []
        self.joint_rows = []
        self.all_joint_rows = []
        for output_axis in range(4):
            for reference_axis in range(4):
                if not output_axis and not reference_axis:
                    continue
                target = logical_axes[output_axis].copy()
                target[reference] = reference_axis
                support = self.ideal.solve_flow_measurements([stim.Flow(output=target)])[0]
                if support is None:
                    continue
                flow = stim.Flow(output=target, measurements=support)
                sign = int(not self.ideal.has_flow(flow))
                signed = stim.Flow(output=target * (-1 if sign else 1), measurements=support)
                if not self.ideal.has_flow(signed):
                    raise ValueError("joint Choi flow has no consistent sign")
                self.certificates.append(signed)
                row = self.instrument.row(signed, int(target.sign == -1))
                joint = stim.PauliString("_XYZ"[output_axis] + "_XYZ"[reference_axis])
                self.all_joint_rows.append((joint, row))
                x, z = masks(joint)
                key = x | z << 2
                while key:
                    pivot = key.bit_length() - 1
                    if pivot not in joint_basis:
                        joint_basis[pivot] = key
                        self.joint_paulis.append(joint)
                        self.joint_rows.append(row)
                        break
                    key ^= joint_basis[pivot]
        if len(self.joint_paulis) != 2 or len(self.all_joint_rows) != 3:
            raise ValueError("logical output and reference are not a pure two-qubit Choi state")
        if any(p.weight != 2 for p, _ in self.all_joint_rows):
            raise ValueError("injection is not a full-rank logical Clifford map")
        # All other wires must be fixed product spectators, including the
        # measured resource wire. This certifies the logical Choi reduction.
        output_checks = [p + stim.PauliString(2) for p in boundary.checks]
        output_checks.append(pauli(virtual.width, "Z", [resource]))
        for target in output_checks:
            support = self.ideal.solve_flow_measurements([stim.Flow(output=target)])[0]
            if support is None:
                raise ValueError("injection output has no code or spectator preparation flow")
            positive = stim.Flow(output=target, measurements=support)
            signed = stim.Flow(
                output=target * (1 if self.ideal.has_flow(positive) else -1), measurements=support
            )
            if not self.ideal.has_flow(signed):
                raise ValueError("injection output preparation has no consistent sign")
            self.certificates.append(signed)
        # Dropping the postselected virtual bit must leave independent record
        # constraints. Then every physical history has a normalized affine set
        # of original outcomes, each with probability 2**(1-random_power).
        pivots = {}
        for row in self.constraints:
            key = row.records & ~(1 << self.postselection_event)
            while key:
                pivot = key.bit_length() - 1
                if pivot not in pivots:
                    pivots[pivot] = key
                    break
                key ^= pivots[pivot]
            if not key:
                raise ValueError("virtual postselection restricts physical fault reachability")
        self.free_outcomes = len(self.event_map) - len(pivots)
        if self.free_outcomes != self.random_power - 1:
            raise ValueError("injection record distribution is not normalized")
        self.matrices = []
        for signs in range(4):
            checks = [p * (-1 if signs >> k & 1 else 1) for k, p in enumerate(self.joint_paulis)]
            state = np.asarray(
                stim.Tableau.from_stabilizers(checks).to_state_vector(endian="little"),
                dtype=complex,
            )
            state /= np.linalg.norm(state)
            matrix = 2 * 2.0 ** (-self.random_power / 2) * state.reshape(2, 2).T
            self.matrices.append(matrix)
        angle = -np.pi / 4 if rotations[0].name == "T_DAG" else np.pi / 4
        self.magic = np.array([1, np.exp(1j * angle)]) / np.sqrt(2)

    def bind(self, outcomes, history):
        self.program.validate_history(history)
        if len(outcomes) <= max(k for k, _ in self.event_map):
            raise ValueError("injection outcomes are incomplete")
        record_bits = 0
        for source, target in self.event_map:
            if outcomes[source] not in (0, 1):
                raise ValueError("injection outcomes must be binary")
            record_bits |= int(outcomes[source]) << target
        # The virtual resource measurement is always postselected to zero;
        # matrices include both Bell normalization and the virtual T replacement.
        faults = self.instrument.bits.encode(
            self.virtual,
            {self.site_map[k]: choice for k, choice in history.items() if k in self.site_map},
        )
        if any(row.evaluate(record_bits, faults) for row in self.constraints):
            return None
        return sum(row.evaluate(record_bits, faults) << k for k, row in enumerate(self.joint_rows))

    def evaluate(self, outcomes, history):
        signs = self.bind(outcomes, history)
        return np.zeros(2, dtype=complex) if signs is None else self.matrices[signs] @ self.magic
