"""Sparse CSS contraction at the actual final MSC code projection.

The coherent cultivation/terminal terms are supplied together. Construction
uses CSS supports and small tableaux, never dense nineteen-qubit operators.
The Python research kernel is not a production hot executor.
"""

import numpy as np
import stim
from fold_blocks import masks
from fold_contraction import ROOTS
from msc_boundaries import pauli
from msc_growth_contraction import local_pauli


class TerminalContraction:
    def __init__(self, program, source_boundary):
        self.program = program
        self.data = tuple(source_boundary.data)
        self.width = len(self.data)
        if self.width not in (7, 19) or source_boundary.code_count != self.width - 1:
            raise ValueError("unsupported terminal CSS block")
        self.ancillas = tuple(q for q in range(program.width) if q not in self.data)
        self.measurement = next(op for op in reversed(program.instructions) if op.name == "MPP")
        self.checks = []
        self.events = []
        for offset, word in enumerate(self.measurement.targets):
            p = stim.PauliString(program.width)
            for factor in word.split("*"):
                p[int(factor[1:])] = factor[0]
            if any(p[q] for q in self.ancillas):
                raise ValueError("terminal check includes a spectator")
            self.checks.append(local_pauli(p, self.data))
            self.events.append(program.measurement_at[self.measurement.line, offset])
        source_checks = [
            local_pauli(p, self.data) for p in source_boundary.checks[: source_boundary.code_count]
        ]
        if self.checks != source_checks:
            raise ValueError("terminal checks differ from the source code convention")
        lx = pauli(self.width, "X", range(self.width))
        lz = pauli(self.width, "Z", range(self.width))
        if any(not p.commutes(lx) or not p.commutes(lz) for p in self.checks):
            raise ValueError("all-data logical axes do not preserve the code")
        tableau = stim.Tableau.from_stabilizers([*self.checks, lz])
        span = {0}
        for p in self.checks:
            x, z = masks(p)
            if p.sign != 1 or (x and z) or not (x or z):
                raise ValueError("terminal code is not positive CSS")
            if x:
                span |= {b ^ x for b in span}
        logical_x = (1 << self.width) - 1
        self.supports = [tuple(sorted(span)), tuple(sorted(b ^ logical_x for b in span))]
        if span & set(self.supports[1]) or any(b.bit_count() & 1 for b in span):
            raise ValueError("CSS support does not separate the logical basis")
        self.decoder = {b: logical for logical, row in enumerate(self.supports) for b in row}
        self.normalization = 1 / len(span)
        self.duals = []
        for k in range(len(self.checks)):
            correction = tableau.x_output(k)
            if not correction.commutes(lx):
                correction *= lz
            self.duals.append(masks(correction))
        if any(
            op.name not in {"DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "SHIFT_COORDS"}
            for op in program.instructions
            if op.line > self.measurement.line
        ):
            raise ValueError("quantum operations follow the terminal projection")

    def contract(self, logical, frame, terms, syndrome):
        """Project the complete monomial sum and return weighted logical amplitudes."""
        logical = np.asarray(logical, dtype=complex)
        if logical.shape != (2,) or not np.all(np.isfinite(logical)):
            raise ValueError("two finite logical amplitudes are required")
        if len(syndrome) != len(self.duals) or any(b not in (0, 1) for b in syndrome):
            raise ValueError("one true outcome is required per terminal check")
        x, z = frame
        limit = 1 << self.width
        if not 0 <= x < limit or not 0 <= z < limit:
            raise ValueError("input frame exceeds data width")
        cx = cz = 0
        for sign, (dx, dz) in zip(syndrome, self.duals, strict=True):
            if sign:
                cx ^= dx
                cz ^= dz
        result = np.zeros(2, dtype=complex)
        if not terms:
            return result
        anchor = terms[0].ancillas
        for term in terms:
            op = term.data
            if (
                op.width != self.width
                or op.edges
                or len(op.linear) != self.width
                or not 0 <= op.flips < limit
            ):
                raise ValueError("monomial is outside the terminal diagonal/flip family")
            if term.ancillas.shape != (len(self.ancillas), 2) or not np.allclose(
                np.sum(np.abs(term.ancillas) ** 2, axis=1), 1, atol=1e-12, rtol=0
            ):
                raise ValueError("one normalized state is required per spectator")
            scalar = term.weight
            for q in range(len(self.ancillas)):
                overlap = np.vdot(anchor[q], term.ancillas[q])
                if abs(abs(overlap) - 1) > 1e-10:
                    raise ValueError("spectator differs between coherent terminal terms")
                scalar *= overlap
            for source, row in enumerate(self.supports):
                for b in row:
                    output, phase = op.column(b ^ x)
                    phase += 4 * ((z & b).bit_count() & 1)
                    output ^= cx
                    phase += 4 * ((cz & output).bit_count() & 1)
                    target = self.decoder.get(output)
                    if target is not None:
                        result[target] += (
                            scalar * logical[source] * ROOTS[phase % 8] * self.normalization
                        )
        return result
