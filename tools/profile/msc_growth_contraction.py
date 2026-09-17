"""Sparse code/ancilla contraction for the certified original-MSC growth map.

Stim is used only during construction. Evaluation sums monomial terms on a
fixed sparse code basis and scalar ancilla overlaps. This Python research
kernel is not an allocation-free native executor or a complete sampler.
"""

from dataclasses import dataclass

import numpy as np
import stim
from fold_blocks import masks
from fold_check import PhaseMonomial
from fold_contraction import ROOTS
from msc_boundaries import pauli


@dataclass
class FactoredTerm:
    data: PhaseMonomial
    ancillas: np.ndarray
    weight: complex = 0.5


def local_pauli(p, data):
    result = stim.PauliString("".join("_XYZ"[p[q]] for q in data))
    result.sign = p.sign
    return result


def sparse_basis(checks, lx, lz):
    """Fix the logical phase by defining |1_L> = X_L |0_L>."""
    tableau = stim.Tableau.from_stabilizers([*checks, lz])
    zero = np.asarray(tableau.to_state_vector(endian="little"), dtype=complex)
    # Stim's single-precision amplitudes need renormalizing before contraction.
    zero /= np.linalg.norm(zero)
    one = lx.to_unitary_matrix(endian="little") @ zero
    result = []
    for vector in (zero, one):
        support = np.flatnonzero(np.abs(vector) > 1e-10)
        result.append([(int(b), complex(vector[b])) for b in support])
    return result


class GrowthContraction:
    def __init__(self, source_boundary, instrument):
        self.width = instrument.program.width
        self.data = tuple(source_boundary.data)
        if len(self.data) != 7:
            raise ValueError("growth contraction requires a seven-wire input block")
        self.ancillas = tuple(q for q in range(self.width) if q not in self.data)
        self.data_rows = []
        self.ancilla_rows = []
        checks = []
        seen = set()
        for k, p in enumerate(instrument.input_paulis):
            support = {q for q in range(self.width) if p[q]}
            if support and support <= set(self.data):
                self.data_rows.append(k)
                checks.append(local_pauli(p, self.data))
            elif len(support) == 1 and not support & set(self.data):
                q = next(iter(support))
                if q in seen:
                    raise ValueError("multiple input projectors on one ancilla")
                seen.add(q)
                matrix = stim.PauliString("_XYZ"[p[q]]).to_unitary_matrix(endian="little")
                bras = []
                for sign in (0, 1):
                    projector = (np.eye(2) + (-1) ** sign * matrix) / 2
                    column = max(projector.T, key=np.linalg.norm)
                    bras.append(column.conj() / np.linalg.norm(column))
                self.ancilla_rows.append((k, self.ancillas.index(q), np.array(bras)))
            else:
                raise ValueError("input projector couples data and ancillas")
        if len(checks) != 6:
            raise ValueError("input projector does not define one encoded qubit")
        self.unprojected = tuple(self.ancillas.index(q) for q in self.ancillas if q not in seen)
        self.logical = []
        for p in instrument.logical_paulis:
            if any(p[q] for q in self.ancillas):
                raise ValueError("logical pullback acts on spectators")
            self.logical.append(local_pauli(p, self.data))
        lx, ly, lz = self.logical
        if lx.commutes(lz) or any(not c.commutes(p) for c in checks for p in self.logical):
            raise ValueError("logical pullbacks do not preserve the input code")
        product = 1j * lx * lz
        if product not in (ly, -ly):
            raise ValueError("logical Y differs by a nonconstant syndrome factor")
        self.y_sign = int(product == -ly)
        self.source_basis = sparse_basis(
            [local_pauli(p, self.data) for p in source_boundary.checks[:6]],
            pauli(7, "X", range(7)),
            pauli(7, "Z", range(7)),
        )
        tableau = stim.Tableau.from_stabilizers([*checks, lz])
        self.duals = []
        for k in range(6):
            correction = tableau.x_output(k)
            if not correction.commutes(lx):
                correction *= lz
            self.duals.append(masks(correction))
        # Syndrome duals select every sector from one basis. Storing a separate
        # basis for each sector would grow exponentially with the check count.
        self.decoder = np.zeros((2, 128), dtype=complex)
        basis = sparse_basis(checks, lx, lz)
        for logical, row in enumerate(basis):
            for b, coefficient in row:
                self.decoder[logical, b] = coefficient.conjugate()
        if any(len(row) != 8 for row in self.source_basis + basis):
            raise ValueError("unexpected sparse CSS support")

    def contract(self, logical, frame, terms, bound):
        """Return weighted output logical amplitudes, up to one common phase.

        The input is a common encoded qubit and frame, followed by a coherent
        sum of monomials with product ancillas. Any unprojected spectator must
        be common across terms; a relative scalar phase is retained.
        """
        logical = np.asarray(logical, dtype=complex)
        if logical.shape != (2,) or not np.all(np.isfinite(logical)):
            raise ValueError("two finite logical amplitudes are required")
        x, z = frame
        if not 0 <= x < 128 or not 0 <= z < 128:
            raise ValueError("input frame exceeds seven wires")
        if not bound.reachable or not terms:
            return np.zeros(2, dtype=complex)
        correction_x = correction_z = 0
        for k, (dx, dz) in zip(self.data_rows, self.duals, strict=True):
            if bound.input_signs[k]:
                correction_x ^= dx
                correction_z ^= dz
        sx, sy, sz = bound.logical_signs
        if sy != (sx ^ sz ^ self.y_sign):
            raise ValueError("logical signs are inconsistent with a single-qubit isometry")
        result = np.zeros(2, dtype=complex)
        anchor = terms[0].ancillas
        for term in terms:
            op = term.data
            if op.width != 7 or op.edges or len(op.linear) != 7 or not 0 <= op.flips < 128:
                raise ValueError("monomial is outside the compiled diagonal/flip family")
            if term.ancillas.shape != (len(self.ancillas), 2) or not np.allclose(
                np.sum(np.abs(term.ancillas) ** 2, axis=1), 1, atol=1e-12, rtol=0
            ):
                raise ValueError("one normalized state is required per ancilla")
            scalar = term.weight
            for k, q, bras in self.ancilla_rows:
                scalar *= bras[bound.input_signs[k]] @ term.ancillas[q]
            for q in self.unprojected:
                overlap = np.vdot(anchor[q], term.ancillas[q])
                if abs(abs(overlap) - 1) > 1e-10:
                    raise ValueError("unprojected spectator differs between coherent terms")
                scalar *= overlap
            for source, row in enumerate(self.source_basis):
                for b, coefficient in row:
                    output, phase = op.column(b ^ x)
                    phase += 4 * ((z & b).bit_count() & 1)
                    output ^= correction_x
                    phase += 4 * ((correction_z & output).bit_count() & 1)
                    result += (
                        scalar
                        * logical[source]
                        * coefficient
                        * ROOTS[phase % 8]
                        * self.decoder[:, output]
                    )
        # X^sz Z^sx maps the pullback X/Z convention to the output code axes.
        if sx:
            result[1] *= -1
        if sz:
            result = result[::-1].copy()
        return np.sqrt(bound.probability_scale) * result
