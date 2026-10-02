"""Study symbolic Pauli faults with fixed samplers and bounded branch banks.

The symbolic forms are compact even when there are many fault selectors. A
branch bank is an exact small-model experiment: every leaf is compiled before
sampling, and the existing Clifft executor handles each immutable leaf. Its
exponential leaf cap is intentional; it is not a proposed production backend.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from tools.prototypes.phase_core_study import (
    Check,
    compress_phase,
    pauli_derivative,
    stabilizer_kernel,
    trace,
)
from tools.prototypes.phase_polynomial import (
    Polynomial,
    _bits,
    _coordinates,
    _cz_phase,
    _encode,
    _parity_phase,
    _synthesize_phase,
)


@dataclass(frozen=True)
class FaultSite:
    after_line: int
    pauli: str
    qubit: int
    probability: float


@dataclass(frozen=True)
class Mod4Form:
    """A modulo-four sum of weighted affine fault parities."""

    constant: int = 0
    terms: tuple[tuple[int, int], ...] = ()

    def evaluate(self, pattern: int) -> int:
        return (
            self.constant
            + sum(weight * ((mask & pattern).bit_count() % 2) for mask, weight in self.terms)
        ) % 4

    def odd_mask(self) -> int:
        out = 0
        for mask, weight in self.terms:
            if weight % 2:
                out ^= mask
        return out


def _add_form(terms: dict[int, int], mask: int, weight: int) -> None:
    if mask and weight % 4:
        value = (terms.get(mask, 0) + weight) % 4
        if value:
            terms[mask] = value
        else:
            terms.pop(mask, None)


def _form(terms: dict[int, int], constant: int = 0) -> Mod4Form:
    return Mod4Form(constant % 4, tuple(sorted(terms.items())))


def _subtract(left: Mod4Form, right: Mod4Form) -> Mod4Form:
    out = dict(left.terms)
    for mask, weight in right.terms:
        _add_form(out, mask, -weight)
    return _form(out, left.constant - right.constant)


def _span_basis(rows: list[int]) -> list[int]:
    pivots: dict[int, int] = {}
    for vector in rows:
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = vector
                break
            vector ^= pivots[pivot]
    return list(pivots.values())


def _rank(rows: list[int]) -> int:
    return len(_span_basis(rows))


@dataclass(frozen=True)
class MeasurementForm:
    x: int
    z: int
    z_selectors: tuple[int, ...]
    sign: Mod4Form

    def evaluate(self, pattern: int) -> tuple[int, int, int]:
        z = self.z
        for j, selector in enumerate(self.z_selectors):
            if (selector & pattern).bit_count() % 2:
                z ^= 1 << j
        phase = self.sign.evaluate(pattern)
        assert phase in (0, 2)
        return self.x, z, phase // 2


@dataclass(frozen=True)
class SymbolicFrame:
    source: str
    sites: tuple[FaultSite, ...]
    width: int
    core_width: int
    physical_qubits: int
    preparation: tuple[str, ...]
    base_s: tuple[int, ...]
    s: tuple[Mod4Form, ...]
    base_cz: tuple[int, ...]
    cz: tuple[tuple[int, int], ...]
    offsets: tuple[int, ...]
    encoding: tuple[str, ...]
    records: tuple[str | MeasurementForm, ...]

    def correction_phase(self, pattern: int) -> Polynomial:
        out = {}
        for j, form in enumerate(self.s):
            coefficient = 2 * form.evaluate(pattern)
            if coefficient:
                out[1 << j] = coefficient
        for mask, selector in self.cz:
            if (selector & pattern).bit_count() % 2:
                out[mask] = 4
        return out

    def render(self, pattern: int) -> str:
        """Bind coefficients into a source circuit during branch-bank construction."""
        if not 0 <= pattern < 1 << len(self.sites):
            raise ValueError("fault pattern is outside this model")
        output = list(self.preparation)
        names = {1: "S", 2: "Z", 3: "S_DAG"}
        for j, form in enumerate(self.s):
            coefficient = (self.base_s[j] + form.evaluate(pattern)) % 4
            if coefficient:
                output.append(f"{names[coefficient]} {j}")
        selectors = dict(self.cz)
        for mask in sorted(set(self.base_cz) | selectors.keys()):
            active = int(mask in self.base_cz)
            active ^= (selectors.get(mask, 0) & pattern).bit_count() % 2
            if active:
                output.append("CZ " + " ".join(str(bit.bit_length() - 1) for bit in _bits(mask)))
        encoded = False
        for record in self.records:
            if isinstance(record, MeasurementForm):
                x, z, negative = record.evaluate(pattern)
                factors = []
                for bit in _bits(x | z):
                    axis = "Y" if x & bit and z & bit else "X" if x & bit else "Z"
                    factors.append(f"{axis}{bit.bit_length() - 1}")
                output.append("MPP " + ("!" if negative else "") + "*".join(factors))
            else:
                if record.startswith("MX") and not encoded:
                    output.extend(self.encoding)
                    output.extend(
                        f"X {q}"
                        for q, selector in enumerate(self.offsets)
                        if (selector & pattern).bit_count() % 2
                    )
                    encoded = True
                output.append(record)
        if not encoded:
            output.extend(self.encoding)
            output.extend(
                f"X {q}"
                for q, selector in enumerate(self.offsets)
                if (selector & pattern).bit_count() % 2
            )
        return "\n".join(output) + "\n"

    def original(self, pattern: int | None = None) -> str:
        """Render either the original stochastic model or one selected fault path."""
        insertions: dict[int, list[str]] = {}
        for j, site in enumerate(self.sites):
            if pattern is None:
                gate = f"{site.pauli}_ERROR({site.probability:.17g})"
            elif pattern & (1 << j):
                gate = site.pauli
            else:
                continue
            insertions.setdefault(site.after_line, []).append(f"{gate} {site.qubit}")
        lines = []
        for j, line in enumerate(self.source.splitlines()):
            lines.append(line)
            lines.extend(insertions.get(j, ()))
        return "\n".join(lines) + "\n"

    def geometry_rank(self) -> int:
        return _rank([form.odd_mask() for form in self.s] + [mask for _, mask in self.cz])

    def fixed_frame_core_width(self) -> int:
        """Common Pauli-body bound for the unmeasured preparation, not sampling."""
        constraints: dict[tuple[int, int], int] = {}
        for j, form in enumerate(self.s):
            if j >= self.core_width:
                for selector in _bits(form.odd_mask()):
                    constraints[j, selector] = constraints.get((j, selector), 0) ^ (1 << j)
        for mask, selectors in self.cz:
            left, right = (bit.bit_length() - 1 for bit in _bits(mask))
            for selector in _bits(selectors):
                if right >= self.core_width:
                    constraints[left, selector] = constraints.get((left, selector), 0) ^ (
                        1 << right
                    )
                if left >= self.core_width:
                    constraints[right, selector] = constraints.get((right, selector), 0) ^ (
                        1 << left
                    )
        return self.core_width + _rank(list(constraints.values()))

    def stats(self) -> dict[str, int]:
        selectors = [mask for form in self.s for mask, _ in form.terms]
        selectors.extend(mask for _, mask in self.cz)
        selectors.extend(self.offsets)
        for record in self.records:
            if isinstance(record, MeasurementForm):
                selectors.extend(record.z_selectors)
                selectors.extend(mask for mask, _ in record.sign.terms)
        return {
            "fault_selectors": len(self.sites),
            "prepared_variables": self.width,
            "core_width": self.core_width,
            "fixed_frame_prepared_core_width": self.fixed_frame_core_width(),
            "s_parity_terms": sum(len(form.terms) for form in self.s),
            "cz_forms": len(self.cz),
            "geometry_rank": self.geometry_rank(),
            "measurement_forms": sum(isinstance(r, MeasurementForm) for r in self.records),
            "selector_words_64": sum((mask.bit_length() + 63) // 64 for mask in selectors),
        }


def compile_frame(source: str, sites: list[FaultSite], *, max_variables: int = 32) -> SymbolicFrame:
    ideal = trace(source)
    if ideal.width > max_variables:
        raise ValueError("prepared-variable cap exceeded")
    magic, clifford, coordinate_encoding = compress_phase(ideal.phase, ideal.width)
    n, width = len(ideal.rows), ideal.width
    lines = source.splitlines()
    h_lines = [j for j, line in enumerate(lines) if line.strip().startswith("H ")]
    mx_lines = [j for j, line in enumerate(lines) if line.strip().startswith("MX ")]
    last_h, first_mx = max(h_lines, default=-1), min(mx_lines, default=len(lines))
    at_line: dict[int, list[tuple[int, FaultSite]]] = {}
    for j, site in enumerate(sites):
        if site.pauli not in ("X", "Y", "Z") or not 0 <= site.qubit < n:
            raise ValueError("invalid selected Pauli fault site")
        if not math.isfinite(site.probability) or not 0 <= site.probability <= 1:
            raise ValueError("invalid fault probability")
        if not last_h <= site.after_line < first_mx:
            raise ValueError("fault sites must follow preparation and precede terminal MX")
        at_line.setdefault(site.after_line, []).append((1 << j, site))
    coordinates = [1 << j for j in range(width)]
    for instruction in coordinate_encoding:
        gate, a_text, b_text = instruction.split()
        left, right = int(a_text), int(b_text)
        if gate == "CX":
            coordinates[right] ^= coordinates[left]
        else:
            coordinates[left], coordinates[right] = coordinates[right], coordinates[left]
    rows = [0] * n
    offsets = [0] * n
    s_terms: list[dict[int, int]] = [{} for _ in range(width)]
    cz_terms: dict[int, int] = {}
    poly: Polynomial = {}
    prepared = 0
    pending: list[str | tuple[Check, tuple[Mod4Form, ...], dict[int, int]]] = []
    terminal = False
    for line_index, line in enumerate(lines):
        clean = line.split("#", 1)[0].strip()
        if clean:
            gate, *args = clean.split()
            if gate.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
                pending.append(clean)
            elif gate == "MPP":
                if terminal:
                    raise ValueError("checks after terminal MX are outside this study")
                for product in args:
                    physical = sum(1 << int(q[1:]) for q in product.split("*"))
                    flip = _coordinates(rows, width, physical)
                    if flip is None:
                        raise ValueError("check leaves the prepared subspace")
                    pending.append(
                        (
                            Check(clean, flip, dict(poly)),
                            tuple(_form(t) for t in s_terms),
                            dict(cz_terms),
                        )
                    )
            elif gate == "MX":
                terminal = True
                pending.append(clean)
            else:
                targets = list(map(int, args))
                if gate == "H":
                    for q in targets:
                        rows[q] = coordinates[prepared]
                        prepared += 1
                elif gate in ("CX", "CNOT", "SWAP", "CZ"):
                    for a, b in zip(targets[::2], targets[1::2]):
                        if gate in ("CX", "CNOT"):
                            rows[b] ^= rows[a]
                            offsets[b] ^= offsets[a]
                        elif gate == "SWAP":
                            rows[a], rows[b] = rows[b], rows[a]
                            offsets[a], offsets[b] = offsets[b], offsets[a]
                        else:
                            _cz_phase(poly, rows[a], rows[b])
                            for bit in _bits(rows[b]):
                                _add_form(s_terms[bit.bit_length() - 1], offsets[a], 2)
                            for bit in _bits(rows[a]):
                                _add_form(s_terms[bit.bit_length() - 1], offsets[b], 2)
                elif gate in ("T", "T_DAG", "S", "S_DAG", "Z"):
                    c = {"T": 1, "T_DAG": 7, "S": 2, "S_DAG": 6, "Z": 4}[gate]
                    for q in targets:
                        _parity_phase(poly, rows[q], c)
                        bits = _bits(rows[q])
                        for bit in bits:
                            _add_form(s_terms[bit.bit_length() - 1], offsets[q], -c)
                        if c % 2:
                            for a, b in itertools.combinations(bits, 2):
                                cz_terms[a | b] = cz_terms.get(a | b, 0) ^ offsets[q]
                elif gate != "I":
                    raise ValueError(f"unsupported symbolic input gate {gate}")
        for selector, site in at_line.get(line_index, ()):
            if site.pauli in ("Y", "Z"):
                for bit in _bits(rows[site.qubit]):
                    _add_form(s_terms[bit.bit_length() - 1], selector, 2)
            if site.pauli in ("X", "Y"):
                offsets[site.qubit] ^= selector
    final_s = tuple(_form(t) for t in s_terms)
    final_cz = {mask: selector for mask, selector in cz_terms.items() if selector}
    records: list[str | MeasurementForm] = []
    for record in pending:
        if isinstance(record, str):
            records.append(record)
            continue
        check, prefix_s, prefix_cz = record
        suffix = dict(poly)
        for mask, coefficient in check.prefix.items():
            value = (suffix.get(mask, 0) - coefficient) % 8
            if value:
                suffix[mask] = value
            else:
                suffix.pop(mask, None)
        derivative = pauli_derivative(suffix, check.flip)
        if derivative is None:
            raise ValueError("non-Pauli ideal check suffix")
        constant, base_z = derivative
        delta_s = [_subtract(final, prefix) for final, prefix in zip(final_s, prefix_s)]
        delta_cz = {
            mask: final_cz.get(mask, 0) ^ prefix_cz.get(mask, 0)
            for mask in final_cz.keys() | prefix_cz.keys()
        }
        z_selectors = [0] * width
        sign_terms: dict[int, int] = {}
        for j, form in enumerate(delta_s):
            if check.flip & (1 << j):
                z_selectors[j] ^= form.odd_mask()
                for mask, weight in form.terms:
                    _add_form(sign_terms, mask, weight)
        for mask, selector in delta_cz.items():
            left, right = _bits(mask)
            if check.flip & left:
                z_selectors[right.bit_length() - 1] ^= selector
            if check.flip & right:
                z_selectors[left.bit_length() - 1] ^= selector
            if check.flip & mask == mask:
                _add_form(sign_terms, selector, 2)
        sign_constant = constant // 2
        for j, selector in enumerate(z_selectors):
            if check.flip & (1 << j):
                if base_z & (1 << j):
                    sign_constant -= 1
                    _add_form(sign_terms, selector, 1)
                else:
                    _add_form(sign_terms, selector, -1)
        records.append(
            MeasurementForm(
                check.flip, base_z, tuple(z_selectors), _form(sign_terms, sign_constant)
            )
        )
    # The phase pieces overlap on linear and quadratic terms, so compare their sum.
    reconstructed = dict(clifford)
    for mask, coefficient in magic.items():
        reconstructed[mask] = (reconstructed.get(mask, 0) + coefficient) % 8
    assert poly == reconstructed
    base_s = tuple(clifford.get(1 << j, 0) // 2 for j in range(width))
    base_cz = tuple(
        sorted(mask for mask, c in clifford.items() if mask.bit_count() == 2 and c == 4)
    )
    preparation = [f"I {n - 1}"] if n else []
    if width:
        preparation.append("H " + " ".join(map(str, range(width))))
    preparation.extend(_synthesize_phase(magic, list(range(width))))
    return SymbolicFrame(
        source,
        tuple(sites),
        width,
        max((mask.bit_length() for mask in magic), default=0),
        n,
        tuple(preparation),
        base_s,
        final_s,
        base_cz,
        tuple(sorted(final_cz.items())),
        tuple(offsets),
        tuple(_encode(rows, [1 << j for j in range(width)])),
        tuple(records),
    )


class BranchBank:
    """Exact small-model experiment with all native programs compiled up front."""

    def __init__(self, frame: SymbolicFrame, mask: list[int], *, max_patterns: int = 256):
        import clifft

        patterns = 1 << len(frame.sites)
        if patterns > max_patterns:
            raise ValueError("exact branch bank exceeds its leaf cap")
        self.frame = frame
        self.programs: list[Any] = []
        cached: dict[str, Any] = {}
        for pattern in range(patterns):
            source = frame.render(pattern)
            if source not in cached:
                manager = clifft.default_hir_pass_manager()
                manager.add(clifft.ActiveWidthSchedulePass())
                cached[source] = clifft.compile(source, postselection_mask=mask, hir_passes=manager)
            self.programs.append(cached[source])
        self.unique_programs = len(cached)
        self.probabilities = np.array([site.probability for site in frame.sites])
        self.num_observables = self.programs[0].num_observables
        self.num_measurements = self.programs[0].num_measurements
        self.postselected = any(mask)

    def _counts(self, shots: int, seed: int) -> np.ndarray:
        if shots < 0:
            raise ValueError("shots must be nonnegative")
        rng = np.random.default_rng(seed)
        faults = rng.random((shots, len(self.frame.sites))) < self.probabilities
        patterns = faults.astype(np.int64) @ (1 << np.arange(len(self.frame.sites)))
        return np.bincount(patterns, minlength=len(self.programs))

    def sample_survivors(self, shots: int, *, seed: int, batch_size: int = 1) -> dict[str, Any]:
        import clifft

        counts = self._counts(shots, seed)
        passed = 0
        ones = np.zeros(self.num_observables, dtype=np.int64)
        for pattern, count in enumerate(counts):
            if count:
                result = clifft.sample_survivors(
                    self.programs[pattern],
                    shots=int(count),
                    seed=seed + pattern + 1,
                    threads=1,
                    batch_size=batch_size,
                )
                passed += result.passed_shots
                ones += result.observable_ones.astype(np.int64)
        return {"total_shots": shots, "passed_shots": passed, "observable_ones": ones}

    def sample_measurements(self, shots: int, *, seed: int) -> np.ndarray:
        import clifft

        if self.postselected:
            raise ValueError("sample_measurements requires a bank without postselection")
        counts = self._counts(shots, seed)
        records = []
        for pattern, count in enumerate(counts):
            if count:
                result = clifft.sample(
                    self.programs[pattern],
                    shots=int(count),
                    seed=seed + pattern + 1,
                    threads=1,
                    batch_size=1,
                )
                records.append(result.measurements)
        return (
            np.concatenate(records)
            if records
            else np.empty((0, self.num_measurements), dtype=np.uint8)
        )


def _half_affine(form: Mod4Form, variables: int) -> tuple[int, int] | None:
    """Recognize an even modulo-four form as twice an affine Boolean form."""
    if form.constant % 2 or form.odd_mask():
        return None
    linear = [0] * variables
    quadratic: dict[int, int] = {}
    for mask, weight in form.terms:
        for bit in _bits(mask):
            j = bit.bit_length() - 1
            linear[j] = (linear[j] + weight) % 4
            if weight % 2:
                quadratic[j] = quadratic.get(j, 0) ^ (mask ^ bit)
    if any(quadratic.values()):
        return None
    assert all(coefficient % 2 == 0 for coefficient in linear)
    return form.constant // 2, sum(
        1 << j for j, coefficient in enumerate(linear) if coefficient == 2
    )


def affine_record_flips(frame: SymbolicFrame) -> tuple[int, ...] | None:
    """Find a fixed-measurement Pauli orbit, without enumerating fault patterns.

    For independent T states, X after T reproduces S_DAG on |+>.
    Ancilla corrections must be Paulis and all measurement axes must be fixed.
    Quadratic fault-dependent Pauli signs are rejected, not approximated.
    """
    variables = len(frame.sites)
    linear_magic = all(line.split()[0] in ("I", "H", "T") for line in frame.preparation)
    magic_targets = (
        {int(line.split()[1]) for line in frame.preparation if line.startswith("T ")}
        if linear_magic
        else set()
    )
    x = [0] * frame.physical_qubits
    z = [0] * frame.physical_qubits
    for j, form in enumerate(frame.s):
        terms = dict(form.terms)
        if form.odd_mask():
            if j not in magic_targets:
                return None
            x[j] = form.odd_mask()
            _add_form(terms, x[j], 1)
        reduced = _half_affine(_form(terms, form.constant), variables)
        if reduced is None:
            return None
        constant, z[j] = reduced
        assert constant == 0
    if frame.cz:
        return None
    for j, coefficient in enumerate(frame.base_s):
        if coefficient % 2:
            z[j] ^= x[j]
    for mask in frame.base_cz:
        left, right = (bit.bit_length() - 1 for bit in _bits(mask))
        z[left] ^= x[right]
        z[right] ^= x[left]
    flips = []
    encoded = False
    for record in frame.records:
        if isinstance(record, MeasurementForm):
            if any(record.z_selectors):
                return None
            sign = _half_affine(record.sign, variables)
            if sign is None:
                return None
            _, flip = sign
            for bit in _bits(record.x):
                flip ^= z[bit.bit_length() - 1]
            for bit in _bits(record.z):
                flip ^= x[bit.bit_length() - 1]
            flips.append(flip)
        elif record.startswith("MX"):
            if not encoded:
                for instruction in frame.encoding:
                    gate, a, b = instruction.split()
                    left, right = int(a), int(b)
                    if gate == "CX":
                        x[right] ^= x[left]
                        z[left] ^= z[right]
                    else:
                        x[left], x[right] = x[right], x[left]
                        z[left], z[right] = z[right], z[left]
                encoded = True
            flips.extend(z[int(q)] for q in record.split()[1:])
    return tuple(flips)


class AffineRecordSampler:
    """Research wrapper using one fixed native sampler and precomputed record flips.

    The NumPy fault draw and parity matrices are intentionally outside native
    hot dispatch. This tests the representation, not an optimized Python API.
    """

    def __init__(self, frame: SymbolicFrame, mask: list[int]):
        import clifft

        flips = affine_record_flips(frame)
        if flips is None:
            raise ValueError("fault model is outside the fixed Pauli-orbit representation")
        manager = clifft.default_hir_pass_manager()
        manager.add(clifft.ActiveWidthSchedulePass())
        self.program = clifft.compile(frame.render(0), hir_passes=manager)
        self.probabilities = np.array([site.probability for site in frame.sites])
        self.flip_matrix = np.array(
            [[(selector >> j) & 1 for selector in flips] for j in range(len(frame.sites))],
            dtype=np.uint8,
        ).reshape(len(frame.sites), len(flips))
        self.mask = np.array(mask, dtype=np.bool_)
        if len(mask) != self.program.num_detectors:
            raise ValueError("postselection mask does not match detector count")
        detectors: list[list[int]] = []
        observables: list[list[int]] = [[] for _ in range(self.program.num_observables)]
        records = 0
        for record in frame.records:
            if isinstance(record, MeasurementForm):
                records += 1
            elif record.startswith("MX"):
                records += len(record.split()) - 1
            elif record.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
                gate, *args = record.split()
                targets = [records + int(arg[4:-1]) for arg in args]
                if gate.startswith("DETECTOR"):
                    detectors.append(targets)
                else:
                    observable = int(gate.split("(")[1][:-1])
                    observables[observable].extend(targets)
        assert records == self.program.num_measurements == len(flips)
        self.detectors = np.zeros((records, len(detectors)), dtype=np.uint8)
        self.observables = np.zeros((records, len(observables)), dtype=np.uint8)
        for j, targets in enumerate(detectors):
            for target in targets:
                self.detectors[target, j] ^= 1
        for j, targets in enumerate(observables):
            for target in targets:
                self.observables[target, j] ^= 1

    def _samples(self, shots: int, seed: int) -> Iterator[np.ndarray]:
        import clifft

        if shots < 0:
            raise ValueError("shots must be nonnegative")
        rng = np.random.default_rng(seed)
        chunk = max(1, min(4096, 4000000 // max(len(self.probabilities), 1)))
        for start in range(0, shots, chunk):
            count = min(chunk, shots - start)
            faults = (rng.random((count, len(self.probabilities))) < self.probabilities).astype(
                np.uint8
            )
            result = clifft.sample(
                self.program, shots=count, seed=seed + start + 1, threads=1, batch_size=1
            )
            yield np.asarray(
                result.measurements ^ ((faults @ self.flip_matrix) & 1), dtype=np.uint8
            )

    def sample_measurements(self, shots: int, *, seed: int) -> np.ndarray:
        records = list(self._samples(shots, seed))
        return (
            np.concatenate(records)
            if records
            else np.empty((0, self.program.num_measurements), dtype=np.uint8)
        )

    def sample_survivors(self, shots: int, *, seed: int) -> dict[str, Any]:
        passed_count = 0
        ones = np.zeros(self.program.num_observables, dtype=np.int64)
        for records in self._samples(shots, seed):
            detectors = (records @ self.detectors) & 1
            passed = ~np.any(detectors[:, self.mask], axis=1)
            observables = (records @ self.observables) & 1
            passed_count += int(np.count_nonzero(passed))
            ones += observables[passed].sum(axis=0, dtype=np.int64)
        return {"total_shots": shots, "passed_shots": passed_count, "observable_ones": ones}


def gate_noise_geometry_stats(source: str) -> dict[str, int]:
    """Rank all X/Y fault influences after each unitary gate target, offline.

    Reverse propagation carries the effect on the diagonal Clifford matrix,
    not a quantum state. Z faults change only Pauli signs. Measurements and
    reset gadgets are not noisy in this simplified gate-target model.
    """
    result = trace(source)
    width = result.width
    pair_index = {
        (i, j): k for k, (i, j) in enumerate((i, j) for i in range(width) for j in range(i, width))
    }
    rows = [0] * len(result.rows)
    prepared = 0
    events: list[tuple[str, list[int], list[int]]] = []
    for line in source.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        gate, *args = clean.split()
        if gate.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")) or gate in ("MPP", "MX", "I"):
            continue
        targets = list(map(int, args))
        if gate == "H":
            for q in targets:
                rows[q] = 1 << prepared
                prepared += 1
        elif gate in ("CX", "CNOT", "SWAP"):
            for left, right in zip(targets[::2], targets[1::2]):
                if gate == "SWAP":
                    rows[left], rows[right] = rows[right], rows[left]
                else:
                    rows[right] ^= rows[left]
        if gate in ("T", "T_DAG"):
            geometry = []
            for q in targets:
                bits = [bit.bit_length() - 1 for bit in _bits(rows[q])]
                geometry.append(sum(1 << pair_index[a, b] for a in bits for b in bits if a <= b))
        else:
            geometry = []
        events.append((gate, targets, geometry))
    influence = [0] * len(rows)
    columns: list[int] = []
    for gate, targets, geometry in reversed(events):
        if gate in ("CX", "CNOT", "SWAP"):
            for left, right in reversed(list(zip(targets[::2], targets[1::2]))):
                columns.extend((influence[left], influence[right]))
                if gate == "SWAP":
                    influence[left], influence[right] = influence[right], influence[left]
                else:
                    influence[left] ^= influence[right]
        elif gate in ("T", "T_DAG"):
            for q, effect in reversed(list(zip(targets, geometry))):
                columns.append(influence[q])
                influence[q] ^= effect
        else:
            columns.extend(influence[q] for q in targets)
    basis = _span_basis(columns)
    kernel = stabilizer_kernel(result.phase, width)
    projected = []
    for effect in basis:
        matrix_rows = [0] * width
        for (i, j), index in pair_index.items():
            if effect & (1 << index):
                matrix_rows[i] ^= 1 << j
                if i != j:
                    matrix_rows[j] ^= 1 << i
        for row in matrix_rows:
            projected.append(
                sum(
                    ((row & translation).bit_count() % 2) << j
                    for j, translation in enumerate(kernel)
                )
            )
    return {
        "geometry_rank": len(basis),
        "fixed_frame_prepared_core_width": width - len(kernel) + _rank(projected),
    }


def gate_noise_geometry_rank(source: str) -> int:
    return gate_noise_geometry_stats(source)["geometry_rank"]
