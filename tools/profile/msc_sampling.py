"""Complete MSC research sampler using static logical-block plans.

The Clifford instruments sample their input projectors and then uniform free
record bits. Terminal CSS measurements use a sparse Fourier transform. All
Pauli coordinates and parity elimination are compiled before sampling. Python
allocations make this a correctness prototype, not a production hot executor.
"""

import random
from copy import deepcopy

import numpy as np
from fold_blocks import masks
from fold_contraction import ROOTS
from msc_boundaries import Row
from msc_growth_contraction import local_pauli
from study_msc_injection import FixedHistoryPlan


def parity(bits, mask):
    return (bits & mask).bit_count() & 1


class AffineRecords:
    """Pre-eliminated binary equations with a variable right-hand side."""

    def __init__(self, events, rows):
        self.events, self.rows = tuple(events), tuple(rows)
        event_mask = sum(1 << k for k in events)
        basis = {}
        for k, row in enumerate(rows):
            key, rhs = row.records, 1 << k
            if key & ~event_mask:
                raise ValueError("record equation refers outside its instrument")
            while key:
                pivot = key.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = (key ^ (1 << pivot), rhs)
                    break
                other, value = basis[pivot]
                key ^= other | (1 << pivot)
                rhs ^= value
            if not key:
                raise ValueError("record equations are dependent")
        self.pivots = tuple((k, *basis[k]) for k in sorted(basis))
        self.free = tuple(k for k in events if k not in basis)

    def sample(self, faults, values, rng):
        rhs = sum(
            (row.evaluate(0, faults) ^ value) << k
            for k, (row, value) in enumerate(zip(self.rows, values, strict=True))
        )
        records = sum(rng.getrandbits(1) << k for k in self.free)
        for pivot, row, transform in self.pivots:
            records |= (parity(records, row) ^ parity(rhs, transform)) << pivot
        return records


def draw(weights, rng):
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if np.min(weights) < -1e-10 or not np.all(np.isfinite(weights)):
        raise ValueError("invalid quantum probabilities")
    weights = np.maximum(weights, 0)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("zero-probability sampling input")
    threshold = rng.random() * total
    cumulative = 0.0
    for k, value in enumerate(weights):
        cumulative += value
        if cumulative > threshold:
            return k, float(value / total)
    raise AssertionError("probability accumulation failed")


class SparseCode:
    """CSS coordinates with at most one coset per coherent monomial term."""

    def __init__(self, checks, width):
        self.width = width
        self.xchecks: list[int] = []
        self.zchecks: list[int] = []
        self.order = []
        for p in checks:
            x, z = masks(p)
            if (x and z) or not (x or z) or p.sign != 1:
                raise ValueError("sparse sampler requires positive CSS checks")
            family = self.xchecks if x else self.zchecks
            self.order.append((bool(x), len(family)))
            family.append(x or z)
        span = [0]
        basis = {}
        for k, x in enumerate(self.xchecks):
            span += [b ^ x for b in span]
            key, coords = x, 1 << k
            while key:
                pivot = key.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = key, coords
                    break
                row, value = basis[pivot]
                key ^= row
                coords ^= value
            if not key:
                raise ValueError("dependent X checks")
        self.coordinates = [0] * len(self.xchecks)
        for q in range(width):
            key, coords = 1 << q, 0
            for pivot in sorted(basis, reverse=True):
                if key >> pivot & 1:
                    row, value = basis[pivot]
                    key ^= row
                    coords ^= value
            for k in range(len(self.xchecks)):
                if coords >> k & 1:
                    self.coordinates[k] |= 1 << q
        self.span = np.asarray(span, dtype=np.int64)
        self.all_x = (1 << width) - 1
        if len(self.xchecks) + len(self.zchecks) != width - 1 or any(
            b.bit_count() & 1 for b in span
        ):
            raise ValueError("CSS code does not have the required logical convention")
        self.sources = np.asarray([self.span, self.span ^ self.all_x])
        self.source_coordinates = np.asarray(
            [
                [
                    sum(parity(int(b), mask) << k for k, mask in enumerate(self.coordinates))
                    for b in row
                ]
                for row in self.sources
            ]
        )
        if any(parity(self.all_x, z) for z in self.zchecks):
            raise ValueError("all-data X does not preserve code")

    def amplitudes(self, logical, frame, terms):
        """Fixed coset slots, with coefficient interference handled by equality masks."""
        x, z = frame
        tensors, labels = [], []
        roots = np.asarray(ROOTS)
        for term in terms:
            op = term.data
            inputs = self.sources ^ x
            phase = np.full(inputs.shape, op.global_phase, dtype=np.int64)
            for q, c in enumerate(op.linear):
                phase += c * ((inputs >> q) & 1)
            phase += 4 * np.asarray([[parity(int(b), z) for b in row] for row in self.sources])
            offset = x ^ op.flips
            labels.append(sum(parity(offset, mask) << k for k, mask in enumerate(self.zchecks)))
            shift = sum(parity(offset, mask) << k for k, mask in enumerate(self.coordinates))
            tensor = np.zeros((2, len(self.span)), dtype=complex)
            for source in (0, 1):
                tensor[
                    source ^ (offset.bit_count() & 1), self.source_coordinates[source] ^ shift
                ] = logical[source] * roots[phase[source] % 8] / np.sqrt(len(self.span))
            tensors.append(tensor)
        return labels, np.asarray(tensors)

    @staticmethod
    def gram(terms):
        result = np.zeros((len(terms), len(terms)), dtype=complex)
        for a, left in enumerate(terms):
            for b, right in enumerate(terms):
                result[a, b] = (
                    left.weight.conjugate()
                    * right.weight
                    * np.prod(np.sum(left.ancillas.conj() * right.ancillas, axis=1))
                )
        return result

    def norm(self, logical, frame, terms):
        labels, tensors = self.amplitudes(logical, frame, terms)
        gram = self.gram(terms)
        return float(
            sum(
                gram[a, b] * np.vdot(tensors[a], tensors[b])
                for a in range(len(terms))
                for b in range(len(terms))
                if labels[a] == labels[b]
            ).real
        )

    def syndrome_weights(self, logical, frame, terms):
        labels, tensors = self.amplitudes(logical, frame, terms)
        stride = 1
        while stride < len(self.span):
            blocks = tensors.reshape(len(terms), 2, -1, 2, stride)
            a, b = blocks[..., 0, :].copy(), blocks[..., 1, :].copy()
            blocks[..., 0, :], blocks[..., 1, :] = (a + b) / np.sqrt(2), (a - b) / np.sqrt(2)
            stride *= 2
        gram = self.gram(terms)
        weights = np.zeros((len(terms), len(self.span)))
        # Duplicate slots carry zero mass. Comparisons only select coefficient
        # interference; they do not plan a new basis or discover dependencies.
        for a in range(len(terms)):
            if labels[a] in labels[:a]:
                continue
            for b in range(len(terms)):
                for c in range(len(terms)):
                    if labels[b] == labels[a] == labels[c]:
                        weights[a] += (
                            gram[b, c] * np.sum(tensors[b].conj() * tensors[c], axis=0)
                        ).real
        return labels, weights

    def sample_syndrome(self, logical, frame, terms, rng):
        labels, weights = self.syndrome_weights(logical, frame, terms)
        selected, probability = draw(weights, rng)
        slot, xs = divmod(selected, len(self.span))
        zs = labels[slot]
        return [((xs if x else zs) >> k) & 1 for x, k in self.order], probability


class Sampler(FixedHistoryPlan):
    def __init__(self, program):
        super().__init__(program)
        injection = self.injection
        # Translate virtual Choi event indices to the original exposed events.
        rows = []
        for row in injection.constraints:
            records = sum(
                ((row.records >> virtual) & 1) << source for source, virtual in injection.event_map
            )
            rows.append(Row(records, row.faults, row.constant))
        self.injection_records = AffineRecords([s for s, _ in injection.event_map], rows)
        self.terminal_code = SparseCode(self.terminal.checks, self.terminal.width)
        if self.growth is not None:
            self.growth_records = AffineRecords(
                self.instrument.events, self.instrument.input_rows + self.instrument.constraints
            )
            if len(self.growth_records.free) != self.instrument.random_power:
                raise ValueError("growth free-record count differs")
            boundary = self.boundaries[0]
            self.growth_code = SparseCode(
                [local_pauli(p, boundary.data) for p in boundary.checks[:6]], 7
            )
            self.projectors = []
            for k in self.growth.data_rows:
                p = local_pauli(self.instrument.input_paulis[k], self.growth.data)
                x, z = masks(p)
                phase = 1j ** (x & z).bit_count()
                self.projectors.append(
                    (
                        k,
                        np.arange(128) ^ x,
                        np.asarray([phase * (-1) ** parity(b, z) for b in range(128)]),
                    )
                )

        assigned = list(self.injection_records.events) + list(self.terminal.events)
        payloads = [self.terminal_payload]
        if self.growth is not None:
            assigned.extend(self.growth_records.events)
            payloads.append(self.growth_payload)
        for payload in payloads:
            for kind, step in payload.steps:
                if kind == "measure":
                    assigned.append(step[1])
                elif kind == "gadget":
                    assigned.append(step[0].event)
                    if step[0].hidden is not None:
                        assigned.append(step[0].hidden)
        if sorted(assigned) != list(range(len(program.measurements))):
            raise ValueError("every true event must have exactly one sampling owner")

    def payload_sample(self, plan, code, logical, history, outcomes, rng):
        frame, payload = plan.start(history, outcomes)
        faults = plan.bits.encode(self.program, history)
        probability = 1.0
        for kind, step in plan.steps:
            if kind == "fault":
                plan.fault(payload, step, faults)
            elif kind == "gadget":
                candidates = [plan.expand(payload, step, faults, bit) for bit in (0, 1)]
                weights = [code.norm(logical, frame, candidate) for candidate in candidates]
                bit, chance = draw(weights, rng)
                payload = candidates[bit]
                probability *= chance
                for term in payload:
                    term.weight /= np.sqrt(weights[bit])
                gadget = step[0]
                outcomes[gadget.event] = bit
                if gadget.hidden is not None:
                    outcomes[gadget.hidden] = bit ^ parity(faults, gadget.reset_flip)
            else:
                j, event = step
                candidates = [deepcopy(payload), deepcopy(payload)]
                for bit in (0, 1):
                    plan.measure(candidates[bit], j, bit)
                weights = [code.norm(logical, frame, candidate) for candidate in candidates]
                bit, chance = draw(weights, rng)
                payload = candidates[bit]
                probability *= chance
                outcomes[event] = bit
                for term in payload:
                    term.weight /= np.sqrt(weights[bit])
        return frame, payload, probability

    def growth_signs(self, logical, frame, payload, rng):
        assert self.growth is not None
        vectors = np.zeros((len(payload), 128), dtype=complex)
        x, z = frame
        for k, term in enumerate(payload):
            for source, row in enumerate(self.growth.source_basis):
                for b, coefficient in row:
                    output, phase = term.data.column(b ^ x)
                    vectors[k, output] += (
                        logical[source] * coefficient * ROOTS[(phase + 4 * parity(b, z)) % 8]
                    )
        ancillas = np.asarray([term.ancillas.copy() for term in payload])
        weights = np.asarray([term.weight for term in payload])

        def norm(data, spectators):
            return float(
                sum(
                    weights[a].conjugate()
                    * weights[b]
                    * np.vdot(data[a], data[b])
                    * np.prod(np.sum(spectators[a].conj() * spectators[b], axis=1))
                    for a in range(len(payload))
                    for b in range(len(payload))
                ).real
            )

        signs = [0] * len(self.instrument.input_rows)
        probability = 1.0
        for k, q, bras in self.growth.ancilla_rows:
            candidates = []
            for bit in (0, 1):
                projected = ancillas.copy()
                projected[:, q] = (ancillas[:, q] @ bras[bit])[:, None] * bras[bit].conj()
                candidates.append(projected)
            bit, chance = draw([norm(vectors, c) for c in candidates], rng)
            signs[k] = bit
            probability *= chance
            ancillas = candidates[bit]
        for k, targets, phases in self.projectors:
            action = np.zeros_like(vectors)
            action[:, targets] = vectors * phases
            candidates = [(vectors + action) / 2, (vectors - action) / 2]
            bit, chance = draw([norm(c, ancillas) for c in candidates], rng)
            signs[k] = bit
            probability *= chance
            vectors = candidates[bit]
        return signs, probability

    def sample(self, seed, history=None):
        rng = random.Random(seed)
        if history is None:
            history = {
                k: rng.choice(site.choices)
                for k, site in enumerate(self.program.sites)
                if rng.random() < site.probability
            }
        self.program.validate_history(history)
        outcomes = [0] * len(self.program.measurements)
        injection = self.injection
        virtual_faults = injection.instrument.bits.encode(
            injection.virtual,
            {injection.site_map[k]: v for k, v in history.items() if k in injection.site_map},
        )
        record_bits = self.injection_records.sample(
            virtual_faults, [0] * len(self.injection_records.rows), rng
        )
        for k in self.injection_records.events:
            outcomes[k] = (record_bits >> k) & 1
        logical = injection.evaluate(outcomes, history)
        probability = float(np.vdot(logical, logical).real)
        logical /= np.sqrt(probability)
        if self.growth is not None:
            frame, payload, chance = self.payload_sample(
                self.growth_payload, self.growth_code, logical, history, outcomes, rng
            )
            probability *= chance
            signs, chance = self.growth_signs(logical, frame, payload, rng)
            faults = self.instrument.bits.encode(self.program, history)
            records = self.growth_records.sample(
                faults, signs + [0] * len(self.instrument.constraints), rng
            )
            for k in self.growth_records.events:
                outcomes[k] = (records >> k) & 1
            logical = self.growth.contract(
                logical, frame, payload, self.instrument.bind(outcomes, history)
            )
            branch_weight = float(np.vdot(logical, logical).real)
            expected = chance * 2.0 ** -len(self.growth_records.free)
            if abs(branch_weight - expected) > 1e-10 * expected:
                raise AssertionError(("sampled growth weight differs", branch_weight, expected))
            probability *= expected
            logical /= np.sqrt(branch_weight)
        frame, payload, chance = self.payload_sample(
            self.terminal_payload, self.terminal_code, logical, history, outcomes, rng
        )
        probability *= chance
        syndrome, chance = self.terminal_code.sample_syndrome(logical, frame, payload, rng)
        for k, bit in zip(self.terminal.events, syndrome, strict=True):
            outcomes[k] = bit
        logical = self.terminal.contract(logical, frame, payload, syndrome)
        weight = float(np.vdot(logical, logical).real)
        if abs(weight - chance) > 1e-10 * chance:
            raise AssertionError(("sampled terminal weight differs", weight, chance))
        probability *= chance
        logical /= np.sqrt(weight)
        return {
            "faults": history,
            "outcomes": outcomes,
            "probability": probability,
            "logical": logical,
            **self.program.outputs(outcomes, history),
        }
