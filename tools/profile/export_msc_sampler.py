"""Serialize static MSC research plans for the standalone native sampler."""

import argparse
from pathlib import Path

from msc_boundaries import Row
from msc_sampling import Sampler
from study_msc_protocol import load


class Writer:
    def __init__(self):
        self.tokens = []

    def put(self, *values):
        self.tokens.extend(map(str, values))

    def vector(self, values, write=None):
        self.put(len(values))
        for value in values:
            (write or self.put)(value)

    def mask(self, value):
        words = [
            (k, (value >> (64 * k)) & ((1 << 64) - 1))
            for k in range((value.bit_length() + 63) // 64)
            if (value >> (64 * k)) & ((1 << 64) - 1)
        ]
        self.vector(words, lambda pair: self.put(*pair))

    def row(self, row):
        self.mask(row.records)
        self.mask(row.faults)
        self.put(row.constant)

    def complex(self, z):
        self.put(float(z.real), float(z.imag))


def translate_injection(plan, row):
    injection = plan.injection
    inverse = {virtual: original for original, virtual in injection.site_map.items()}
    fault_mask = 0
    for (site, q, axis), slot in injection.instrument.bits.slots.items():
        if row.faults >> slot & 1:
            fault_mask ^= plan.terminal_payload.bits.bit(inverse[site], q, axis)
    records = sum(((row.records >> target) & 1) << source for source, target in injection.event_map)
    return Row(records, fault_mask, row.constant)


def translate_boundary(plan, row):
    records, faults = 0, row.faults
    for k, event in enumerate(plan.program.records):
        if row.records >> k & 1:
            records ^= 1 << event
            measurement = plan.program.measurements[event]
            site = plan.program.site_at.get((measurement.line, measurement.offset))
            if site is not None:
                faults ^= plan.terminal_payload.bits.bit(site, -1, "readout")
    return Row(records, faults, row.constant)


def affine(w, records, transform=lambda row: row):
    w.vector(records.free)
    w.vector(records.events)
    w.put(len(records.pivots))
    for pivot, mask, rhs in records.pivots:
        row = Row(records=mask)
        for k, source in enumerate(records.rows):
            if rhs >> k & 1:
                source = transform(source)
                row = Row(row.records, row.faults ^ source.faults, row.constant ^ source.constant)
        w.put(pivot)
        w.row(row)
        w.mask(rhs)


def code(w, plan):
    w.put(plan.width)
    for values in (plan.xchecks, plan.zchecks, plan.coordinates, plan.span.tolist()):
        w.vector(values)
    w.vector(plan.order, lambda pair: w.put(int(pair[0]), pair[1]))
    w.vector(plan.source_coordinates.reshape(-1).tolist())


def payload(w, sampler, plan):
    w.put(plan.width, len(plan.ancillas))
    w.vector(plan.boundary.rows, lambda row: w.row(translate_boundary(sampler, row)))
    w.vector(plan.boundary.duals, lambda pair: w.put(*pair))
    w.vector(plan.initial, lambda item: w.put(*item))
    w.put(len(plan.steps))
    for kind, step in plan.steps:
        w.put({"fault": 0, "measure": 1, "gadget": 2}[kind])
        if kind == "fault":
            data, j, x, z = step
            w.put(int(data), j, x.bit_length() - 1, z.bit_length() - 1)
        elif kind == "measure":
            w.put(*step)
        else:
            gadget, dm, am = step
            w.put(
                gadget.event,
                -1 if gadget.hidden is None else gadget.hidden,
                gadget.outcome_bit.bit_length() - 1,
            )
            w.mask(gadget.reset_flip)
            w.vector(dm)
            w.vector(am, lambda pair: w.put(*pair))
            for flips, phase, linear in gadget.branches:
                w.vector(flips, w.mask)
                for polynomial in (phase, *linear):
                    w.put(len(polynomial.terms))
                    for (a, b), c in polynomial.terms.items():
                        w.put(c)
                        w.mask(a)
                        w.mask(b)


def export(plan, path):
    w = Writer()
    program, bits = plan.program, plan.terminal_payload.bits
    w.put(len(program.measurements), len(bits.slots), int(plan.growth is not None))
    w.put(len(program.sites))
    for k, site in enumerate(program.sites):
        w.put(site.probability, len(site.choices))
        for choice in site.choices:
            mask = bits.encode(program, {k: choice})
            w.vector([j for j in range(len(bits.slots)) if mask >> j & 1])
    affine(w, plan.injection_records, lambda row: translate_injection(plan, row))
    w.vector(plan.injection.joint_rows, lambda row: w.row(translate_injection(plan, row)))
    for matrix in plan.injection.matrices:
        for value in matrix @ plan.injection.magic:
            w.complex(value)
    if plan.growth:
        payload(w, plan, plan.growth_payload)
        code(w, plan.growth_code)
        affine(w, plan.growth_records)
        w.vector(plan.instrument.input_rows, w.row)
        w.vector(plan.instrument.logical_rows, w.row)
        w.put(plan.instrument.random_power, plan.growth.y_sign)
        w.vector(plan.growth.data_rows)
        w.vector(plan.growth.duals, lambda pair: w.put(*pair))
        w.put(len(plan.growth.ancilla_rows))
        for k, q, bras in plan.growth.ancilla_rows:
            w.put(k, q)
            for value in bras.reshape(-1):
                w.complex(value)
        w.vector(plan.growth.unprojected)
        for value in plan.growth.decoder.reshape(-1):
            w.complex(value)
        w.put(len(plan.projectors))
        for k, targets, phases in plan.projectors:
            w.put(k)
            for target, phase in zip(targets, phases, strict=True):
                w.put(target)
                w.complex(phase)
    payload(w, plan, plan.terminal_payload)
    code(w, plan.terminal_code)
    w.vector(plan.terminal.duals, lambda pair: w.put(*pair))
    w.vector(plan.terminal.events)
    w.put(len(program.records))
    for event in program.records:
        item = program.measurements[event]
        site = program.site_at.get((item.line, item.offset))
        w.put(event, -1 if site is None else bits.slots[site, -1, "readout"])
    w.vector(program.detectors, w.vector)
    w.vector(list(program.observables.values()), w.vector)
    path.write_text("\n".join(w.tokens) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distance", type=int, choices=[3, 5])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(Sampler(load(args.distance)), args.output)
