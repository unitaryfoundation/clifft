"""Conservative research triage using compiled width and certified block cost.

This does not install an executor routing policy or predict wall-clock speed.
Small-control measurements justify a narrow fallback; larger in-budget cases
still require timing both implementations on the same complete circuit.
"""

import argparse
import json
import subprocess
from pathlib import Path

from fold_contraction import Plan
from msc_factor_sampling import FactorCode
from msc_sampling import Sampler
from study_msc_protocol import FIXTURES, load


def screen(peak_width, coefficient_bytes, complex_products, *, certified, budget=1 << 30):
    dense_bytes = 16 << peak_width
    if not certified:
        decision = "no certified block replacement"
    elif peak_width <= 10:
        decision = "keep ordinary Clifft based on measured small controls"
    elif dense_bytes > budget:
        decision = "bounded factor trial because dense allocation exceeds budget"
    else:
        decision = "benchmark both complete implementations"
    return {
        "peak_active_width": peak_width,
        "dense_coefficient_bytes": dense_bytes,
        "factor_coefficient_bytes": coefficient_bytes,
        "factor_complex_products": complex_products,
        "dense_budget_bytes": budget,
        "certified_family": certified,
        "decision": decision,
        "limitations": "Research triage only; no general speed prediction or production routing.",
    }


def study(baseline, factor_data):
    cases = []
    for distance in (3, 5):
        path = FIXTURES / f"msc_d{distance}_inject_cultivate_p1e-3.stim"
        planned = json.loads(
            subprocess.check_output([str(baseline), str(path), "0", "plan"], text=True)
        )
        sampler = Sampler(load(distance))
        profile = FactorCode(sampler.terminal.checks, sampler.terminal.width).statistics()
        cases.append(
            {
                "name": f"original MSC d{distance}",
                **screen(
                    planned["peak_active_width"],
                    16 * max(p["coefficient_slots"] for p in profile["prefixes"]),
                    20 * sum(p["gather_entries"] for p in profile["prefixes"]),
                    certified="diagonal monomial CSS blocks",
                ),
            }
        )
    fixture = FIXTURES.parent / "fold_cultivation_f7.stim"
    planned = json.loads(
        subprocess.check_output([str(baseline), str(fixture), "0", "plan"], text=True)
    )
    fold = Plan(7).statistics()
    cases.append(
        {
            "name": "reconstructed f7",
            "scope": "Existing fold contraction, not the CSS factor sampler",
            **screen(
                planned["peak_active_width"],
                16 * fold["coefficient_slots"],
                fold["complex_multiplications_per_input"],
                certified="existing fold blocks",
            ),
        }
    )
    for profile in factor_data:
        cases.append(
            {
                "name": f"synthetic {profile['data_width']}-data-qubit planning block",
                **screen(
                    profile["original_gate_plan"]["peak_active_width"],
                    16 * max(p["coefficient_slots"] for p in profile["prefixes"]),
                    20 * sum(p["gather_entries"] for p in profile["prefixes"]),
                    certified="diagonal monomial CSS blocks",
                ),
            }
        )
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--factor-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(study(args.baseline, json.loads(args.factor_data.read_text())), indent=2) + "\n"
    )
