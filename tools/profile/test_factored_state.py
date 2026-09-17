"""Offline model checks; run with unittest discover in this directory."""

import unittest

from factored_state import analyze


def promote(width):
    return {"kind": "promote", "before": width, "after": width + 1, "dense_passes": 1}


def operation(kind, width, support, pivot=0):
    return {
        "kind": kind,
        "before": width,
        "after": width - (kind == "measure"),
        "support": support,
        "pivot": pivot,
        "dense_passes": 2 if kind == "measure" else 1,
    }


def plan(actions, peak=3, initial=0):
    return {"initial_width": initial, "peak_width": peak, "actions": actions}


class ProductAnalysisTest(unittest.TestCase):
    def test_independent_promotions(self):
        result = analyze(plan([promote(i) for i in range(12)], peak=12))
        self.assertEqual(result["factor_peak_live_coefficients"], 24)
        self.assertEqual(result["dense_peak_coefficients"], 4096)
        self.assertEqual(result["peak_component_width"], 1)
        self.assertEqual(result["merges"], 0)

    def test_noncontiguous_measurement_merge_and_coordinate_compaction(self):
        actions = [promote(i) for i in range(5)]
        actions += [operation("measure", 5, 0b10101, pivot=2)]
        # The old {0,4} component becomes {0,3}; rotating it must not merge again.
        actions += [operation("rotate", 4, 0b1001)]
        result = analyze(plan(actions, peak=5))
        self.assertEqual(result["merges"], 1)
        self.assertEqual(result["peak_component_width"], 3)
        self.assertEqual(result["trace"][-1]["components"], [2, 1, 1])
        self.assertEqual(result["factor_peak_live_coefficients"], 12)

    def test_measurement_does_not_assume_disentanglement(self):
        actions = [promote(i) for i in range(3)]
        actions += [operation("rotate", 3, 7), operation("measure", 3, 1)]
        result = analyze(plan(actions))
        self.assertEqual(result["trace"][-1]["components"], [2])
        self.assertEqual(result["merge_output_coefficients"], 8)

    def test_expectation_does_not_merge_factors(self):
        actions = [promote(i) for i in range(3)]
        actions += [operation("expectation", 3, 7)]
        result = analyze(plan(actions))
        self.assertEqual(result["merges"], 0)
        self.assertEqual(result["factor_unfused_visits"], 12)

    def test_unknown_initial_state_stays_monolithic(self):
        result = analyze(plan([operation("rotate", 3, 1)], initial=3))
        self.assertEqual(result["peak_component_width"], 3)

    def test_instruments_decline_instead_of_guessing(self):
        result = analyze(
            plan([{"kind": "unsupported_instrument", "before": 0, "after": 0, "dense_passes": 0}])
        )
        self.assertFalse(result["eligible"])

    def test_bad_support_and_unknown_actions_fail_closed(self):
        for action in [operation("rotate", 2, 4), operation("new_kind", 2, 1)]:
            with self.assertRaises(ValueError):
                analyze(plan([action], initial=2))


if __name__ == "__main__":
    unittest.main()
