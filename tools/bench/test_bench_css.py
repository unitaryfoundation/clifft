"""Many-shot regressions for synthetic five-check logical-block controls."""

from pathlib import Path

import pytest

import clifft


@pytest.mark.parametrize("distance,enabled", [(7, False), (7, True), (9, True)])
def test_sample_css_five_checks(benchmark, distance, enabled):
    text = (Path(__file__).parent / f"fixtures/css_five_check_d{distance}.stim").read_text()
    program = clifft.compile(text, logical_blocks=enabled)
    assert program.num_css_blocks == (5 if enabled else 0)
    benchmark.pedantic(
        lambda: clifft.sample(program, 512, seed=781, threads=1, batch_size=1),
        rounds=3,
        iterations=1,
        warmup_rounds=1,
    )
