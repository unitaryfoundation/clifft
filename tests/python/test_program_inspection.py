"""Tests for the Program inspection bindings (inspect / inspect_action)."""

import re

import pytest

import clifft

CIRCUIT = "H 0\nCX 0 1\nM 0 1"


def test_inspect_starts_with_header() -> None:
    program = clifft.compile(CIRCUIT)
    assert program.inspect().startswith("executable_plan backend=")


def test_inspect_is_deterministic_across_compiles() -> None:
    first = clifft.compile(CIRCUIT)
    second = clifft.compile(CIRCUIT)
    assert first.inspect() == second.inspect()


def test_inspect_action_starts_with_uppercase_mnemonic() -> None:
    program = clifft.compile(CIRCUIT)
    text = program.inspect_action(0)
    assert text
    assert re.match(r"^[A-Z][A-Z0-9_]+\b", text)


def test_inspect_action_out_of_range_raises_index_error() -> None:
    program = clifft.compile(CIRCUIT)
    with pytest.raises(IndexError):
        program.inspect_action(program.num_actions)
