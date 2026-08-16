"""Ensure every HIR op type is documented in docs/opcodes.json.

This test parses the C++ HIR enum and verifies that docs/opcodes.json has a matching entry.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
HIR_HEADER = ROOT / "src" / "clifft" / "frontend" / "hir.h"
OPCODES_JSON = ROOT / "docs" / "opcodes.json"

# Sentinel values that should NOT have documentation entries
SENTINELS = {"NUM_OP_TYPES"}


def _extract_enum_members(header: Path, enum_name: str) -> list[str]:
    """Extract member names from a C++ enum class."""
    text = header.read_text()
    pattern = rf"enum\s+class\s+{enum_name}\s*:\s*\w+\s*\{{(.*?)\}}"
    match = re.search(pattern, text, re.DOTALL)
    assert match, f"Could not find 'enum class {enum_name}' in {header}"
    body = match.group(1)
    members = re.findall(r"^\s*(\w+)", body, re.MULTILINE)
    return [m for m in members if m not in SENTINELS]


@pytest.fixture(scope="module")
def opcodes_data() -> dict[str, Any]:
    with open(OPCODES_JSON) as f:
        return json.load(f)  # type: ignore[no-any-return]


class TestHirDocCompleteness:
    """Every C++ OpType enum member must have a docs/opcodes.json entry."""

    def test_all_hir_ops_documented(self, opcodes_data: dict[str, Any]) -> None:
        cpp_ops = set(_extract_enum_members(HIR_HEADER, "OpType"))
        json_ops = set(opcodes_data["hir_ops"].keys())
        missing = cpp_ops - json_ops
        assert not missing, (
            f"HIR OpTypes in C++ but missing from docs/opcodes.json: {sorted(missing)}. "
            f"Add entries for these op types to keep documentation complete."
        )

    def test_no_stale_hir_docs(self, opcodes_data: dict[str, Any]) -> None:
        cpp_ops = set(_extract_enum_members(HIR_HEADER, "OpType"))
        json_ops = set(opcodes_data["hir_ops"].keys())
        stale = json_ops - cpp_ops
        assert not stale, (
            f"HIR OpTypes in docs/opcodes.json but removed from C++: {sorted(stale)}. "
            f"Remove these stale entries."
        )


class TestJsonStructure:
    """Validate that each entry has the required fields."""

    def test_hir_entries_have_required_fields(self, opcodes_data: dict[str, Any]) -> None:
        for name, doc in opcodes_data["hir_ops"].items():
            assert "category" in doc, f"{name} missing 'category'"
            assert "summary" in doc, f"{name} missing 'summary'"
            assert "detail" in doc, f"{name} missing 'detail'"
