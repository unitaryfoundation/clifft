"""Ensure optimization pass docs stay aligned with the C++ pass registry."""

import ast
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
PASS_REGISTRY = ROOT / "src" / "clifft" / "optimizer" / "pass_registry.h"
DOCS_MACROS = ROOT / "docs" / "macros.py"


def _extract_cpp_passes() -> dict[str, dict[str, object]]:
    """Extract registered optimization pass metadata from pass_registry.h."""
    text = PASS_REGISTRY.read_text()
    match = re.search(
        r"inline\s+const\s+PassInfo\s+kRegisteredPasses\[\]\s*=\s*\{(.*?)\};",
        text,
        re.DOTALL,
    )
    assert match, f"Could not find kRegisteredPasses in {PASS_REGISTRY}"

    passes: dict[str, dict[str, object]] = {}
    for entry_match in re.finditer(r"\{(.*?)\},", match.group(1), re.DOTALL):
        entry = entry_match.group(1)
        name_match = re.search(r'\.name\s*=\s*"([^"]+)"', entry)
        kind_match = re.search(r"\.kind\s*=\s*PassKind::(\w+)", entry)
        default_match = re.search(r"\.default_enabled\s*=\s*(true|false)", entry)

        assert name_match, f"Registered pass entry missing .name: {entry}"
        assert kind_match, f"Registered pass entry missing .kind: {entry}"
        assert default_match, f"Registered pass entry missing .default_enabled: {entry}"

        passes[name_match.group(1)] = {
            "kind": kind_match.group(1),
            "default_enabled": default_match.group(1) == "true",
        }

    return passes


def _extract_docs_passes() -> dict[str, dict[str, Any]]:
    """Extract optimization pass metadata from docs/macros.py."""
    tree = ast.parse(DOCS_MACROS.read_text(), filename=str(DOCS_MACROS))
    pass_entries: list[dict[str, Any]] | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "passes" for target in node.targets
        ):
            continue
        pass_entries = ast.literal_eval(node.value)
        break

    assert pass_entries is not None, f"Could not find passes metadata in {DOCS_MACROS}"
    return {entry["name"]: entry for entry in pass_entries}


class TestOptimizationPassDocCompleteness:
    """Every registered optimization pass must be documented."""

    def test_all_registered_passes_documented(self) -> None:
        cpp_passes = set(_extract_cpp_passes())
        docs_passes = set(_extract_docs_passes())
        missing = cpp_passes - docs_passes
        assert not missing, (
            f"Passes in pass_registry.h but missing from docs/macros.py: {sorted(missing)}. "
            f"Add entries for these passes to keep optimization docs complete."
        )

    def test_no_stale_pass_docs(self) -> None:
        cpp_passes = set(_extract_cpp_passes())
        docs_passes = set(_extract_docs_passes())
        stale = docs_passes - cpp_passes
        assert not stale, (
            f"Passes in docs/macros.py but removed from pass_registry.h: {sorted(stale)}. "
            f"Remove these stale docs entries."
        )

    def test_pass_kind_and_default_enabled_metadata_match(self) -> None:
        cpp_passes = _extract_cpp_passes()
        docs_passes = _extract_docs_passes()

        for name, cpp_metadata in cpp_passes.items():
            assert name in docs_passes, f"{name} missing from docs/macros.py"
            docs_metadata = docs_passes[name]
            assert docs_metadata["kind"] == cpp_metadata["kind"], (
                f"{name} has kind {docs_metadata['kind']!r} in docs/macros.py but "
                f"{cpp_metadata['kind']!r} in pass_registry.h."
            )
            assert docs_metadata["default_enabled"] == cpp_metadata["default_enabled"], (
                f"{name} has default_enabled={docs_metadata['default_enabled']!r} in "
                f"docs/macros.py but {cpp_metadata['default_enabled']!r} in pass_registry.h."
            )


class TestOptimizationPassDocStructure:
    """Validate optimization pass docs entries have the required fields."""

    def test_pass_entries_have_required_fields(self) -> None:
        required_fields = {"name", "kind", "default_enabled", "python_name", "summary", "detail"}

        for name, metadata in _extract_docs_passes().items():
            missing = required_fields - set(metadata)
            assert not missing, f"{name} missing required docs metadata fields: {sorted(missing)}"
