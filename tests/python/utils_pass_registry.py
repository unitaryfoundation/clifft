"""Read the production pass inventory for documentation and coverage guards."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PASS_REGISTRY = ROOT / "src" / "clifft" / "optimizer" / "pass_registry.h"


def registered_hir_passes() -> dict[str, dict[str, object]]:
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
        default_match = re.search(r"\.default_enabled\s*=\s*(true|false)", entry)
        record_order_match = re.search(
            r"\.record_order\s*=\s*k(Preserves|Breaks)RecordOrder", entry
        )
        prefix_match = re.search(
            r"\.instrument_prefix\s*=\s*k(Preserves|MayChange)InstrumentPrefix", entry
        )

        assert name_match, f"Registered pass entry missing .name: {entry}"
        assert default_match, f"Registered pass entry missing .default_enabled: {entry}"
        assert record_order_match, f"Registered pass entry missing .record_order: {entry}"
        assert prefix_match, f"Registered pass entry missing .instrument_prefix: {entry}"

        passes[name_match.group(1)] = {
            "kind": "HIR",
            "default_enabled": default_match.group(1) == "true",
            "preserves_record_order": record_order_match.group(1) == "Preserves",
            "preserves_instrument_prefix": prefix_match.group(1) == "Preserves",
        }

    assert passes, f"No passes extracted from {PASS_REGISTRY}"
    return passes
