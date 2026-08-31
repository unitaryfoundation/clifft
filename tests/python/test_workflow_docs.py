"""Validate the workflow and backend contracts used to render user docs."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS = ROOT / "docs"
CONTRACTS_PATH = DOCS / "workflow_contracts.json"
HIP_EXECUTABLE_PLAN_HEADER = ROOT / "src" / "clifft" / "sampling" / "hip" / "executable_plan.h"

EXPECTED_WORKFLOWS = {
    "ordinary_sampling",
    "survivor_sampling",
    "basis_probabilities",
    "record_probabilities",
    "fixed_fault_sampling",
    "noncomputational_trajectories",
}
EXPECTED_WORKFLOW_GROUPS = {"sampling": 3, "exact": 2, "specialized": 1}
EXPECTED_INSPECTION_TOOLS = {"statevector"}
EXPECTED_BACKEND_STATUS = {
    "cpu": "Stable default",
    "hip": "Experimental",
    "cuda": "Planned, not implemented",
}


@pytest.fixture(scope="module")
def workflow_contracts() -> dict[str, Any]:
    """Load the metadata rendered into the workflow and GPU guides."""
    with open(CONTRACTS_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _resolve_public_api(dotted_name: str) -> object:
    """Resolve a public API path without assuming every submodule is eager."""
    parts = dotted_name.split(".")
    value: object = importlib.import_module(parts[0])
    for part in parts[1:]:
        value = getattr(value, part)
    return value


def test_contract_entries_are_complete(
    workflow_contracts: dict[str, Any],
) -> None:
    """Every documented route has user-facing content, APIs, and a valid guide."""
    workflows = workflow_contracts["workflows"]
    assert len(workflows) == len(EXPECTED_WORKFLOWS)
    assert {entry["id"] for entry in workflows} == EXPECTED_WORKFLOWS
    assert {
        group: sum(entry["group"] == group for entry in workflows)
        for group in EXPECTED_WORKFLOW_GROUPS
    } == EXPECTED_WORKFLOW_GROUPS

    for entry in workflows:
        for field in ("id", "group", "prompt", "route", "guide_label", "guide"):
            assert entry[field], f"{entry['id']} has an empty {field}"
            assert "\n" not in entry[field], f"{entry['id']} {field} must fit one route"
        assert entry["public_apis"], f"{entry['id']} has no public API contract"
        assert (DOCS / entry["guide"]).is_file(), f"{entry['id']} guide does not exist"

    inspection_tools = workflow_contracts["inspection_tools"]
    assert {entry["id"] for entry in inspection_tools} == EXPECTED_INSPECTION_TOOLS
    for entry in inspection_tools:
        for field in ("id", "purpose", "contract", "guide"):
            assert entry[field], f"{entry['id']} has an empty {field}"
        assert entry["public_apis"], f"{entry['id']} has no public API contract"
        assert (DOCS / entry["guide"]).is_file(), f"{entry['id']} guide does not exist"

    backends = workflow_contracts["backends"]
    assert len(backends) == len(EXPECTED_BACKEND_STATUS)
    assert {entry["id"]: entry["status"] for entry in backends} == EXPECTED_BACKEND_STATUS
    for entry in backends:
        for field in ("id", "name", "status", "selection", "distribution"):
            assert entry[field], f"{entry['id']} has an empty {field}"
            assert (
                "|" not in entry[field]
            ), f"{entry['id']} {field} contains an unescaped table pipe"
        assert bool(entry["public_apis"]) == (entry["id"] != "cuda")

    capabilities = workflow_contracts["backend_capabilities"]
    assert len({entry["feature"] for entry in capabilities}) == len(capabilities)
    for entry in capabilities:
        for field in ("feature", "cpu", "hip"):
            assert entry[field], f"capability has an empty {field}"
            assert "\n" not in entry[field], f"capability {field} must fit one table row"
            assert "|" not in entry[field], f"capability {field} contains an unescaped table pipe"


def test_every_documented_public_api_resolves(workflow_contracts: dict[str, Any]) -> None:
    """Catch stale names in workflow and backend routing before release."""
    documented_apis = {
        api
        for section in ("workflows", "inspection_tools", "backends")
        for entry in workflow_contracts[section]
        for api in entry["public_apis"]
    }

    for dotted_name in sorted(documented_apis):
        assert callable(_resolve_public_api(dotted_name)), f"{dotted_name} is not callable"


def test_hip_width_contract_matches_implementation(workflow_contracts: dict[str, Any]) -> None:
    """Keep the experimental HIP limit aligned with its lowering guard."""
    header = HIP_EXECUTABLE_PLAN_HEADER.read_text()
    match = re.search(r"kThreadPerShotMaxActiveWidth\s*=\s*(\d+)", header)
    assert match, "Could not find the HIP active-width limit"
    implementation_limit = int(match.group(1))

    backends = {entry["id"]: entry for entry in workflow_contracts["backends"]}
    documented_limit = backends["hip"]["limits"]["peak_active_width"]
    assert documented_limit == implementation_limit

    capability = next(
        entry
        for entry in workflow_contracts["backend_capabilities"]
        if entry["feature"] == "Peak active width"
    )
    assert capability["hip"] == f"`k <= {implementation_limit}`"


def test_contract_metadata_renders_the_canonical_routes_and_tables() -> None:
    """Prevent the validated data from becoming an unused sidecar."""
    chooser = (DOCS / "getting-started" / "choosing-a-workflow.md").read_text()
    gpu_guide = (DOCS / "guide" / "gpu-execution.md").read_text()

    assert chooser.count("{% for workflow in workflow_contracts['workflows']") == len(
        EXPECTED_WORKFLOW_GROUPS
    )
    for field in ("prompt", "route", "guide_label", "guide"):
        assert f"workflow['{field}']" in chooser
    assert "get_statevector" not in chooser
    assert "|---|---|---|---|\n{% for backend in workflow_contracts['backends'] -%}" in gpu_guide
    assert (
        "|---|---|---|\n"
        "{% for capability in workflow_contracts['backend_capabilities'] -%}" in gpu_guide
    )
