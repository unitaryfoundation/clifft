"""Opt-in behavioral coverage across declared sampling configurations.

Mark a test using sampling_api with sampling_conformance and its behaviors.
Each configuration must explicitly support or explain exclusion of every
registered behavior. Run the full suite with --sampling-coverage to audit the
collected matrix; add --collect-only to list its test IDs without execution.
"""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import pytest
from utils_conformance import CPU_SAMPLING_MODES, CpuSamplingMode

import clifft

FEATURES = frozenset(
    {
        "terminal-measurements",
        "bell-correlations",
        "nonclifford-probabilities",
        "measurement-reset",
        "readout-noise",
        "correlated-noise",
        "record-feedback",
        "syndrome-snapshots",
        "expectation-values",
        "survivor-accounting",
        "survivor-records",
    }
)

# Separate from the inventory so a new behavior requires a support decision.
# None means supported; a string explains why a configuration cannot run it.
CPU_SUPPORT: dict[str, str | None] = {
    "terminal-measurements": None,
    "bell-correlations": None,
    "nonclifford-probabilities": None,
    "measurement-reset": None,
    "readout-noise": None,
    "correlated-noise": None,
    "record-feedback": None,
    "syndrome-snapshots": None,
    "expectation-values": None,
    "survivor-accounting": None,
    "survivor-records": None,
}


@dataclass(frozen=True)
class SamplingConfiguration:
    api: CpuSamplingMode
    support: dict[str, str | None]

    @property
    def name(self) -> str:
        return self.api.name


CONFIGURATIONS = tuple(SamplingConfiguration(mode, CPU_SUPPORT) for mode in CPU_SAMPLING_MODES)
REQUIRED_CPU_MODES = {"single-shot": 1, "packed-65": 65, "automatic": "auto"}
_REPORT = pytest.StashKey[dict[tuple[str, str], list[str]]]()


def validate_inventory() -> None:
    modes = {configuration.name: configuration for configuration in CONFIGURATIONS}
    if len(modes) != len(CONFIGURATIONS):
        raise pytest.UsageError("Duplicate sampling configuration names")
    for name, batch_size in REQUIRED_CPU_MODES.items():
        if name not in modes or modes[name].api.batch_size != batch_size:
            raise pytest.UsageError(
                f"Required CPU sampling configuration missing or changed: {name}"
            )
    for configuration in CONFIGURATIONS:
        missing = FEATURES - configuration.support.keys()
        stale = configuration.support.keys() - FEATURES
        if missing or stale:
            raise pytest.UsageError(
                f"Sampling support decision required for {configuration.name}: "
                f"missing={sorted(missing)}, stale={sorted(stale)}"
            )
        for feature, reason in configuration.support.items():
            if reason is not None and (not isinstance(reason, str) or not reason.strip()):
                raise pytest.UsageError(
                    f"An exclusion reason is required: {configuration.name}/{feature}"
                )


def _features(node: Any) -> frozenset[str] | None:
    marks = list(node.iter_markers("sampling_conformance"))
    if not marks:
        return None
    if len(marks) != 1 or marks[0].kwargs:
        raise pytest.UsageError(
            f"Use one sampling_conformance marker with behavior names: {node.nodeid}"
        )
    features = frozenset(marks[0].args)
    if not features or features - FEATURES:
        raise pytest.UsageError(
            f"Unknown or empty sampling behaviors on {node.nodeid}: {sorted(features)}"
        )
    return features


def _compatible(features: frozenset[str]) -> list[SamplingConfiguration]:
    return [
        configuration
        for configuration in CONFIGURATIONS
        if all(configuration.support[feature] is None for feature in features)
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sampling-coverage",
        action="store_true",
        help="Require complete shared sampling coverage; use with the full Python suite.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "sampling_conformance(*behaviors): shared sampling semantic assertions"
    )


class _SamplingApi:
    def __init__(self, api: CpuSamplingMode):
        self._api = api
        self.sampled = False

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._api, name)
        if name not in {"sample", "sample_survivors"}:
            return attribute

        def sample(*args: Any, **kwargs: Any) -> Any:
            result = attribute(*args, **kwargs)
            self.sampled = True
            return result

        return sample


@pytest.fixture
def sampling_api(request: pytest.FixtureRequest) -> Iterator[Any]:
    if request.param is clifft:
        yield clifft
    else:
        api = _SamplingApi(request.param)
        yield api
        # A marker must not claim mode coverage when the test bypasses the
        # adapter and calls the default sampler directly.
        assert (
            api.sampled
        ), f"Shared test did not sample through its selected mode: {request.node.nodeid}"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    features = _features(metafunc.definition)
    if "sampling_api" not in metafunc.fixturenames:
        if features is not None:
            raise pytest.UsageError(
                f"Shared sampling test needs sampling_api: {metafunc.definition.nodeid}"
            )
        return
    if features is None:
        metafunc.parametrize("sampling_api", [clifft], ids=["symbolic-coordinate"], indirect=True)
        return
    validate_inventory()
    configurations = _compatible(features)
    if not configurations:
        raise pytest.UsageError(
            f"No supporting sampling configuration: {metafunc.definition.nodeid}"
        )
    metafunc.parametrize(
        "sampling_api",
        [configuration.api for configuration in configurations],
        ids=[configuration.name for configuration in configurations],
        indirect=True,
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("sampling_coverage"):
        return
    validate_inventory()
    coverage: dict[tuple[str, str], list[str]] = defaultdict(list)
    cases: dict[tuple[str, str], tuple[frozenset[str], set[str]]] = {}
    for item in items:
        features = _features(item)
        if features is None:
            continue
        callspec = cast(Any, item).callspec
        api = callspec.params["sampling_api"]
        if item.get_closest_marker("skip") or item.get_closest_marker("skipif"):
            raise pytest.UsageError(
                f"Declare sampling exclusions instead of skipping: {item.nodeid}"
            )
        if item.get_closest_marker("xfail"):
            raise pytest.UsageError(
                f"Expected failures do not establish sampling coverage: {item.nodeid}"
            )
        parameters = repr(
            sorted((k, v) for k, v in callspec.indices.items() if k != "sampling_api")
        )
        key = (item.nodeid.split("[")[0], parameters)
        cases.setdefault(key, (features, set()))[1].add(api.name)
        for feature in features:
            coverage[feature, api.name].append(item.nodeid)
    for (nodeid, parameters), (features, actual) in cases.items():
        expected = {configuration.name for configuration in _compatible(features)}
        if actual != expected:
            raise pytest.UsageError(
                f"Incomplete sampling modes for {nodeid} {parameters}: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
    for feature in sorted(FEATURES):
        for configuration in CONFIGURATIONS:
            if configuration.support[feature] is None and not coverage[feature, configuration.name]:
                raise pytest.UsageError(
                    f"No shared tests for supported behavior: {feature}/{configuration.name}"
                )
    config.stash[_REPORT] = dict(coverage)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    coverage = terminalreporter.config.stash.get(_REPORT, None)
    if coverage is None:
        return
    terminalreporter.section("Shared sampling coverage")
    for feature in sorted(FEATURES):
        for configuration in CONFIGURATIONS:
            reason = configuration.support[feature]
            nodeids = coverage.get((feature, configuration.name), [])
            summary = f"excluded: {reason}" if reason is not None else f"{len(nodeids)} tests"
            terminalreporter.write_line(f"{feature} / {configuration.name}: {summary}")
            if terminalreporter.config.option.collectonly:
                for nodeid in nodeids:
                    terminalreporter.write_line(f"  {nodeid}")
