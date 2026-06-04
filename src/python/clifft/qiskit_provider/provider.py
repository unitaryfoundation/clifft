"""ClifftProvider: minimal provider exposing ClifftBackend."""

from __future__ import annotations

from .backend import ClifftBackend


class ClifftProvider:
    """Provider for the Clifft near-Clifford simulator.

    Usage::

        from clifft.qiskit_provider import ClifftProvider

        provider = ClifftProvider()
        backend = provider.get_backend("clifft")
        job = backend.run(circuit, shots=1000)
        counts = job.result().get_counts()
    """

    def __init__(self) -> None:
        self._backend = ClifftBackend()

    def backends(self, name: str | None = None, **kwargs: object) -> list[ClifftBackend]:
        if name is None or name == "clifft":
            return [self._backend]
        return []

    def get_backend(self, name: str = "clifft", **kwargs: object) -> ClifftBackend:
        if name != "clifft":
            raise ValueError(f"Unknown backend '{name}'. Only 'clifft' is available.")
        return self._backend
