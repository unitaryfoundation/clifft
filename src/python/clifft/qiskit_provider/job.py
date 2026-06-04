"""ClifftJob: Qiskit JobV1 wrapper around a synchronous Clifft result."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from qiskit.providers import JobV1
from qiskit.providers.jobstatus import JobStatus

if TYPE_CHECKING:
    from qiskit.providers import Backend
    from qiskit.result import Result


class ClifftJob(JobV1):
    """Synchronous Qiskit job wrapping a pre-computed Clifft result.

    Because Clifft executes synchronously, the result is available
    immediately after construction.  ``status()`` always returns DONE
    and ``result()`` never blocks.
    """

    def __init__(self, backend: Backend, result: Result) -> None:
        super().__init__(backend, str(uuid.uuid4()))
        self._result = result

    def result(self) -> Result:
        return self._result

    def cancel(self) -> None:
        pass

    def status(self) -> JobStatus:
        return JobStatus.DONE

    def submit(self) -> None:
        pass
