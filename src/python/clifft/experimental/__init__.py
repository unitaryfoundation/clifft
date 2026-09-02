"""Experimental APIs that may change without compatibility guarantees."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import clifft.experimental.cuda as cuda
    import clifft.experimental.hip as hip

_LAZY_SUBMODULES = frozenset({"cuda", "hip"})


def __getattr__(name: str) -> ModuleType:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"clifft.experimental.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cuda", "hip"]
