"""Griffe extensions used by the API reference build."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from griffe import Attribute, Class, Extension, Function, Inspector, Module, visit


def _merge_stub_signatures(dynamic: Module | Class, stub: Module | Class) -> None:
    for name, stub_member in stub.members.items():
        dynamic_member = dynamic.members.get(name)

        if isinstance(dynamic_member, Function) and isinstance(stub_member, Function):
            dynamic_member.parameters = deepcopy(stub_member.parameters)
            dynamic_member.returns = deepcopy(stub_member.returns)
        elif isinstance(dynamic_member, Attribute) and isinstance(stub_member, Attribute):
            dynamic_member.annotation = deepcopy(stub_member.annotation)
        elif isinstance(dynamic_member, Class) and isinstance(stub_member, Class):
            if "__init__" not in stub_member.members and "__init__" not in stub_member.overloads:
                dynamic_member.members.pop("__init__", None)
            _merge_stub_signatures(dynamic_member, stub_member)


class NanobindStubSignatures(Extension):
    """Prefer generated stub signatures when Griffe inspects nanobind objects."""

    def on_module_members(
        self,
        *,
        node: Any,
        mod: Module,
        agent: Any,
        **kwargs: Any,
    ) -> None:
        if mod.path != "clifft._clifft_core" or not isinstance(agent, Inspector):
            return

        module_file = getattr(node.obj, "__file__", None)
        if module_file is None:
            raise RuntimeError("clifft._clifft_core has no module file")

        stub_path = Path(module_file).with_name("_clifft_core.pyi")
        if not stub_path.is_file():
            raise FileNotFoundError(f"nanobind stub not found: {stub_path}")

        stub = visit("_clifft_core_stub", stub_path, stub_path.read_text())
        _merge_stub_signatures(mod, stub)
