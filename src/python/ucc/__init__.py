"""UCC - Universal Compiler Collection.

A multi-level AOT compiler and VM for quantum circuits (Clifford + T and beyond).
"""

from ucc._ucc_core import (
    AstNode,
    Circuit,
    GateType,
    ParseError,
    Target,
    max_sim_qubits,
    parse,
    parse_file,
    version,
)

__all__ = [
    "AstNode",
    "Circuit",
    "GateType",
    "ParseError",
    "Target",
    "max_sim_qubits",
    "parse",
    "parse_file",
    "version",
]

__version__ = version()
