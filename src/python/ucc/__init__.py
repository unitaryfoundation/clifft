"""UCC - Unitary Compiler Collection.

A multi-level AOT compiler and Schrodinger Virtual Machine for quantum circuits
(Clifford + T and beyond).
"""

from ucc._ucc_core import (
    AstNode,
    Circuit,
    GateType,
    HirModule,
    ParseError,
    Pass,
    PassManager,
    PeepholeFusionPass,
    Program,
    State,
    Target,
    compile,
    default_pass_manager,
    execute,
    get_statevector,
    lower,
    max_sim_qubits,
    parse,
    parse_file,
    sample,
    trace,
    version,
)

__all__ = [
    "AstNode",
    "Circuit",
    "GateType",
    "HirModule",
    "Pass",
    "PassManager",
    "ParseError",
    "PeepholeFusionPass",
    "Program",
    "State",
    "Target",
    "compile",
    "default_pass_manager",
    "execute",
    "get_statevector",
    "lower",
    "max_sim_qubits",
    "parse",
    "parse_file",
    "sample",
    "trace",
    "version",
]

__version__ = version()
