"""UCC - Universal Compiler Collection.

A multi-level AOT compiler and VM for quantum circuits (Clifford + T and beyond).
"""

from ucc._ucc_core import max_sim_qubits, version

__all__ = [
    "version",
    "max_sim_qubits",
]

__version__ = version()
