export interface StarterCircuit {
  id: string;
  title: string;
  description: string;
  source: string;
}

export const STARTER_CIRCUITS: StarterCircuit[] = [
  {
    id: "bell-pair",
    title: "Bell pair",
    description: "Two-qubit entanglement with correlated Z measurements.",
    source: `H 0
CNOT 0 1
M 0 1
`,
  },
  {
    id: "ghz-state",
    title: "GHZ state",
    description: "Three-qubit fanout circuit with shared parity.",
    source: `H 0
CNOT 0 1
CNOT 1 2
M 0 1 2
`,
  },
  {
    id: "t-gate-probability",
    title: "T-gate interference",
    description: "T between Hadamards changes the final Z-basis probability.",
    source: `H 0
T 0
H 0
M 0
`,
  },
  {
    id: "zz-rotation",
    title: "ZZ rotation",
    description: "Arbitrary-angle two-qubit rotation outside the Clifford gate set.",
    source: `H 0
H 1
R_ZZ(0.25) 0 1
M 0 1
`,
  },
];
