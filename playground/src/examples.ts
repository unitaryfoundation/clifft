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
    title: "T-gate probability",
    description: "Small non-Clifford circuit that activates Clifft's VM.",
    source: `H 0
T 0
H 0
M 0
`,
  },
  {
    id: "noisy-bell",
    title: "Noisy Bell",
    description: "Bell circuit with depolarizing noise before measurement.",
    source: `H 0
DEPOLARIZE1(0.01) 0
CNOT 0 1
DEPOLARIZE2(0.01) 0 1
M 0 1
`,
  },
];
