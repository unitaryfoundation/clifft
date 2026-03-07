// Lightweight Monarch tokenizers for UCC's three editor panes.
// All rules are structural (regex patterns) rather than enumerative,
// so they automatically cover new gates, opcodes, or HIR ops without updates.

import type { languages } from "monaco-editor";
import type { Monaco } from "@monaco-editor/react";

// --- Stim circuit language ---
export const stimLanguage: languages.IMonarchLanguage = {
  tokenizer: {
    root: [
      // Comments
      [/#.*$/, "comment"],
      // Gate/instruction names at line start (ALL_CAPS with digits/underscores)
      [/^[A-Z][A-Z0-9_]*/, "keyword"],
      // Parenthesized noise probabilities: (0.001)
      [/\(/, "delimiter.parenthesis", "@parens"],
      // Braces for REPEAT blocks
      [/[{}]/, "delimiter.brace"],
      // Numeric literals (integers and floats)
      [/\b\d+\.\d+\b/, "number.float"],
      [/\b\d+\b/, "number"],
      // Record references: rec[...]
      [/rec\[/, "type", "@bracket"],
    ],
    parens: [
      [/\d+\.\d+/, "number.float"],
      [/\d+/, "number"],
      [/\)/, "delimiter.parenthesis", "@pop"],
    ],
    bracket: [
      [/\d+/, "number"],
      [/]/, "type", "@pop"],
    ],
  },
};

// --- Heisenberg IR language ---
export const hirLanguage: languages.IMonarchLanguage = {
  tokenizer: {
    root: [
      // Op-type keywords at line start
      [/^(T_DAG|T|S_DAG|S|MEASURE|IF|THEN|NOISE|READOUT_NOISE|DETECTOR|OBSERVABLE)\b/, "keyword"],
      // Pauli terms: +X0, -Z3, +I, -Y12*Z3
      [/[+-]/, "operator"],
      [/[XYZI]\d*/, "type"],
      [/\*/, "operator"],
      // Record/detector/observable refs: rec[...], det[...], obs[...]
      [/(rec|det|obs)\[/, "variable", "@bracket"],
      // Arrows
      [/->/, "operator"],
      // Annotations in parens: (hidden), (identity)
      [/\(\w+\)/, "comment"],
      // Numbers
      [/\b\d+\.\d+\b/, "number.float"],
      [/\b\d+\b/, "number"],
    ],
    bracket: [
      [/last/, "variable"],
      [/\d+/, "number"],
      [/]/, "variable", "@pop"],
    ],
  },
};

// --- VM Bytecode language ---
export const bytecodeLanguage: languages.IMonarchLanguage = {
  tokenizer: {
    root: [
      // Opcodes: OP_FRAME_CNOT, OP_EXPAND, OP_MEAS_ACTIVE_INTERFERE, etc.
      [/OP_[A-Z_]+/, "keyword"],
      // Record/detector/observable refs: rec[...], det[...], obs[...]
      [/(rec|det|obs)\[/, "variable", "@bracket"],
      // Named parameters: cp_mask=, cp_site=, cp_targets=, cp_entry=
      [/\b(cp_mask|cp_site|cp_targets|cp_entry)=/, "attribute"],
      // Keywords
      [/\b(if)\b/, "keyword"],
      // Arrows
      [/->/, "operator"],
      // Annotations in parens
      [/\(\w+\)/, "comment"],
      // Numbers
      [/\b\d+\.\d+\b/, "number.float"],
      [/\b\d+\b/, "number"],
    ],
    bracket: [
      [/\d+/, "number"],
      [/]/, "variable", "@pop"],
    ],
  },
};

// Track registration so we only register once even if beforeMount fires
// multiple times (e.g. StrictMode, multiple editors).
let registered = false;

export function registerLanguages(monaco: Monaco): void {
  if (registered) return;
  registered = true;

  monaco.languages.register({ id: "stim" });
  monaco.languages.setMonarchTokensProvider("stim", stimLanguage);

  monaco.languages.register({ id: "ucc-hir" });
  monaco.languages.setMonarchTokensProvider("ucc-hir", hirLanguage);

  monaco.languages.register({ id: "ucc-bytecode" });
  monaco.languages.setMonarchTokensProvider("ucc-bytecode", bytecodeLanguage);
}
