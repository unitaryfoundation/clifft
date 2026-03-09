// Lightweight Monarch tokenizers for UCC's three editor panes.
// All rules are structural (regex patterns) rather than enumerative,
// so they automatically cover new gates, opcodes, or HIR ops without updates.

import type { languages, editor, IMarkdownString, Position } from "monaco-editor";
import type { Monaco } from "@monaco-editor/react";
import opcodesData from "@docs/opcodes.json";

interface OpDoc {
  category: string;
  summary: string;
  detail: string;
  operands?: string;
  display?: string[];
}

const opcodeMap = opcodesData.opcodes as Record<string, OpDoc>;
const hirMap = opcodesData.hir_ops as Record<string, OpDoc>;

// Build a reverse lookup from HIR display names (T, T_DAG, MEASURE, etc.) to docs
const hirDisplayMap: Record<string, OpDoc> = {};
for (const [, doc] of Object.entries(hirMap)) {
  if (doc.display) {
    for (const name of doc.display) {
      hirDisplayMap[name] = doc;
    }
  }
}

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

/** Format an opcode doc entry as Monaco-flavored Markdown for the hover widget. */
function formatOpcodeHover(name: string, doc: OpDoc): IMarkdownString {
  const lines = [
    `**\`${name}\`** &mdash; _${doc.category}_`,
    "",
    doc.summary,
  ];
  if (doc.detail) {
    lines.push("", doc.detail);
  }
  if (doc.operands) {
    lines.push("", `**Operands:** \`${doc.operands}\``);
  }
  return { value: lines.join("\n"), isTrusted: true };
}

export function registerLanguages(monaco: Monaco): void {
  if (registered) return;
  registered = true;

  monaco.languages.register({ id: "stim" });
  monaco.languages.setMonarchTokensProvider("stim", stimLanguage);

  monaco.languages.register({ id: "ucc-hir" });
  monaco.languages.setMonarchTokensProvider("ucc-hir", hirLanguage);

  monaco.languages.register({ id: "ucc-bytecode" });
  monaco.languages.setMonarchTokensProvider("ucc-bytecode", bytecodeLanguage);

  // --- Hover providers ---

  // VM Bytecode: hover over OP_* tokens
  monaco.languages.registerHoverProvider("ucc-bytecode", {
    provideHover(
      model: editor.ITextModel,
      position: Position,
    ) {
      const word = model.getWordAtPosition(position);
      if (!word) return null;

      // getWordAtPosition splits on underscore by default; expand to full OP_* token
      const line = model.getLineContent(position.lineNumber);
      const match = line.match(/OP_[A-Z_]+/);
      if (!match) return null;

      const opName = match[0];
      const startCol = match.index! + 1; // Monaco columns are 1-based
      const endCol = startCol + opName.length;

      // Only show hover if cursor is within the opcode token
      if (position.column < startCol || position.column > endCol) return null;

      const doc = opcodeMap[opName];
      if (!doc) return null;

      return {
        range: {
          startLineNumber: position.lineNumber,
          startColumn: startCol,
          endLineNumber: position.lineNumber,
          endColumn: endCol,
        },
        contents: [formatOpcodeHover(opName, doc)],
      };
    },
  });

  // HIR: hover over op-type keywords (T, T_DAG, S, S_DAG, MEASURE, etc.)
  monaco.languages.registerHoverProvider("ucc-hir", {
    provideHover(
      model: editor.ITextModel,
      position: Position,
    ) {
      const word = model.getWordAtPosition(position);
      if (!word) return null;

      // Expand to full keyword token (letters, digits, underscores)
      const line = model.getLineContent(position.lineNumber);
      const match = line.match(/^[A-Z][A-Z0-9_]*/);
      if (!match) return null;

      const kwName = match[0];
      const startCol = match.index! + 1;
      const endCol = startCol + kwName.length;

      if (position.column < startCol || position.column > endCol) return null;

      const doc = hirDisplayMap[kwName];
      if (!doc) return null;

      return {
        range: {
          startLineNumber: position.lineNumber,
          startColumn: startCol,
          endLineNumber: position.lineNumber,
          endColumn: endCol,
        },
        contents: [formatOpcodeHover(kwName, doc)],
      };
    },
  });
}
