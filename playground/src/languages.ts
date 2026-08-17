// Lightweight Monarch tokenizers for Clifft's three editor panes.

import type { languages, editor, IMarkdownString, Position } from "monaco-editor";
import type { Monaco } from "@monaco-editor/react";
import hirMetadata from "@docs/opcodes.json";

interface OpDoc {
  category: string;
  summary: string;
  detail: string;
  operands?: string;
  display?: string[];
}

const hirMap = hirMetadata.hir_ops as Record<string, OpDoc>;

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

// --- SamplingPlan and prepared WASM program language ---
export const planLanguage: languages.IMonarchLanguage = {
  tokenizer: {
    root: [
      [/\b[serdl]\d+\b/, "variable"],
      [/\b(active_width|dense_passes|half_turns|sign|outcome|value|source|correction|kernel|mode|descriptor|pivot|pairing_bit|branch|record|detector|observable|exp_val|site|flip|p01|p10|postselected|cosine|sine|noise|symbol_prefix_size)=/, "attribute"],
      [/\b[a-z][a-z0-9_]*(?=\s|$)/, "keyword"],
      [/0x[0-9a-f]+/, "number.hex"],
      [/->/, "operator"],
      [/\.\./, "operator"],
      [/\b(?:\d+\.\d+(?:e[+-]?\d+)?|\d+e[+-]?\d+)\b/i, "number.float"],
      [/\b\d+\b/, "number"],
    ],
  },
};

// Track registration so we only register once even if beforeMount fires
// multiple times (e.g. StrictMode, multiple editors).
let registered = false;

/** Format an HIR operation as Monaco-flavored Markdown for the hover widget. */
function formatOperationHover(name: string, doc: OpDoc): IMarkdownString {
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

  monaco.languages.register({ id: "clifft-hir" });
  monaco.languages.setMonarchTokensProvider("clifft-hir", hirLanguage);

  monaco.languages.register({ id: "clifft-plan" });
  monaco.languages.setMonarchTokensProvider("clifft-plan", planLanguage);

  // --- Hover providers ---

  // HIR: hover over op-type keywords (T, T_DAG, S, S_DAG, MEASURE, etc.)
  monaco.languages.registerHoverProvider("clifft-hir", {
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
        contents: [formatOperationHover(kwName, doc)],
      };
    },
  });
}
