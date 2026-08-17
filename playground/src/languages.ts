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

// --- SamplingPlan compact inspection language ---
// Grammar: w<k>[-><k'>] <MNEMONIC> [<pauli>] key=value... [postselect] [passes=<n>]
export const planLanguage: languages.IMonarchLanguage = {
  tokenizer: {
    root: [
      // Width prefix: w0 or w0->1
      [/\bw\d+(?:->\d+)?\b/, "keyword"],
      // Action mnemonics (explicit list; avoids matching arbitrary identifiers)
      [
        /\b(?:ROTATE_ACTIVE|ROTATE_PHASE|PROMOTE_DORMANT|MEASURE_ACTIVE|MEASURE_DORMANT|RECORD_CLASSICAL|DEFINE_SYMBOL|READOUT_NOISE|WRITE_DETECTOR|WRITE_OBSERVABLE|WRITE_EXPECTATION|APPLY_INSTRUMENT|INSTRUMENT_BOUNDARY)\b/,
        "keyword",
      ],
      // Flag words
      [/\b(?:postselect|zero)\b/, "keyword"],
      // Pauli factors: X0, Z1, Y2, and standalone identity I
      [/\b[XYZ]\d+\b/, "type"],
      [/\bI\b/, "type"],
      // Typed ids: symbols, records, detectors, observables, expectation slots
      [/\b[srdov]\d+\b/, "variable"],
      // Attribute keys (key= as one token, matching how they appear in the plan text)
      [
        /\b(?:half_turns|sign|outcome|value|source|record|detector|observable|exp_val|site|mode|flip|p01|p10|pivot|branch|next_noise_site|symbol_prefix_size|passes)=/,
        "attribute",
      ],
      // Truncated affine-expression tail: ...(+10)
      [/\.\.\.\(\+\d+\)/, "comment"],
      // Operators
      [/->/, "operator"],
      [/[*^]/, "operator"],
      // Numbers: integers, decimals, negatives, scientific notation
      [/[+-]?\b\d+\.\d+(?:e[+-]?\d+)?\b/i, "number.float"],
      [/[+-]?\b\d+e[+-]?\d+\b/i, "number.float"],
      [/[+-]?\b\d+\b/, "number"],
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
