import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Allotment } from "allotment";
import "allotment/dist/style.css";
import Editor, { type BeforeMount, type OnMount } from "@monaco-editor/react";
import type { editor as monacoEditor } from "monaco-editor";
import LZString from "lz-string";
import { useUccWasm } from "./hooks/useUccWasm";
import { useTheme } from "./hooks/useTheme";
import { Toolbar } from "./components/Toolbar";
import { KHistoryChart } from "./components/KHistoryChart";
import { HistogramChart } from "./components/HistogramChart";
import { GuidedTour } from "./components/GuidedTour";
import { registerLanguages } from "./languages";
import { isCompileSuccess, isSimulateSuccess } from "./types";
import type { CompileResult, CompileSuccess, SimulateResult } from "./types";

const DEFAULT_SOURCE = `H 0
CNOT 0 1
T 0
T 1
M 0
M 1
`;

const DEBOUNCE_MS = 200;
const DEFAULT_SHOTS = 10000;
const TOUR_SEEN_KEY = "ucc-tour-seen";

function getInitialSource(): string {
  const params = new URLSearchParams(window.location.search);
  const code = params.get("code");
  if (code) {
    try {
      const decoded = LZString.decompressFromEncodedURIComponent(code);
      if (decoded) return decoded;
    } catch {
      // ignore bad encoded data
    }
  }
  return DEFAULT_SOURCE;
}

// Decoration style for highlighted lines
const HIGHLIGHT_CLASS = "source-map-highlight";

export default function App() {
  const { status: wasmStatus, compile, simulate } = useUccWasm();
  const { theme, toggle: toggleTheme, chartColors } = useTheme();
  const monacoTheme = theme === "dark" ? "vs-dark" : "vs";
  const rulerColor = theme === "dark" ? "#4fc3f7" : "#0277bd";

  const [source, setSource] = useState(getInitialSource);
  const [optimize, setOptimize] = useState(true);
  const [compileResult, setCompileResult] = useState<CompileResult | null>(null);
  const [simResult, setSimResult] = useState<SimulateResult | null>(null);
  const [simElapsedMs, setSimElapsedMs] = useState<number | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [shots, setShots] = useState(DEFAULT_SHOTS);
  const [cursorSourceLine, setCursorSourceLine] = useState<number | null>(null);
  const [tourOpen, setTourOpen] = useState(
    () => !localStorage.getItem(TOUR_SEEN_KEY),
  );

  // Editor refs
  const sourceEditorRef = useRef<monacoEditor.IStandaloneCodeEditor | null>(null);
  const hirEditorRef = useRef<monacoEditor.IStandaloneCodeEditor | null>(null);
  const bcEditorRef = useRef<monacoEditor.IStandaloneCodeEditor | null>(null);

  // Decoration collections
  const hirDecosRef = useRef<monacoEditor.IEditorDecorationsCollection | null>(null);
  const bcDecosRef = useRef<monacoEditor.IEditorDecorationsCollection | null>(null);
  const bcPcDecosRef = useRef<monacoEditor.IEditorDecorationsCollection | null>(null);
  const sourceDecosRef = useRef<monacoEditor.IEditorDecorationsCollection | null>(null);

  // Ref to avoid stale closures in editor cursor handlers
  const compileResultRef = useRef<CompileResult | null>(null);

  // Derived: which bytecode PC maps to the cursor source line (for k-history highlight)
  const highlightPC = useMemo(() => {
    if (!compileResult || !isCompileSuccess(compileResult) || cursorSourceLine === null)
      return null;
    const idx = compileResult.bytecode_source_map.findIndex((lines: number[]) =>
      lines.includes(cursorSourceLine),
    );
    return idx >= 0 ? idx : null;
  }, [cursorSourceLine, compileResult]);

  // --- Debounced compilation ---
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (wasmStatus !== "ready") return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      const result = compile(source, optimize);
      setCompileResult(result);
    }, DEBOUNCE_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [source, optimize, wasmStatus, compile]);

  // Keep ref in sync for cursor handlers that close over mount-time state
  useEffect(() => {
    compileResultRef.current = compileResult;
  }, [compileResult]);

  // Register custom languages once before any editor mounts
  const handleBeforeMount: BeforeMount = useCallback((monaco) => {
    registerLanguages(monaco);
  }, []);

  // Update read-only editor text while preserving cursor/selection
  const updateReadOnlyEditor = useCallback(
    (editor: monacoEditor.IStandaloneCodeEditor | null, value: string) => {
      if (!editor) return;
      const sel = editor.getSelection();
      editor.setValue(value);
      if (sel) editor.setSelection(sel);
    },
    [],
  );

  // --- Update read-only editors when compile result changes ---
  useEffect(() => {
    if (!compileResult || !isCompileSuccess(compileResult)) {
      const errorText = compileResult?.error ? `; Error: ${compileResult.error}` : "";
      updateReadOnlyEditor(hirEditorRef.current, errorText);
      updateReadOnlyEditor(bcEditorRef.current, errorText);
      return;
    }
    updateReadOnlyEditor(hirEditorRef.current, compileResult.hir_ops.join("\n"));
    updateReadOnlyEditor(bcEditorRef.current, compileResult.bytecode.join("\n"));
  }, [compileResult, updateReadOnlyEditor]);

  // --- Bidirectional highlighting ---
  useEffect(() => {
    if (!compileResult || !isCompileSuccess(compileResult) || cursorSourceLine === null) {
      hirDecosRef.current?.clear();
      bcDecosRef.current?.clear();
      bcPcDecosRef.current?.clear();
      sourceDecosRef.current?.clear();
      return;
    }

    const srcLine = cursorSourceLine;

    // Find HIR ops that map to this source line
    const hirLines: number[] = [];
    compileResult.hir_source_map.forEach((lines: number[], i: number) => {
      if (lines.includes(srcLine)) hirLines.push(i + 1); // Monaco is 1-indexed
    });

    // Find bytecode instructions that map to this source line
    const bcLines: number[] = [];
    compileResult.bytecode_source_map.forEach((lines: number[], i: number) => {
      if (lines.includes(srcLine)) bcLines.push(i + 1);
    });

    // Highlight the source line itself
    sourceDecosRef.current?.set([
      {
        range: { startLineNumber: srcLine, startColumn: 1, endLineNumber: srcLine, endColumn: 1 },
        options: {
          isWholeLine: true,
          className: HIGHLIGHT_CLASS,
          overviewRuler: { color: rulerColor, position: 2 },
        },
      },
    ]);

    hirDecosRef.current?.set(
      hirLines.map((ln) => ({
        range: { startLineNumber: ln, startColumn: 1, endLineNumber: ln, endColumn: 1 },
        options: {
          isWholeLine: true,
          className: HIGHLIGHT_CLASS,
          overviewRuler: { color: rulerColor, position: 2 },
        },
      })),
    );

    // Blue scrollbar markers on bytecode (consistent with source/HIR)
    bcDecosRef.current?.set(
      bcLines.map((ln) => ({
        range: { startLineNumber: ln, startColumn: 1, endLineNumber: ln, endColumn: 1 },
        options: {
          isWholeLine: true,
          className: HIGHLIGHT_CLASS,
          overviewRuler: { color: rulerColor, position: 2 },
        },
      })),
    );

    // Yellow gutter glyph on the first matching bytecode line (PC for k-history)
    if (bcLines.length > 0) {
      bcPcDecosRef.current?.set([
        {
          range: {
            startLineNumber: bcLines[0],
            startColumn: 1,
            endLineNumber: bcLines[0],
            endColumn: 1,
          },
          options: {
            glyphMarginClassName: "pc-glyph-marker",
          },
        },
      ]);
    } else {
      bcPcDecosRef.current?.clear();
    }
  }, [cursorSourceLine, compileResult, rulerColor]);

  // --- Editor mount handlers ---
  const onSourceMount: OnMount = (editor) => {
    sourceEditorRef.current = editor;
    sourceDecosRef.current = editor.createDecorationsCollection();
    editor.onDidChangeCursorPosition((e) => {
      setCursorSourceLine(e.position.lineNumber);
    });
  };

  const onHirMount: OnMount = (editor) => {
    hirEditorRef.current = editor;
    hirDecosRef.current = editor.createDecorationsCollection();
    editor.onDidChangeCursorPosition((e) => {
      const cr = compileResultRef.current;
      if (!cr || !isCompileSuccess(cr)) return;
      const hirIdx = e.position.lineNumber - 1;
      const srcLines = cr.hir_source_map[hirIdx];
      // Filter out sentinel 0 values (Monaco lines are 1-based)
      const validLine = srcLines?.find((l: number) => l >= 1);
      if (validLine !== undefined) {
        setCursorSourceLine(validLine);
      }
    });
  };

  const onBcMount: OnMount = (editor) => {
    bcEditorRef.current = editor;
    bcDecosRef.current = editor.createDecorationsCollection();
    bcPcDecosRef.current = editor.createDecorationsCollection();
    editor.onDidChangeCursorPosition((e) => {
      const cr = compileResultRef.current;
      if (!cr || !isCompileSuccess(cr)) return;
      const bcIdx = e.position.lineNumber - 1;
      const srcLines = cr.bytecode_source_map[bcIdx];
      const validLine = srcLines?.find((l: number) => l >= 1);
      if (validLine !== undefined) {
        setCursorSourceLine(validLine);
      }
    });
  };

  // --- Simulation ---
  const handleSimulate = useCallback(() => {
    setSimulating(true);
    // Defer to next frame so the UI updates before blocking on Wasm
    requestAnimationFrame(() => {
      const t0 = performance.now();
      const result = simulate(source, shots, optimize);
      const elapsed = performance.now() - t0;
      setSimResult(result);
      setSimElapsedMs(elapsed);
      setSimulating(false);
    });
  }, [source, shots, optimize, simulate]);

  const handleTourClose = useCallback(() => {
    setTourOpen(false);
    localStorage.setItem(TOUR_SEEN_KEY, "1");
  }, []);

  // --- Stats bar ---
  const stats: CompileSuccess | null =
    compileResult && isCompileSuccess(compileResult) ? compileResult : null;

  return (
    <div className="app">
      <Toolbar
        wasmStatus={wasmStatus}
        source={source}
        optimize={optimize}
        onOptimizeChange={setOptimize}
        onSimulate={handleSimulate}
        simulating={simulating}
        shots={shots}
        onShotsChange={setShots}
        onTourOpen={() => setTourOpen(true)}
        theme={theme}
        onThemeToggle={toggleTheme}
      />

      {stats && (
        <div className="stats-bar">
          <span>Qubits: {stats.num_qubits}</span>
          <span>Peak k: {stats.peak_rank}</span>
          <span>T gates: {stats.num_t_gates}</span>
          <span>Measurements: {stats.num_measurements}</span>
          <span>HIR ops: {stats.hir_ops.length}</span>
          <span>Bytecode: {stats.bytecode.length}</span>
        </div>
      )}

      <div className="workspace">
        <Allotment vertical defaultSizes={[65, 35]}>
          {/* Top: 3-column editors */}
          <Allotment.Pane>
            <Allotment defaultSizes={[34, 33, 33]}>
              <Allotment.Pane>
                <div className="editor-pane">
                  <div className="editor-label">Source (.stim)</div>
                  <Editor
                    defaultLanguage="stim"
                    defaultValue={source}
                    theme={monacoTheme}
                    onChange={(v) => setSource(v ?? "")}
                    beforeMount={handleBeforeMount}
                    onMount={onSourceMount}
                    options={{
                      minimap: { enabled: false },
                      fontSize: 13,
                      lineNumbers: "on",
                      scrollBeyondLastLine: false,
                      wordWrap: "off",
                      automaticLayout: true,
                    }}
                  />
                </div>
              </Allotment.Pane>
              <Allotment.Pane>
                <div className="editor-pane">
                  <div className="editor-label">HIR (Heisenberg IR)</div>
                  <Editor
                    defaultLanguage="ucc-hir"
                    defaultValue=""
                    theme={monacoTheme}
                    beforeMount={handleBeforeMount}
                    onMount={onHirMount}
                    options={{
                      readOnly: true,
                      minimap: { enabled: false },
                      fontSize: 13,
                      lineNumbers: "on",
                      scrollBeyondLastLine: false,
                      wordWrap: "off",
                      automaticLayout: true,
                      domReadOnly: true,
                    }}
                  />
                </div>
              </Allotment.Pane>
              <Allotment.Pane>
                <div className="editor-pane">
                  <div className="editor-label">VM Bytecode (RISC)</div>
                  <Editor
                    defaultLanguage="ucc-bytecode"
                    defaultValue=""
                    theme={monacoTheme}
                    beforeMount={handleBeforeMount}
                    onMount={onBcMount}
                    options={{
                      readOnly: true,
                      minimap: { enabled: false },
                      fontSize: 13,
                      lineNumbers: "on",
                      scrollBeyondLastLine: false,
                      wordWrap: "off",
                      automaticLayout: true,
                      domReadOnly: true,
                      glyphMargin: true,
                    }}
                  />
                </div>
              </Allotment.Pane>
            </Allotment>
          </Allotment.Pane>

          {/* Bottom: k-history chart + simulation results */}
          <Allotment.Pane>
            <Allotment defaultSizes={[50, 50]}>
              <Allotment.Pane>
                <div className="chart-pane">
                  <div className="chart-label">Active Dimensions (k) Timeline</div>
                  <div className="chart-container">
                    <KHistoryChart
                      history={stats?.active_k_history ?? []}
                      highlightPC={highlightPC}
                      colors={chartColors}
                    />
                  </div>
                </div>
              </Allotment.Pane>
              <Allotment.Pane>
                <div className="chart-pane">
                  <div className="chart-label">
                    Simulation Results
                    {simResult && isSimulateSuccess(simResult) && (
                      <span className="chart-label-detail">
                        {" "}({simResult.shots.toLocaleString()} shots)
                      </span>
                    )}
                  </div>
                  <div className="chart-container">
                    <HistogramChart result={simResult} elapsedMs={simElapsedMs} colors={chartColors} />
                  </div>
                </div>
              </Allotment.Pane>
            </Allotment>
          </Allotment.Pane>
        </Allotment>
      </div>
      {tourOpen && <GuidedTour onClose={handleTourClose} />}
    </div>
  );
}
